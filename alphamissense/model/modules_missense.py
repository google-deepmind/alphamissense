# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and code used in the core part of AlphaMissense."""

import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import tree
from alphamissense.common import residue_constants
from alphamissense.model import common_modules
from alphamissense.model import folding_multimer
from alphamissense.model import layer_stack
from alphamissense.model import modules
from alphamissense.model import prng
from alphamissense.model import utils

FeatureDict = MutableMapping[str, jnp.ndarray]
OutputDict = MutableMapping[str, Any]


def gumbel_noise(key: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  """Generate Gumbel Noise of given Shape.

  This generates samples from Gumbel(0, 1).

  Args:
    key: Jax random number key.
    shape: Shape of noise to return.

  Returns:
    Gumbel noise of given shape.
  """
  epsilon = 1e-6
  uniform = utils.padding_consistent_rng(jax.random.uniform)
  uniform_noise = uniform(
      key, shape=shape, dtype=jnp.float32, minval=0., maxval=1.)
  gumbel = -jnp.log(-jnp.log(uniform_noise + epsilon) + epsilon)
  return gumbel


def gumbel_argsort_sample_idx(key: jnp.ndarray,
                              logits: jnp.ndarray) -> jnp.ndarray:
  """Samples with replacement from a distribution given by 'logits'.

  This uses Gumbel trick to implement the sampling an efficient manner. For a
  distribution over k items this samples k times without replacement, so this
  is effectively sampling a random permutation with probabilities over the
  permutations derived from the logprobs.

  Args:
    key: prng key.
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.

  Returns:
    Sample from logprobs in one-hot form.
  """
  z = gumbel_noise(key, logits.shape)
  # This construction is equivalent to jnp.argsort, but using a non stable sort,
  # since stable sort's aren't supported by jax2tf.
  axis = len(logits.shape) - 1
  iota = jax.lax.broadcasted_iota(jnp.int64, logits.shape, axis)
  _, perm = jax.lax.sort_key_val(
      logits + z, iota, dimension=-1, is_stable=False)
  return perm[::-1]


def sample_msa(key: prng.SafeKey, batch: FeatureDict, max_seq: int
               ) -> FeatureDict:
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.

  Args:
    key: safe key for random number generation.
    batch: batch to sample msa from.
    max_seq: number of sequences to sample.
  Returns:
    Protein with sampled msa.
  """
  # Sample uniformly among sequences with at least one non-masked position.
  logits = (jnp.clip(jnp.sum(batch['msa_mask'], axis=-1), 0., 1.) - 1.) * 1e6
  # cluster_bias_mask can be used to ensure that the marked MSA rows are always
  # sampled and their positions are preserved. In AlphaMissense this ensures
  # that first row is always the target sequence and second row is the
  # masked variant row.
  cluster_bias_mask = batch['cluster_bias_mask']
  logits += cluster_bias_mask * 1e6
  index_order = gumbel_argsort_sample_idx(key.get(), logits)
  sel_idx = jnp.where(cluster_bias_mask[:max_seq],
                      jnp.arange(max_seq),
                      index_order[:max_seq])
  extra_idx = index_order[max_seq:]

  for k in ('msa', 'deletion_matrix', 'msa_mask', 'bert_mask', 'true_msa'):
    if k in batch:
      if k == 'true_msa':
        # extra_msa has no BERT mask
        batch['extra_msa'] = batch[k][extra_idx]
      if k not in ['msa']:
        batch['extra_' + k] = batch[k][extra_idx]
      batch[k] = batch[k][sel_idx]

  return batch


def create_extra_msa_feature(batch: FeatureDict, num_extra_msa: int
                             ) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Arguments:
    batch: a dictionary with the following keys:
     * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
       centre. Note, that this is not one-hot encoded.
     * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
       the left of each position in the extra MSA.
     * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
       the left of each position in the extra MSA.
    num_extra_msa: Number of sequences to include in the extra MSA features.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  deletion_matrix = batch['extra_deletion_matrix'][:num_extra_msa]
  extra_msa_1hot = jax.nn.one_hot(batch['extra_msa'][:num_extra_msa], 23)
  has_deletion = jnp.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (jnp.arctan(deletion_matrix / 3.) * (2. / jnp.pi))[..., None]
  extra_msa_mask = batch['extra_msa_mask'][:num_extra_msa]
  return jnp.concatenate([extra_msa_1hot, has_deletion, deletion_value],
                         axis=-1), extra_msa_mask


def clipped_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    clip_negative_at_logit: float,
    clip_positive_at_logit: float,
    epsilon: float = 1e-07,
    ) -> jnp.ndarray:
  """Computes sigmoid xent loss with clipped input logits.

  Args:
    logits: The predicted values.
    labels: The ground truth values.
    clip_negative_at_logit: clip the loss to 0 if prediction smaller than this
      value for the negative class.
    clip_positive_at_logit: clip the loss to this value if prediction smaller
      than this value for the positive class.
    epsilon: A small increment to add to avoid taking a log of zero.

  Returns:
    Loss value.
  """
  prob = jax.nn.sigmoid(logits)
  prob = jnp.clip(prob, epsilon, 1. - epsilon)
  loss = -labels * jnp.log(
      prob) - (1. - labels) * jnp.log(1. - prob)
  loss_at_clip = math.log(math.exp(clip_negative_at_logit) + 1)
  loss = jnp.where(
      (1 - labels) * (logits < clip_negative_at_logit), loss_at_clip, loss)
  loss_at_clip = math.log(math.exp(-clip_positive_at_logit) + 1)
  loss = jnp.where(
      labels * (logits < clip_positive_at_logit), loss_at_clip, loss)
  return loss


class LogitDiffPathogenicityHead(hk.Module):
  """Variant pathogenicity classification head."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'logit_diff_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.num_output = len(residue_constants.restypes_with_x_and_gap)
    self.variant_row = 1

  def __call__(self,
               representations: FeatureDict,
               batch: FeatureDict,
               is_training: bool
               ) -> FeatureDict:
    logits = common_modules.Linear(
        self.num_output,
        initializer='linear',
        name='logits')(
            representations['msa'][self.variant_row])

    ref_score = jnp.einsum('ij, ij->i', logits, jax.nn.one_hot(
        batch['aatype'], num_classes=self.num_output))
    variant_score = jnp.einsum('ij, ij->i', logits, jax.nn.one_hot(
        batch['variant_aatype'], num_classes=self.num_output))
    logit_diff = ref_score - variant_score
    variant_pathogenicity = jnp.sum(logit_diff * batch['variant_mask'])
    return {'variant_row_logit_diff': logit_diff,
            'variant_pathogenicity': variant_pathogenicity}

  def loss(self, value: FeatureDict, batch: FeatureDict) -> jnp.ndarray:
    loss = clipped_sigmoid_cross_entropy(logits=value['variant_row_logit_diff'],
                                         labels=batch['pathogenicity'],
                                         clip_negative_at_logit=0.0,
                                         clip_positive_at_logit=-1.0)
    loss = (jnp.sum(loss * batch['variant_mask'], axis=(-2, -1)) /
            (1e-8 + jnp.sum(batch['variant_mask'], axis=(-2, -1))))
    return loss


class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file. Each head also returns a
  loss which is combined as a weighted sum to produce the total loss.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch: FeatureDict,
               is_training: bool,
               safe_key: prng.SafeKey,
               ) -> OutputDict:

    # Compute representations for each batch element and average.
    evoformer_module = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)
    representations = evoformer_module(batch, is_training, safe_key)

    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa': modules.MaskedMsaHead,
          'distogram': modules.DistogramHead,
          'structure_module': folding_multimer.StructureModule,
          'logit_diff': LogitDiffPathogenicityHead,
      }[head_name]
      self.heads[head_name] = (head_config,
                               head_factory(head_config, self.global_config))
    ret = {}
    ret['representations'] = representations
    for name, (_, module) in self.heads.items():
      ret[name] = module(representations, batch, is_training)
    return ret


class AlphaFold(hk.Module):
  """AlphaFold model with recycling.

  Changes relative to the original AlphaFold 2 implementation are described in
  the Methods section of Cheng et al. (2023).
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               name: str = 'alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch: FeatureDict,
      *,
      is_training: bool,
      return_representations: bool,
      ) -> OutputDict:

    impl = AlphaFoldIteration(self.config, self.global_config)
    num_residues = batch['aatype'].shape[0]
    safe_key = prng.SafeKey(hk.next_rng_key())

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          is_training=is_training,
          safe_key=safe_key)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      prev['prev_pos'] = jnp.zeros(
          [num_residues, residue_constants.atom_type_num, 3])
    if emb_config.recycle_features:
      prev['prev_pair'] = jnp.zeros(
          [num_residues, num_residues, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # Value for each ensemble batch is the same, so arbitrarily taking 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that even when ensembling, we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, self.config.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = self.config.num_recycle

      def recycle_body(x):
        i, prev, safe_key = x
        safe_key1, safe_key2 = safe_key.split()
        ret = apply_network(prev=prev, safe_key=safe_key2)
        return i + 1, get_prev(ret), safe_key1

      if hk.running_init():
        num_recycles, prev, safe_key = recycle_body((0, prev, safe_key))
      else:
        num_recycles, prev, safe_key = hk.while_loop(
            lambda x: x[0] < num_iter,
            recycle_body,
            (0, prev, safe_key))
    else:
      # No recycling.
      num_recycles = 0

    # Run extra iteration.
    ret = apply_network(prev=prev, safe_key=safe_key)

    if not return_representations:
      del ret['representations']
    ret['num_recycles'] = num_recycles

    return ret


class EmbeddingsAndEvoformer(hk.Module):
  """Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  Changes relative to the original AlphaFold 2 implementation are described in
  the Methods section of Cheng et al. (2023).
  """

  def __init__(self, config, global_config, name='evoformer'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(self, batch: FeatureDict) -> jnp.ndarray:
    """Add relative Position encodings.

    Feed the Pair stack a relative feature. For position (i, j), its value
    is (i-j) clipped to [-k, k] and one-hotted.

    When not using 'use_chain_relative' the residue indices are used as is, e.g.
    for heteromers relative positions will be computed using the positions in
    the corresponding chains.

    When using 'use_chain_relative' we add an extra bin that denotes
    'different chain'. Furthermore we also provide the relative chain index
    (i.e. sym_id) clipped and one-hotted to the network. And an extra feature
    which denotes whether they belong to the same chain type, i.e. it's 0 if
    they are in different heteromer chains and 1 otherwise.

    Args:
      batch: batch.
    Returns:
      Feature embedding using the features as described before.
    """
    c = self.config
    rel_feats = []
    pos = batch['residue_index']
    offset = pos[:, None] - pos[None, :]

    clipped_offset = jnp.clip(
        offset + c.max_relative_idx, a_min=0, a_max=2 * c.max_relative_idx)

    if c.use_chain_relative:
      asym_id = batch['seq_mask'] * batch.get(
          'asym_id', jnp.ones_like(batch['residue_index']))
      asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])

      final_offset = jnp.where(asym_id_same, clipped_offset,
                               (2 * c.max_relative_idx + 1) *
                               jnp.ones_like(clipped_offset))

      rel_pos = jax.nn.one_hot(final_offset, 2 * c.max_relative_idx + 2)

      rel_feats.append(rel_pos)

      entity_id = batch['seq_mask'] * batch.get(
          'entity_id', jnp.ones_like(batch['residue_index']))
      entity_id_same = jnp.equal(entity_id[:, None], entity_id[None, :])
      rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

      sym_id = batch['seq_mask'] * batch.get(
          'sym_id', jnp.ones_like(batch['residue_index']))
      rel_sym_id = sym_id[:, None] - sym_id[None, :]

      max_rel_chain = c.max_relative_chain

      clipped_rel_chain = jnp.clip(
          rel_sym_id + max_rel_chain, a_min=0, a_max=2 * max_rel_chain)

      final_rel_chain = jnp.where(entity_id_same, clipped_rel_chain,
                                  (2 * max_rel_chain + 1) *
                                  jnp.ones_like(clipped_rel_chain))
      rel_chain = jax.nn.one_hot(final_rel_chain, 2 * c.max_relative_chain + 2)

      rel_feats.append(rel_chain)

    else:
      rel_pos = jax.nn.one_hot(clipped_offset, 2 * c.max_relative_idx + 1)
      rel_feats.append(rel_pos)

    rel_feat = jnp.concatenate(rel_feats, axis=-1)
    return common_modules.Linear(
        c.pair_channel, name='position_activations')(
            rel_feat.astype(jnp.float32))

  def __call__(self,
               batch: FeatureDict,
               is_training: bool,
               safe_key: prng.SafeKey
               ) -> FeatureDict:

    c = self.config
    gc = self.global_config

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    num_residues = batch['aatype'].shape[0]
    target_feat = jax.nn.one_hot(batch['aatype'], 21)
    preprocess_1d = jnp.zeros((num_residues, c.msa_channel))
    left_single = common_modules.Linear(
        c.pair_channel, name='left_single')(target_feat)
    right_single = common_modules.Linear(
        c.pair_channel, name='right_single')(target_feat)

    profile_feat = jnp.concatenate([
        batch['msa_profile'], batch['deletion_mean'][..., None]], axis=-1)
    preprocess_1d += common_modules.Linear(
        c.msa_channel, name='profile_preprocess_1d',
        use_bias=False, initializer='zeros')(profile_feat)
    left_single += common_modules.Linear(
        c.pair_channel, name='profile_left_single',
        use_bias=False, initializer='zeros')(profile_feat)
    right_single += common_modules.Linear(
        c.pair_channel, name='profile_right_single',
        use_bias=False, initializer='zeros')(profile_feat)

    safe_key, sample_key = safe_key.split()
    batch = sample_msa(sample_key, batch, c.num_msa)
    preprocess_msa = common_modules.Linear(
        c.msa_channel, name='preprocess_msa')(jax.nn.one_hot(batch['msa'], 23))
    msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa
    pair_activations = left_single[:, None] + right_single[None]
    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

    if 'prev_pos' in batch:
      prev_pseudo_beta = modules.pseudo_beta_fn(batch['aatype'],
                                                batch['prev_pos'],
                                                None)
      dgram = modules.dgram_from_positions(prev_pseudo_beta,
                                           **self.config.prev_pos)
      pair_activations += common_modules.Linear(c.pair_channel,
                                                name='prev_pos_linear')(
                                                    dgram)

    if 'prev_pair' in batch:
      pair_activations += common_modules.LayerNorm(axis=[-1],
                                                   create_scale=True,
                                                   create_offset=True,
                                                   name='prev_pair_norm')(
                                                       batch['prev_pair'])

    pair_activations += self._relative_encoding(batch)

    if c.enable_extra_msa_stack:
      extra_msa_feat, extra_msa_mask = create_extra_msa_feature(batch,
                                                                c.num_extra_msa)
      extra_msa_activations = common_modules.Linear(
          c.extra_msa_channel,
          name='extra_msa_activations')(
              extra_msa_feat)

      # Extra MSA Stack.
      # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
      extra_msa_stack_input = dict(msa=extra_msa_activations,
                                   pair=pair_activations)
      extra_msa_stack_iteration = modules.EvoformerIteration(
          c.evoformer, gc, is_extra_msa=True, name='extra_msa_stack')

      def extra_msa_stack_fn(x):
        act, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        extra_evoformer_output = extra_msa_stack_iteration(
            activations=act,
            masks=dict(msa=extra_msa_mask, pair=mask_2d),
            is_training=is_training,
            safe_key=safe_subkey)
        return (extra_evoformer_output, safe_key)

      if gc.use_remat:
        extra_msa_stack_fn = hk.remat(extra_msa_stack_fn)

      extra_msa_stack = layer_stack.layer_stack(
          c.extra_msa_stack_num_block)(
              extra_msa_stack_fn)
      extra_msa_output, safe_key = extra_msa_stack(
          (extra_msa_stack_input, safe_key))
      pair_activations = extra_msa_output['pair']

    evoformer_input = dict(msa=msa_activations, pair=pair_activations)
    evoformer_masks = dict(msa=batch['msa_mask'], pair=mask_2d)

    evoformer_iteration = modules.EvoformerIteration(
        c.evoformer, gc, is_extra_msa=False, name='evoformer_iteration')

    def evoformer_fn(x):
      act, safe_key = x
      safe_key, safe_subkey = safe_key.split()
      evoformer_output = evoformer_iteration(
          activations=act,
          masks=evoformer_masks,
          is_training=is_training,
          safe_key=safe_subkey)
      return (evoformer_output, safe_key)

    if gc.use_remat:
      evoformer_fn = hk.remat(evoformer_fn)

    evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(
        evoformer_fn)
    evoformer_output, _ = evoformer_stack((evoformer_input, safe_key))

    msa_activations = evoformer_output['msa']
    pair_activations = evoformer_output['pair']
    diag_act = jnp.diagonal(pair_activations, axis1=0, axis2=1).T
    single_activations = common_modules.Linear(
        c.seq_channel, name='single_act_from_pair_diag')(diag_act)

    return dict(single=single_activations,
                pair=pair_activations,
                msa=msa_activations)


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               *,
               params: Optional[Mapping[str, Mapping[str, jax.Array]]]):
    self.config = config
    self.params = params
    def _forward_fn(batch):
      model = AlphaFold(self.config.model)
      return model(batch, is_training=False, return_representations=False)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = hk.transform(_forward_fn).init

  def init_params(self, feat: FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        pipeline_missense.DataPipeline.process.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def eval_shape(self, feat: FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.info('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def predict(self,
              feat: FeatureDict,
              random_seed: int,
              ) -> OutputDict:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        pipeline_missense.DataPipeline.process.
      random_seed: The random seed to use when running the model. In the
        multimer model this controls the MSA sampling.

    Returns:
      A dictionary of model outputs.
    """
    self.init_params(feat)
    logging.info('Running predict with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)

    # This block is to ensure benchmark timings are accurate. Some blocking is
    # already happening when computing get_confidence_metrics, and this ensures
    # all outputs are blocked on.
    jax.tree_map(lambda x: x.block_until_ready(), result)
    return result
