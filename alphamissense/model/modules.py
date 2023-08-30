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

"""Modules and code used in the core part of AlphaFold.

The structure generation code is in 'folding.py'.
"""

import functools
from typing import Any, MutableMapping, Union

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
from alphamissense.common import residue_constants
from alphamissense.model import common_modules
from alphamissense.model import mapping
from alphamissense.model import prng
from alphamissense.model import utils

_SOFTMAX_MASK = -1e9

FeatureDict = MutableMapping[str, jnp.ndarray]


def softmax_cross_entropy(logits: jnp.ndarray,
                          labels: jnp.ndarray
                          ) -> jnp.ndarray:
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)


def apply_dropout(*,
                  tensor: jnp.ndarray,
                  safe_key: prng.SafeKey,
                  rate: float,
                  is_training: bool,
                  broadcast_dim: int | None = None,
                  ) -> jnp.ndarray:
  """Applies dropout to a tensor."""
  if is_training and rate != 0.0:
    shape = list(tensor.shape)
    if broadcast_dim is not None:
      shape[broadcast_dim] = 1
    keep_rate = 1.0 - rate
    keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
    return keep * tensor / keep_rate
  else:
    return tensor


def dropout_wrapper(module: Any,
                    input_act: jnp.ndarray,
                    mask: jnp.ndarray,
                    safe_key: prng.SafeKey,
                    global_config: ml_collections.ConfigDict,
                    output_act: jnp.ndarray | None = None,
                    is_training: bool = True,
                    **kwargs
                    ) -> jnp.ndarray:
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, mask, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

  # Will override `is_training` to True if want to use dropout.
  should_apply_dropout = True if gc.eval_dropout else is_training

  if module.config.shared_dropout:
    if module.config.orientation == 'per_row':
      broadcast_dim = 0
    else:
      broadcast_dim = 1
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=should_apply_dropout,
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act


class Transition(hk.Module):
  """Transition layer.

  Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
  Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'transition_block'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               act: jnp.ndarray,
               mask: jnp.ndarray,
               is_training: bool = True
               ) -> jnp.ndarray:
    """Builds Transition module.

    Arguments:
      act: A tensor of queries of size [batch_size, N_res, N_channel].
      mask: A tensor denoting the mask of size [batch_size, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      A float32 tensor of size [batch_size, N_res, N_channel].
    """
    del mask  # Not applied since operation is element-wise.
    _, _, nc = act.shape

    num_intermediate = int(nc * self.config.num_intermediate_factor)

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='input_layer_norm')(
            act)

    transition_module = hk.Sequential([
        common_modules.Linear(
            num_intermediate,
            initializer='relu',
            name='transition1'), jax.nn.relu,
        common_modules.Linear(
            nc,
            initializer=utils.final_init(self.global_config),
            name='transition2')
    ])

    act = mapping.inference_subbatch(
        transition_module,
        self.global_config.subbatch_size,
        batched_args=[act],
        nonbatched_args=[],
        low_memory=not is_training)

    return act


def glorot_uniform():
  return hk.initializers.VarianceScaling(scale=1.0,
                                         mode='fan_avg',
                                         distribution='uniform')


class Attention(hk.Module):
  """Multihead attention."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               output_dim: int,
               name: str = 'attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim

  def __call__(self,
               q_data: jnp.ndarray,
               m_data: jnp.ndarray,
               mask: jnp.ndarray,
               nonbatched_bias: jnp.ndarray | None = None
               ) -> jnp.ndarray:
    """Builds Attention module.

    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      mask: A mask for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].

    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())

    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k)
    if nonbatched_bias is not None:
      logits += jnp.expand_dims(nonbatched_bias, axis=0)
    logits = jnp.where(mask, logits, _SOFTMAX_MASK)
    weights = utils.stable_softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                               gating_weights) + gating_bias

      gate_values = jax.nn.sigmoid(gate_values)

      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)
    o_bias = hk.get_parameter(
        'output_b', shape=(self.output_dim,),
        dtype=q_data.dtype,
        init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output


class GlobalAttention(hk.Module):
  """Global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               output_dim: int,
               name: str = 'attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim

  def __call__(self,
               q_data: jnp.ndarray,
               m_data: jnp.ndarray,
               q_mask: jnp.ndarray,
               ) -> jnp.ndarray:
    """Builds GlobalAttention module.

    Arguments:
      q_data: A tensor of queries with size [batch_size, N_queries,
        q_channels]
      m_data: A tensor of memories from which the keys and values
        projected. Size [batch_size, N_keys, m_channels]
      q_mask: A binary mask for q_data with zeros in the padded sequence
        elements and ones otherwise. Size [batch_size, N_queries, q_channels]
        (or broadcastable to this shape).

    Returns:
      A float32 tensor of size [batch_size, N_queries, output_dim].
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())

    v = jnp.einsum('bka,ac->bkc', m_data, v_weights)

    q_avg = utils.mask_mean(q_mask, q_data, axis=1)

    q = jnp.einsum('ba,ahc->bhc', q_avg, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ac->bkc', m_data, k_weights)
    bias = q_mask[:, None, :, 0]
    logits = jnp.einsum('bhc,bkc->bhk', q, k)
    logits = jnp.where(bias, logits, _SOFTMAX_MASK)
    weights = utils.stable_softmax(logits)
    weighted_avg = jnp.einsum('bhk,bkc->bhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)
    o_bias = hk.get_parameter(
        'output_b', shape=(self.output_dim,),
        dtype=q_data.dtype,
        init=hk.initializers.Constant(0.0))

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data, gating_weights)
      gate_values = jax.nn.sigmoid(gate_values + gating_bias)
      weighted_avg = weighted_avg[:, None] * gate_values
      output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias
    else:
      output = jnp.einsum('bhc,hco->bo', weighted_avg, o_weights) + o_bias
      output = output[:, None]
    return output


class MSARowAttentionWithPairBias(hk.Module):
  """MSA per-row attention biased by the pair representation.

  Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'msa_row_attention_with_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act: jnp.ndarray,
               msa_mask: jnp.ndarray,
               pair_act: jnp.ndarray,
               is_training: bool = False
               ) -> jnp.ndarray:
    """Builds MSARowAttentionWithPairBias module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      pair_act: [N_res, N_res, c_z] pair representation.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_row'

    mask = msa_mask[:, None, None, :]
    assert len(mask.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    pair_act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='feat_2d_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=msa_act.dtype,
        init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)

    attn_mod = Attention(
        c, self.global_config, msa_act.shape[-1])
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, mask],
        nonbatched_args=[nonbatched_bias],
        low_memory=not is_training)

    return msa_act


class MSAColumnAttention(hk.Module):
  """MSA per-column attention.

  Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'msa_column_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act: jnp.ndarray,
               msa_mask: jnp.ndarray,
               is_training: bool = False
               ) -> jnp.ndarray:
    """Builds MSAColumnAttention module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m]
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_column'

    msa_act = jnp.swapaxes(msa_act, -2, -3)
    msa_mask = jnp.swapaxes(msa_mask, -1, -2)

    mask = msa_mask[:, None, None, :]
    assert len(mask.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    attn_mod = Attention(
        c, self.global_config, msa_act.shape[-1])
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, mask],
        nonbatched_args=[],
        low_memory=not is_training)

    msa_act = jnp.swapaxes(msa_act, -2, -3)

    return msa_act


class MSAColumnGlobalAttention(hk.Module):
  """MSA per-column global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'msa_column_global_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act: jnp.ndarray,
               msa_mask: jnp.ndarray,
               is_training: bool = False
               ) -> jnp.ndarray:
    """Builds MSAColumnGlobalAttention module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_column'

    msa_act = jnp.swapaxes(msa_act, -2, -3)
    msa_mask = jnp.swapaxes(msa_mask, -1, -2)

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    attn_mod = GlobalAttention(
        c, self.global_config, msa_act.shape[-1],
        name='attention')
    # [N_seq, N_res, 1]
    msa_mask = jnp.expand_dims(msa_mask, axis=-1)
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, msa_mask],
        nonbatched_args=[],
        low_memory=not is_training)

    msa_act = jnp.swapaxes(msa_act, -2, -3)

    return msa_act


class TriangleAttention(hk.Module):
  """Triangle Attention.

  Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
  Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
  """

  def __init__(self, config, global_config, name='triangle_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask, is_training=False):
    """Builds TriangleAttention module.

    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    c = self.config

    assert len(pair_act.shape) == 3
    assert len(pair_mask.shape) == 2
    assert c.orientation in ['per_row', 'per_column']

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)
      pair_mask = jnp.swapaxes(pair_mask, -1, -2)

    mask = pair_mask[:, None, None, :]
    assert len(mask.shape) == 4

    pair_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=pair_act.dtype,
        init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)

    attn_mod = Attention(
        c, self.global_config, pair_act.shape[-1])
    pair_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[pair_act, pair_act, mask],
        nonbatched_args=[nonbatched_bias],
        low_memory=not is_training)

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)

    return pair_act


class MaskedMsaHead(hk.Module):
  """Head to predict MSA at the masked locations.

  The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
  version of the full MSA, based on a linear projection of
  the MSA representation.
  Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'masked_msa_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

    if global_config.multimer_mode:
      self.num_output = len(residue_constants.restypes_with_x_and_gap)
    else:
      self.num_output = config.num_output

  def __call__(self,
               representations: FeatureDict,
               batch: FeatureDict,
               is_training: bool,
               ) -> FeatureDict:
    """Builds MaskedMsaHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'msa': MSA representation, shape [N_seq, N_res, c_m].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_seq, N_res, N_aatype] with
            (unnormalized) log probabilies of predicted aatype at position.
    """
    del batch
    logits = common_modules.Linear(
        self.num_output,
        initializer=utils.final_init(self.global_config),
        name='logits')(
            representations['msa'])
    return dict(logits=logits)

  def loss(self, value: FeatureDict, batch: FeatureDict) -> jnp.ndarray:
    errors = softmax_cross_entropy(
        labels=jax.nn.one_hot(batch['true_msa'], num_classes=self.num_output),
        logits=value['logits'])
    loss = (jnp.sum(errors * batch['bert_mask'], axis=(-2, -1)) /
            (1e-8 + jnp.sum(batch['bert_mask'], axis=(-2, -1))))
    return loss


def _layer_norm(axis: int = -1, name: str = 'layer_norm'):
  return common_modules.LayerNorm(
      axis=axis,
      create_scale=True,
      create_offset=True,
      eps=1e-5,
      use_fast_variance=True,
      scale_init=hk.initializers.Constant(1.),
      offset_init=hk.initializers.Constant(0.),
      param_axis=axis,
      name=name)


class TriangleMultiplication(hk.Module):
  """Triangle multiplication layer ("outgoing" or "incoming").

  Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
  Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name: str = 'triangle_multiplication'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               left_act: jnp.ndarray,
               left_mask: jnp.ndarray,
               is_training: bool = True
               ) -> jnp.ndarray:
    """Builds TriangleMultiplication module.

    Arguments:
      left_act: Pair activations, shape [N_res, N_res, c_z]
      left_mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Outputs, same shape/type as left_act.
    """
    del is_training

    if self.config.fuse_projection_weights:
      return self._fused_triangle_multiplication(left_act, left_mask)
    else:
      return self._triangle_multiplication(left_act, left_mask)

  @hk.transparent
  def _triangle_multiplication(self,
                               left_act: jnp.ndarray,
                               left_mask: jnp.ndarray
                               ) -> jnp.ndarray:
    """Implementation of TriangleMultiplication used in AF2 and AF-M<2.3."""
    c = self.config
    gc = self.global_config

    mask = left_mask[..., None]

    act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True,
        name='layer_norm_input')(left_act)
    input_act = act

    left_projection = common_modules.Linear(
        c.num_intermediate_channel,
        name='left_projection')
    left_proj_act = mask * left_projection(act)

    right_projection = common_modules.Linear(
        c.num_intermediate_channel,
        name='right_projection')
    right_proj_act = mask * right_projection(act)

    left_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='left_gate')(act))

    right_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='right_gate')(act))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    # "Outgoing" edges equation: 'ikc,jkc->ijc'
    # "Incoming" edges equation: 'kjc,kic->ijc'
    # Note on the Suppl. Alg. 11 & 12 notation:
    # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
    # For the "incoming" edges, it's swapped:
    #   b = left_proj_act and a = right_proj_act
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='center_layer_norm')(
            act)

    output_channel = int(input_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = jax.nn.sigmoid(common_modules.Linear(
        output_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='gating_linear')(input_act))
    act *= gate_values

    return act

  @hk.transparent
  def _fused_triangle_multiplication(self,
                                     left_act: jnp.ndarray,
                                     left_mask: jnp.ndarray,
                                     ) -> jnp.ndarray:
    """TriangleMultiplication with fused projection weights."""
    mask = left_mask[..., None]
    c = self.config
    gc = self.global_config

    left_act = _layer_norm(axis=-1, name='left_norm_input')(left_act)

    # Both left and right projections are fused into projection.
    projection = common_modules.Linear(
        2*c.num_intermediate_channel, name='projection')
    proj_act = mask * projection(left_act)

    # Both left + right gate are fused into gate_values.
    gate_values = common_modules.Linear(
        2 * c.num_intermediate_channel,
        name='gate',
        bias_init=1.,
        initializer=utils.final_init(gc))(left_act)
    proj_act *= jax.nn.sigmoid(gate_values)

    left_proj_act = proj_act[:, :, :c.num_intermediate_channel]
    right_proj_act = proj_act[:, :, c.num_intermediate_channel:]
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = _layer_norm(axis=-1, name='center_norm')(act)

    output_channel = int(left_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = common_modules.Linear(
        output_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='gating_linear')(left_act)
    act *= jax.nn.sigmoid(gate_values)

    return act


class DistogramHead(hk.Module):
  """Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               name='distogram_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               representations: FeatureDict,
               batch: FeatureDict,
               is_training: bool
               ) -> FeatureDict:
    """Builds DistogramHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(
            representations['pair'])

    logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
    breaks = jnp.linspace(self.config.first_break, self.config.last_break,
                          self.config.num_bins - 1)

    return dict(logits=logits, bin_edges=breaks)

  def loss(self,
           value: FeatureDict,
           batch: FeatureDict
           ) -> jnp.ndarray:
    return _distogram_log_loss(value['logits'], value['bin_edges'],
                               batch, self.config.num_bins)


def _distogram_log_loss(logits: jnp.ndarray,
                        bin_edges: jnp.ndarray,
                        batch: FeatureDict,
                        num_bins: int
                        ) -> jnp.ndarray:
  """Log loss of a distogram."""

  assert len(logits.shape) == 3
  positions = batch['pseudo_beta']
  mask = batch['pseudo_beta_mask']

  assert positions.shape[-1] == 3

  sq_breaks = jnp.square(bin_edges)

  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2) -
          jnp.expand_dims(positions, axis=-3)),
      axis=-1,
      keepdims=True)

  true_bins = jnp.sum(dist2 > sq_breaks, axis=-1)

  errors = softmax_cross_entropy(
      labels=jax.nn.one_hot(true_bins, num_bins), logits=logits)

  square_mask = jnp.expand_dims(mask, axis=-2) * jnp.expand_dims(mask, axis=-1)

  avg_error = (
      jnp.sum(errors * square_mask, axis=(-2, -1)) /
      (1e-6 + jnp.sum(square_mask, axis=(-2, -1))))
  return avg_error


class OuterProductMean(hk.Module):
  """Computes mean outer product.

  Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               num_output_channel: int,
               name: str = 'outer_product_mean'):
    super().__init__(name=name)
    self.global_config = global_config
    self.config = config
    self.num_output_channel = num_output_channel

  def __call__(self,
               act: jnp.ndarray,
               mask: jnp.ndarray,
               is_training: bool = True
               ) -> jnp.ndarray:
    """Builds OuterProductMean module.

    Arguments:
      act: MSA representation, shape [N_seq, N_res, c_m].
      mask: MSA mask, shape [N_seq, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair representation, shape [N_res, N_res, c_z].
    """
    gc = self.global_config
    c = self.config

    mask = mask[..., None]
    act = common_modules.LayerNorm(
        [-1], True, True, name='layer_norm_input')(act)

    left_act = mask * common_modules.Linear(
        c.num_outer_channel,
        initializer='linear',
        name='left_projection')(
            act)

    right_act = mask * common_modules.Linear(
        c.num_outer_channel,
        initializer='linear',
        name='right_projection')(
            act)

    if gc.zero_init:
      init_w = hk.initializers.Constant(0.0)
    else:
      init_w = hk.initializers.VarianceScaling(scale=2., mode='fan_in')

    output_w = hk.get_parameter(
        'output_w',
        shape=(c.num_outer_channel, c.num_outer_channel,
               self.num_output_channel),
        dtype=act.dtype,
        init=init_w)
    output_b = hk.get_parameter(
        'output_b', shape=(self.num_output_channel,),
        dtype=act.dtype,
        init=hk.initializers.Constant(0.0))

    def compute_chunk(left_act):
      # This is equivalent to
      #
      # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
      # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
      #
      # but faster.
      left_act = jnp.transpose(left_act, [0, 2, 1])
      act = jnp.einsum('acb,ade->dceb', left_act, right_act)
      act = jnp.einsum('dceb,cef->dbf', act, output_w) + output_b
      return jnp.transpose(act, [1, 0, 2])

    act = mapping.inference_subbatch(
        compute_chunk,
        c.chunk_size,
        batched_args=[left_act],
        nonbatched_args=[],
        low_memory=True,
        input_subbatch_dim=1,
        output_subbatch_dim=0)

    epsilon = 1e-3
    norm = jnp.einsum('abc,adc->bdc', mask, mask)
    act /= epsilon + norm

    return act


def dgram_from_positions(positions: jnp.ndarray,
                         num_bins: int, min_bin: float, max_bin: float
                         ) -> jnp.ndarray:
  """Compute distogram from amino acid positions.

  Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.

  Returns:
    Distogram with the specified number of bins.
  """

  def squared_difference(x, y):
    return jnp.square(x - y)

  lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate([lower_breaks[1:],
                                  jnp.array([1e8], dtype=jnp.float32)], axis=-1)
  dist2 = jnp.sum(
      squared_difference(
          jnp.expand_dims(positions, axis=-2),
          jnp.expand_dims(positions, axis=-3)),
      axis=-1, keepdims=True)

  dgram = ((dist2 > lower_breaks).astype(jnp.float32) *
           (dist2 < upper_breaks).astype(jnp.float32))
  return dgram


def pseudo_beta_fn(aatype: jnp.ndarray,
                   all_atom_positions: jnp.ndarray,
                   all_atom_masks: jnp.ndarray | None,
                   ) -> Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
  """Create pseudo beta features."""

  is_gly = jnp.equal(aatype, residue_constants.restype_order['G'])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = jnp.where(
      jnp.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = jnp.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(jnp.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta


class EvoformerIteration(hk.Module):
  """Single iteration (block) of Evoformer stack.

  Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
  """

  def __init__(self,
               config: ml_collections.ConfigDict,
               global_config: ml_collections.ConfigDict,
               is_extra_msa: bool,
               name: str = 'evoformer_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.is_extra_msa = is_extra_msa

  def __call__(self,
               activations: FeatureDict,
               masks: FeatureDict,
               is_training: bool,
               safe_key: prng.SafeKey
               ) -> FeatureDict:
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    c = self.config
    gc = self.global_config

    msa_act, pair_act = activations['msa'], activations['pair']

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    msa_mask, pair_mask = masks['msa'], masks['pair']

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=gc)

    unused_safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)

    outer_module = OuterProductMean(
        config=c.outer_product_mean,
        global_config=self.global_config,
        num_output_channel=int(pair_act.shape[-1]),
        name='outer_product_mean')
    if c.outer_product_mean.first:
      pair_act = dropout_wrapper_fn(
          outer_module,
          msa_act,
          msa_mask,
          safe_key=next(sub_keys),
          output_act=pair_act)

    msa_act = dropout_wrapper_fn(
        MSARowAttentionWithPairBias(
            c.msa_row_attention_with_pair_bias, gc,
            name='msa_row_attention_with_pair_bias'),
        msa_act,
        msa_mask,
        safe_key=next(sub_keys),
        pair_act=pair_act)

    if not self.is_extra_msa:
      attn_mod = MSAColumnAttention(
          c.msa_column_attention, gc, name='msa_column_attention')
    else:
      attn_mod = MSAColumnGlobalAttention(
          c.msa_column_attention, gc, name='msa_column_global_attention')
    msa_act = dropout_wrapper_fn(
        attn_mod,
        msa_act,
        msa_mask,
        safe_key=next(sub_keys))

    msa_act = dropout_wrapper_fn(
        Transition(c.msa_transition, gc, name='msa_transition'),
        msa_act,
        msa_mask,
        safe_key=next(sub_keys))

    if not c.outer_product_mean.first:
      pair_act = dropout_wrapper_fn(
          outer_module,
          msa_act,
          msa_mask,
          safe_key=next(sub_keys),
          output_act=pair_act)

    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                               name='triangle_multiplication_outgoing'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                               name='triangle_multiplication_incoming'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_starting_node, gc,
                          name='triangle_attention_starting_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_ending_node, gc,
                          name='triangle_attention_ending_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    pair_act = dropout_wrapper_fn(
        Transition(c.pair_transition, gc, name='pair_transition'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    return {'msa': msa_act, 'pair': pair_act}
