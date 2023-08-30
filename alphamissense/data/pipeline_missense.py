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

"""Functions for building the input features for the AlphaMissense model."""

import dataclasses
# Internal import (7717).
import os
import subprocess
import tempfile
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from absl import logging
import numpy as np

# Internal import (7716).
from alphamissense.common import residue_constants
from alphamissense.data import mmcif_parsing
from alphamissense.data import msa_identifiers
from alphamissense.data import parsers
from alphamissense.data import templates
from alphamissense.data.tools import jackhmmer


FeatureDict = MutableMapping[str, np.ndarray]

# SEQ_FEATURES have shape [num_res, ...].
# NUM_SEQ_MSA_FEATURES have shape [num_msa, ...].
# NUM_SEQ_NUM_RES_MSA_FEATURES have shape [num_msa, num_res, ...].
NUM_SEQ_NUM_RES_MSA_FEATURES = ('msa', 'msa_mask', 'deletion_matrix',)
NUM_SEQ_MSA_FEATURES = ('cluster_bias_mask',)
SEQ_FEATURES = (
    'residue_index', 'all_atom_positions', 'msa_profile', 'deletion_mean',
    'aatype', 'all_atom_mask', 'seq_mask', 'variant_aatype', 'variant_mask')
TOKENIZER = {
    r: i for i, r in enumerate(residue_constants.restypes + ['X', '-'])}

AFDB_BUCKET = 'gs://public-datasets-deepmind-alphafold-v4'


@dataclasses.dataclass(frozen=True)
class ProteinVariant:
  """A class containing definition of protein variant."""
  sequence: str  # The original protein sequence as extracted from FASTA file.
  position: int  # 1-based aa-position within protein.
  reference_aa: str  # May be different from AA in `sequence`.
  alternate_aa: str
  protein_id: str
  pathogenicity: Optional[bool]

  def __post_init__(self):
    if len(self.reference_aa) != 1 or len(self.alternate_aa) != 1:
      raise ValueError(
          'Reference and alternate amino acids must be of length 1.')

    if self.position < 1 or self.position > len(self.sequence):
      raise ValueError(
          f'{self.protein_id} Variant position {self.position} is invalid. '
          f'Must be 1 to {len(self.sequence)} (both included).')

    if self.sequence[self.position - 1] != self.reference_aa:
      sequence_aa = self.sequence[self.position - 1]
      raise ValueError(f'Mismatch between {sequence_aa=} and input '
                       f'{self.reference_aa=} at position {self.position}. '
                       'Note that position must follow 1-based indexing.')

  @property
  def alternate_sequence(self) -> str:
    return self.sequence[:self.position - 1] + (
        self.alternate_aa + self.sequence[self.position:])

  @property
  def reference_sequence(self) -> str:
    return self.sequence


class Cropper:
  """Crops and pads features to enforce uniform input sizes."""

  def __init__(self, crop_size: int, msa_crop_size: int):
    self.crop_size = crop_size
    self.msa_crop_size = msa_crop_size

  def _apply_crop(self, features: FeatureDict) -> FeatureDict:
    residue_keep_mask = self._choose_crop_region(features)
    for k in features:
      if k in SEQ_FEATURES:
        features[k] = features[k][residue_keep_mask]
      elif k in NUM_SEQ_NUM_RES_MSA_FEATURES:
        features[k] = features[k][:self.msa_crop_size, residue_keep_mask]
      elif k in NUM_SEQ_MSA_FEATURES:
        features[k] = features[k][:self.msa_crop_size]
    return features

  def _apply_pad(self, features: FeatureDict) -> FeatureDict:
    """Pads num residues and num MSA sequences."""
    num_res_before_pad = features['aatype'].shape[0]
    num_msa_seq_before_pad = features['msa'].shape[0]
    paddable_features = frozenset(
        SEQ_FEATURES + NUM_SEQ_MSA_FEATURES + NUM_SEQ_NUM_RES_MSA_FEATURES)

    def _empty_padding_shape(feature: np.ndarray) -> list[list[int]]:
      padding = []
      for _ in range(feature.ndim):
        padding.append([0, 0])
      return padding

    for k, v in features.items():
      if k not in paddable_features:
        logging.info('Skipping padding for feature %s', k)
        continue

      pad_size = _empty_padding_shape(v)
      if k in frozenset(NUM_SEQ_MSA_FEATURES + NUM_SEQ_NUM_RES_MSA_FEATURES):
        assert v.shape[0] == num_msa_seq_before_pad, (
            f'Unexpected shape for MSA feature {k=}: {v.shape}.')
        pad_size[0][1] = self.msa_crop_size - v.shape[0]

      seq_axis = 1 if k in NUM_SEQ_NUM_RES_MSA_FEATURES else 0
      if k in frozenset(SEQ_FEATURES + NUM_SEQ_NUM_RES_MSA_FEATURES):
        assert v.shape[seq_axis] == num_res_before_pad, (
            f'Unexpected shape for MSA feature {k=}: {v.shape}.')
        pad_size[seq_axis][1] = self.crop_size - v.shape[seq_axis]

      features[k] = np.pad(v, pad_size, mode='constant')
    return features

  def crop(self, features: FeatureDict) -> FeatureDict:
    features = self._apply_crop(features)
    features = self._apply_pad(features)

    features['seq_length'] = np.asarray(self.crop_size, dtype=np.int32)
    features['num_alignments'] = np.asarray(self.msa_crop_size, dtype=np.int32)
    return features

  def _contiguous_crop_selection(self, chain: FeatureDict) -> np.ndarray:
    """Selects a contiguous crop around the variant position."""
    variant_pos = chain['variant_position'] + 1
    sequence_len = chain['seq_length']
    # Determine the crop by placing the variant in the center.
    if variant_pos > sequence_len // 2:
      crop_end = min(sequence_len, variant_pos + self.crop_size // 2)
      crop_start = max(0, crop_end - self.crop_size)
    else:
      crop_start = max(0, variant_pos - self.crop_size // 2)
      crop_end = min(sequence_len, crop_start + self.crop_size)
    keep_mask = np.zeros(chain['seq_length'], dtype=bool)
    keep_mask[crop_start:crop_end] = True
    return keep_mask

  def _spatial_crop_selection(self, chain: FeatureDict) -> np.ndarray:
    """Crops around the variant position."""
    all_positions = chain['all_atom_positions']
    all_masks = chain['all_atom_mask']
    variant_index = chain['variant_position']

    variant_positions = all_positions[variant_index]
    variant_mask = all_masks[variant_index]
    variant_positions = variant_positions[variant_mask.astype(bool)]

    distances = np.linalg.norm(
        all_positions[:, :, None] - variant_positions, axis=-1)
    # Do not include distance from masked atoms.
    distances = np.where(all_masks[:, :, None], distances, np.inf)
    # Sort by minimum over atoms in both residues.
    min_distances = np.min(distances, axis=(1, 2))

    # Take crop_size residues closest to the variant. The first argsort returns
    # a mapping from distance-rank (in ascending order) to residue-id. The
    # second returns the mapping from residue-id to distance-rank.
    include_mask = np.argsort(np.argsort(min_distances)) < self.crop_size
    included_res_ids = chain['residue_index'][include_mask]

    # Make the keep masks.
    chain_keep_mask = np.zeros(chain['seq_length'], dtype=bool)
    chain_keep_mask[included_res_ids] = True
    return chain_keep_mask

  def _choose_crop_region(self, chain: FeatureDict) -> np.ndarray:
    if (chain['all_atom_mask'] == 0).all():  # Structure not found.
      chain_keep_masks = self._contiguous_crop_selection(chain)
    else:
      chain_keep_masks = self._spatial_crop_selection(chain)

    total_residues = chain_keep_masks.sum()
    if total_residues == 0:
      raise ValueError('Selected keep masks are all false!')
    elif total_residues > self.crop_size:
      raise ValueError(f'Invalid crop: {total_residues=}, {self.crop_size=}.')
    return chain_keep_masks


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append([TOKENIZER[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix'] = np.array(deletion_matrix, dtype=np.float32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_mask'] = np.ones_like(features['msa'], dtype=bool)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_runner is None:
      raise ValueError(f'MSA results not found in {msa_out_path=} '
                       'and msa_runner is not set. Either pre-fill MSA cache '
                       'file or provide a valid jackhmmer_binary_path.')
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result


def variant_sequence_features(variant: ProteinVariant) -> FeatureDict:
  """Computes features for a single protein sequence."""

  def get_aatype(seq: str) -> np.ndarray:
    return np.array([
        residue_constants.restype_order_with_x[s] for s in seq], np.int32)

  variant_mask = np.zeros(len(variant.sequence), dtype=np.int32)
  variant_mask[variant.position - 1] = 1

  aatype = get_aatype(variant.reference_sequence)
  num_res = len(variant.sequence)
  features = {
      'seq_length': np.array(num_res, dtype=np.int32),
      'aatype': aatype,
      'seq_mask': np.ones_like(aatype, dtype=np.int32),
      'residue_index': np.arange(0, num_res, dtype=np.int32),
      'variant_aatype': get_aatype(variant.alternate_sequence),
      'variant_position': np.array(variant.position - 1, dtype=np.int32),
      'variant_mask': variant_mask,
  }
  if variant.pathogenicity is not None:
    features['pathogenicity'] = np.array(variant.pathogenicity,
                                         dtype=np.float32)
  return features


def make_variant_masked_msa(features: FeatureDict) -> FeatureDict:
  """Add masked mutant sequence in the second line of MSA."""
  # Remove the last line of the MSA to preserve cropped shape.
  features['msa'] = np.concatenate([
      features['msa'][:1],
      features['variant_aatype'][None],
      features['msa'][1:-1]])
  # For other features, duplicate the first row.
  for k in ['deletion_matrix', 'msa_mask']:
    features[k] = np.concatenate([
        features[k][:1], features[k][:1], features[k][1:-1]])

  features['cluster_bias_mask'] = np.zeros(features['msa'].shape[0])
  features['cluster_bias_mask'][:2] = 1.

  # Only mask target sequence at the variant position.
  features['msa'][1, features['variant_position']] = 22
  return features


def read_from_gcloud(source: str) -> str:
  """Download and read text file from gcloud."""
  with tempfile.NamedTemporaryFile(mode='w+t') as temp_file:
    cmd = ('gcloud', 'storage', 'cp', source, temp_file.name)

    logging.info('Launching subprocess "%s"', ' '.join(cmd))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    retcode = process.wait()

    if retcode:
      raise RuntimeError(
          'gcloud storage cp failed\nstderr:\n%s\n' % stderr.decode('utf-8'))

    with open(temp_file.name, 'r') as f:
      cif_string = f.read()
  return cif_string


def get_atom_positions(protein_id: str) -> FeatureDict:
  """Reads the positions predicted by AFDB. Used only to crop the sequence."""
  source_file = f'{AFDB_BUCKET}/AF-{protein_id}-F1-model_v4.cif'

  cif_string = read_from_gcloud(source_file)
  parsing_result = mmcif_parsing.parse(file_id=protein_id,
                                       mmcif_string=cif_string)
  if parsing_result.mmcif_object is None:
    raise ValueError('Unable to parse cif file.')

  chain_id = list(parsing_result.mmcif_object.chain_to_seqres.keys())[0]
  atom_positions, atom_mask = templates.get_atom_positions(
      parsing_result.mmcif_object, chain_id, max_ca_ca_distance=150.0)
  return {'all_atom_positions': atom_positions, 'all_atom_mask': atom_mask}


def get_empty_atom_positions(num_res: int) -> FeatureDict:
  return {'all_atom_positions': np.zeros([num_res, 37, 3], dtype=np.float32),
          'all_atom_mask': np.zeros([num_res, 37], dtype=np.int32)}


def make_msa_profile(batch: FeatureDict) -> FeatureDict:
  """Returns chains with the profile feature added."""
  def _one_hot(idx, n_classes, dtype=np.float32):
    """Numpy implementation of jax.nn.one_hot."""
    arr = np.concatenate([np.eye(n_classes, dtype=dtype),
                          np.zeros([1, n_classes], dtype=dtype)])
    idx = np.clip(idx, -1, n_classes)
    return arr[idx]

  batch['msa_profile'] = np.mean(_one_hot(batch['msa'], 30), axis=0)
  batch['deletion_mean'] = np.mean(batch['deletion_matrix'], axis=0)
  return batch


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               *,
               jackhmmer_binary_path: str,
               protein_sequence_file: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               small_bfd_database_path: str,
               mgnify_max_hits: int = 5_000,
               uniref_max_hits: int = 10_000,
               small_bfd_max_hits: int = 5_000,
               sequence_crop_size: int = 256,
               msa_crop_size: int = 2048,
               use_precomputed_msas: bool = True):
    """Initializes the data pipeline."""
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.small_bfd_max_hits = small_bfd_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.cropper = Cropper(crop_size=sequence_crop_size,
                           msa_crop_size=msa_crop_size)

    with open(protein_sequence_file) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    self.protein_sequences = {k: v for k, v in zip(input_descs, input_seqs)}

    self.jackhmmer_uniref90_runner = None
    self.jackhmmer_small_bfd_runner = None
    self.jackhmmer_mgnify_runner = None
    if jackhmmer_binary_path:
      if uniref90_database_path:
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path)
      if small_bfd_database_path:
        self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=small_bfd_database_path)
      if mgnify_database_path:
        self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=mgnify_database_path)

  def gather_msa_features(self, variant: ProteinVariant, msa_output_dir: str):
    """Runs all msa tools and returns the postprocessed features."""
    if not os.path.exists(msa_output_dir):
      os.makedirs(msa_output_dir)

    msa_results = []
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.fasta') as query_file:
      query_file.write(f'>{variant.protein_id}\n{variant.sequence}\n')
      query_file.seek(0)

      if self.jackhmmer_uniref90_runner is not None:
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=self.jackhmmer_uniref90_runner,
            input_fasta_path=query_file.name,
            msa_out_path=os.path.join(msa_output_dir, 'uniref90_hits.sto'),
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.uniref_max_hits)
        uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        msa_results.append(uniref90_msa)

      if self.jackhmmer_mgnify_runner is not None:
        jackhmmer_mgnify_result = run_msa_tool(
            msa_runner=self.jackhmmer_mgnify_runner,
            input_fasta_path=query_file.name,
            msa_out_path=os.path.join(msa_output_dir, 'mgnify_hits.sto'),
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.mgnify_max_hits)
        mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
        logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
        msa_results.append(mgnify_msa)

      if self.jackhmmer_small_bfd_runner is not None:
        jackhmmer_small_bfd_result = run_msa_tool(
            msa_runner=self.jackhmmer_small_bfd_runner,
            input_fasta_path=query_file.name,
            msa_out_path=os.path.join(msa_output_dir, 'small_bfd_hits.sto'),
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.small_bfd_max_hits)
        bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
        logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
        msa_results.append(bfd_msa)

    msa_features = make_msa_features(msa_results)
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return msa_features

  def process(self,
              *,
              protein_id: str,
              reference_aa: str,
              alternate_aa: str,
              position: int,
              msa_output_dir: str,
              pathogenicity: Optional[bool] = None,
              ) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    if protein_id not in self.protein_sequences:
      raise ValueError(f'{protein_id=} not in input protein_sequence_file')

    variant = ProteinVariant(sequence=self.protein_sequences[protein_id],
                             protein_id=protein_id,
                             position=position,
                             reference_aa=reference_aa,
                             alternate_aa=alternate_aa,
                             pathogenicity=pathogenicity)

    features = variant_sequence_features(variant=variant)
    features.update(self.gather_msa_features(variant, msa_output_dir))
    features = make_msa_profile(features)

    try:
      features.update(get_atom_positions(protein_id))
    except (ValueError, FileNotFoundError):
      logging.warning('Unable to find atom positions from AFDB.')
      features.update(get_empty_atom_positions(num_res=len(variant.sequence)))
    features = self.cropper.crop(features)

    # Correct variant_position after crop and size fix based on variant mask.
    features['variant_position'] = np.argmax(features['variant_mask'])
    # If variant is in second row of MSA, first row must be the target sequence.
    if not np.array_equal(features['msa'][0], features['aatype']):
      logging.warning('First row of MSA is not the target sequence diff=%s.',
                      np.sum(features['msa'][0] != features['aatype']))
      features['msa'][0] = features['aatype']
    # Mark that first and second row of MSA should not be affected by sampling.
    features['cluster_bias_mask'] = np.pad(
        np.zeros(features['msa'].shape[0] - 2), (2, 0), constant_values=1.)

    features = make_variant_masked_msa(features)

    return features
