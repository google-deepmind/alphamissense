# Copyright 2021 DeepMind Technologies Limited
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

"""Functions for getting templates and calculating template features."""

import numpy as np

from alphamissense.common import residue_constants
from alphamissense.data import mmcif_parsing


class Error(Exception):
  """Base class for exceptions."""


class NoChainsError(Error):
  """An error indicating that template mmCIF didn't have any chains."""


class SequenceNotInTemplateError(Error):
  """An error indicating that template mmCIF didn't contain the sequence."""


class NoAtomDataInTemplateError(Error):
  """An error indicating that template mmCIF didn't contain atom positions."""


class TemplateAtomMaskAllZerosError(Error):
  """An error indicating that template mmCIF had all atom positions masked."""


class QueryToTemplateAlignError(Error):
  """An error indicating that the query can't be aligned to the template."""


class CaDistanceError(Error):
  """An error indicating that a CA atom distance exceeds a threshold."""


class MultipleChainsError(Error):
  """An error indicating that multiple chains were found for a given ID."""


# Prefilter exceptions.
class PrefilterError(Exception):
  """A base class for template prefilter exceptions."""


class DateError(PrefilterError):
  """An error indicating that the hit date was after the max allowed date."""


class AlignRatioError(PrefilterError):
  """An error indicating that the hit align ratio to the query was too small."""


class DuplicateError(PrefilterError):
  """An error indicating that the hit was an exact subsequence of the query."""


class LengthError(PrefilterError):
  """An error indicating that the hit was too short."""


TEMPLATE_FEATURES = {
    'template_aatype': np.float32,
    'template_all_atom_masks': np.float32,
    'template_all_atom_positions': np.float32,
    'template_domain_names': object,
    'template_sequence': object,
    'template_sum_probs': np.float32,
}


def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
  """Checks if the distance between unmasked neighbor residues is ok."""
  ca_position = residue_constants.atom_order['CA']
  prev_is_unmasked = False
  prev_calpha = None
  for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
    this_is_unmasked = bool(mask[ca_position])
    if this_is_unmasked:
      this_calpha = coords[ca_position]
      if prev_is_unmasked:
        distance = np.linalg.norm(this_calpha - prev_calpha)
        if distance > max_ca_ca_distance:
          raise CaDistanceError(
              'The distance between residues %d and %d is %f > limit %f.' % (
                  i, i + 1, distance, max_ca_ca_distance))
      prev_calpha = this_calpha
    prev_is_unmasked = this_is_unmasked


def get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float) -> tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""
  num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

  relevant_chains = [c for c in mmcif_object.structure.get_chains()
                     if c.id == auth_chain_id]
  if len(relevant_chains) != 1:
    raise MultipleChainsError(
        f'Expected exactly one chain in structure with id {auth_chain_id}.')
  chain = relevant_chains[0]

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)
  for res_index in range(num_res):
    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
    res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
    if not res_at_position.is_missing:
      assert res_at_position.position is not None
      res = chain[(res_at_position.hetflag,
                   res_at_position.position.residue_number,
                   res_at_position.position.insertion_code)]
      for atom in res.get_atoms():
        atom_name = atom.get_name()
        x, y, z = atom.get_coord()
        if atom_name in residue_constants.atom_order.keys():
          pos[residue_constants.atom_order[atom_name]] = [x, y, z]
          mask[residue_constants.atom_order[atom_name]] = 1.0
        elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
          # Put the coordinates of the selenium atom in the sulphur column.
          pos[residue_constants.atom_order['SD']] = [x, y, z]
          mask[residue_constants.atom_order['SD']] = 1.0

      # Fix naming errors in arginine residues where NH2 is incorrectly
      # assigned to be closer to CD than NH1.
      cd = residue_constants.atom_order['CD']
      nh1 = residue_constants.atom_order['NH1']
      nh2 = residue_constants.atom_order['NH2']
      if (res.get_resname() == 'ARG' and
          all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
          (np.linalg.norm(pos[nh1] - pos[cd]) >
           np.linalg.norm(pos[nh2] - pos[cd]))):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

    all_positions[res_index] = pos
    all_positions_mask[res_index] = mask
  _check_residue_distances(
      all_positions, all_positions_mask, max_ca_ca_distance)
  return all_positions, all_positions_mask
