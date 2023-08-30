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
"""Script to test installation of AlphaMissense."""

import pathlib

import jax
import jax.numpy as jnp
import tree

from alphamissense.data import pipeline_missense
from alphamissense.model import config
from alphamissense.model import modules_missense

CWD = str(pathlib.Path(__file__).parent)


def array_tree_summary(array_tree):
  """Creates flatten list of keys with shapes and dtype."""
  array_strs = []
  for path, array in tree.flatten_with_path(array_tree):
    path = '.'.join(map(str, path))
    if array is None:
      array_strs.append(f'{path}(None)')
    else:
      array_strs.append(f'{path}(shape={array.shape}, dtype={array.dtype})')
  return array_strs


def main():

  pipeline = pipeline_missense.DataPipeline(
      jackhmmer_binary_path='/usr/bin/jackhmmer',
      protein_sequence_file=CWD + '/test_sequence.fasta',
      uniref90_database_path=CWD + '/test_sequence.fasta',
      mgnify_database_path='',
      small_bfd_database_path='',
  )

  sample = pipeline.process(
      protein_id='Q08708',  # Sequence identifier in the FASTA file.
      reference_aa='A',
      alternate_aa='C',
      position=3,  # Note that the position is 1-based!
      msa_output_dir=CWD + '/msa_output_dir',
  )
  print('Pipeline produced sample with features:')
  for sample_str in sorted(array_tree_summary(sample)):
    print(sample_str)

  cfg = config.model_config()
  cfg.model.num_recycle = 1  # Speeds up testing.
  model_runner = modules_missense.RunModel(cfg, params=None)
  sample = jax.tree_map(jnp.asarray, sample)
  output = model_runner.predict(sample, random_seed=0)
  print('Model inizialized parameters:')
  for param_str in sorted(array_tree_summary(model_runner.params)):
    print(param_str)

  print('Model produced outputs:')
  for output_str in sorted(array_tree_summary(output)):
    print(output_str)


if __name__ == '__main__':
  main()
