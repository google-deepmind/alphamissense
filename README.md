# AlphaMissense

This package provides the AlphaMissense model implementation. This implementation is provided for reference alongside the [AlphaMissense 2023 publication](https://www.science.org/doi/10.1126/science.adg7492) and will not be actively maintained moving forward.

We forked the [AlphaFold repository](https://github.com/google-deepmind/alphafold/tree/v2.3.2) and modified it to implement AlphaMissense.

What we provide:
*   Detailed implementation of the AlphaMissense model and training losses ([modules_missense.py](https://github.com/deepmind/alphamissense/blob/main/alphamissense/model/modules_missense.py))

*   The data pipeline to create input features for inference ([pipeline_missense.py](https://github.com/deepmind/alphamissense/blob/main/alphamissense/data/pipeline_missense.py)). The data pipeline requires access to genetic databases for multiple sequence alignments and, if using spatial cropping, protein structures of the AlphaFold Database hosted in Google Cloud Storage. Please see the section for genetic databases and [AFDB readme file](https://github.com/deepmind/alphafold/tree/main/afdb)) to learn how to access these datasets.

* Pre-computed predictions for all possible human amino acid substitutions and missense variants ([hosted here](https://console.cloud.google.com/storage/browser/dm_alphamissense)).

What we don’t provide:
*   The trained AlphaMissense model weights.


## Access AlphaMissense predictions:
Predictions for human major transcripts and isoforms are provided [here](https://console.cloud.google.com/storage/browser/dm_alphamissense).
You can use these files with the Ensembl VEP tool and [AlphaMissense plug-in](https://www.ensembl.org/info/docs/tools/vep/script/vep_plugins.html).


## Installation

1. Install all dependencies:
```bash
sudo apt install python3.11-venv aria2 hmmer
```

2. Clone this repository and `cd` into it.
```bash
git clone https://github.com/deepmind/alphamissense.git
cd ./alphamissense
```

3. Set up a Python virtual environment and install the Python dependencies:
```bash
python3 -m venv ./venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
```

4. Test the installation
```bash
venv/bin/python test/test_installation.py
```


## Usage
Because we are not releasing the trained model weights, the code is not meant to be used for making new predictions, but serve as an implementation reference. We are releasing the data pipeline, model and loss function code.

The data pipeline requires a FASTA file (i.e. `protein_sequence_file`) which should contain all target sequences, and the genetic sequence databases outlined in the next section.
```python
from alphamissense.data import pipeline_missense

protein_sequence_file = ...
pipeline = pipeline_missense.DataPipeline(
    jackhmmer_binary_path=...,  # Typically '/usr/bin/jackhmmer'.
    protein_sequence_file=protein_sequence_file,
    uniref90_database_path=DATABASES_DIR + '/uniref90/uniref90.fasta',
    mgnify_database_path=DATABASES_DIR + '/mgnify/mgy_clusters_2022_05.fa',
    small_bfd_database_path=DATABASES_DIR + '/small_bfd/bfd-first_non_consensus_sequences.fasta',
)

sample = pipeline.process(
    protein_id=...,  # Sequence identifier in the FASTA file.
    reference_aa=...,  # Single capital letter, e.g. 'A'.
    alternate_aa=...,
    position=...,  # Integer, note that the position is 1-based!
    msa_output_dir=msa_output_dir,
)
```

The model is implemented as a JAX module and can be instantiated for example as:
```python
from alphamissense.model import config
from alphamissense.model import modules_missense

def _forward_fn(batch):
    model = modules_missense.AlphaMissense(config.model_config().model)
    return model(batch, is_training=False, return_representations=False)

random_seed = 0
prng = jax.random.PRNGKey(random_seed)

params = hk.transform(_forward_fn).init(prng, sample)
apply = jax.jit(hk.transform(_forward_fn).apply)
output = apply(params, prng, sample)
```
For example, at this point the score of the variant would be stored in `output['logit_diff']['variant_pathogenicity']`.


## Genetic databases

AlphaMissense used multiple genetic (sequence) databases for multiple sequence alignments:

*   [BFD](https://bfd.mmseqs.com/),
*   [MGnify](https://www.ebi.ac.uk/metagenomics/),
*   [UniRef90](https://www.uniprot.org/help/uniref).

We refer to the [AlphaFold repository](https://github.com/deepmind/alphafold) for instructions on how to download these databases.


## Citing this work
Any publication that discloses findings arising from using this source code should cite:

```bibtex
@article {AlphaMissense2023,
  author       = {Jun Cheng, Guido Novati, Joshua Pan, Clare Bycroft, Akvilė Žemgulytė, Taylor Applebaum, Alexander Pritzel, Lai Hong Wong, Michal Zielinski, Tobias Sargeant, Rosalia G. Schneider, Andrew W. Senior, John Jumper, Demis Hassabis, Pushmeet Kohli, Žiga Avsec},
  journal      = {Science},
  title        = {Accurate proteome-wide missense variant effect prediction with AlphaMissense},
  year         = {2023},
  doi          = {10.1126/science.adg7492},
  URL          = {https://www.science.org/doi/10.1126/science.adg7492},
}
```


## Acknowledgements
AlphaMissense communicates with and/or references the following separate libraries and packages:
*   Abseil
*   Biopython
*   HMMER Suite
*   Haiku
*   Immutabledict
*   JAX
*   Matplotlib
*   NumPy
*   Pandas
*   SciPy
*   Tree
*   Zstandard
We thank all their contributors and maintainers!


## License and Disclaimer
This is not an officially supported Google product.

The information within the AlphaMissense Database and the model implementation is provided for theoretical modelling only, caution should be exercised in its use. This information is not intended to be a substitute for professional medical advice, diagnosis, or treatment, and does not constitute medical or other professional advice.

Copyright 2023 DeepMind Technologies Limited.


### AlphaMissense Code License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### AlphaMissense predictions License
AlphaMissense predictions are made available under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc-sa/4.0/

### Third-party software
Use of the third-party software, libraries or code referred to in the [Acknowledgements](#acknowledgements) section above may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.
