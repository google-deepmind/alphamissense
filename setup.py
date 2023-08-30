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
"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup
from alphamissense import version

setup(
    name='alphamissense',
    version=version.__version__,
    description=(
        'An implementation of the inference pipeline of AlphaMissense.'
    ),
    author='DeepMind',
    author_email='alphamissense@deepmind.com',
    license='Apache License, Version 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'biopython',
        'dm-haiku',
        'dm-tree',
        'immutabledict',
        'jax',
        'jaxlib',
        'ml-collections',
        'numpy',
        'scipy',
        'typing_extensions',
    ],
    tests_require=[
        'mock',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
