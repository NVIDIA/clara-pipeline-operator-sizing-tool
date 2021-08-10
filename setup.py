# Copyright 2021 NVIDIA Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Install required packages from requirements.txt file
requirements_relative_path = "/requirements.txt"
package_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = package_folder + requirements_relative_path
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

# Extract version number from VERSION file
release_version = "0.0.0"
if os.path.exists('VERSION'):
    with open('VERSION') as version_file:
        release_version = version_file.read().strip()

setuptools.setup(
    name="nvidia-clara-cpost",
    author="NVIDIA Clara Deploy",
    version=release_version,
    description="Python package to run Clara Pipeline Operator Sizing Tool (cpost)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab-master.nvidia.com/Clara/sdk/-/tree/main/Tools/cpost",
    install_requires=install_requires,
    packages=setuptools.find_packages('.'),
    entry_points={
        'console_scripts': [
                'cpost = src.main:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
