[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

[![NVIDIA](https://github.com/NVIDIA/clara-platform-python-client/blob/main/ext/NVIDIA_horo_white.png?raw=true)](https://docs.nvidia.com/clara/deploy/index.html)

# CPOST (Clara Pipeline Operator Sizing Tool)
## Tool to measure resource usage of Clara Platform pipeline operators

Cpost is a tool that will help you run your pipeline locally and provides you with the CPU and memory usage of each operators ran for the given input payload. Opeartors are ran one at a time and CPU and memory usage are sampled. The CPU and memory usage metrics are provided in a .csv format which allows further data analytics as needed.

##  System Requirements
* Clara Compiler (downloadable from [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:clara:clara_cli))
* Docker 20.10 or higher due to cgroup v2 constraints
* System must be using cgroup v2 (See [Docker Control Groups](https://docs.docker.com/config/containers/runmetrics/#control-groups) for more information)
* Python 3.8.0 or higher
*Do not have a Triton instance running on the same machine that CPOST is running on. CPOST will provision it's own Triton instance and the two instances could conflict and cause failures.

## Usage
The following is the help message of cpost:
```
usage: cpost [-h] [--metrics_dir METRICS_DIR] [--models_dir MODELS_DIR] [-v] [--force [{none,cont,stop}]] <pipeline_path> <input_dir>

Clara Pipeline Sizing Tool CLI

positional arguments:
  <pipeline_path>       pipeline definition file path
  <input_dir>           input payload directory

optional arguments:
  -h, --help            show this help message and exit
  --metrics_dir METRICS_DIR
                        metrics output directory, if not specified, write to stdout
  --models_dir MODELS_DIR
                        directory for Triton models, required if pipeline uses Triton
  -v, --verbose         verbose output (DEBUG level). If not specified, default output is INFO level.
  --force [{none,cont,stop}]
                        force continue or stop when operator failure occurs. (default: none, which will prompt the user for each failure).
```

## Quick Start Guide

### Download CPOST
#### Method 1: From NGC
Download the cpost wheel file from NGC. (Available soon)

#### Method 2: Build from Source Repository
1. Clone this repository.
2. In the source folder, run `python3 setup.py sdist bdist_wheel` and you should see a wheel file in `./dist`. Use this file to `pip install` in your desired virtual environment. For example:
```
$ ls 
CONTRIBUTING.md  demo  dist  LICENSE  README.md  requirements-dev.txt  requirements.txt  setup.cfg  setup.py  src  tests
$ ls dist
nvidia_clara_cpost-0.0.0-py3-none-any.whl  nvidia-clara-cpost-0.0.0.tar.gz
```

### Run CPOST in a virtual environment (or I guess you can install it globally as well)
After you have downloaded the wheel from [Download CPOST](#download-cpost), create a virtual environment to work with.
```
$ mkdir ./demo
$ cd demo
$ python3.8 -m venv venv
$ source venv/bin/activate
$ pip install -U pip
$ pip install ../dist/nvidia_clara_cpost-0.0.0-py3-none-any.whl  # or any other path to the wheel file
```
After pip install has completed, run `cpost` and you should see the help message.

### Prepare Pipeline Data

Let's prepare some source data to work with. We will use the AI Spleen Segementation Pipeline as an example

Download the [Clara AI Spleen Segmentation Pipeline](https://ngc.nvidia.com/catalog/resources/nvidia:clara:clara_ai_spleen_pipeline) to a directory (e.g. `./demo`). Download instructions are available on the linked page

Once we have the spleen downloaded, go into the folder and unzip the model and input data.
```
$ cd clara_ai_spleen_pipeline_v${VERSION_ON_NGC}
$ ls clara_ai_spleen_pipeline_v${VERSION_ON_NGC}
app_spleen-input_v1.zip  app_spleen-model_v1.zip  source.zip  spleen-pipeline-model-repo.yaml  spleen-pipeline.yaml
$ unzip app_spleen-input_v1.zip -d app_spleen-input_v1
$ unzip app_spleen-model_v1.zip -d app_spleen-model_v1
```
Now we're ready to run cpost!

The simplest way to run `cpost` is to provide a pipeline definition file and input payload data as shown below. The resulting metrics and console logs are written to standard output directly. In the demo folder:
```
$ cpost --models_dir clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/app_spleen-model_v1 clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/spleen-pipeline.yaml clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/app_spleen-input_v1
```

If raw metrics are desired, then a valid directory can be specified with `--metrics_dir` and the resulting metrics csv files will be stored in the given directory for each executed operator.
```
$ mkdir metrics
$ cpost--metrics_dir metrics --models_dir clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/app_spleen-model_v1 clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/spleen-pipeline.yaml clara_ai_spleen_pipeline_v${VERSION_ON_NGC}/app_spleen-input_v1
```

### Interpreting the Result
After running the above command, you should see below as output:

```
All software dependencies are fullfilled.

______________Executing Operator dicom-reader_______________
Running operator ...
The container id is: 47ca2626929006154a5515eba841755993df3f298de0abcdc5b9b951971470ca
Results are stored in /home/magzhang/code/sdk/Tools/cpost/demo/metrics/dicom-reader_final_result.csv
_______________Operator dicom-reader Summary________________
+--------+-----------+------------+-------------+
| Metric |  Average  |  Maximum   |  Resource   |
+--------+-----------+------------+-------------+
|  CPU   | 124.714 % | 1097.941 % |   cpu: 11   |
| Memory | 91.057 MB | 405.242 MB | memory: 512 |
+--------+-----------+------------+-------------+

___________Executing Operator spleen-segmentation___________
Running operator ...
The container id is: 270f486475aa4584b4fb5911a0db23a10b4eaf0eb26a14daa3fa8951c6a77c95
Results are stored in /home/magzhang/code/sdk/Tools/cpost/demo/metrics/spleen-segmentation_final_result.csv
____________Operator spleen-segmentation Summary____________
+--------+-------------+-------------+--------------+
| Metric |   Average   |   Maximum   |   Resource   |
+--------+-------------+-------------+--------------+
|  CPU   |  150.649 %  | 1134.358 %  |   cpu: 12    |
| Memory | 1630.311 MB | 4455.412 MB | memory: 4608 |
+--------+-------------+-------------+--------------+

______________Executing Operator dicom-writer_______________
Running operator ...
The container id is: 32cf46da42111c75dfa1856ec35e4724e22d9e6d246e64ab3089fc212f049a4a
Results are stored in /home/magzhang/code/sdk/Tools/cpost/demo/metrics/dicom-writer_final_result.csv
_______________Operator dicom-writer Summary________________
+--------+------------+------------+-------------+
| Metric |  Average   |  Maximum   |  Resource   |
+--------+------------+------------+-------------+
|  CPU   | 190.224 %  | 1017.747 % |   cpu: 11   |
| Memory | 278.678 MB | 552.313 MB | memory: 768 |
+--------+------------+------------+-------------+

__Executing Operator register-volume-images-for-rendering___
Running operator ...
The container id is: 2ad135d27cd827de8f687791c9c70ca88229d5eec912be1d20c1a66993ecbb1a
Results are stored in /home/magzhang/code/sdk/Tools/cpost/demo/metrics/register-volume-images-for-rendering_final_result.csv
Operator failed with exitcode is: 126
___Operator register-volume-images-for-rendering Summary____
+--------+----------+----------+-------------+
| Metric | Average  | Maximum  |  Resource   |
+--------+----------+----------+-------------+
|  CPU   | 12.667 % | 14.923 % |   cpu: 1    |
| Memory | 2.633 MB | 3.783 MB | memory: 256 |
+--------+----------+----------+-------------+
Operator register-volume-images-for-rendering failed with exitcode 126
+--------------------------------------+----------+-------------+-------------+--------------+
| Operator                             | Metric   | Average     | Maximum     | Resource     |
+======================================+==========+=============+=============+==============+
| dicom-reader                         | CPU      | 124.714 %   | 1097.941 %  | cpu: 11      |
|                                      | Memory   | 91.057 MB   | 405.242 MB  | memory: 512  |
+--------------------------------------+----------+-------------+-------------+--------------+
| spleen-segmentation                  | CPU      | 150.649 %   | 1134.358 %  | cpu: 12      |
|                                      | Memory   | 1630.311 MB | 4455.412 MB | memory: 4608 |
+--------------------------------------+----------+-------------+-------------+--------------+
| dicom-writer                         | CPU      | 190.224 %   | 1017.747 %  | cpu: 11      |
|                                      | Memory   | 278.678 MB  | 552.313 MB  | memory: 768  |
+--------------------------------------+----------+-------------+-------------+--------------+
| register-volume-images-for-rendering | CPU      | 12.667 %    | 14.923 %    | cpu: 1       |
| (Non-zero exitcode)                  | Memory   | 2.633 MB    | 3.783 MB    | memory: 256  |
+--------------------------------------+----------+-------------+-------------+--------------+
```
The last column in the last table is what you can put into the pipeline definition file's `requests`.
Please note that there maybe some small differences between each execution. You can run multiple times to see what are the best numbers to fill.


## Troubleshooting
### Docker pull error
```
Docker pull command for `nvcr.io/nvstaging/clara/dicom-reader:0.8.1-2108.1` returned with code 1
stdout is:
stderr is: Error response from daemon: unauthorized: authentication required

Please verify docker access and the pipeline definition
```
**Resolution**: CPOST performs a local check to match with the given image and tag. If this fails, CPOST performs a docker pull. Thus, please do a `docker login` to the correct registry or ensure that you have the correct docker image locally. 

### Docker network error
```
Error response from daemon: network with name cpost_net already exists

cpost_net already exist, please remove the network and rerun cpost
```
**Resolution**: This occurs because the docker network with name "cpost_net" already exist, which could either because you happen to have this network or because CPOST failed to clean up in one of the previous runs. Please do a `docker network rm cpost_net` and `docker network ls` to ensure this network is cleaned up.

For all other problems, please submit an issue in the repository and we will resolve this as soon as possible.

### Warning from container ID timeout
```
Running operator ...
Obtaining docker ID timed out. Operator spleen-segmentation failed
Operator spleen-segmentation failed with exitcode -15
```
**Resolution**: This occurs when CPOST tries to run the container in detached mode and times out during when waiting for the container ID to return. The exitcode `-15` means that cpost terminated the docker container because it speculates that something has gone wrong. This could happen due to a lot of reasons, and you can run in `-v` (verbose) mode to see the full `docker run` command and run it yourself and hopefully this will provides you some insights on why CPOST couldn't obtain a docker ID.

##  Running from Source Code During Development

The environment must have Python 3.8 installed and should have the necessary packages required by cpost installed. The `requirements.txt` contains all the necessary packages and can be used to install them. The tools used for development can be found in `requirements-dev.txt`

Once virtual environment are created successfully and have been activated. Install the `requirements.txt` with `pip` or `conda`, etc..  The following command can be run directly as cpost:
```
python src/main.py 
```

### Test Coverage

To see test coverage, activate the virtual environment and install the development tools from `requirements-dev.txt`. 
From the root of repository, run the command below will provide the unittest coverage report.
```
coverage run -m pytest tests && coverage report
```
