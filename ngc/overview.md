# CPOST (Clara Pipeline Operator Sizing Tool)
## Tool to measure resource usage of Clara Platform pipeline operators

CPOST is a tool that will help you run your pipeline locally and provides you with the CPU and memory usage of each operators ran for the given input payload. Opeartors are ran one at a time and CPU and memory usage are sampled. The CPU and memory usage metrics are provided in a .csv format which allows further data analytics as needed.

##  System Requirements
* Clara Compiler (downloadable from [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:clara:clara_cli))
* Docker 20.10 or higher due to cgroup v2 constraints
* System must be using cgroup v2 (See [Docker Control Groups](https://docs.docker.com/config/containers/runmetrics/#control-groups) for more information)
* Python 3.8.0 or higher
*Do not have a Triton instance running on the same machine that CPOST is running on. CPOST will provision it's own Triton instance and the two instances could conflict and cause failures.

CPOST is available on [GitHub](https://github.com/NVIDIA/clara-pipeline-operator-sizing-tool)