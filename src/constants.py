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
import sys
from pathlib import Path

B_MB_FACTOR = 1e6

SYSFS_PATH = Path("/sys/fs/cgroup")

ON_POSIX = 'posix' in sys.builtin_module_names

NS_PER_S = 1e9
CLOCK_TICKS_PER_S = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
ONLINE_CPUS = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])

ID_WAITING_TIME_SECONDS = 15
METRIC_SAMPLING_PERIOD_SECONDS = 0.2  # i.e 200ms


TRITON_IMAGE_TAG = "nvcr.io/nvidia/tritonserver:20.07-v1-py3"
TRITON_READY_TIMEOUT_SECONDS = 30
TRITON_WAIT_TIME_SECONDS = 15
TRITON_WAIT_SLEEP_TIME_SECONDS = 1
TRITON_HTTP_ENV_VAR = "NVIDIA_TRITON_HTTPURI"
TRITON_HTTP_PORT = 8000
TRITON_GRPC_ENV_VAR = "NVIDIA_TRITON_GRPCURI"
TRITON_GRPC_PORT = 8001
LEGACY_TRTIS_HTTP_ENV_VAR = "NVIDIA_CLARA_TRTISURI"
LEGACY_TRITON_HTTP_ENV_VAR = "CLARA_TRITON_URI"
