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


from dataclasses import dataclass
from dataclasses import fields as data_fields
from datetime import datetime

import psutil
from constants import B_MB_FACTOR, NS_PER_S, ONLINE_CPUS, SYSFS_PATH


@dataclass
class Metrics:
    timestamp: float
    cpu_percent: float
    memory: float  # in MB


METRICS_HEADER = [obj.name for obj in data_fields(Metrics)]


@dataclass
class RawMetrics:
    timestamp: float
    cpu: float
    per_cpu: bytes
    sys_cpu: tuple
    memory: float  # in bytes


class Container:

    def __init__(self) -> None:
        """Initializes the Container object with id, metrics_path, raw_metrics, and metrics.

        Args:
            None

        Returns:
            None
        """
        self.id = ""
        self.metric_paths = ()  # Tuple[Path, Path, Path]
        self.raw_metrics = []  # List[RawMetrics]
        self.metrics = []

    def construct_metrics_path(self):
        """Constructs metrics reading paths in a tuple based on self.id attribute.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError if id is not set when this is called
        """
        if self.id:
            _cpu_path = SYSFS_PATH / "cpuacct" / "docker" / self.id / "cpuacct.usage"
            _per_cpu_path = SYSFS_PATH / "cpuacct" / "docker" / self.id / "cpuacct.usage_percpu"
            _mem_path = SYSFS_PATH / "memory" / "docker" / self.id / "memory.usage_in_bytes"
            self.metric_paths = (_cpu_path, _per_cpu_path, _mem_path)
        else:
            raise RuntimeError("Container ID is not set when creating paths")

    def metrics_path_exists(self) -> bool:
        """Checks if all the paths in the container.metrics_path attribute exist.

        Args:
            None

        Returns:
            A boolean value for whether all metrics_paths exist on the system.
        """
        return self.metric_paths[0].exists() and self.metric_paths[1].exists() and self.metric_paths[2].exists()

    def _read_raw_metrics(self) -> RawMetrics:
        """Reads raw metrics data based on the self.metric_path and timestamp it.

        Args:
            None

        Returns:
            A RawMetrics object
        """
        timestamp = datetime.utcnow().timestamp()
        # Rationale for raw_sys_cpu arithmetic: getSystemCPUUsage() in docker/daemon/stats_collector_unix.go
        # in https://github.com/rancher/docker
        raw_sys_cpu = sum(psutil.cpu_times()[:7])  # in seconds
        # Note: Converting to float takes an extra 1000ns
        raw_cpu = float(self.metric_paths[0].read_bytes())
        # If we know this len is the same as the system cpu num, then we don't need per_cpu anymore
        raw_per_cpu = self.metric_paths[1].read_bytes()
        raw_mem = float(self.metric_paths[2].read_bytes())
        return RawMetrics(timestamp, raw_cpu, raw_per_cpu, raw_sys_cpu, raw_mem)

    def sample_metrics(self) -> None:
        """Samples raw metrics data and append to self.raw_metrics list.

        FileNotFoundError and OSError errno 19 implies that the file no longer
        exist and thus these are bypassed.

        Args:
            None

        Returns:
            None or metric, which is a Metrics object

        Raises:
            RuntimeError if self.metric_paths is not set when this is called
        """
        if self.metric_paths:
            try:
                raw_metrics = self._read_raw_metrics()
                self.raw_metrics.append(raw_metrics)
                # process metrics starting at second item
                if len(self.raw_metrics) >= 2:
                    metric = self._process_raw_data(self.raw_metrics[-2], self.raw_metrics[-1])
                    self.metrics.append(metric)
                    return metric
                else:
                    return
            except FileNotFoundError:
                return
            except OSError as err:
                if err.errno == 19:  # no such device error
                    return
                else:
                    raise(err)
        else:
            raise RuntimeError("Metrics paths must constructed before sampling.")

    @staticmethod
    def _process_raw_data(prev, cur):
        """Process the given data and convert units.
        Computation according to https://docs.docker.com/engine/api/v1.41/#operation/ContainerStats

        Args:
            prev: the prior RawMetrics object
            cur: the current RawMetrics object

        Returns:
            result: A list of MetricsData object
        """
        ts_avg = (prev.timestamp + cur.timestamp) / 2.0
        cpu_percent = 0.0
        # Convert from nanoseconds to seconds
        cpu_delta = (cur.cpu - prev.cpu) / NS_PER_S
        # Below does not need div by CLOCK_TICKS_PER_S because it has been done in psutils
        sys_cpu_delta = cur.sys_cpu - prev.sys_cpu

        if cpu_delta > 0.0 and sys_cpu_delta > 0.0:
            cpu_percent = (cpu_delta / sys_cpu_delta) * ONLINE_CPUS * 100.0

        # Since we're averaging the cpu, we also need to average the memory to match the averaged timestamp
        memory_avg = (prev.memory + cur.memory) / 2.0 / B_MB_FACTOR

        return Metrics(ts_avg, cpu_percent=cpu_percent, memory=memory_avg)
