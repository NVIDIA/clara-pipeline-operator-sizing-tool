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
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from container import Container, Metrics, RawMetrics  # nopep8  # noqa: E402


def is_empty(any_structure):
    """Helper method to check if structure is empty."""
    if any_structure:
        return False
    else:
        return True


TEMP_DIR = Path(tempfile.gettempdir())
TEST_SYS_FS = TEMP_DIR / "test_sys_fs"


@patch("container.SYSFS_PATH", TEST_SYS_FS)
class TestContainer:

    def test_init_container(self):
        container = Container()
        assert isinstance(container, Container)
        assert is_empty(container.id)
        assert is_empty(container.raw_metrics)
        assert is_empty(container.metric_paths)

    def test_create_metrics_path_no_id(self):
        container = Container()
        with pytest.raises(RuntimeError):
            container.construct_metrics_path()

    def test_create_metrics_path_with_id(self):
        container = Container()
        container.id = "testID1"
        container.construct_metrics_path()

        cpu_path = TEST_SYS_FS / "cpuacct" / "docker" / container.id / "cpuacct.usage"
        per_cpu_path = TEST_SYS_FS / "cpuacct" / "docker" / container.id / "cpuacct.usage_percpu"
        mem_path = TEST_SYS_FS / "memory" / "docker" / container.id / "memory.usage_in_bytes"

        assert container.metric_paths == (cpu_path, per_cpu_path, mem_path)

    def test_metrics_path_exists(self, tmp_path):
        container = Container()
        p1, p2, p3 = tmp_path / "p1", tmp_path / "p2", tmp_path / "p3"
        container.metric_paths = (p1, p2, p3)

        assert not container.metrics_path_exists()
        p1.touch()
        assert not container.metrics_path_exists()
        p2.touch()
        assert not container.metrics_path_exists()
        p3.touch()
        assert container.metrics_path_exists()

    @patch("container.psutil.cpu_times")
    def test_read_raw_metrics(self, mock_cpu, tmp_path):
        mock_cpu_data = [10, 20, 10, 20, 10, 20, 10, 20]
        mock_cpu.return_value = mock_cpu_data
        container = Container()
        p1, p2, p3 = tmp_path / "p1", tmp_path / "p2", tmp_path / "p3"
        content1, content2, content3 = b'123', b'456', b'789'
        p1.write_bytes(content1)
        p2.write_bytes(content2)
        p3.write_bytes(content3)
        container.metric_paths = (p1, p2, p3)

        raw_metrics = container._read_raw_metrics()
        assert isinstance(raw_metrics, RawMetrics)
        assert isinstance(raw_metrics.timestamp, float)
        assert raw_metrics.cpu == float(content1)
        assert raw_metrics.per_cpu == content2
        assert raw_metrics.sys_cpu == sum(mock_cpu_data[:7])
        assert raw_metrics.memory == float(content3)

    def test_sample_metrics_no_path(self):
        container = Container()
        with pytest.raises(RuntimeError):
            container.sample_metrics()

    @patch("container.Container._read_raw_metrics")
    @patch("container.Container._process_raw_data")
    def test_sample_metrics(self, mock_process_data, mock_read_metrics):
        container = Container()
        container.metric_paths = (1, 2, 3)
        mock_read_metrics.side_effect = [1, 2, 3]

        def sum_two(prev, cur):
            return (prev + cur)
        mock_process_data.side_effect = sum_two

        container.sample_metrics()
        assert container.raw_metrics == [1]
        assert container.metrics == []
        container.sample_metrics()
        assert container.raw_metrics == [1, 2]
        assert container.metrics == [3]
        container.sample_metrics()
        assert container.raw_metrics == [1, 2, 3]
        assert container.metrics == [3, 5]

    @patch("container.ONLINE_CPUS", 4)
    def test_process_raw_data(self):
        container = Container()
        raw_data = [
            RawMetrics(
                timestamp=2.0, cpu=800000.0, per_cpu=b'300000 0 0 500000 \n', sys_cpu=14000000.00,
                memory=6500000),
            RawMetrics(
                timestamp=3.0, cpu=1000000.0, per_cpu=b'500000 0 0 500000 \n', sys_cpu=14000000.60,
                memory=8500000)]
        post_data = container._process_raw_data(raw_data[0], raw_data[1])
        cpu_delta = (raw_data[1].cpu - raw_data[0].cpu) / 1e9
        sys_delta = raw_data[1].sys_cpu - raw_data[0].sys_cpu

        assert post_data == Metrics(
            timestamp=2.5,
            cpu_percent=(cpu_delta/sys_delta)*4*100,
            memory=7.50,
        )
