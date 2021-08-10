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


import logging
import os
import re
import sys
import time
from multiprocessing import Manager, Queue
from random import uniform as rand_float
from unittest.mock import MagicMock, call, patch

import pytest
from src.cli import ContinueOptions

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from clarac_utils import OperatorConfig, ServiceConfig  # nopep8  # noqa: E402
from container import Metrics  # nopep8  # noqa: E402
from pipeline_utils import (_enqueue_output, build_operator_cmd, clean_up_containers,  # nopep8  # noqa: E402
                            get_output_writers, print_operator_summary, print_pipeline_summary, run_pipeline,
                            run_pipeline_alone, run_pipeline_with_services, sample_operator, start_operator,
                            start_pipeline_services)
from triton_utils import RUN_MODE  # nopep8  # noqa: E402


def test_enqueue_output(tmp_path):
    file_path = tmp_path / "test_enqueue"
    data = b"1255\n1233\n"
    file_path.write_bytes(data)
    q = Queue()
    opened_file = file_path.open("rb")
    _enqueue_output(opened_file, q)
    assert q.get(timeout=1) == b"1255\n"
    assert q.get(timeout=1) == b"1233\n"


@patch("pipeline_utils.Popen")
def test_start_operator(mock_popen):
    raw_container_id = b'8c0b4110ae930dbe26b258de9bc34a03f98056ed6f27f991d32919bfe401d7c5\n'
    actual_container_id = raw_container_id.decode('utf-8').strip()
    mock_popen.return_value = MagicMock(**{"returncode": 0,
                                           "poll.return_value": None,
                                           "stdout.readline.return_value": raw_container_id,
                                           "stdout.close.return_value": None})

    manager = Manager()
    expected_container_id = manager.Value('c_wchar_p', '')
    mock_event = MagicMock()
    cmd = ["some", "docker", "run", "command"]

    start_operator(expected_container_id, mock_event, cmd)

    assert actual_container_id == expected_container_id.value
    mock_event.set.assert_called_once()


@patch("pipeline_utils.Process")
@patch("pipeline_utils.Popen")
def test_start_operator_popen_error(mock_popen, mock_multi_process):
    mock_exit_msg = "exiting because of error"
    mock_popen.return_value = MagicMock(**{"returncode": 1, "stderr.read.return_value": mock_exit_msg.encode("UTF-8")})

    manager = Manager()
    expected_container_id = manager.Value('c_wchar_p', '')
    mock_event = MagicMock()
    cmd = ['some', 'docker', 'run', 'command']

    with pytest.raises(SystemExit) as exc:
        start_operator(expected_container_id, mock_event, cmd)
    assert mock_exit_msg in str(exc.value)
    mock_event.set.assert_not_called()


@pytest.mark.parametrize("mock_exitcode, expected_code", [(b'0\n', None), (b'125\n', 125), (b'error\n', 1)])
@patch("pipeline_utils.Queue")
@patch("pipeline_utils.subproc_run")
@patch("pipeline_utils.Process")
@patch("pipeline_utils.Popen")
def test_start_operator_docker_error(
        mock_popen, mock_multi_process, mock_subproc_run, mock_q, mock_exitcode, expected_code):
    mock_popen.return_value = MagicMock(**{"returncode": None, "poll.return_value": None})

    mock_q.return_value = MagicMock(
        **{"get_nowait.return_value": b'8c0b4110ae930dbe26b258de9bc34a03f98056ed6f27f991d32919bfe401d7c5\n'})

    mock_subproc_run.return_value = MagicMock(**{"returncode": None, "stdout": mock_exitcode})

    manager = Manager()
    expected_container_id = manager.Value('c_wchar_p', '')
    mock_event = MagicMock()
    cmd = ['some', 'docker', 'run', 'command']

    if mock_exitcode == b'0\n':
        start_operator(expected_container_id, mock_event, cmd)
    else:
        with pytest.raises(SystemExit) as exc:
            start_operator(expected_container_id, mock_event, cmd)
        assert exc.value.code == expected_code
    mock_event.set.assert_called_once()


def test_sample_operator_logic():
    mock_q = MagicMock()
    mock_container = MagicMock()
    mock_container.metrics_path_exists.side_effect = [0, 1, 1, 0]
    mock_container.sample_metrics.return_value = None
    sample_operator(mock_container, mock_q)
    assert mock_container.method_calls == [
        call.metrics_path_exists(),
        call.metrics_path_exists(),
        call.metrics_path_exists(),
        call.sample_metrics(),
        call.metrics_path_exists(),
    ]

    assert mock_q.put.call_count == 2
    assert mock_q.put.call_args_list == [call(None), call(0)]


@pytest.mark.parametrize("sampling_time,expected", [(rand_float(0.0001, 0.19), [0.2]), (0.3, [0.3])])
def test_sample_operator_sampling_rate(sampling_time, expected):
    mock_q = MagicMock()
    mock_container = MagicMock()
    sampling_num = 10
    mock_container.metrics_path_exists.side_effect = [0, 1] + [1] * sampling_num + [0]
    result_timestamps = []

    def mock_sample():
        """Mock sampling function that appends a timestamp to a list."""
        timestamp = time.perf_counter()
        time.sleep(sampling_time)
        result_timestamps.append(timestamp)

    mock_container.sample_metrics = mock_sample
    sample_operator(mock_container, mock_q)
    assert len(result_timestamps) == sampling_num, "The number of samples does not match with expected."

    result_diffs = [round(j - i, 1) for i, j in zip(result_timestamps[:-1], result_timestamps[1:])]
    assert result_diffs == expected * (sampling_num - 1), "Something is wrong with the accuracy of time.sleep()"


# autopep8: off
@pytest.mark.parametrize(
    "op_config, expected_args",
    [
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, {"VAR0": 2, "VAR1": "hi"}, [{"path": "/input"}], [{"path": "/output"}]),
            ["--env", "VAR0=2", "--env", "VAR1=hi", "-v", "%tmp%/app_data:/input", "-v", "%tmp%/op_name:/output", "image:tag"], id="with_ENV_VAR"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, None, None),
            ["image:tag"], id="no_input_output"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, [{"path": "/input"}], [{"path": "/output"}]),
            ["-v", "%tmp%/app_data:/input", "-v", "%tmp%/op_name:/output", "image:tag"], id="min_input_output"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, [{"from": "liver", "path": "/input"}], None),
            ["-v", "%tmp%/liver:/input", "image:tag"], id="input_contains_from"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, [{"from": "liver", "name": "classification", "path": "/input"}, {"path": "/dcm"}], None),
            ["-v", "%tmp%/liver/classification:/input", "-v", "%tmp%/app_data:/dcm", "image:tag"], id="double_inputs"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, [{"path": "/input"}], [{"name": "logs", "path": "/output"}]),
            ["-v", "%tmp%/app_data:/input", "-v", "%tmp%/op_name/logs:/output", "image:tag"], id="named_output"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", ["some", "command"], None, [{"path": "/input"}], [{"path": "/output"}]),
            ["-v", "%tmp%/app_data:/input", "-v", "%tmp%/op_name:/output", "image:tag", "some", "command"], id="image_with_command"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, None, None, ["model1"]),
            ["--env", "NVIDIA_TRITON_HTTPURI=localhost:8000", "--env", "CLARA_TRITON_URI=localhost:8000", "--env", "NVIDIA_CLARA_TRTISURI=localhost:8000", "--env", "NVIDIA_TRITON_GRPCURI=localhost:8001", "image:tag"], id="model_repo"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", None, None, None, None, None, [ServiceConfig("name", "it", None, None)]),
            ["image:tag"], id="pipeline_services"
        ),
        pytest.param(
            OperatorConfig("op_name", "image:tag", ["some", "command"], {"VAR0": 2}, 
                [{"from": "liver", "name": "classification", "path": "/input"}, {"path": "/dcm"}], 
                [{"name": "dicom", "path": "/output"}, {"name": "logs", "path": "/logs"}]),
                ["--env", "VAR0=2", "-v", "%tmp%/liver/classification:/input", "-v", "%tmp%/app_data:/dcm", 
                    "-v", "%tmp%/op_name/dicom:/output", "-v", "%tmp%/op_name/logs:/logs", "image:tag", "some", "command"], 
            id="all_in_one"
        ),
    ],
)
# autopep8: on
def test_build_operator_cmd(tmp_path, op_config, expected_args):
    input_path = tmp_path / "app_data"

    def swap_tmp(temp_dir, args):
        return [re.sub(r'%tmp%', temp_dir, i) for i in args]
    expected_args = swap_tmp(str(tmp_path), expected_args)
    config = op_config

    result_cmd = build_operator_cmd(input_path, tmp_path, config, "localhost")

    assert (tmp_path / "op_name").is_dir()

    assert result_cmd == ["docker", "run", "-d", "--rm", "--env", "NVIDIA_CLARA_NOSYNCLOCK=1"] + expected_args


def test_print_operator_summary(caplog):
    metrics = [Metrics(1.5, 10, 20), Metrics(1.5, 20, 20), Metrics(1.5, 30, 25)]
    with caplog.at_level(logging.INFO):
        print_operator_summary(metrics, "opeartor_name")
        # [1] only gets the table section
        messages = [rec.getMessage() for rec in caplog.records][1]

    messages = messages.split("\n")
    cpu_line = messages[3]
    mem_line = messages[4]
    assert "CPU" in cpu_line
    assert "20" in cpu_line
    assert "30" in cpu_line
    assert "Memory" in mem_line
    assert "21.6" in mem_line
    assert "25" in mem_line


@pytest.mark.parametrize("run_mode", [RUN_MODE.NO_INFERENCE_SERVER, RUN_MODE.MODEL_REPO, RUN_MODE.PIPELINE_SERVICES])
@patch("pipeline_utils.run_pipeline_with_services")
@patch("pipeline_utils.run_pipeline_alone")
@patch("pipeline_utils.run_triton_model_repo")
@patch("pipeline_utils.decide_method_to_run_triton")
def test_run_pipeline(mock_decide, mock_run_triton, mock_run_alone, mock_run_services, run_mode):
    mock_decide.return_value = run_mode
    mock_run_triton.return_value.__enter__.return_value = MagicMock()
    run_pipeline([], None, None, None, ContinueOptions.NONE)
    if run_mode == RUN_MODE.NO_INFERENCE_SERVER:
        mock_run_triton.assert_not_called()
        mock_run_alone.assert_called_once()
        mock_run_services.assert_not_called()
    elif run_mode == RUN_MODE.MODEL_REPO:
        mock_run_triton.assert_called_once()
        mock_run_alone.assert_called_once()
        mock_run_services.assert_not_called()
    elif run_mode == RUN_MODE.PIPELINE_SERVICES:
        mock_run_triton.assert_not_called()
        mock_run_alone.assert_not_called()
        mock_run_services.assert_called_once()


def test_get_output_writers(tmp_path):
    mock_writer = MagicMock(**{"join.return_value": None})
    with get_output_writers(tmp_path) as writers:
        assert writers == []
        writers.append(mock_writer)
    assert mock_writer.join.call_count == 1


def test_get_no_output_writers():
    with get_output_writers(None) as writers:
        assert writers is None


@patch("pipeline_utils.build_operator_cmd")
@patch("pipeline_utils.run_operator")
@patch("pipeline_utils.TemporaryDirectory")
def test_run_pipeline_alone(mock_temp_file, mock_run_operator, mock_build_cmd, tmp_path):
    mock_temp_file.return_value.__enter__.return_value = "tmp_file_name"
    mock_run_operator.side_effect = [None, True, None]
    m1, m2, m3 = MagicMock(**{"name": "1"}), MagicMock(**{"name": "2"}), MagicMock(**{"name": "3"})
    execution_order = [m1, m2, m3]
    run_pipeline_alone(execution_order, tmp_path, None, ContinueOptions.NONE, None)
    assert len(mock_run_operator.call_args_list) == 2
    assert m1 in mock_run_operator.call_args_list[0].args
    assert m2 in mock_run_operator.call_args_list[1].args


@patch("pipeline_utils.subproc_run_wrapper")
def test_clean_up_containers(mock_subproc_run_wrapper):
    running_containers = {"image1": ("ID1", "ip_address")}
    clean_up_containers(running_containers)
    assert mock_subproc_run_wrapper.call_args.args[0] == ["docker", "kill", "ID1"]
    assert running_containers == {}


@patch("pipeline_utils.start_triton")
@patch("pipeline_utils.clean_up_containers")
def test_start_pipeline_services(mock_clean_up_containers, mock_start_triton):
    container_info = ("container_id_123", "ip_address")
    mock_start_triton.return_value = container_info

    service_config_1 = ServiceConfig("trtis", "image_tag", ["some", "cmd"], {"VAR": "port_num"})
    op_config_1 = OperatorConfig("name", None, None, None, None, None, None, [service_config_1])
    services_dict = {}
    start_pipeline_services(op_config_1, services_dict, "some-dir")
    assert services_dict["image_tag some cmd"] == container_info
    assert op_config_1.variables == {"VAR": "ip_address:port_num"}
    assert mock_start_triton.call_count == 1

    # Same service -> no new services created
    start_pipeline_services(op_config_1, services_dict, "some-dir")
    assert services_dict["image_tag some cmd"] == container_info
    assert op_config_1.variables == {"VAR": "ip_address:port_num"}
    assert mock_start_triton.call_count == 1

    # Different service -> new service created
    service_config_2 = ServiceConfig("trtis", "image_tag2", ["some", "cmd"], {"VAR": "port_num2"})
    op_config_2 = OperatorConfig("name", None, None, None, None, None, None, [service_config_2])
    start_pipeline_services(op_config_2, services_dict, "some-dir")
    mock_clean_up_containers.assert_called_once()
    assert services_dict["image_tag2 some cmd"] == container_info
    assert op_config_2.variables == {"VAR": "ip_address:port_num2"}
    assert mock_start_triton.call_count == 2


@patch("pipeline_utils.start_triton")
@patch("pipeline_utils.clean_up_containers")
def test_start_service_not_supported(mock_clean_up_containers, mock_start_triton, caplog):
    service_config_1 = ServiceConfig("other service", "image_tag", ["some", "cmd"], {"VAR": "value"})
    op_config_1 = OperatorConfig("name", None, None, None, None, None, None, [service_config_1])
    services_dict = {}

    with caplog.at_level(logging.WARNING):
        start_pipeline_services(op_config_1, services_dict, "some-dir")
        messages = [rec.getMessage() for rec in caplog.records]
    mock_clean_up_containers.assert_not_called()
    mock_start_triton.assert_not_called()
    assert "does not support" in messages[0]
    assert "Skipping `other service`" in messages[1]


@patch("pipeline_utils.clean_up_containers")
@patch("pipeline_utils.build_operator_cmd")
@patch("pipeline_utils.start_pipeline_services")
@patch("pipeline_utils.run_operator")
@patch("pipeline_utils.TemporaryDirectory")
def test_run_pipeline_with_services(
        mock_temp_file, mock_run_operator, mock_start_pipeline_services, mock_build_cmd, mock_clean_up_containers,
        tmp_path):

    def mock_add_dict(op, services_dict, *args):
        services_dict["name"] = "cont_id"
    mock_start_pipeline_services.side_effect = mock_add_dict

    mock_temp_file.return_value.__enter__.return_value = "tmp_file_name"
    mock_run_operator.side_effect = [None, True, None]
    mock_config1 = MagicMock(**{"services": True})
    mock_config2 = MagicMock(**{"services": False})
    execution_order = [mock_config1, mock_config2, mock_config2]
    run_pipeline_with_services(execution_order, tmp_path, None, tmp_path, ContinueOptions.NONE)
    assert len(mock_run_operator.call_args_list) == 2
    mock_start_pipeline_services.assert_called_once()
    assert mock_build_cmd.call_count == 2
    mock_clean_up_containers.assert_called_once()


@patch("pipeline_utils.tabulate")
def test_print_pipeline_summary(mock_tabulate):
    raw_data = {
        'dicom-reader':
        [['CPU', '130.407 %', '732.975 %', 'cpu: 8'],
         ['Memory', '109.309 MB', '431.407 MB', 'memory: 512']],
        'spleen-segmentation':
        [['CPU', '126.747 %', '1144.132 %', 'cpu: 12'],
            ['Memory', '1403.712 MB', '4339.55 MB', 'memory: 8192']],
        'dicom-writer':
        [['CPU', '168.027 %', '676.498 %', 'cpu: 7'],
            ['Memory', '481.506 MB', '866.976 MB', 'memory: 1024']],
        'register-dicom-output\n(Non-zero exitcode)':
        [['CPU', '14.524 %', '18.102 %', 'cpu: 1'],
            ['Memory', '2.074 MB', '2.589 MB', 'memory: 4']]}

    print_pipeline_summary(raw_data)
    # This format is desired to keep the display result from tabulate clean
    assert mock_tabulate.call_args.args[0] == [
        ['dicom-reader', 'CPU\nMemory', '130.407 %\n109.309 MB', '732.975 %\n431.407 MB', 'cpu: 8\nmemory: 512'],
        ['spleen-segmentation', 'CPU\nMemory', '126.747 %\n1403.712 MB', '1144.132 %\n4339.55 MB', 'cpu: 12\nmemory: 8192'],
        ['dicom-writer', 'CPU\nMemory', '168.027 %\n481.506 MB', '676.498 %\n866.976 MB', 'cpu: 7\nmemory: 1024'],
        ['register-dicom-output\n(Non-zero exitcode)', 'CPU\nMemory', '14.524 %\n2.074 MB', '18.102 %\n2.589 MB',
         'cpu: 1\nmemory: 4']]
