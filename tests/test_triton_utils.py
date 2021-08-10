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
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from triton_utils import (RUN_MODE, _extract_models_from_configs, check_models_directory,  # nopep8  # noqa: E402
                          check_triton_status, decide_method_to_run_triton, inspect_ip_address, run_triton_model_repo,
                          start_triton)


@pytest.fixture(scope="function")
def create_triton_models_dir(tmp_path):
    """Custom Pytest fixture to mock triton models directory.

    Args:
        model_names: List of str representing triton model names.

    Returns:
        None
    """
    def _func(model_names):
        # Create the folders needed and some extra models in that directory
        for dir_name in model_names:
            config_file = tmp_path / "models" / dir_name / "config.pbtxt"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            file_content = f'name: "{dir_name}"\n'
            config_file.write_text(file_content)
    yield _func


def test_fixture_create_models_dir(tmp_path, create_triton_models_dir):
    names = ["liver", "heart"]
    create_triton_models_dir(names)
    assert sorted(os.listdir(str(tmp_path / "models"))) == sorted(names)


@dataclass
class MockConfig:
    models: List[str] = None


@pytest.mark.parametrize("configs, expected", [
    ([MockConfig(), MockConfig(["m1", "m2"]), MockConfig(["m3"]), MockConfig(), MockConfig(["m4", "m5", "m6"])],
     ["m1", "m2", "m3", "m4", "m5", "m6"]),
    ([MockConfig(), MockConfig()], []),
    ([MockConfig(["m1", "m2"]), MockConfig(["m1"])], ["m1", "m2"])
])
def test_extract_models_from_configs(configs, expected):
    result = _extract_models_from_configs(configs)
    assert sorted(result) == expected


@patch("triton_utils._extract_models_from_configs")
def test_check_model_repository_no_models_needed(mock_models, tmp_path):
    mock_models.return_value = []
    mock_configs = MagicMock()
    result = check_models_directory(mock_configs, tmp_path)
    assert result == []


@patch("triton_utils._extract_models_from_configs")
def test_check_model_repository_no_model_dir(mock_models):
    mock_models.return_value = ["liver", "spleen", "heart"]
    mock_configs = MagicMock()
    with pytest.raises(SystemExit):
        check_models_directory(mock_configs, None)


@pytest.mark.parametrize("mock_models, dir_name, file_content", [
    pytest.param(["liver"], "liver", 'name: "segmentation_liver_v1"\n', id="content_not_match"),
    pytest.param(["liver"], "liver_seg", 'name: "liver"\n', id="dir_name_not_match"),
    pytest.param(["liver", "heart"], "liver", 'name: "liver"\n', id="missing_model")
])
@patch("triton_utils._extract_models_from_configs")
def test_check_model_repository_bad_input(mock_func, mock_models, dir_name, file_content, tmp_path):
    mock_func.return_value = mock_models
    mock_configs = MagicMock()
    config_file = tmp_path / "models" / dir_name / "config.pbtxt"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(file_content)
    with pytest.raises(SystemExit):
        check_models_directory(mock_configs, config_file.parents[1])


@pytest.mark.parametrize("mock_models", [
    pytest.param(["liver"], id="one_model"),
    pytest.param(["liver", "spleen", "heart"], id="three_models"),
])
@patch("triton_utils._extract_models_from_configs")
def test_check_model_repository_good_input(mock_func, mock_models, tmp_path, create_triton_models_dir):
    mock_func.return_value = mock_models
    mock_configs = MagicMock()

    create_triton_models_dir(mock_models + ["eyes", "lung"])

    result = check_models_directory(mock_configs, tmp_path / "models")
    assert sorted(result) == sorted(mock_models)


@pytest.mark.parametrize("mock_configs, exp_mode", [
    pytest.param([MagicMock(**{"models": True, "services": None})], RUN_MODE.MODEL_REPO, id="model_repo"),
    pytest.param([MagicMock(**{"models": None, "services": True})], RUN_MODE.PIPELINE_SERVICES, id="services"),
    pytest.param([MagicMock(**{"models": None, "services": None})], RUN_MODE.NO_INFERENCE_SERVER, id="neither"),
])
def test_decide_method_to_run_triton(mock_configs, exp_mode):
    assert decide_method_to_run_triton(mock_configs) == exp_mode


def test_decide_method_to_run_triton_error():
    mock_configs = [MagicMock(**{"models": True, "services": True})]
    with pytest.raises(SystemExit):
        decide_method_to_run_triton(mock_configs)


@pytest.mark.parametrize(
    "model_names, mock_reponses",
    [
        pytest.param(
            [],
            [MagicMock(**{"status_code": 200})],
            id="no_model_names"),
        pytest.param(
            ["model1"],
            [MagicMock(**{"status_code": 200, "text": None}), MagicMock(**{"status_code": 200, "text": None})],
            id="1_model_name"),
    ]
)
@patch("triton_utils.TRITON_WAIT_SLEEP_TIME_SECONDS", 0)
@patch("triton_utils.TRITON_WAIT_TIME_SECONDS", 0)
@patch("triton_utils.requests")
def test_check_triton_status_200(mock_requests, model_names, mock_reponses):
    mock_requests.configure_mock(**{"ConnectionError": ValueError})
    mock_requests.get.side_effect = mock_reponses
    check_triton_status(triton_models_names=model_names, host="some_host", port="1234")
    assert f"http://some_host:1234" in mock_requests.get.call_args.args[0]


@pytest.mark.parametrize(
    "model_names, mock_reponses, exp_msg",
    [
        pytest.param(
            [],
            [MagicMock(**{"status_code": 400, "text": "some msg"})],
            "Triton is not working", id="no_model_names"),
        pytest.param(
            ["model1"],
            [MagicMock(**{"status_code": 200, "text": None}), MagicMock(**{"status_code": 400, "text": "some msg"})],
            "Error:", id="1_model_name"),
    ]
)
@patch("triton_utils.TRITON_WAIT_SLEEP_TIME_SECONDS", 0)
@patch("triton_utils.TRITON_WAIT_TIME_SECONDS", 0)
@patch("triton_utils.requests")
def test_check_triton_status_error(mock_requests, model_names, mock_reponses, exp_msg):

    mock_requests.configure_mock(**{"ConnectionError": ValueError})
    mock_requests.get.side_effect = mock_reponses
    with pytest.raises(SystemExit) as exc:
        check_triton_status(triton_models_names=model_names)
    assert exp_msg in str(exc.value)


@patch("triton_utils.subproc_run_wrapper")
def test_inspect_ip_address(mock_subproc_run_wrapper):
    mock_subproc_run_wrapper.return_value = "'125.12.199.0'"
    result = inspect_ip_address("container_name")
    assert result == "125.12.199.0"


@pytest.mark.parametrize("model_names", [["spleen", "arm", "legs"], []])
@patch("triton_utils.check_triton_status")
@patch("triton_utils.inspect_ip_address")
@patch("triton_utils.subproc_run_wrapper")
def test_start_triton(mock_subproc_run_wrapper, mock_inspect, mock_check_triton_status, model_names):
    mock_subproc_run_wrapper.return_value = "container_id"
    mock_inspect.return_value = "ip_address"
    result = start_triton("models", ["some", "command"], triton_models_names=model_names)
    assert result == ("container_id", "ip_address")

    # Check that all the models used are listed in the call_args for Popen
    if model_names != []:
        for name in model_names:
            assert f"--load-model={name}" in mock_subproc_run_wrapper.call_args_list[0].args[0]


@patch("triton_utils.subproc_run_wrapper")
@patch("triton_utils.check_models_directory")
@patch("triton_utils.start_triton")
def test_run_triton_model_repo(mock_start_triton, mock_check_dir, mock_subproc_run_wrapper):
    triton_models_names = ["spleen", "arm", "legs"]
    mock_check_dir.return_value = triton_models_names

    process_mock = MagicMock()
    process_mock.configure_mock(**{"returncode": None, "terminate.return_value": None})
    mock_start_triton.return_value = ("container_id", "ip_address")

    with run_triton_model_repo([], "some_dir"):
        pass

    mock_subproc_run_wrapper.assert_called_once()
    assert "container_id" in mock_subproc_run_wrapper.call_args_list[0].args[0]
