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
from unittest.mock import MagicMock, patch

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from clarac_utils import OperatorConfig, PipelineConfig, ServiceConfig, run_clarac  # nopep8  # noqa: E402


@pytest.mark.parametrize("og_variables, exp_variables",
                         [(None, {"a": 1, "b": 2}),
                          ({"c": 3},
                           {"c": 3, "a": 1, "b": 2})])
def test_op_config_update_variables(og_variables, exp_variables):
    new_variables = {"a": 1, "b": 2}
    op = OperatorConfig("op1", "image_tag", None, og_variables, None, None)
    op.update_variables(new_variables)
    assert op.variables == exp_variables


@patch("clarac_utils.subproc_run")
def test_run_clarac_subproc_error(mock_subproc_run, tmp_path):
    mock_subproc_run.return_value = MagicMock(**{"returncode": 1, "stderr": "some error"})
    with pytest.raises(SystemExit):
        run_clarac(tmp_path)


@patch("clarac_utils.NamedTemporaryFile")
@patch("clarac_utils.subproc_run")
def test_run_clarac_yaml_error(mock_subproc_run, mock_temp_file, tmp_path):
    mock_subproc_run.return_value = MagicMock(**{"returncode": 0, "stdout": "some output"})
    mock_file = tmp_path / "bad.yaml"
    mock_file.touch()
    mock_file.write_text("api-version: '0.4.0'\n name: null-pipeline")
    with open(mock_file) as mock_file_obj:
        mock_temp_file.return_value.__enter__.return_value = mock_file_obj

        with pytest.raises(SystemExit):
            run_clarac(tmp_path)


@pytest.mark.skip("Skipping due to pipeline setup for clarac is incomplete")
def test_run_clarac():
    pipeline_file = Path(__file__).parent / "pipelines" / ("nullpipeline.yaml")

    config = run_clarac(pipeline_file)
    assert isinstance(config, PipelineConfig)
    assert config.name == "null-pipeline"
    assert len(config.operators) == 3
    op = config.operators[0]
    assert op.name == "null-reader"
    assert op.image_n_tag == "null-pipeline/operator-py:0.8.1"
    assert op.command is None
    assert op.variables == {"CLARA_TRACE": 2}
    assert op.inputs == [{"name": None, "path": "/input"}]
    assert op.outputs == [{"name": None, "path": "/output"}]
    assert op.models is None
    assert op.services is None

    op = config.operators[1]
    assert op.name == "null-inference"
    assert op.image_n_tag == "null-pipeline/operator-py:0.8.1"
    assert op.command is None
    assert op.variables == {"CLARA_TRACE": 2}
    assert op.inputs == [{"from": "null-reader", "name": None, "path": "/input"}]
    assert op.outputs == [{"name": None, "path": "/output"}]
    assert op.models is None
    assert op.services is None

    op = config.operators[2]
    assert op.name == "null-writer"
    assert op.image_n_tag == "null-pipeline/operator-py:0.8.1"
    assert op.command is None
    assert op.variables == {"CLARA_TRACE": 2}
    assert op.inputs == [{"from": "null-inference", "name": None, "path": "/input"}]
    assert op.outputs == [{"name": None, "path": "/output"}]
    assert op.models is None
    assert op.services is None


@pytest.mark.skip("Skipping due to pipeline setup for clarac is incomplete")
def test_run_clarac_with_triton_models():
    pipeline_file = Path(__file__).parent / "pipelines" / ("operator_with_model.yaml")

    config = run_clarac(pipeline_file)
    assert isinstance(config, PipelineConfig)
    assert config.name == "null-pipeline"
    assert len(config.operators) == 1
    op = config.operators[0]
    assert op.name == "null-reader"
    assert op.image_n_tag == "null-pipeline/operator-py:0.8.1"
    assert op.inputs == [{"name": None, "path": "/input"}]
    assert op.outputs == [{"name": None, "path": "/output"}]
    assert op.command == ["python", "register.py", "--agent", "renderserver"]
    assert op.models == ["segmentation_ct_spleen_v1", "segmentation_ct_liver_v1"]
    assert op.services is None


@pytest.mark.skip("Skipping due to pipeline setup for clarac is incomplete")
def test_run_clarac_with_pipeline_services():
    pipeline_file = Path(__file__).parent / "pipelines" / ("operator_with_services.yaml")

    config = run_clarac(pipeline_file)
    assert isinstance(config, PipelineConfig)
    assert config.name == "null-pipeline"
    assert len(config.operators) == 1
    op = config.operators[0]
    assert op.name == "null-reader"
    assert op.image_n_tag == "null-pipeline/operator-py:0.8.1"
    assert op.inputs == [{"name": None, "path": "/input"}]
    assert op.outputs == [{"name": None, "path": "/output"}]
    assert op.command is None
    assert op.models is None
    assert len(op.services) == 1
    op_service = op.services[0]
    assert isinstance(op_service, ServiceConfig)
    assert op_service.name == "trtis"
    assert op_service.image_n_tag == "nvcr.io/nvidia/tritonserver:latest"
    assert op_service.command == ["some", "command"]
    assert op_service.http_connections == {"NVIDIA_CLARA_TRTISURI": 8000}
