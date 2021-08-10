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


import io
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from clarac_utils import OperatorConfig  # nopep8  # noqa: E402
from utils import (assert_installed, check_images_and_tags, convert_percent_to_cores,  # nopep8  # noqa: E402
                   prompt_yes_or_no, round_up_to_multiple, subproc_run_wrapper, write_to_csv)


@pytest.mark.parametrize("data_in, base, data_out",
                         [(3, 2, 4),
                          (4, 2, 4),
                          (15.5, 5, 20),
                          (148.05, 256, 256),
                          (256.05, 256, 512)])
def test_round_up_to_multiple(data_in, base, data_out):
    assert round_up_to_multiple(data_in, base) == data_out


@pytest.mark.parametrize("data_in, data_out",
                         [(100.05, 2),
                          (1343.5, 14),
                          (50.55, 1)])
def test_convert_percent_to_cores(data_in, data_out):
    assert convert_percent_to_cores(data_in) == data_out


@pytest.mark.parametrize("program, exist", [("echo", True), ("clara", True), ("claraabc", False)])
def test_assert_installed(program, exist):
    if program == "clara":
        pytest.skip()
    if exist:
        assert assert_installed(program) is None
    else:
        with pytest.raises(SystemExit) as exc:
            assert_installed(program)
        assert exc.value.code == 1


@pytest.mark.parametrize("mocked_return, run_called_count", [
    pytest.param(MagicMock(**{"stdout": b'tag1\n'}), 2, id="exists_locally"),
    pytest.param(MagicMock(**{"stdout": b'', "returncode": 0}), 4, id="can_be_pulled"),
    pytest.param(MagicMock(**{"stdout": b'', "returncode": 1, "stderr": b'error message'}), 2, id="pull_failed"),
])
@patch("utils.subproc_run")
def test_check_images_and_tags(mock_subproc_run, mocked_return, run_called_count):
    mock_subproc_run.return_value = mocked_return
    mock_service = [MagicMock(**{"image_n_tag": "tag1"})]
    op1 = OperatorConfig("Input1", "tag1", None, None, [{"path": "/input"}], None, None, mock_service)
    if mocked_return.returncode == 1:
        with pytest.raises(SystemExit):
            check_images_and_tags([op1])
    else:
        check_images_and_tags([op1])
    assert mock_subproc_run.call_count == run_called_count


@patch("utils.TRITON_IMAGE_TAG", "triton-tag")
@pytest.mark.parametrize("mocked_return, expect_exit, run_called_count",
                         [
                             pytest.param([MagicMock(**{"stdout": b'triton-tag\n'})],
                                          False, 2, id="exists_locally"),
                             pytest.param(
                                 [MagicMock(**{"stdout": b''}),
                                  MagicMock(**{"stdout": b'', "returncode": 0})],
                                 False, 3, id="can_be_pulled"),
                             pytest.param(
                                 [MagicMock(**{"stdout": b''}),
                                  MagicMock(**{"stdout": b'', "returncode": 1, "stderr": b'error message'})],
                                 True, 3, id="pull_failed"),

                         ])
@patch("utils.subproc_run")
def test_check_images_and_tags_with_triton(mock_subproc_run, mocked_return, expect_exit, run_called_count):
    mock_subproc_run.side_effect = [MagicMock(**{"stdout": b'tag1\n'})] + mocked_return
    op1 = OperatorConfig("Input1", "tag1", None, None, [{"path": "/input"}], None, ["model1"])
    if expect_exit:
        with pytest.raises(SystemExit):
            check_images_and_tags([op1])
    else:
        check_images_and_tags([op1])
    assert mock_subproc_run.call_count == run_called_count


@pytest.mark.parametrize("mocked_return", [
    pytest.param(MagicMock(**{"stdout": b'container_id\n', "returncode": 0}), id="all_good"),
    pytest.param(MagicMock(**{"stderr": b'error message', "returncode": 1}), id="error")
])
@patch("utils.subproc_run")
def test_subproc_run_wrapper(mock_subproc_run, mocked_return):
    mock_subproc_run.return_value = mocked_return
    if mocked_return.returncode == 1:
        with pytest.raises(SystemExit):
            subproc_run_wrapper(["some", "cmd"])
    else:
        result = subproc_run_wrapper(["some", "cmd"])
        assert result == "container_id"


@pytest.mark.parametrize("choice, expected_result", [
    ("y", True),
    ("Y", True),
    ("yes", True),
    ("YES", True),
    ("yup", True),
    ("n", False),
    ("N", False),
    ("no", False),
    ("NO", False),
    ("nope", False),
    ("j\nx\nyeeee", True),
    ("exxxy\nadsfa\nnaaah", False),
    ("\nx\ny", True)
])
def test_prompt_yes_or_no(choice, expected_result):
    sys.stdin = io.StringIO(choice)
    assert prompt_yes_or_no("Please give your response") == expected_result


def test_write_to_csv(tmp_path):

    @ dataclass
    class MockMetric:
        field1: str
        field2: int

    mock_q = MagicMock()
    mock_q.get.side_effect = [None, MockMetric("abc", 12), MockMetric("fdvc", 15), 0]
    output_dir = tmp_path / "sub_dir" / "test_write_to_csv"
    field_names = ["field1", "field2"]
    write_to_csv(mock_q, field_names, output_dir)

    assert output_dir.read_text() == "field1,field2\nabc,12\nfdvc,15\n"
