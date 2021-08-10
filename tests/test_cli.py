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
import re
import sys
from argparse import ArgumentTypeError

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from cli import ContinueOptions, parse_args  # nopep8  # noqa: E402


@pytest.fixture(scope="function")
def file_maker(tmp_path):
    """A function scoped pytest fixture to return the path of a temporary file."""
    file_path = tmp_path / "pipeline_defn"
    file_path.touch()
    return str(file_path)


def swap_pattern(pattern, substitute, args):
    """Helper method to substitute a pattern in args for cleaner tests."""
    return [re.sub(pattern, substitute, i) for i in args]


def test_swap_pattern():
    args = ["%tmp_file%", "some_input_dir", "%tmp%", "hello", "%tmp%"]
    result = swap_pattern("%tmp%", "abc", args)
    assert result == ["%tmp_file%", "some_input_dir", "abc", "hello", "abc"]


@pytest.mark.parametrize("input_args", [["%tmp_file%"], [], ["-x"], ["-v"]])
def test_missing_required_args(input_args, file_maker, capsys):
    input_args = swap_pattern(r'%tmp_file%', file_maker, input_args)

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parse_args(input_args)
    out, err = capsys.readouterr()

    assert "" == out
    assert "error: the following arguments are required" in err
    assert "usage: cpost" in err
    assert pytest_wrapped_e.value.code == 2


@pytest.mark.parametrize("input_args, error",
                         [
                             (["some_pipeline_path", "some_input_dir"], ArgumentTypeError),
                             (["/tmp", "/tmp"], ArgumentTypeError),
                             (["%tmp_file%", "some_input_dir"], ArgumentTypeError),
                             (["%tmp_file%", "%tmp_file%"], ArgumentTypeError),
                             (["%tmp_file%", "/tmp", "--metrics_dir", "some_dir"], ArgumentTypeError),
                             (["%tmp_file%", "/tmp", "--metrics_dir", "%tmp_file%"], ArgumentTypeError),
                             (["%tmp_file%", "/tmp", "--models_dir", "some_dir"], ArgumentTypeError),
                             (["%tmp_file%", "/tmp", "--models_dir", "%tmp_file%"], ArgumentTypeError)
                         ])
def test_invalid_path(input_args, error, file_maker):
    input_args = swap_pattern(r'%tmp_file%', file_maker, input_args)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        with pytest.raises(error) as excinfo:
            parse_args(input_args)
        assert "No such" in str(excinfo.value)
    assert pytest_wrapped_e.value.code == 2


@pytest.mark.parametrize("optional_dir_specified", [True, False])
def test_valid_path(optional_dir_specified, tmp_path, file_maker):
    input_dir = tmp_path / test_valid_path.__name__
    input_dir.mkdir()
    pipeline = file_maker

    if not optional_dir_specified:
        input_args = [pipeline, str(input_dir)]
        parsed = parse_args(input_args)
        assert parsed.input_dir == input_dir
        assert str(parsed.pipeline_path) == pipeline
        assert parsed.metrics_dir is None
        assert parsed.models_dir is None
        assert parsed.force == ContinueOptions.NONE
    else:
        metrics_dir = tmp_path / "test_output_metrics"
        metrics_dir.mkdir()
        models_dir = tmp_path / "model_repo"
        models_dir.mkdir()
        input_args = [pipeline, str(input_dir), "--metrics_dir", str(metrics_dir), "--models_dir", str(models_dir)]
        parsed = parse_args(input_args)
        assert parsed.input_dir == input_dir
        assert str(parsed.pipeline_path) == pipeline
        assert parsed.metrics_dir == metrics_dir
        assert parsed.models_dir == models_dir
        assert parsed.force == ContinueOptions.NONE


@pytest.mark.parametrize("force_args, exp_option",
                         [(["--force", "cont"], ContinueOptions.CONT),
                          (["--force=cont"], ContinueOptions.CONT),
                          ([], ContinueOptions.NONE),
                          (["--force", "none"], ContinueOptions.NONE),
                          (["--force", "stop"], ContinueOptions.STOP)])
def test_parse_force_options(force_args, exp_option, tmp_path, file_maker):
    input_dir = tmp_path / test_parse_force_options.__name__
    input_dir.mkdir()
    pipeline = file_maker
    input_args = force_args + [pipeline, str(input_dir)]
    parsed = parse_args(input_args)
    assert parsed.input_dir == input_dir
    assert str(parsed.pipeline_path) == pipeline
    assert parsed.metrics_dir is None
    assert parsed.models_dir is None
    assert parsed.force == exp_option


@pytest.mark.parametrize("force_args, err_msg",
                         [(["--force", "continue"], "argument --force: invalid choice: 'continue'"),
                          (["--force"], "argument --force: invalid choice:"),
                          (["--force", "aaaa"], "argument --force: invalid choice: 'aaaa'")])
def test_parse_force_options_error(force_args, err_msg, tmp_path, capsys, file_maker):
    input_dir = tmp_path / test_parse_force_options_error.__name__
    input_dir.mkdir()
    pipeline = file_maker
    input_args = force_args + [pipeline, str(input_dir)]
    with pytest.raises(SystemExit):
        parse_args(input_args)
    out, err = capsys.readouterr()
    assert err_msg in err
