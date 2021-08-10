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
from unittest.mock import MagicMock, patch

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from main import main  # nopep8  # noqa: E402


@patch("main.run_pipeline")
@patch("main.topo_sort_pipeline")
@patch("main.check_images_and_tags")
@patch("main.run_clarac")
@patch("main.assert_installed")
@patch("main.set_up_logging")
@patch("main.parse_args")
def test_main(mock_parse, mock_set_logging, mock_assert_install, mock_run_clarac, mock_check, mock_sort, mock_run):

    mock_parse.return_value = MagicMock(**{"verbose": 2, "pipeline_path": "some_path"})
    mock_run_clarac.return_value = MagicMock(**{"operators": "operators"})
    main()
    mock_set_logging.assert_called_with(2)
    assert mock_assert_install.call_count == 2
    mock_run_clarac.assert_called_with("some_path")
    mock_check.assert_called_with("operators")
    mock_sort.assert_called_with("operators")
    mock_run.assert_called_once()
