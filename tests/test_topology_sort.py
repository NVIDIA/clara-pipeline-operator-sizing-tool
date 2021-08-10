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

import pytest

sys.path.append("{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "../src"))
from clarac_utils import OperatorConfig  # nopep8  # noqa: E402
from topology_sort import PipelineDAG, topo_sort_pipeline  # nopep8  # noqa: E402


def test_topo_sort():
    g = PipelineDAG()
    g.add_input_edge(2, 5)
    g.add_input_edge(0, 5)
    g.add_input_edge(0, 4)
    g.add_input_edge(1, 4)
    g.add_input_edge(3, 2)
    g.add_input_edge(1, 3)
    assert g.topological_sort() == [5, 4, 2, 0, 3, 1]


def test_topo_sort_2():
    g = PipelineDAG()
    g.add_input_edge(2, 1)
    g.add_input_edge(3, 2)
    g.add_input_edge(4, 3)
    assert g.topological_sort() == [1, 2, 3, 4]


def test_topo_sort_error():
    g = PipelineDAG()
    g.add_input_edge(2, 1)
    g.add_input_edge(3, 2)
    g.add_input_edge(1, 3)
    with pytest.raises(RuntimeError):
        g.topological_sort()


def test_a_pipeline():
    op1 = OperatorConfig("Input1", "tag", None, None, [{"path": "/input"}], None)
    op2 = OperatorConfig("Input2", "tag", None, None, [{"from": "Input1", "path": "/input"}], None)
    op3 = OperatorConfig("Input3", "tag", None, None, [{"from": "Input2", "path": "/input"}], None)

    sequence = topo_sort_pipeline([op2, op3, op1])
    assert sequence == [op1, op2, op3]


def test_a_single_operator_pipeline():
    op1 = OperatorConfig("Input1", "tag", None, None, [{"path": "/input"}], None)

    sequence = topo_sort_pipeline([op1])
    assert sequence == [op1]


def test_twp_operator_pipeline():
    op1 = OperatorConfig("Input1", "tag", None, None, [{"path": "/input"}], None)
    op2 = OperatorConfig("Input2", "tag", None, None, [{"from": "Input1", "path": "/input"}], None)

    sequence = topo_sort_pipeline([op2, op1])
    assert sequence == [op1, op2]


def test_complex_pipeline():
    op1 = OperatorConfig("Input1", "tag", None, None, [{"path": "/input"}], None)
    op2 = OperatorConfig("Input2", "tag", None, None, [{"path": "/input"}], None)
    op3 = OperatorConfig("Input3", "tag", None, None,
                         [{"from": "Input1", "path": "/input"},
                          {"from": "Input2", "path": "/input"}],
                         None)
    op4 = OperatorConfig("Input4", "tag", None, None, [{"from": "Input2", "path": "/input"}], None)
    op5 = OperatorConfig("Input5", "tag", None, None,
                         [{"from": "Input3", "path": "/input"},
                          {"from": "Input4", "path": "/input"}],
                         None)

    sequence = topo_sort_pipeline([op3, op4, op1, op2, op5])
    assert sequence == [op1, op2, op3, op4, op5]
