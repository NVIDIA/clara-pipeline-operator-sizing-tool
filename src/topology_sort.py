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
from collections import defaultdict


class PipelineDAG:
    """Class for the Pipeline DAG used for sorting."""

    def __init__(self):
        self.input_deg_graph = defaultdict(lambda: 0)
        self.output_graph = defaultdict(list)  # dictionary containing adjacency List

    def add_input_edge(self, node: str, input_node: str):
        """Add the node by giving its input node.

        Args:
            node: Node to be added
            input_node: One of its dependency nodes

        Returns:
            None
        """
        self.output_graph[input_node].append(node)
        # Update the input_degree_graph as we are adding each node
        self.input_deg_graph[input_node] += 0
        self.input_deg_graph[node] += 1

    def topological_sort(self):
        """Topologically sort the given graph based on Kahn's algorithm.

        Args:
            None

        Returns:
            A list that is the topological order of the current graph

        Raises:
            Runtime Error if the graph contains cycles
        """
        visited_count = 0
        topo_order = []
        # Create a list for all node with in-degree 0
        zero_indegree = [node for node, length in self.input_deg_graph.items() if length == 0]

        # Pick zero-in-degree node one by one and check if any new zero-in-degree node shows up
        while zero_indegree:
            # Get the first zero in-degree node and add it to topo_order
            cur_node = zero_indegree.pop(0)
            topo_order.append(cur_node)

            # Iterate through output nodes of cur_node and decrease their in-degree by 1
            for i in self.output_graph[cur_node]:
                self.input_deg_graph[i] -= 1
                # If in-degree becomes zero, add it to zero_indegree
                if self.input_deg_graph[i] == 0:
                    zero_indegree.append(i)

            visited_count += 1

        # Check for a cycle in the graph
        if visited_count != len(self.output_graph.keys()):
            raise RuntimeError("There exists a cycle in the given graph")

        return topo_order


def topo_sort_pipeline(operators):
    """Topologically sort the given operators.

    Args:
        operators: List of OperatorConfig objects

    Returns:
        A topologically ordered OperatorConfig objects
    """
    logging.debug(f"Topolocally order the given input: {operators}")
    if len(operators) == 1:
        result = operators.copy()
    else:
        # Construct a dictionary from operators so that we can convert names back to OperatorConfigs later
        op_dict = {op.name: op for op in operators}
        dag = PipelineDAG()
        for op in operators:
            for input_path in op.inputs:
                if input_path.get("from"):
                    dag.add_input_edge(op.name, input_path.get("from"))
        sequence = dag.topological_sort()
        result = [op_dict[op_name] for op_name in sequence]
    logging.debug(f"Topologically order result is: {result}")
    return result
