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
import sys
from dataclasses import dataclass
from subprocess import run as subproc_run
from tempfile import NamedTemporaryFile
from typing import Dict, List

import yaml


@dataclass
class ServiceConfig:
    name: str
    image_n_tag: str
    command: List[str]
    http_connections: Dict


@dataclass
class OperatorConfig:
    name: str
    image_n_tag: str
    command: List[str]
    variables: Dict
    inputs: List
    outputs: List
    models: List[str] = None
    services: List[ServiceConfig] = None

    def update_variables(self, var_dict):
        """Update the variable attribute with the given dictionary."""
        if self.variables:
            self.variables = {**var_dict, **self.variables}
        else:
            self.variables = {**var_dict}


@dataclass
class PipelineConfig:
    name: str
    operators: List[OperatorConfig]


def run_clarac(source_file: str) -> PipelineConfig:
    """Run Clara Complier in a subprocess using the given pipeline definition and parse the results.

    Args:
        source_file: path to the pipeline definition file

    Returns:
        A PipelineConfig object
    """
    def _extract_services(services):
        """Extract services section in pipeline definition into list of ServiceConfig."""
        result = []
        for service in services:
            service_image_n_tag = service["container"]["image"] + ":" + service["container"]["tag"]
            command = service["container"].get("command")
            if command:
                command = [c.replace("$(NVIDIA_CLARA_SERVICE_DATA_PATH)", "") for c in command]
            op_service = ServiceConfig(
                name=service["name"],
                image_n_tag=service_image_n_tag,
                command=command,
                http_connections={con["name"]: con["port"] for con in service["connections"].get("http")})
            result.append(op_service)
        return result

    logging.debug("Running Clara Complier to validate the pipeline definition ...")
    with NamedTemporaryFile() as result_file:
        cmd = ["clarac", "-p", source_file, "-o", result_file.name, "--resolve-imports"]
        proc = subproc_run(cmd)
        if proc.returncode != 0:
            logging.error(proc.stderr)
            sys.exit(proc.returncode)
        else:
            logging.debug(f"stdout from Clara Complier: {proc.stdout}")
            logging.debug(f"Clara Complier returned with error code {proc.returncode}, loading result as python object")

        try:
            config = yaml.load(result_file, yaml.FullLoader)
        except yaml.YAMLError as exc:
            logging.error(f"Error in configuration file from Clara Complier: {exc}")
            sys.exit(2)
        logging.debug(f"The content loaded from Clara Complier is: {config}")

    operators = []
    # Get the objects of interest, construct a list, and return it
    for op in config["operators"]:
        # Get services and names of triton models used by this operator
        op_models = [model_dict["name"] for model_dict in op.get("models")] if op.get("models") else None
        op_services = _extract_services(op.get("services")) if op.get("services") else None

        image_n_tag = op["container"]["image"] + ":" + op["container"]["tag"]
        cmd = op["container"].get("command")
        operator = OperatorConfig(name=op["name"], image_n_tag=image_n_tag, command=cmd, variables=op.get(
            "variables"), inputs=op["input"], outputs=op.get("output"), models=op_models, services=op_services)
        operators.append(operator)

    return PipelineConfig(name=config["name"], operators=operators)
