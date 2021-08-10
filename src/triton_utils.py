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
import time
from contextlib import contextmanager
from enum import Enum, auto
from typing import List

import requests
from clarac_utils import OperatorConfig
from constants import (TRITON_HTTP_PORT, TRITON_IMAGE_TAG, TRITON_READY_TIMEOUT_SECONDS,
                       TRITON_WAIT_SLEEP_TIME_SECONDS, TRITON_WAIT_TIME_SECONDS)
from utils import subproc_run_wrapper


class RUN_MODE(Enum):
    NO_INFERENCE_SERVER = auto()
    MODEL_REPO = auto()
    PIPELINE_SERVICES = auto()


def _extract_models_from_configs(op_configs: List[OperatorConfig]):
    """Helper method to obtain models from list of OperatorConfig.

    Args:
        op_configs: List of OperatorConfigs to extract information from

    Returns:
        List of string which represents the names of each model with no repeating models
    """
    logging.debug("Abstracting model form pipeline definition")
    result = list(set([model for op in op_configs if op.models for model in op.models]))
    logging.debug(f"The models present are `{result}`")
    return result


def check_models_directory(op_configs, models_dir) -> List[str]:
    """Checks if the model directory contains the models needed in the pipeline.

    Args:
        op_configs: List of OperatorConfigs to extract information from
        models_dir: A directory that contains Triton models

    Returns:
        model_names: List of model names used by this pipeline
    """
    logging.info("Checking model directory for dependent models ...")
    required_models = _extract_models_from_configs(op_configs)
    if required_models == []:
        logging.debug("Pipeline did not specify any Triton models, skipping check for models_dir")
        return []
    else:
        logging.debug("Examining model directory ...")
        if models_dir is None:
            sys.exit(f"Model directory must be provided since your pipeline uses: {required_models}")

        # The directory can contain more models than what's needed
        model_names = []
        for model_name in required_models:
            logging.debug(f"Checking for model `{model_name}` ...")
            matching_config = list(models_dir.glob(f"{model_name}/config.pbtxt"))
            if len(matching_config) == 0:
                sys.exit(f"Model `{model_name}` is missing in the models directory")
            elif len(matching_config) > 1:
                logging.warning(
                    f"Found more than one matching config file for model `{model_name}`. Using the first occurrence.")
            model_path = matching_config[0]
            with open(model_path) as f:
                name_in_file = f.readline().split(":")[1].strip()[1:-1]
                if name_in_file != model_path.parent.name:
                    sys.exit(
                        f"Expected name in config {name_in_file} to be equal to directory name {model_path.parent.name}")
            model_names.append(model_path.parent.name)

        logging.info("All model directory checks are complete!")
        return model_names


def decide_method_to_run_triton(op_configs) -> RUN_MODE:
    """Decide how to run triton based on the given op_configs.

    Args:
        op_configs: List of OperatorConfig objects

    Return:
        RUN_MODE.MODEL_REPO, RUN_MODE.PIPELINE_SERVICES or RUN_MODE.NO_INFERENCE_SERVER

    Raises:
        SystemExit if both models and services are present in the op_config
    """
    model_repo = False
    services = False
    for op in op_configs:
        if op.models:
            model_repo = True
        if op.services:
            services = True
    if model_repo and services:
        sys.exit("CPOST does not support model_repository and pipeline services at the same time")
    if model_repo:
        return RUN_MODE.MODEL_REPO
    elif services:
        return RUN_MODE.PIPELINE_SERVICES
    return RUN_MODE.NO_INFERENCE_SERVER


def check_triton_status(triton_models_names=[], host="localhost", port=TRITON_HTTP_PORT):
    """Check status of Triton server via http.

    Kwargs:
        triton_models_names: list of triton model names to verify, default: []
        host: ip address of triton, default: localhost
        port: the port to query http status, default: "8000"

    Returns:
        None

    Raises:
        SystemExit if requests.get returned with a non-200 status
    """
    logging.debug("Waiting and checking Triton status ...")
    time.sleep(TRITON_WAIT_TIME_SECONDS)
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < TRITON_READY_TIMEOUT_SECONDS:
        time.sleep(TRITON_WAIT_SLEEP_TIME_SECONDS)
        try:
            ready = requests.get(f"http://{host}:{port}/api/status")
            if ready.status_code != 200:
                sys.exit(f"Triton is not working, status code = {ready.status_code} with message {ready.text}")
            break
        except requests.ConnectionError:
            continue
    else:
        raise TimeoutError("Timeout when waiting for triton to be ready.")

    # Verify that each model is ready
    for model_name in triton_models_names:
        ready = requests.get(
            f"http://{host}:{port}/api/status/{model_name}", timeout=TRITON_READY_TIMEOUT_SECONDS)
        if ready.status_code != 200:
            sys.exit(f"Error: {ready.status_code} {ready.reason}, {ready.headers}")
    logging.debug("Triton is ready to be used")


def inspect_ip_address(container_name):
    """Inspect and obtain the IP address for the given container.

    Args:
        container_name: docker name or docker container ID

    Returns:
        network_ip: the IP address of the container
    """
    cmd = ["docker", "inspect", "--format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'", container_name]
    output = subproc_run_wrapper(cmd)
    network_ip = output[1:-1]  # Strip away the quotes around the returned IP address
    logging.debug(f"{container_name} can be communicated on address {network_ip}")
    return network_ip


def start_triton(models_dir, command, image_tag=TRITON_IMAGE_TAG, triton_models_names=[]):
    """Starts triton container and wait for it to be ready.

    Args:
        models_dir: Absolute path of models_directory
        command: list of commands to run for the container

    Kwargs:
        image_tag: The image and tag for the container, e.g. image:tag, default to TRITON_IMAGE_TAG
        triton_models_names: List of triton model names to load, default = []

    Returns:
       triton_container_id, ip_address: Tuple of string
    """
    # build triton command
    loading_models = [f"--load-model={name}" for name in triton_models_names]
    cmd = ["docker", "run", "--gpus=1", "--rm", "-d", "-p8000:8000", "-p8001:8001", "-p8002:8002",
           "-v", f"{models_dir}:/models", image_tag] + command + loading_models
    logging.debug(f"Spinning up Triton with {cmd}")
    triton_container_id = subproc_run_wrapper(cmd)
    ip_address = inspect_ip_address(triton_container_id)
    check_triton_status(triton_models_names=triton_models_names, host=ip_address)
    return triton_container_id, ip_address


@contextmanager
def run_triton_model_repo(execution_order, models_dir):
    """Run Triton in a context manager if pipeline requires Triton.

    Args:
        execution_order: List of OperatorConfigs to extract information from
        models_dir: Absolute path of models_directory

    Yields:
        ip_address
    """
    try:
        triton_models_names = check_models_directory(execution_order, models_dir)
        command = ["tritonserver", "--model-repository=/models", "--model-control-mode=explicit"]
        triton_container_id, ip_address = start_triton(models_dir, command, triton_models_names=triton_models_names)
        yield ip_address
    finally:
        logging.debug("Stopping Triton ...")
        subproc_run_wrapper(["docker", "kill", triton_container_id])
        logging.debug("Finished cleaning up Triton")
