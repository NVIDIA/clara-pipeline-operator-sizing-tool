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


import csv
import dataclasses
import logging
import math
import shutil
import sys
from pathlib import Path
from subprocess import PIPE, Popen
from subprocess import run as subproc_run
from typing import List

from clarac_utils import OperatorConfig
from constants import ON_POSIX, TRITON_IMAGE_TAG


def round_up_to_multiple(x, base):
    """Round up the given number to the nearest multiple of the given base number."""
    return math.ceil(float(x) / float(base)) * base


def convert_percent_to_cores(x):
    "Convert the given percentage to CPU cores."
    return int(math.ceil(x / 100.0))


def assert_installed(prog: str):
    """Check if the given program is installed, terminate if not.

    Args:
        prog: Name of the commandline program

    Returns:
        None. If program is not installed, sys.exit(1)
    """
    logging.debug(f"Checking for dependency {prog} ...")
    if not shutil.which(prog):
        sys.stderr.write(f"error: {prog} not installed, please install {prog}\n")
        sys.exit(1)
    logging.debug(f"Dependency {prog} fulfilled")


def set_up_logging(verbose):
    """Setup logging for cpost to standard out.

    Args:
        verbose: Boolean value indicating whether log level will be debug or not

    Returns:
        None.
    """
    if verbose:  # pragma: no cover
        level = logging.DEBUG
    else:    # pragma: no cover
        level = logging.INFO
    # logging config are default to StreamHandlers
    logging.basicConfig(format='%(message)s', level=level)  # pragma: no cover


def check_images_and_tags(operators: List[OperatorConfig]):
    """For the image and tag of each operator, examine local images and pull if not found locally.

    Args:
        operators: List of OperatorConfig objects

    Returns:
        None

    Raises:
        sys.exit if the docker pull command errorred out
    """
    uses_triton_model_repo = False
    logging.info("Checking for container images and tags needed for the pipeline...")

    def _check_image_exists_locally(image_and_tag):
        logging.debug(f"Checking if `{image_and_tag}` are in local images...")
        local_check_proc = subproc_run(
            ["docker", "images", image_and_tag, "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True)
        result = local_check_proc.stdout.decode('UTF-8')
        if image_and_tag in result:
            logging.debug(f"`{image_and_tag}` found.")
            return True
        else:
            return False

    def _pull_image(image_and_tag):
        logging.debug(f"`{image_and_tag}` not found, try pulling from registry ...")
        pull_proc = subproc_run(["docker", "pull", image_and_tag], capture_output=True)
        if pull_proc.returncode == 0:
            logging.debug(f"Docker pull command for `{image_and_tag}` returned with code {pull_proc.returncode}")
            logging.debug(f"stdout is: \n{pull_proc.stdout.decode('UTF-8').strip()}")
        else:
            logging.error(f"Docker pull command for `{image_and_tag}` returned with code {pull_proc.returncode}")
            logging.error(f"stdout is: {pull_proc.stdout.decode('UTF-8')}")
            logging.error(f"stderr is: {pull_proc.stderr.decode('UTF-8')}")
            sys.exit("Please verify docker access and the pipeline definition")

    for operator in operators:
        if not _check_image_exists_locally(operator.image_n_tag):
            _pull_image(operator.image_n_tag)
        if operator.models:
            uses_triton_model_repo = True
        if operator.services:
            for op_service in operator.services:
                if not _check_image_exists_locally(op_service.image_n_tag):
                    _pull_image(op_service.image_n_tag)
    if uses_triton_model_repo:
        if not _check_image_exists_locally(TRITON_IMAGE_TAG):
            _pull_image(TRITON_IMAGE_TAG)

    logging.info("All container images are ready to be used.")


def subproc_run_wrapper(cmd, **kwargs):
    sub_proc = subproc_run(cmd, capture_output=True, **kwargs)
    if sub_proc.returncode == 0:
        std_out = sub_proc.stdout.decode('UTF-8').strip()
        logging.debug(f"Subprocess returned with stdout {std_out}")
        return std_out
    else:
        logging.error(
            f"Running {cmd} returned with {sub_proc.returncode} with error {sub_proc.stderr}")
        return sys.exit(f"Failed to run subprocess with command {cmd}")


def prompt_yes_or_no(condition: str):
    """Prompt the user with a question and waits for the y/n input.

    Args:
        condition: Condition that needs user's input

    Returns:
        Boolean value corresponding to yes or no
    """
    while "the answer is invalid":
        reply = input(condition + ' (y/n): ').lower().strip()
        if reply:
            if reply[0] == 'y':
                return True
            if reply[0] == 'n':
                return False


def write_to_csv(que, field_names, output_file):
    """Write data in que to the output file in csv format.

    Args:
        que: a multiprocess.Queue contains the data to be written
        field_names: Header for the csv file
        output_file: String or Path of the output file location

    Returns:
        None
    """
    output_file = Path(output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    with open(output_file, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writeheader()
        while True:
            item = que.get()
            if item is None:
                continue
            if item == 0:
                que.close()
                break
            csv_writer.writerow(dataclasses.asdict(item))
            f.flush()
    logging.info(f"Results are stored in {output_file}")
