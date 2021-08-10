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
import sys

sys.path.append('{}/{}'.format(os.path.dirname(os.path.realpath(__file__)), '../src'))
from clarac_utils import run_clarac  # nopep8  # noqa: E402
from pipeline_utils import run_pipeline  # nopep8  # noqa: E402
from topology_sort import topo_sort_pipeline  # nopep8  # noqa: E402
from utils import assert_installed, check_images_and_tags, set_up_logging  # nopep8  # noqa: E402

from cli import parse_args  # nopep8  # noqa: E402


def main():
    parsed_args = parse_args(sys.argv[1:])

    set_up_logging(parsed_args.verbose)

    assert_installed("clarac")
    assert_installed("docker")
    logging.info("All software dependencies are fullfilled.")

    pipeline_config = run_clarac(parsed_args.pipeline_path)

    check_images_and_tags(pipeline_config.operators)

    execution_order = topo_sort_pipeline(pipeline_config.operators)

    run_pipeline(execution_order, parsed_args.input_dir, parsed_args.metrics_dir,
                 parsed_args.models_dir, parsed_args.force)


if __name__ == "__main__":    # pragma: no cover
    main()
