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


import argparse
import sys
from enum import IntEnum
from pathlib import Path


class ContinueOptions(IntEnum):
    """Enum to organize options to prompt user, continue execution, or stop execution when operator fails."""
    NONE = 0  # prompt user y/n
    CONT = 1  # continue execution
    STOP = 2  # stop execution

    # methods for compatible with argparse and error message

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return ContinueOptions[s.upper()]
        except KeyError:  # To be used with `choices` in add_argument()
            return s


class MyParser(argparse.ArgumentParser):
    """Custom parser class to override the error method."""

    def error(self, message):
        """Overriding the default error method to print help message before exiting."""
        sys.stderr.write('error: %s\n' % message)
        self.print_help(sys.stderr)
        self.exit(2)


def valid_file(path):
    """Helper method for parse_args to convert to Path and verify if the file path exists.

    Args:
        path: path to file from parse_args()

    Returns:
        The absolute path of the given file path if it exists

    Raises:
        argparse.ArgumentTypeError if the file given does not exist
    """
    path = Path(path)
    if path.exists() and path.is_file():
        return path.absolute()
    raise argparse.ArgumentTypeError(f"No such file or the given path is not a file: '{path}'")


def valid_dir(path):
    """Helper method for parse_args to convert to Path and verify if the directory exists.

    Args:
        path: path to directory from parse_args()

    Returns:
        The absolute path of the given directory if it exists

    Raises:
        argparse.ArgumentTypeError if the directory given does not exist or if not a directory
    """
    path = Path(path)
    if path.exists() and path.is_dir():
        return path.absolute()
    raise argparse.ArgumentTypeError(f"No such directory or the given path is not a directory: '{path}'")


def parse_args(args):
    """Create an argument parser and parse the command-line arguments.

    Args:
        args: A list of arguments to parse

    Returns:
        A parser object containing parsed arguments
    """

    parser = MyParser(prog="cpost", description="Clara Pipeline Sizing Tool CLI")

    parser.add_argument("pipeline_path", metavar="<pipeline_path>",
                        type=valid_file, help="pipeline definition file path")

    parser.add_argument("input_dir", metavar="<input_dir>", type=valid_dir, help="input payload directory")

    parser.add_argument("--metrics_dir", type=valid_dir,
                        help="metrics output directory, if not specified, write to stdout")

    parser.add_argument("--models_dir", type=valid_dir,
                        help="directory for Triton models, required if pipeline uses Triton")

    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="verbose output (DEBUG level). If not specified, default output is INFO level.")

    parser.add_argument(
        "--force", default=ContinueOptions.NONE, const=ContinueOptions.NONE, nargs='?', type=ContinueOptions.argparse,
        choices=list(ContinueOptions),
        help='force continue or stop when operator failure occurs. \
            (default: %(default)s, which will prompt the user for each failure).')

    return parser.parse_args(args)
