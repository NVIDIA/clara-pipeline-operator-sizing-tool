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
from dataclasses import astuple
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from queue import Empty
from subprocess import PIPE, Popen
from subprocess import run as subproc_run
from tempfile import TemporaryDirectory

from clarac_utils import OperatorConfig
from constants import (ID_WAITING_TIME_SECONDS, LEGACY_TRITON_HTTP_ENV_VAR, LEGACY_TRTIS_HTTP_ENV_VAR,
                       METRIC_SAMPLING_PERIOD_SECONDS, ON_POSIX, TRITON_GRPC_ENV_VAR, TRITON_GRPC_PORT,
                       TRITON_HTTP_ENV_VAR, TRITON_HTTP_PORT)
from container import METRICS_HEADER, Container
from tabulate import tabulate
from triton_utils import (RUN_MODE, check_triton_status, decide_method_to_run_triton, inspect_ip_address,
                          run_triton_model_repo, start_triton)
from utils import convert_percent_to_cores, prompt_yes_or_no, round_up_to_multiple, subproc_run_wrapper, write_to_csv

from cli import ContinueOptions


def _enqueue_output(out, queue):
    """Reads the file content, add to queue, and close the file handler when done.

    Args:
        out: opened file handler or stdout
        queue: multiprocessing.Queue object

    Returns:
        None
    """
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def start_operator(container_id, id_returned_event, cmd):
    """Runs the given docker command and assign docker ID to the given shared value.

    Args:
        container_id: A multiprocessing.Value object to allow sharing of values.
        id_returned_event: A multiprocess.Event object, set when container_id is assigned.
        cmd: The full docker command to run an image.

    Returns:
        None
    """
    cmd_proc = Popen(cmd, stdout=PIPE, stderr=PIPE, close_fds=ON_POSIX)
    logging.info("Running operator ...")
    q = Queue()

    checker = Process(target=_enqueue_output, args=(cmd_proc.stdout, q), daemon=True)
    checker.start()

    while cmd_proc.poll() is None:
        try:
            raw_id = q.get_nowait()
        except Empty:
            continue
        else:
            # Validate the result, expect length to be 64 + 1 from '\n'
            if len(raw_id) == 65:
                container_id.value = raw_id.decode('utf-8').strip()
                logging.info(f"The container id is: {container_id.value}")
                id_returned_event.set()
                break
            else:
                sys.exit(f"The output of docker should be the 64 bit container ID, got {raw_id} instead.")
    else:
        if cmd_proc.returncode != 0:
            checker.terminate()
            checker.join()
            # This means that cmd_proc has errorred and terminated. Log the error and return
            logging.warning(f"Operator failed to start with returncode {cmd_proc.returncode}")
            sys.exit(f"The operator failed with stderr:\n{cmd_proc.stderr.read().decode('UTF-8')}")

    checker.terminate()
    checker.join()
    if cmd_proc.returncode is None:
        logging.debug("Operator is running...")
        # We need to know if docker exited correctly
        docker_wait_proc = subproc_run(["docker", "wait", container_id.value], capture_output=True)
        returned_str = docker_wait_proc.stdout.decode('UTF-8').strip()
        if returned_str == "0":
            logging.debug(f"Operator finished successfully with exitcode {returned_str}")
        else:
            logging.error(f"Operator failed with exitcode is: {returned_str}")
            try:
                return_code = int(returned_str)
                sys.exit(return_code)
            except ValueError:
                sys.exit(1)
    else:
        logging.debug(f"Docker run command returned with {cmd_proc.returncode}")


def sample_operator(container, que):
    """Samples and writes metrics for the given operator as long as its metrics paths exist.
    Sampling frequency is determined by METRIC_SAMPLING_PERIOD_SECONDS.

    Args:
        container: Container object.
        que: None or a multiprocessing.Queue object to store data that needs to be written to csv.

    Returns:
        None
    """
    # Waits for the files to be created by docker
    while not container.metrics_path_exists():
        continue

    # Samples until the files disappear
    logging.debug("Starts sampling container ...")
    before_sample = time.perf_counter()
    while container.metrics_path_exists():
        metric = container.sample_metrics()
        if que:
            que.put(metric)
        after_sample = time.perf_counter()
        sleep_time = METRIC_SAMPLING_PERIOD_SECONDS - (after_sample - before_sample)
        sleep_time = sleep_time if sleep_time > 0 else 0
        if sleep_time == 0:
            logging.info(
                f"Sampling taking longer than sampling period with time of {(after_sample - before_sample)} seconds")
        # NOTE: Due to the inaccurate nature of time.sleep(), our sampling will not be extremely precise
        time.sleep(sleep_time)
        before_sample = time.perf_counter()

    # Signal the end of que
    if que:
        que.put(0)
    logging.debug("Finished sampling container.")


def build_operator_cmd(input_dir: Path, data_folder_name: str, op_config: OperatorConfig, triton_ip: str = None):
    """Constructs the docker command used to run operator.

    Args:
        input_dir: A Path object for the input payload data in local system
        data_folder_name: Name of the data folder to store temporary data
        op_config: A OperatorConfig object containing information about the operator
        triton_ip: None, or Triton's IP address

    Returns:
        cmd: A list of string representing the docker command that can be used to run the operator
    """
    logging.debug(f"Constructing commands for operator {op_config.name} ...")
    op_output_dir = Path(data_folder_name) / op_config.name
    op_output_dir.mkdir()

    cmd = ["docker", "run", "-d", "--rm", "--env", "NVIDIA_CLARA_NOSYNCLOCK=1"]

    # If models is present, then we supply Triton ports to this
    if op_config.models:
        cmd.extend(["--env", f"{TRITON_HTTP_ENV_VAR}={triton_ip}:{TRITON_HTTP_PORT}"])
        cmd.extend(["--env", f"{LEGACY_TRITON_HTTP_ENV_VAR}={triton_ip}:{TRITON_HTTP_PORT}"])
        cmd.extend(["--env", f"{LEGACY_TRTIS_HTTP_ENV_VAR}={triton_ip}:{TRITON_HTTP_PORT}"])
        cmd.extend(["--env", f"{TRITON_GRPC_ENV_VAR}={triton_ip}:{TRITON_GRPC_PORT}"])

    # Add operator specific environment variables
    if op_config.variables:
        for key, value in op_config.variables.items():
            cmd.extend(["--env", f"{key}={value}"])

    # Mount input and output volumes
    def build_volume_mount(local, remote):
        return ["-v", ":".join([local, remote])]

    # Mount input volumes
    if op_config.inputs:
        for input_obj in op_config.inputs:
            # If `from` is not present, we use the input payload directory
            if input_obj.get("from") is None:
                cmd.extend(build_volume_mount(str(input_dir), input_obj["path"]))
            # If `from` is specified, we use the specified operator's output directory as the input for this operator
            else:
                op_input_dir = op_output_dir.parent / input_obj["from"]
                # If `name` is specified, then find the subdirectory and use this as the input
                if input_obj.get("name"):
                    cmd.extend(build_volume_mount(str((op_input_dir / input_obj["name"])), input_obj["path"]))
                else:
                    cmd.extend(build_volume_mount(str(op_input_dir), input_obj["path"]))

    # Mount output volumes
    if op_config.outputs:
        for output_obj in op_config.outputs:
            # If `name` is specified, create a subdirectory with this name
            if output_obj.get("name"):
                sub_dir = Path(op_output_dir / output_obj["name"])
                sub_dir.mkdir(parents=True)
                cmd.extend(build_volume_mount(str(sub_dir), output_obj["path"]))
            else:
                cmd.extend(build_volume_mount(str(op_output_dir), output_obj["path"]))

    # Add the image and tag, and command last
    cmd.append(op_config.image_n_tag)
    if op_config.command:
        cmd.extend(op_config.command)
    logging.debug(f"Docker command for operator {op_config.name} is: {cmd}")
    return cmd


def print_operator_metrics(metrics, metrics_header, op_name):
    """Logs the metrics to console in a table format.

    Args:
        metrics: list of Metrics object
        metrics_header: Header of the metrics data
        op_name: Name of the operator

    Returns:
        None
    """
    logging.info("{:_^60}".format(f"Operator {op_name} Metrics Data"))  # pragma: no cover
    data = [astuple(metric) for metric in metrics]  # pragma: no cover
    logging.info(tabulate(data, metrics_header, tablefmt="pretty"))  # pragma: no cover


def print_operator_summary(metrics, op_name):
    """Calculate and logs the metrics statistics in a readable format.

    Args:
        metrics: list of Metrics object
        op_name: Name of the operator

    Returns:
        None
    """
    logging.info("{:_^60}".format(f"Operator {op_name} Summary"))
    # Calculate metrics for CPU and memory
    cpu_data = [metric.cpu_percent for metric in metrics]
    cpu_avg = round(sum(cpu_data)/len(metrics), 3)
    cpu_max = round(max(cpu_data), 3)

    memory_data = [metric.memory for metric in metrics]
    memory_avg = round(sum(memory_data)/len(metrics), 3)
    memory_max = round(max(memory_data), 3)

    recommended_cpu = convert_percent_to_cores(cpu_max)
    # Add 100MB of buffer memory and round to multiple of base 256
    recommended_memory = round_up_to_multiple(memory_max + 100.0, 256)

    # Log it onto console
    data = [["CPU", f"{cpu_avg} %", f"{cpu_max} %", f"cpu: {recommended_cpu}"], [
        "Memory", f"{memory_avg} MB", f"{memory_max} MB", f"memory: {recommended_memory}"]]
    logging.info(
        tabulate(
            data, ["Metric", "Average", "Maximum", "Resource"],
            tablefmt="pretty"))
    return data


def print_pipeline_summary(pipeline_metrics_dict):
    """Display the pipeline summary table.

    Args:
        pipeline_metrics_dict: Dictionary with key being operator name and values are metrics

    Returns:
        None
    """
    pipeline_data = []
    for op_name, op_summary in pipeline_metrics_dict.items():
        p_sumamry = [op_name] + ["\n".join([str(row1), str(row2)]) for row1, row2 in zip(op_summary[0], op_summary[1])]
        pipeline_data.append(p_sumamry)
    logging.info(
        tabulate(
            pipeline_data, ["Operator", "Metric", "Average", "Maximum", "Resource"],
            tablefmt="grid", numalign="right"))


def run_operator(
        op_config, docker_cmd, output_writers, metrics_output, continue_option,
        pipeline_summary_dict):
    """Run the operator using the directories given.

    Args:
        op_config: a OperatorConfig object
        docker_cmd: List of docker commands to run the operator
        output_writers: List of writers or None
        metrics_output: A Path object for the metrics directory or None
        continue_option: A ContinueOptions Enum object
        pipeline_summary_dict: Dictionary with key being operator name and values are metrics

    Returns:
        True when the operator failed and user wants to stop, otherwise None
    """
    container = Container()
    manager = Manager()
    container_id = manager.Value('c_wchar_p', '')
    id_returned_event = manager.Event()

    if output_writers is not None:
        write_que = Queue()
        writer_process = Process(
            target=write_to_csv,
            args=(write_que, METRICS_HEADER, (metrics_output / f"{op_config.name}_final_result.csv")))
        writer_process.start()
        output_writers.append(writer_process)
    else:
        write_que = None

    p_start = Process(target=start_operator, args=(container_id, id_returned_event, docker_cmd))
    before_id = time.perf_counter()  # timing
    p_start.start()

    if id_returned_event.wait(timeout=ID_WAITING_TIME_SECONDS):
        # Event.wait() returns true if it has been set
        after_id = time.perf_counter()  # timing
        container.id = container_id.value
        container.construct_metrics_path()
        sample_operator(container, write_que)
        end = time.perf_counter()  # timing
        logging.debug(f"Time it takes to get container ID: {after_id-before_id} s")
        logging.debug(f"Waiting and Sampling Time: {end-after_id} s")

        p_start.join()

        # print metrics to console if not written to csv
        if output_writers is None:
            print_operator_metrics(container.metrics, METRICS_HEADER, op_config.name)
        operator_summary = print_operator_summary(container.metrics, op_config.name)
        pipeline_summary_dict[op_config.name] = operator_summary

    else:
        logging.warning(f"Obtaining docker ID timed out. Operator {op_config.name} failed")
        p_start.terminate()
        p_start.join()
        if output_writers is not None:
            writer_process.terminate()

    if p_start.exitcode != 0:  # i.e. container_id timed out
        logging.warning(f"Operator {op_config.name} failed with exitcode {p_start.exitcode}")
        if pipeline_summary_dict.get(op_config.name):
            new_key = f"{op_config.name}\n(Non-zero exitcode)"
            pipeline_summary_dict[new_key] = pipeline_summary_dict.pop(op_config.name)
        if continue_option == ContinueOptions.CONT:
            return
        if continue_option == ContinueOptions.STOP:
            return True
        if not prompt_yes_or_no(
                "Would you like to continue execution at the risk of the rest of pipeline failing (y)? If (n), cpost will stop and cleanup."):
            # When user says no, we exit the for-loop and return
            return True


def run_pipeline(execution_order, input_data_dir, metrics_output, models_dir, continue_option):
    """Run the pipeline operators in the given execution_order using the directories given.

    Args:
        execution_order: List of OperatorConfig objects in the order of execution
        input_data_dir: Path to the input payload directory
        metrics_output: A Path object for the metrics directory or stdout
        models_dir: A directory that contains Triton models
        continue_option: A ContinueOptions Enum object

    Returns:
        None
    """

    triton_mode = decide_method_to_run_triton(execution_order)

    if triton_mode == RUN_MODE.NO_INFERENCE_SERVER:
        return run_pipeline_alone(execution_order, input_data_dir, metrics_output, continue_option)
    if triton_mode == RUN_MODE.MODEL_REPO:
        with run_triton_model_repo(execution_order, models_dir) as triton_ip:
            run_pipeline_alone(execution_order, input_data_dir, metrics_output, continue_option, triton_ip)
    else:  # PIPELINE_SERVICES
        run_pipeline_with_services(execution_order, input_data_dir, metrics_output,
                                   models_dir, continue_option)


@contextmanager
def get_output_writers(metrics_output):
    """Context manager for keeping a list of output writers and cleaning up.
    The list is used to keep output_writer processes which are threads/multiprocessing.Process.

    Args:
        metrics_output: a pathlib.Path object or None

    Yields:
        None if metrics_output is None. Empty list if metrics_output is Path
    """
    try:
        write_csv_flag = True if isinstance(metrics_output, Path) else False
        if write_csv_flag:
            output_writers = []
            yield output_writers
        else:
            yield None

    finally:
        if write_csv_flag:
            for writer in output_writers:
                writer.join()


def run_pipeline_alone(execution_order, input_data_dir, metrics_output, continue_option, triton_ip=None):
    """Run the pipeline operators in the given execution_order using the directories given.

    Args:
        execution_order: List of OperatorConfig objects in the order of execution
        input_data_dir: Path to the input payload directory
        metrics_output: A Path object for the metrics directory or stdout
        continue_option: A ContinueOptions Enum object
        triton_ip: None, or Triton's IP address

    Returns:
        None
    """
    with TemporaryDirectory() as data_folder_name:
        with get_output_writers(metrics_output) as output_writers:
            pipeline_summary_dict = {}
            for op_config in execution_order:
                logging.info("\n{:_^60}".format(f"Executing Operator {op_config.name}"))
                docker_cmd = build_operator_cmd(input_data_dir, data_folder_name, op_config, triton_ip)
                exit = run_operator(op_config, docker_cmd, output_writers,
                                    metrics_output, continue_option, pipeline_summary_dict)
                if exit:
                    break
            print_pipeline_summary(pipeline_summary_dict)


def clean_up_containers(running_dict):
    """Kill the containers in the given dictionary and remove the item from the dictionary.

    Args:
        running_dict: Dictionary where key is image name and value is (container ID, ip_address)

    Returns:
        None
    """
    for old_key, container_info in running_dict.items():
        logging.debug(f"Tear down unused services {old_key}")
        if container_info:
            subproc_run_wrapper(["docker", "kill", container_info[0]])
    running_dict.clear()


def start_pipeline_services(op_config, running_dict, models_dir):
    """Start the pipeline services for the given op_config.

    Args:
        op_config: A OperatorConfig object
        running_dict: Dictionary for keep track of currently running services
        models_dir: A directory that contains Triton models

    Return:
        None
    """
    for service in op_config.services:
        logging.debug(f"Checking service with name {service.name}")
        key = service.image_n_tag + " " + " ".join(service.command)
        if running_dict.get(key):
            # Add the connection variables
            ip_address = running_dict[key][1]
            http_connections_dict = {k: f"{ip_address}:{v}" for k, v in service.http_connections.items()}
            op_config.update_variables(http_connections_dict)
            logging.debug("Found running services that suit the needs")
        else:
            logging.debug("Didn't find matching service, starting new service")
            if len(running_dict) != 0:  # tear down current services before spin up another one
                clean_up_containers(running_dict)
            if "trtis" in service.name or "triton" in service.name:
                triton_container_id, ip_address = start_triton(models_dir, service.command, service.image_n_tag)
                running_dict[key] = (triton_container_id, ip_address)
                http_connections_dict = {k: f"{ip_address}:{v}" for k, v in service.http_connections.items()}
                op_config.update_variables(http_connections_dict)
            else:
                logging.warning("CPOST currently does not support services other than triton or trtis.")
                logging.warning(f"Skipping `{service.name}`, operator may fail because of this.")


def run_pipeline_with_services(
        execution_order, input_data_dir, metrics_output, models_dir, continue_option):
    """Run the pipeline operators in the given execution_order using the directories given.

    Args:
        execution_order: List of OperatorConfig objects in the order of execution
        input_data_dir: Path to the input payload directory
        metrics_output: A Path object for the metrics directory or stdout
        models_dir: A directory that contains Triton models
        continue_option: A ContinueOptions Enum object

    Returns:
        None
    """
    with TemporaryDirectory() as data_folder_name:
        with get_output_writers(metrics_output) as output_writers:
            try:
                running_services = {}
                pipeline_summary_dict = {}
                for op_config in execution_order:
                    if op_config.services:
                        start_pipeline_services(op_config, running_services, models_dir)
                    logging.info("\n{:_^60}".format(f"Executing Operator {op_config.name}"))
                    docker_cmd = build_operator_cmd(input_data_dir, data_folder_name, op_config)
                    exit = run_operator(op_config, docker_cmd, output_writers,
                                        metrics_output, continue_option, pipeline_summary_dict)
                    if exit:
                        break
                print_pipeline_summary(pipeline_summary_dict)
            finally:
                # Stop any currently running services
                clean_up_containers(running_services)
