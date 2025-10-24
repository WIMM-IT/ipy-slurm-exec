from __future__ import print_function

import argparse
import datetime
import inspect
import io
import json
import os
import pickle
import re
import shlex
import shutil
import sys
import textwrap
import time
import types
import uuid
from pathlib import Path
from subprocess import PIPE, Popen

import pandas
from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class


def modal(func):
    def wrapped_func(obj, line):
        result = func(obj, line)
        if obj._display == "pandas":
            return pandas.read_table(
                io.StringIO(result), sep="\s+", on_bad_lines="warn"
            )
        else:
            return result

    wrapped_func.__doc__ = func.__doc__
    return wrapped_func


@magics_class
class IPySlurmExec(Magics):
    def __init__(self, shell=None, **kwargs):
        super(IPySlurmExec, self).__init__(shell, **kwargs)
        self._display = "pandas"
        self._jobs_root = Path.cwd() / "slurm_jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)

    @line_magic
    def slurm(self, line):
        chunks = line.lower().split()
        variable, arguments = chunks[0], chunks[1:]
        if variable == "display":
            return self._configure_display(arguments)

    def _configure_display(self, arguments):
        if arguments:
            mode = arguments[0]
            if mode not in ["pandas", "raw"]:
                raise ValueError("Unknown Slurm magics display mode", mode)
            self._display = mode
        return self._display

    @line_cell_magic
    def slurm_exec(self, line, cell=None):
        """Execute Python code on a Slurm allocation and capture the result back into the notebook."""
        if cell is None:
            raise UsageError("%%slurm_exec must be used as a cell magic.")
        args = self._parse_slurm_exec_args(line)
        if args.output_var is not None and not args.output_var.isidentifier():
            raise UsageError("Output variable name must be a valid Python identifier.")

        # Disable capture-all variables, some issues to iron-out.
        # Until then, user must specify variables.
        capture_all = False

        if args.output_var is None and args.inputs:
            raise UsageError(
                "Specify an output variable before listing input variables."
            )

        if capture_all:
            inputs = self._collect_all_user_variables()
        else:
            if args.output_var is None:
                raise UsageError("Output variable name must be provided.")
            inputs = self._collect_input_variables(args.inputs)
        inputs = self._collect_input_variables(args.inputs)
        payload = self._build_slurm_exec_payload(
            output_var=args.output_var,
            inputs=inputs,
            cell=cell,
            capture_all=capture_all,
        )
        job_dir, job_label = self._create_job_directory(args.job_name)
        payload_path = job_dir / "payload.pkl"
        self._write_payload(payload_path, payload)
        driver_path = self._write_driver_script(job_dir)

        submit_path = self._write_submit_script(
            job_dir=job_dir,
            driver_path=driver_path,
            job_label=job_label,
            python_executable=args.python_executable,
            partition=args.partition,
            gpus=args.gpus,
            time_limit=args.time,
            qos=args.qos,
            account=args.account,
            ntasks=args.ntasks,
            cpus_per_task=args.cpus_per_task,
            modules=args.modules,
            sbatch_directives=args.sbatch_directives,
        )

        metadata = {
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "job_label": job_label,
            "job_dir": str(job_dir),
            "inputs": list(inputs.keys()),
            "output_var": args.output_var,
            "python_executable": args.python_executable,
            "partition": args.partition,
            "gpus": args.gpus,
            "time_limit": args.time,
            "qos": args.qos,
            "account": args.account,
            "ntasks": args.ntasks,
            "cpus_per_task": args.cpus_per_task,
            "modules": args.modules,
            "capture_all": capture_all,
        }
        metadata_path = job_dir / "metadata.json"
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        job_id = self._submit_job(submit_path)
        metadata["job_id"] = job_id
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        print("Submitted Slurm job {job} (folder: {folder})".format(
                  job=job_id, folder=job_dir))

        status = self._wait_for_job_completion(
            job_id=job_id,
            job_dir=job_dir,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
        )

        if status.get("state") != "COMPLETED":
            message = status.get("message", "Slurm job did not complete successfully.")
            raise RuntimeError("Job {job} failed ({state}). See {folder} for details.\n{message}".format(
                                    job=job_id,
                                    state=status.get("state"),
                                    folder=job_dir,
                                    message=message))

        output_path = job_dir / "output.pkl"
        if not output_path.exists():
            raise RuntimeError("Job {job} completed but no output.pkl was produced. Inspect {folder}.".format(
                                    job=job_id, folder=job_dir))

        with open(output_path, "rb") as handle:
            result_payload = pickle.load(handle)

        mode = result_payload.get("mode", "single")
        if mode == "namespace":
            namespace_update = result_payload.get("namespace", {})
            if namespace_update:
                self.shell.push(namespace_update)
            updated_names = sorted(namespace_update.keys())
            if len(updated_names) > 8:
                display_names = ", ".join(updated_names[:8]) + ", …"
            else:
                display_names = ", ".join(updated_names) if updated_names else "<none>"
            print("Job {job} completed. Updated variables: {names}".format(
                    job=job_id,
                    names=display_names)
            )
            return namespace_update
        else:
            result_value = result_payload.get("value")
            self.shell.push({args.output_var: result_value})
            print("Job {job} completed. Result assigned to '{var}'".format(
                    job=job_id, var=args.output_var)
            )
            return result_value

    def _parse_slurm_exec_args(self, line):
        parser = argparse.ArgumentParser(
            prog="%slurm_exec",
            description="Execute Python code on a Slurm allocation and return the result.",
        )
        parser.add_argument(
            "variables",
            nargs="*",
            help=("Optionally provide an output variable followed by the names of"
                  " notebook variables to send to the Slurm job."),
        )
        parser.add_argument("--job-name", dest="job_name", help="Custom Slurm job name.")
        parser.add_argument("--partition", help="Slurm partition.")
        parser.add_argument("--gpus", help="Number of GPUs")
        parser.add_argument("--time", help="Slurm time limit (e.g. 00:10:00).")
        parser.add_argument("--qos", help="Slurm QoS.")
        parser.add_argument("--account", help="Slurm account.")
        parser.add_argument("--ntasks", type=int, default=None, help="Number of tasks for the Slurm job.")
        parser.add_argument("--cpus-per-task", dest="cpus_per_task", type=int, default=None, help="CPUs per task for the Slurm job.")
        parser.add_argument("--modules", action="append", default=[], help="Environment modules to load before execution (repeatable, accepts comma separation).")
        parser.add_argument("--python", dest="python_executable", default=sys.executable, help="Python interpreter to use on the Slurm node.")
        parser.add_argument("--sbatch", dest="sbatch_directives", action="append", default=[], help="Additional SBATCH directives, e.g. --sbatch='--gres=gpu:1'.")
        parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between status checks.")
        parser.add_argument("--max-wait", type=float, default=None, help="Maximum seconds to wait for completion (default: unlimited).")
        try:
            args = parser.parse_args(shlex.split(line))
        except SystemExit as exc:
            raise UsageError("Invalid arguments for %slurm_exec") from exc
        args.modules = self._normalize_module_list(args.modules)
        positional = list(args.variables)
        if positional:
            args.output_var = positional[0]
            args.inputs = positional[1:]
        else:
            args.output_var = None
            args.inputs = []
        delattr(args, "variables")
        return args

    def _collect_input_variables(self, input_names):
        variables = {}
        for name in input_names:
            if name not in self.shell.user_ns:
                raise UsageError(
                    "Variable '{name}' not found in the notebook namespace.".format(
                        name=name
                    )
                )
            variables[name] = self.shell.user_ns[name]
        return variables

    def _collect_all_user_variables(self):
        variables = {}
        hidden = set(getattr(self.shell, "user_ns_hidden", []))
        reserved = {"In", "Out", "get_ipython", "exit", "quit"}
        for name, value in self.shell.user_ns.items():
            if name in hidden or name in reserved:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, types.ModuleType):
                continue
            variables[name] = value
        return variables

    def _normalize_module_list(self, modules):
        normalized = []
        for entry in modules:
            for raw in entry.split(","):
                module = raw.strip()
                if module:
                    normalized.append(module)
        return normalized

    def _build_slurm_exec_payload(self, output_var, inputs, cell, capture_all):
        module_aliases = {}
        for alias, value in self.shell.user_ns.items():
            if isinstance(value, types.ModuleType):
                module_name = value.__name__
                if module_name.startswith("IPython"):
                    continue
                module_aliases[alias] = module_name

        payload = {
            "output_var": output_var,
            "variables": inputs,
            "modules": module_aliases,
            "sys_path": list(sys.path),
            "cell": cell,
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "capture_all": capture_all,
        }
        return payload

    def _create_job_directory(self, requested_name):
        base = requested_name or "slurm-exec"
        base = re.sub(r"[^A-Za-z0-9._-]", "-", base).strip("-")
        if not base:
            base = "slurm-exec"
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        token = uuid.uuid4().hex[:8]
        job_label = "{base}-{token}".format(base=base[:64], token=token)
        job_dir = self._jobs_root / "{base}-{timestamp}-{token}".format(
            base=base, timestamp=timestamp, token=token
        )
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir, job_label[:100]

    def _write_payload(self, path, payload):
        try:
            with open(path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, AttributeError, TypeError) as exc:
            raise UsageError(
                "Failed to serialize notebook variables for %slurm_exec: {error}".format(
                    error=exc
                )
            )

    def _write_driver_script(self, job_dir):
        driver_code = textwrap.dedent(
            """\
            import json
            import pickle
            import sys
            import traceback
            import types
            from pathlib import Path

            JOB_DIR = Path(__file__).resolve().parent
            PAYLOAD_FILE = JOB_DIR / "payload.pkl"
            STATUS_FILE = JOB_DIR / "status.json"
            OUTPUT_FILE = JOB_DIR / "output.pkl"
            TRACEBACK_FILE = JOB_DIR / "traceback.log"

            def main():
                exit_code = 0
                status = {"state": "UNKNOWN"}
                try:
                    with open(PAYLOAD_FILE, "rb") as handle:
                        payload = pickle.load(handle)
                    sys.path = payload["sys_path"]
                    namespace = {}
                    namespace.update(payload["variables"])
                    for alias, module_name in payload["modules"].items():
                        try:
                            module = __import__(module_name)
                        except Exception:
                            continue
                        namespace[alias] = module
                    exec(payload["cell"], namespace)
                    if payload.get("capture_all", False):
                        namespace_payload = {}
                        for name, value in namespace.items():
                            if name == "__builtins__":
                                continue
                            if isinstance(value, types.ModuleType):
                                continue
                            namespace_payload[name] = value
                        with open(OUTPUT_FILE, "wb") as handle:
                            pickle.dump(
                                {"mode": "namespace", "namespace": namespace_payload},
                                handle,
                                protocol=payload["pickle_protocol"],
                            )
                    else:
                        result_var = payload["output_var"]
                        if result_var not in namespace:
                            raise RuntimeError(
                                "Result variable '{var}' was not defined by the job.".format(
                                    var=result_var
                                )
                            )
                        with open(OUTPUT_FILE, "wb") as handle:
                            pickle.dump(
                                {"mode": "single", "value": namespace[result_var]},
                                handle,
                                protocol=payload["pickle_protocol"],
                            )
                    status = {"state": "COMPLETED"}
                except Exception as exc:
                    exit_code = 1
                    status = {"state": "FAILED", "message": str(exc)}
                    traceback.print_exc()
                    with open(TRACEBACK_FILE, "w") as trace_handle:
                        traceback.print_exc(file=trace_handle)
                finally:
                    with open(STATUS_FILE, "w") as status_handle:
                        json.dump(status, status_handle)
                return exit_code

            if __name__ == "__main__":
                raise SystemExit(main())
            """
        )

        driver_path = job_dir / "driver.py"
        with open(driver_path, "w") as handle:
            handle.write(driver_code)
        os.chmod(driver_path, 0o755)
        return driver_path

    def _write_submit_script(
        self,
        job_dir,
        driver_path,
        job_label,
        python_executable,
        partition,
        gpus,
        time_limit,
        qos,
        account,
        ntasks,
        cpus_per_task,
        modules,
        sbatch_directives,
    ):
        submit_path = job_dir / "submit.sh"
        lines = ["#!/bin/bash", "#SBATCH --export=ALL", "#SBATCH --chdir={}".format(job_dir)]
        lines.append("#SBATCH --job-name={}".format(job_label))
        lines.append("#SBATCH --output={}".format(job_dir / "slurm-%j.out"))
        lines.append("#SBATCH --error={}".format(job_dir / "slurm-%j.err"))
        if partition:
            lines.append("#SBATCH --partition={}".format(partition))
        if gpus:
            lines.append("#SBATCH --gpus={}".format(gpus))
        if time_limit:
            lines.append("#SBATCH --time={}".format(time_limit))
        if qos:
            lines.append("#SBATCH --qos={}".format(qos))
        if account:
            lines.append("#SBATCH --account={}".format(account))
        if ntasks:
            lines.append("#SBATCH --ntasks={}".format(ntasks))
        if cpus_per_task:
            lines.append("#SBATCH --cpus-per-task={}".format(cpus_per_task))
        for directive in sbatch_directives:
            lines.append("#SBATCH {}".format(directive))

        lines.append("")
        if modules:
            lines.append("module purge || true")
            for module in modules:
                lines.append("module load {}".format(module))

        # driver_path_relative = str(driver_path).replace(str(job_dir)+'/', './')
        driver_path_relative = driver_path.relative_to(job_dir)
        command = "exec {python} {driver}".format(
            python=shlex.quote(python_executable), driver=shlex.quote(str(driver_path_relative))
        )
        lines.append("")
        lines.append(command)

        with open(submit_path, "w") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")
        os.chmod(submit_path, 0o755)
        return submit_path

    def _submit_job(self, submit_path):
        process = Popen(["sbatch", str(submit_path)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to submit Slurm job:\nSTDOUT: {out}\nSTDERR: {err}".format(
                    out=stdout.decode("utf-8", "ignore"),
                    err=stderr.decode("utf-8", "ignore"),
                )
            )
        match = re.search(r"Submitted batch job (\d+)", stdout.decode("utf-8", "ignore"))
        if not match:
            raise RuntimeError(
                "Could not parse job id from sbatch output: {out}".format(
                    out=stdout.decode("utf-8", "ignore")
                )
            )
        return match.group(1)

    def _wait_for_job_completion(self, job_id, job_dir, poll_interval, max_wait):
        status_path = job_dir / "status.json"
        start_time = time.time()
        last_state = None
        while True:
            if status_path.exists():
                with open(status_path, "r") as handle:
                    try:
                        result = json.load(handle)
                        if last_state is not None:
                            self._clear_status_line()
                        return result
                    except json.JSONDecodeError:
                        pass
            if max_wait is not None and (time.time() - start_time) > max_wait:
                if last_state is not None:
                    self._clear_status_line()
                raise RuntimeError(
                    "Timed out waiting for job {job} to produce a status file. "
                    "Inspect {folder} for details.".format(job=job_id, folder=job_dir)
                )
            if not self._job_active(job_id):
                if status_path.exists():
                    with open(status_path, "r") as handle:
                        if last_state is not None:
                            self._clear_status_line()
                        return json.load(handle)
                if last_state is not None:
                    self._clear_status_line()
                raise RuntimeError(
                    "Job {job} is no longer active but no status file was produced. "
                    "Inspect {folder} for diagnostics.".format(job=job_id, folder=job_dir)
                )
            state, checked = self._current_job_state(job_id)
            if state:
                timestamp = checked or "--:--:--"
                message = "Job {job} status: {state} (checked {time})".format(
                    job=job_id,
                    state=state,
                    time=timestamp,
                )
                self._write_status_line(message)
                last_state = state
            time.sleep(poll_interval)

    def _job_active(self, job_id):
        if not self._command_available("squeue"):
            return True
        process = Popen(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, _ = process.communicate()
        if process.returncode != 0:
            return True
        return bool(stdout.strip())

    def _command_available(self, command):
        return shutil.which(command) is not None

    def _current_job_state(self, job_id):
        if not self._command_available("squeue"):
            return None, None
        process = Popen(["squeue", "-h", "-j", job_id, "-o", "%T",], stdout=PIPE, stderr=PIPE)
        stdout, _ = process.communicate()
        if process.returncode != 0:
            return None, None
        state = stdout.decode("utf-8", "ignore").strip()
        if not state:
            return None, None
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        return state, timestamp

    def _write_status_line(self, message):
        sys.stdout.write("\r{msg}".format(msg=message.ljust(80)))
        sys.stdout.flush()

    def _clear_status_line(self):
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


def load_ipython_extension(ip):
    """Load extension in IPython."""
    slurm_magic = IPySlurmExec(ip)
    ip.register_magics(slurm_magic)
