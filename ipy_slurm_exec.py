# © 2025 The Chancellor, Masters and Scholars of the University of Oxford
# Licensed for internal use only within the CCB HPC facility, WIMM, University of Oxford.

from __future__ import print_function

import argparse
import datetime
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
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Optional, Required

from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ipy_slurm_exec_runtime import SerializeFailure, serialize_variable, restore_from_record


@dataclass
class SBatch:
    account: Optional[str] = None
    qos: Optional[str] = None
    job_name: Optional[str] = None
    partition: Optional[str] = None
    time_limit: Optional[str] = None
    ntasks: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gpus: Optional[str] = None
    mem: Optional[str] = None

@dataclass
class NotebookJob:
    python_executable: Required[str] = None
    sbatch_params: Required[SBatch] = None
    modules: Optional[list[str]] = None
    modules_purge: Optional[bool] = None

    inputs: Optional[list[str]] = None
    outputs: Optional[list[str]] = None
    poll_interval: Optional[int] = None
    max_wait: Optional[float] = None


@magics_class
class IPySlurmExec(Magics):
    def __init__(self, shell=None, **kwargs):
        super(IPySlurmExec, self).__init__(shell, **kwargs)
        self._jobs_root = Path.cwd() / "slurm_exec"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._warned_reportseff_missing = False

    @line_cell_magic
    def slurm_exec(self, line, cell=None):
        """Execute Python code on a Slurm allocation and capture the result back into the notebook."""
        if cell is None:
            raise UsageError("%%slurm_exec must be used as a cell magic.")

        args = self._parse_slurm_exec_args(line)
        capture_all_inputs = args.inputs is None or len(args.inputs) == 0 or (args.inputs[0] == '*')
        capture_all_outputs = args.outputs is None or len(args.outputs) == 0 or (args.outputs[0] == '*')

        if capture_all_inputs:
            inputs = self._collect_all_user_variables()
        else:
            inputs = self._collect_input_variables(args.inputs)

        job_dir, job_label = self._create_job_directory(args.sbatch_params.job_name)

        payload = self._build_slurm_exec_payload(
            outputs = args.outputs,
            inputs = inputs,
            cell = cell,
            capture_all_inputs = capture_all_inputs,
            capture_all_outputs = capture_all_outputs,
            job_dir = job_dir,
        )
        payload_path = job_dir / "payload.pkl"
        self._write_payload(payload_path, payload)
        driver_path = self._write_driver_script(job_dir)
        sbatch_params = args.sbatch_params

        submit_path = self._write_submit_script(
            job_dir=job_dir,
            driver_path=driver_path,
            job_label=job_label,
            python_executable=args.python_executable,
            sbatch_params=sbatch_params,
            modules=args.modules,
            modules_purge=args.modules_purge,
            # sbatch_directives=args.sbatch_directives,
        )

        job_id = self._submit_job(submit_path)

        print("Submitted Slurm job {job} (folder: {folder})".format(job=job_id, folder=job_dir.relative_to(Path.cwd())))

        status = self._wait_for_job_completion(
            job_id=job_id,
            job_dir=job_dir,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
        )

        if status.get("state") == "CANCELLED":
            cancel_message = status.get("message", "Slurm job was cancelled.")
            # UsageError to stop Notebook printing a traceback
            raise UsageError(cancel_message)

        if status.get("state") != "COMPLETED":
            message = status.get("message", "Slurm job did not complete successfully.")
            trace_text = None
            trace_path = job_dir / "traceback.log"
            if trace_path.exists():
                try:
                    trace_text = trace_path.read_text(errors="replace")
                except Exception:
                    trace_text = None
                if trace_text:
                    # Expose traceback to the notebook namespace for inspection.
                    self.shell.push({"_slurm_remote_traceback": trace_text})
                    # Render a coloured traceback if pygments is available.
                    try:
                        from pygments import highlight
                        from pygments.formatters import HtmlFormatter
                        from pygments.lexers import PythonTracebackLexer
                        formatter = HtmlFormatter(noclasses=True)
                        html = highlight(trace_text, PythonTracebackLexer(), formatter)
                        display(HTML(f"<div style='font-family: monospace'>{html}</div>"))
                    except Exception:
                        pass
                    # Override message to avoid duplicating the remote error text.
                    message = ''
            raise RuntimeError("slurm_exec job {job} is {state}. Traceback should be printed above. Job files stored in folder: {folder}.\n{message}".format(
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

        print("Job completed")

        # Update Notebook namespace
        raw_namespace = result_payload.get("namespace", {})
        namespace_update = {}
        local_errors = {}
        for name, value in raw_namespace.items():
            try:
                if isinstance(value, dict) and "mode" in value:
                    namespace_update[name] = restore_from_record(value, job_dir)
                elif isinstance(value, bytes):
                    namespace_update[name] = pickle.loads(value)
                else:
                    namespace_update[name] = value
            except Exception as exc:
                if capture_all_outputs:
                    # Treat as soft error because user didn't specify outputs
                    local_errors[name] = str(exc)
                else:
                    raise
        remote_errors = result_payload.get("errors", {}) or {}
        if namespace_update:
            self.shell.push(namespace_update)

        # Print nice summary of Notebook updates:
        updated_names = sorted(namespace_update.keys())
        if remote_errors:
            serialize_fails = {}
            other_fails = {}
            for name in sorted(remote_errors.keys()):
                err = remote_errors[name]
                if err.startswith('SerializeFailure('):
                    serialize_fails[name] = err
                else:
                    other_fails[name] = err
            if len(serialize_fails) > 0:
                msg = "Slurm job failed to export these variables:\n"
                names = list(serialize_fails.keys())
                for i in range(len(names)):
                    name = names[i]
                    err = serialize_fails[name]
                    m = re.match(r'^SerializeFailure\((.*)\)$', err)
                    if m:
                        err_inner = m.group(1)
                        err_inner = re.sub(r"^'|'$", '', err_inner)
                    else:
                        err_inner = err
                    msg += f"- {err_inner}"
                    if i < len(names)-1:
                        msg += "\n"
                print(msg)
            if len(other_fails):
                err_2_var = {}
                for name, err in other_fails.items():
                    if err not in err_2_var:
                        err_2_var[err] = [name]
                    else:
                        err_2_var[err].append(name)
            print("")

        if local_errors:
            err_2_var = {}
            for name, err in local_errors.items():
                if err not in err_2_var:
                    err_2_var[err] = [name]
                else:
                    err_2_var[err].append(name)
            print("Failed importing these variables into Notebook:")
            for err, names in err_2_var.items():
                print(f"- {', '.join(names)}: '{err}'")
            print("")

        if updated_names is None or len(updated_names) == 0:
            print("No variables imported")
        else:
            lines = []
            line = ''
            # line_width = 70
            line_width = 80
            for i in range(len(updated_names)):
                n = updated_names[i]
                if len(n) + len(line) + 2 > line_width:
                    # overflow
                    lines.append(line + ',')
                    line = n
                else:
                    if i > 0:
                        line += ", "
                    line += f"{n}"
            if line != "":
                lines.append(line)
            # msg = "Imported variables: " + lines[0]
            msg = "Imported: " + lines[0]
            for i in range(1, len(lines)):
                # msg += "\n" + ' '*20 + lines[i]
                msg += "\n" + ' '*10 + lines[i]
            print(msg)

        self._report_job_efficiency(job_id, status.get("state"))

        # if capture_all_outputs:
        #     # avoid printing everything
        #     namespace_update = None
        # Don't print anyway
        namespace_update = None

        return namespace_update

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
        parser.add_argument("--python", dest="python_executable", default=sys.executable, help="Python interpreter to use on the Slurm node.")

        parser.add_argument('-i', "--inputs", action="append", default=[], help="Variables to export to Slurm job. Default = everything.")
        parser.add_argument('-o', "--outputs", action="append", default=[], help="Variables to import into Notebook. Default = everything.")

        parser.add_argument("--job-name", dest="job_name", help="Custom Slurm job name.")
        parser.add_argument("--account", help="Slurm account.")
        parser.add_argument("--qos", help="Slurm QoS.")
        parser.add_argument("--partition", help="Slurm partition.")
        parser.add_argument("--time", help="Slurm time limit (e.g. 00:10:00).")
        parser.add_argument("--ntasks", type=int, default=None, help="Number of tasks for the Slurm job.")
        parser.add_argument("--cpus-per-task", dest="cpus_per_task", type=int, default=None, help="CPUs per task for the Slurm job.")
        parser.add_argument("--mem", help="Memory")
        parser.add_argument("--gpus", help="Number of GPUs")
        # parser.add_argument("--sbatch", dest="sbatch_directives", action="append", default=[], help="Additional SBATCH directives, e.g. --sbatch='--gres=gpu:1'.")

        parser.add_argument("--modules", action="append", default=[], help="Environment modules to load before execution (repeatable, accepts comma separation).")
        
        parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between status checks.")
        parser.add_argument("--max-wait", type=float, default=None, help="Maximum seconds to wait for completion (default: unlimited).")
        args = parser.parse_args(shlex.split(line))
        
        modules = args.modules
        if len(modules) > 0 and modules[0][0] == '+':
            purge = False
            if modules[0] == '+':
                del modules[0]
            else:
                modules[0] = modules[0][1:]
        else:
            purge = True
        
        def _norm_csv_list(x):
            y = []
            for entry in x:
                for raw in entry.split(","):
                    module = raw.strip()
                    if module:
                        y.append(module)
            return y
        args.modules = _norm_csv_list(modules)
        args.modules_purge = purge
    
        args.inputs = _norm_csv_list(args.inputs)
        args.outputs = _norm_csv_list(args.outputs)

        args.sbatch_params = SBatch(
            account=args.account,
            qos=args.qos,
            job_name=args.job_name,
            partition=args.partition,
            time_limit=args.time,
            ntasks=args.ntasks,
            cpus_per_task=args.cpus_per_task,
            gpus=args.gpus,
            mem=args.mem
        )

        return NotebookJob(
            python_executable = args.python_executable,
            sbatch_params = args.sbatch_params,
            modules = args.modules,
            modules_purge = args.modules_purge,

            inputs = args.inputs,
            outputs = args.outputs,
            poll_interval = args.poll_interval,
            max_wait = args.max_wait
        )

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

    def _build_slurm_exec_payload(self, outputs, inputs, cell, capture_all_inputs, capture_all_outputs, job_dir):
        module_aliases = {}
        for alias, value in self.shell.user_ns.items():
            if isinstance(value, types.ModuleType):
                module_name = value.__name__
                if module_name.startswith("IPython"):
                    continue
                module_aliases[alias] = module_name

        serialized_vars = {}
        errors = {}
        for name, value in inputs.items():
            try:
                serialized_vars[name] = serialize_variable(
                    name=name,
                    value=value,
                    root_dir=job_dir,
                    rel_root="inputs",
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            except Exception as exc:
                errors[name] = exc
        if errors:
            serialize_fails = [ exc for exc in errors.values() if isinstance(exc, SerializeFailure) ]
            if len(serialize_fails) > 0 and len(serialize_fails) == len(errors):
                # This should be normal codepath
                # reason = 'Not pickle-safe and lack pair of save/load functions'
                if len(serialize_fails) == 1:
                    msg = "%slurm_exec: Cannot export variable: "
                    msg += f"{str(serialize_fails[0])}"
                    # msg += f" (Inform developer: {reason})"
                else:
                    msg = "%slurm_exec: Cannot export these variables:\n"
                    msg += "\n".join(f"- {str(exc)}" for exc in serialize_fails)
                    # msg += f"\n(Inform developer: {reason})"

                if capture_all_inputs:
                    # treat as soft-error
                    print(msg + "\n")
                else:
                    raise UsageError(msg)

            else:
                detail = ", ".join(f"{name}: {exc}" for name, exc in errors.items())
                msg = "%slurm_exec: Failed to export variables. Reason: " + detail
                if capture_all_inputs:
                    # treat as soft-error
                    print(msg)
                else:
                    raise UsageError(msg)

        return {
            "outputs": outputs,
            "variables": serialized_vars,
            "modules": module_aliases,
            "sys_path": list(sys.path),
            "cell": cell,
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "capture_all_inputs": capture_all_inputs,
            "capture_all_outputs": capture_all_outputs,
        }

    def _create_job_directory(self, requested_name=''):
        base = requested_name or ''
        base = re.sub(r"[^A-Za-z0-9._-]", "-", base).strip("-")
        base64 = base[:64]
        if base != '':
            base += '-'
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M")
        token = uuid.uuid4().hex[:8]
        job_label = f"{base64}{token}"
        job_dir = self._jobs_root / f"{base}{timestamp}-{token}"
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
        helper_src = Path(__file__).with_name("ipy_slurm_exec_runtime.py")
        helper_dest = job_dir / "ipy_slurm_exec_runtime.py"
        shutil.copyfile(helper_src, helper_dest)

        driver_code = textwrap.dedent(
            """\
            import json
            import pickle
            import sys
            import traceback
            import types
            import importlib
            import importlib.util
            from pathlib import Path

            JOB_DIR = Path(__file__).resolve().parent
            PAYLOAD_FILE = JOB_DIR / "payload.pkl"
            STATUS_FILE = JOB_DIR / "status.json"
            OUTPUT_FILE = JOB_DIR / "output.pkl"
            TRACEBACK_FILE = JOB_DIR / "traceback.log"
            CELL_FILE = JOB_DIR / "cell.py"

            HELPER_FILE = JOB_DIR / "ipy_slurm_exec_runtime.py"
            _spec = importlib.util.spec_from_file_location("ipy_slurm_exec_runtime", HELPER_FILE)
            _runtime = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_runtime)

            def main():
                exit_code = 0
                status = {"state": "UNKNOWN"}
                try:
                    with open(PAYLOAD_FILE, "rb") as handle:
                        payload = pickle.load(handle)

                    # Load imports and variables
                    sys.path = payload["sys_path"]
                    capture_all_inputs = payload.get("capture_all_inputs", False)
                    namespace = {}
                    namespace_errors = {}
                    # for name, record in payload["variables"].items():
                    #     try:
                    #         namespace[name] = _runtime.restore_from_record(record, JOB_DIR)
                    #     except Exception as exc:
                    #         mode = record.get("mode", "unknown")
                    #         path = record.get("path")
                    #         extra = f", path={path}" if path else ""
                    #         trace_text = traceback.format_exc()
                    #         namespace_errors[name] = f"{repr(exc)} [mode={mode}{extra}]\\n{trace_text}"
                    # if namespace_errors:
                    #     detail = "\\n".join(f"{var}: {err}" for var, err in sorted(namespace_errors.items()))
                    #     raise RuntimeError(f"Failed to restore input variables:\\n{detail}")
                    # Was the above engineering really necessary?
                    for name, record in payload["variables"].items():
                        try:
                            namespace[name] = _runtime.restore_from_record(record, JOB_DIR)
                        except Exception as exc:
                            if capture_all_inputs:
                                # treat as soft error
                                namespace_errors[name] = repr(exc)
                            else:
                                raise
                    for alias, module_name in payload["modules"].items():
                        try:
                            module = importlib.import_module(module_name)
                        except Exception:
                            continue
                        namespace[alias] = module

                    # Important to print import errors here, 
                    # because they could be reason why cell execution fails next.
                    if namespace_errors:
                        print("Import errors in Slurm job:")
                        for name in sorted(namespace_errors.keys()):
                            print("  {var}: '{err}'".format(var=name, err=namespace_errors[name]))
                    namespace_errors = {}
                            
                    # Execute
                    cell_source = payload["cell"]
                    try:
                        CELL_FILE.write_text(cell_source)
                    except Exception:
                        pass
                    code_obj = compile(cell_source, str(CELL_FILE), "exec")
                    exec(code_obj, namespace)

                    # Extract output variables
                    if payload.get("capture_all_outputs", False):
                        vars_to_capture = []
                        for name, value in namespace.items():
                            if name == "__builtins__":
                                continue
                            if isinstance(value, types.ModuleType):
                                continue
                            vars_to_capture.append(name)
                    else:
                        vars_to_capture = payload["outputs"]
                        for name in vars_to_capture:
                            if name not in namespace:
                                raise RuntimeError("Result variable '{var}' was not defined by the job.".format(var=name))
                    capture_all_outputs = payload.get("capture_all_outputs", False)
                    namespace_payload = {}
                    namespace_errors = {}
                    for name in vars_to_capture:
                        value = namespace[name]
                        try:
                            namespace_payload[name] = _runtime.serialize_variable(
                                name,
                                value,
                                root_dir=JOB_DIR,
                                rel_root="outputs",
                                protocol=payload["pickle_protocol"],
                            )
                        except Exception as exc:
                            if capture_all_outputs:
                                # treat as soft-error
                                namespace_errors[name] = repr(exc)
                            else:
                                raise

                    # Write to pickle file
                    with open(OUTPUT_FILE, "wb") as handle:
                        pickle.dump(
                            {
                                "namespace": namespace_payload,
                                "errors": namespace_errors,
                            },
                            handle,
                            protocol=payload["pickle_protocol"],
                            )
                    status = {"state": "COMPLETED"}
                except Exception as exc:
                    exit_code = 1
                    status = {"state": "FAILED", "message": str(exc)}
                    trimmed_tb = exc.__traceback__
                    driver_path = str(Path(__file__).resolve())
                    # Only drop the driver frame if the next frame refers to the cell code.
                    if trimmed_tb and trimmed_tb.tb_frame and str(trimmed_tb.tb_frame.f_code.co_filename) == driver_path:
                        next_tb = trimmed_tb.tb_next
                        if next_tb and str(next_tb.tb_frame.f_code.co_filename) == str(CELL_FILE):
                            trimmed_tb = next_tb
                    traceback.print_exception(type(exc), exc, trimmed_tb)
                    with open(TRACEBACK_FILE, "w") as trace_handle:
                        traceback.print_exception(type(exc), exc, trimmed_tb, file=trace_handle)
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
        python_executable,
        job_label,
        sbatch_params,
        # sbatch_directives,
        modules,
        modules_purge,
    ):
        submit_path = job_dir / "submit.sh"
        lines = ["#!/bin/bash", "#SBATCH --export=ALL"]
        lines.append("#SBATCH --job-name={}".format(job_label))
        lines.append("#SBATCH --output={}".format(job_dir / "slurm-%j.out"))
        lines.append("#SBATCH --error={}".format(job_dir / "slurm-%j.err"))
        if sbatch_params.account:
            lines.append("#SBATCH --account={}".format(sbatch_params.account))
        if sbatch_params.qos:
            lines.append("#SBATCH --qos={}".format(sbatch_params.qos))
        if sbatch_params.partition:
            lines.append("#SBATCH --partition={}".format(sbatch_params.partition))
        if sbatch_params.time_limit:
            lines.append("#SBATCH --time={}".format(sbatch_params.time_limit))
        if sbatch_params.ntasks:
            lines.append("#SBATCH --ntasks={}".format(sbatch_params.ntasks))
        if sbatch_params.cpus_per_task:
            lines.append("#SBATCH --cpus-per-task={}".format(sbatch_params.cpus_per_task))
        if sbatch_params.mem:
            lines.append("#SBATCH --mem={}".format(sbatch_params.mem))
        if sbatch_params.gpus:
            lines.append("#SBATCH --gpus={}".format(sbatch_params.gpus))
        # for directive in sbatch_directives:
        #     lines.append("#SBATCH {}".format(directive))

        lines.append("")
        if modules:
            if modules_purge:
                lines.append("module purge || true")
            for module in modules:
                lines.append("module load {}".format(module))

        try:
            driver_path_relative = driver_path.relative_to(Path.cwd())
        except ValueError:
            driver_path_relative = driver_path
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
        log_path = job_dir / f"slurm-{job_id}.out"
        start_time = time.time()
        last_state = None
        last_status_line = None
        log_offset = 0
        partial_log_line = ""
        last_log_line_width = 0
        last_carriage_line = None

        def _emit(line, carriage=False):
            nonlocal last_log_line_width, last_carriage_line
            if line == "":
                return
            if carriage:
                last_log_line_width = max(last_log_line_width, len(line))
                sys.stdout.write("\r" + line.ljust(last_log_line_width))
                sys.stdout.flush()
                last_carriage_line = line
            else:
                if last_carriage_line is not None and line == last_carriage_line:
                    # Avoid printing the same progress line twice when a newline follows a carriage update.
                    return
                print(line, flush=True)
                last_log_line_width = 0
                last_carriage_line = None

        def _drain_log(force_flush=False):
            nonlocal log_offset, partial_log_line
            if not log_path.exists():
                return
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(log_offset)
                    chunk = fh.read()
                    log_offset = fh.tell()
            except Exception as exc:
                return
            if not chunk and not force_flush and not partial_log_line:
                return

            text = partial_log_line + chunk.decode("utf-8", "replace")
            idx = 0
            new_partial = ""

            while idx < len(text):
                next_nl = text.find("\n", idx)
                next_cr = text.find("\r", idx)
                candidates = [p for p in (next_nl, next_cr) if p != -1]
                if not candidates:
                    new_partial = text[idx:]
                    break
                delim = min(candidates)
                line = text[idx:delim]
                carriage = (delim == next_cr)
                _emit(line, carriage=carriage)
                idx = delim + 1

            if force_flush and new_partial:
                _emit(new_partial, carriage=False)
                partial_log_line = ""
            else:
                partial_log_line = new_partial

        curr_interval = poll_interval
        while True:
            if status_path.exists():
                with open(status_path, "r") as handle:
                    try:
                        result = json.load(handle)
                        if last_state is not None:
                            self._clear_status_line()
                        _drain_log(force_flush=True)
                        return result
                    except json.JSONDecodeError:
                        pass

            secs_since_start = time.time() - start_time
            if max_wait is not None and secs_since_start > max_wait:
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
                sacct_info = self._query_sacct_job_info(job_id)
                final_state = sacct_info.get("state") if sacct_info else None
                if final_state and final_state.upper().startswith("CANCEL"):
                    cancel_message = "Job {job} has been cancelled (state: {state}).".format(
                        job=job_id, state=final_state,
                    )
                    if last_state is not None:
                        self._clear_status_line()
                    return {"state": "CANCELLED", "message": cancel_message}
                if last_state is not None:
                    self._clear_status_line()
                raise RuntimeError(
                    "Job {job} is no longer active but no status file was produced. "
                    "Inspect {folder} for diagnostics.".format(job=job_id, folder=job_dir)
                )
            state, checked = self._current_job_state(job_id)
            if state:
                timestamp = checked or "--:--:--"
                now = datetime.datetime.now()
                base_state = state.split()[0]
                message = "Job {job} status: {state} (checked {time})".format(
                    job=job_id,
                    state=state,
                    time=timestamp,
                )

                if base_state in {"PENDING", "RUNNING"}:
                    sacct_info = self._query_sacct_job_info(job_id)
                    if sacct_info:
                        if base_state == "PENDING":
                            submit_time = sacct_info.get("submit")
                            if submit_time:
                                duration = self._format_duration((now - submit_time).total_seconds())
                                message = f"Job {job_id} status: {state} for {duration}"
                        else:
                            job_start_time = sacct_info.get("start")
                            elapsed_secs = sacct_info.get("elapsed_secs")
                            running_secs = None
                            if job_start_time:
                                running_secs = (now - job_start_time).total_seconds()
                            elif elapsed_secs is not None:
                                running_secs = elapsed_secs
                            if running_secs is not None:
                                duration = self._format_duration(running_secs)
                                message = f"Job {job_id} status: {state} for {duration}"

                self._write_status_line(message)
                last_state = state
                last_status_line = message

            _drain_log()
            
            # Increase poll interval with time
            if secs_since_start > 60:
                curr_interval = max(10, poll_interval)
            elif secs_since_start > 30:
                curr_interval = max(5, poll_interval)
            elif secs_since_start > 10:
                curr_interval = max(2, poll_interval)

            time.sleep(curr_interval)

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

    def _query_sacct_job_info(self, job_id):
        if not self._command_available("sacct"):
            return {}
        process = Popen([ "sacct", "-j", job_id, "-o", "Submit,Start,Elapsed,State", "--noheader", "-p" ],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return {}
        lines = stdout.decode("utf-8", "ignore").splitlines()
        records = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            submit = self._parse_sacct_timestamp(parts[0])
            start = self._parse_sacct_timestamp(parts[1])
            elapsed = self._parse_sacct_elapsed(parts[2])
            state = parts[3].strip()
            records.append({"submit": submit, "start": start, "elapsed_secs": elapsed, "state": state})
        if not records:
            return {}
        submits = [rec["submit"] for rec in records if rec["submit"]]
        starts = [rec["start"] for rec in records if rec["start"]]
        elapsed_values = [rec["elapsed_secs"] for rec in records if rec["elapsed_secs"] is not None]
        states = [rec["state"] for rec in records if rec["state"]]
        return {
            "submit": min(submits) if submits else None,
            "start": min(starts) if starts else None,
            "elapsed_secs": max(elapsed_values) if elapsed_values else None,
            "state": states[-1] if states else None,
        }

    def _parse_sacct_timestamp(self, value):
        text = (value or "").strip()
        if not text or text == "Unknown":
            return None
        text = text.split(".", 1)[0].rstrip("Z")
        try:
            return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

    def _parse_sacct_elapsed(self, value):
        text = (value or "").strip()
        if not text or text == "Unknown":
            return None
        days = 0
        if "-" in text:
            day_part, time_part = text.split("-", 1)
            try:
                days = int(day_part)
            except ValueError:
                days = 0
        else:
            time_part = text
        parts = time_part.split(":")
        if len(parts) != 3:
            return None
        try:
            hours, minutes, seconds = (int(part) for part in parts)
        except ValueError:
            return None
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def _write_status_line(self, message):
        sys.stdout.write("\r{msg}".format(msg=message.ljust(80)))
        sys.stdout.flush()

    def _format_duration(self, seconds):
        secs = int(max(0, seconds))
        hours, remainder = divmod(secs, 3600)
        minutes, secs = divmod(remainder, 60)
        return "{hours:02}:{minutes:02}:{seconds:02}".format(
            hours=hours,
            minutes=minutes,
            seconds=secs,
        )

    def _clear_status_line(self):
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _report_job_efficiency(self, job_id, final_state):
        if not self._command_available("reportseff"):
            if not self._warned_reportseff_missing:
                # print("Skipping resource efficiency report: 'reportseff' command not found.")
                self._warned_reportseff_missing = True
            return
        
        time.sleep(1)  # Give Slurm a moment to update
        nrepeats = 3
        n = 0
        while n < nrepeats:
            # Can take a few seconds for Slurm accounting to update.
            # For very-short jobs, don't ever expect a value for CPU use.
            process = Popen(["reportseff", str(job_id)], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            cmd = ' '.join(["reportseff", str(job_id)])
            # print(f"# cmd = {cmd}")
            # print("# reportseff raw output:") ; print(stdout)
            n += 1
            if process.returncode != 0:
                if n < nrepeats:
                    time.sleep(1 + n)
                    continue
                error_message = stderr.decode("utf-8", "ignore").strip()
                if error_message:
                    print("Failed to collect resource efficiency (reportseff): {err}".format(err=error_message))
                else:
                    print("Failed to collect resource efficiency: reportseff exited with code {code}".format(code=process.returncode))
                return
            output_text = stdout.decode("utf-8", "ignore")
            output_text = output_text.rstrip()
            parsed = self._parse_reportseff_output(output_text)
            # print("# parsed:") ; print(parsed)
            if parsed['Elapsed'] != '---' and parsed['MemEff'] != '---':
                break
            n += 1
            time.sleep(n+1)

        # state_note = " ({})".format(final_state) if final_state else ""
        # print("Resource efficiency (reportseff{note}):".format(note=state_note))
        print("Resource efficiency:")
        # if output_text:
        print(output_text)
        # values = []
        # cpu_eff = parsed.get("CPUEff") or parsed.get("AveCPU")
        # mem_eff = parsed.get("MemEff") or parsed.get("MaxRSS")
        # if cpu_eff:
        #     values.append("CPUEff {value}".format(value=cpu_eff))
        # if mem_eff:
        #     values.append("MemEff {value}".format(value=mem_eff))
        # if values:
        #     print("  " + ", ".join(values))

    def _parse_reportseff_output(self, output_text):
        lines = [line for line in output_text.splitlines() if line.strip()]
        if len(lines) < 2:
            return {}
        header = self._strip_ansi_codes(lines[0]).split()
        for raw_line in lines[1:]:
            data = self._strip_ansi_codes(raw_line).split()
            if len(header) == len(data):
                return dict(zip(header, data))
        return {}

    def _strip_ansi_codes(self, text):
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_escape.sub("", text)

def load_ipython_extension(ip):
    """Load extension in IPython."""
    slurm_magic = IPySlurmExec(ip)
    ip.register_magics(slurm_magic)
