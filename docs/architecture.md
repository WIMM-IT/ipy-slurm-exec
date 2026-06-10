# Architecture

This package is split into two main modules:

- `ipy_slurm_exec.py` runs in the notebook process. It implements the `%%slurm_exec` IPython magic, prepares job files, submits the Slurm job, streams job output, and imports requested outputs back into the notebook namespace.
- `ipy_slurm_exec_runtime.py` is copied into each job directory and used by both sides. It contains the serialization and restoration helpers for variables, functions, classes, and class instances.

Each `%%slurm_exec` cell creates one job directory under `slurm_exec/`. That directory contains the serialized input payload, a generated Python driver, a generated `submit.sh`, Slurm stdout/stderr logs, and the final output payload.

## Common Execution Flow

For every cell, `IPySlurmExec.slurm_exec()` performs the same high-level steps:

1. Parse the magic line with `_parse_slurm_exec_args()`.
2. Decide whether inputs and outputs mean "all" or an explicit list.
3. Collect notebook inputs from `self.shell.user_ns`.
4. Serialize collected inputs with `_build_slurm_exec_payload()`.
5. Write `payload.pkl`, `driver.py`, and `submit.sh`.
6. Submit the job with `sbatch`.
7. Wait for `status.json` while streaming `slurm-<jobid>.out`.
8. Read `output.pkl`.
9. Restore output records into notebook objects and push them into `self.shell.user_ns`.

The generated `driver.py` performs the remote half:

1. Load `payload.pkl`.
2. Rebuild the remote namespace from serialized modules, functions, classes, and variables.
3. Compile and execute the notebook cell source as `cell.py`.
4. Select requested output names, or all non-module/non-callable names when output capture is automatic.
5. Serialize outputs into `output.pkl`.
6. Write `status.json` as `COMPLETED` or `FAILED`.

## Scenario: 1 Input Variable

Example:

```python
%%slurm_exec -i data -o result
result = len(data)
```

Notebook-side flow:

1. `_collect_inputs({"data"})` looks up `data` in the notebook namespace.
2. Because `data` is not a function or class, it is placed in the `variables` dictionary.
3. `_build_slurm_exec_payload()` calls `serialize_variable("data", data, job_dir, "input_vars")`.

Runtime serialization chooses the first supported path:

1. `pickle_safely()` tries to copy and pickle the object without mutating it.
2. If that works, the payload record is `{"mode": "pickle", "data": <bytes>}`.
3. If plain pickle is not safe, `detect_save_load_pair()` looks for `data.save(path)` and `type(data).load(path)`.
4. If found, the object writes files under `input_vars/data`, and the payload record is `{"mode": "save_load", ...}`.
5. If no save/load pair exists, the final fallback is `serialize_class_instance()`.

There is one important ordering rule: if the variable is an instance of a top-level class that is not importable by module and qualified name, `serialize_variable()` uses `serialize_class_instance()` before trying plain pickle. This matters for notebook-defined model instances such as `model = CTCFConvNet(...)`. A local pickle may succeed in the notebook process by referring to `__main__.CTCFConvNet`, but that pickle would fail in the Slurm driver because the driver has a different `__main__` module.

The class-instance path applies when the variable is an instance of a top-level notebook-defined class. In that case, the variable itself provides the class: `serialize_class_instance()` calls `type(data)`, serializes that class's source with `serialize_class()`, and pickles the instance state from `__getstate__()` or `__dict__`. The resulting record contains both pieces:

- `class_record`: source for the instance's class.
- `state`: pickled instance state.

If `data` is not safely pickleable, has no save/load pair, and is not an instance of a top-level source-restorable class, export fails with `SerializeFailure`.

Slurm-side flow:

1. `driver.py` sees the variable record in `payload["variables"]`.
2. It calls `restore_from_record(record, JOB_DIR, namespace)`.
3. `restore_from_record()` unpickles the bytes, calls `Class.load(path)`, or rebuilds a class instance depending on the record mode.
4. The restored object is assigned to `namespace["data"]`.
5. The cell executes with `data` available as a normal global variable.

## Scenario: 1 Input Function Or Class

Function example:

```python
def score(x):
    return x * 2

%%slurm_exec -i score -o result
result = score(21)
```

Class example:

```python
class Box:
    scale = 2

    def __init__(self, value):
        self.value = value

    def doubled(self):
        return self.value * self.scale

%%slurm_exec -i Box -o result
result = Box(5).doubled()
```

Notebook-side flow:

1. `_collect_inputs(...)` finds the requested name in the notebook namespace.
2. `inspect.isfunction(...)` puts a function input in the `functions` dictionary.
3. `inspect.isclass(...)` puts a class input in the `classes` dictionary.
4. Nested or local definitions are rejected; only top-level notebook functions and classes are supported.
5. `_build_slurm_exec_payload()` serializes functions with `serialize_function(...)` and classes with `serialize_class(...)`.
6. Function records use `mode="function_ref"`; class records use `mode="class_ref"`.

Function and class source are both exported as source code:

1. `_get_source_for_function(score)` first calls `inspect.getsource(score)`.
2. `_get_source_for_class(Box)` first calls `inspect.getsource(Box)`.
3. If `inspect.getsource(...)` fails, both paths fall back to `_get_cached_source_block()`.
4. The fallback reads cached cell source from `linecache`, parses it with `ast.parse()`, finds a top-level AST node with the requested name, and slices that complete source block.
5. The source is dedented with `textwrap.dedent()` before it is stored in the payload.

The difference is how each object points back to the cached cell:

- A function has its own code object, so the fallback uses `score.__code__.co_filename` and `score.__code__.co_firstlineno`.
- A class has no class-level `__code__`, so the fallback uses methods as breadcrumbs. It reads `co_filename` from a method in `Box.__dict__` and uses the earliest method `co_firstlineno` as an anchor inside the enclosing `class Box:` block.

For classes, this means the package extracts the whole top-level class definition from the cached notebook cell. It does not serialize each method separately and join them together.

In a notebook, "the object's file" usually does not mean the `.ipynb` file. Python function and method objects carry code objects; those code objects have `co_filename`, which Python documents as the name of the file where the code object was created. In IPython/Jupyter this is normally a synthetic filename produced by the interactive compiler.

There are two common shapes:

- In plain IPython, the name may be an angle-bracket pseudo-filename such as `<ipython-input-17-...>`. This is not a real filesystem path.
- In ipykernel/Jupyter, the name may look like a real temporary path, for example `/tmp/ipykernel_<pid>/<hash>.py`. The current ipykernel compiler computes this path from `tempfile.gettempdir()`, the kernel process id, and a hash of the cell code. That path is a filename used for compilation, tracebacks, debugger integration, and source lookup. It should not be treated as the notebook file, and it is not a `__pycache__` bytecode file.

The important point for this package is that source recovery does not depend on opening the `.ipynb` file. IPython's `CachingCompiler.cache()` stores the executed cell source in `linecache.cache` under the generated name and passes that same name as the filename argument to `compile()`. Later, function and method code objects point back to that generated name.

This package follows the same mechanism:

1. `_get_source_for_function()` or `_get_source_for_class()` finds a generated filename from a code object.
2. `_get_cached_source_block()` passes that filename to `linecache.getlines()`.
3. `linecache` returns the cached cell source if IPython registered it.
4. The AST search extracts only the requested top-level function or class definition from that cached cell source.

So, for notebook-defined functions and classes, the source normally comes from IPython's in-memory `linecache` entry for the cell that defined the object. The generated filename is the lookup key. It may look like a path in ipykernel, but the reliable source of truth for this package is the `linecache` entry, not the notebook JSON file and not `__pycache__`.

Slurm-side flow:

1. `driver.py` writes function source to `input_sources/function_<name>.py` and class source to `input_sources/class_<name>.py`.
2. Function records are restored with `restore_function_from_record(record, JOB_DIR, namespace)`.
3. Class records are restored with `restore_class_from_record(record, JOB_DIR, namespace)`.
4. Both restore helpers compile and execute the source in the remote namespace.
5. The restored object is available as `namespace[name]`.
6. Restored classes are tagged with `__slurm_exec_source_record__` so later class-instance outputs can refer back to the same source.
7. The cell executes and can call the function or instantiate the class.

Important limitations:

- Functions and classes are transferred as source code, not as closures.
- Globals used by the function or class must be importable modules or explicitly sent as inputs.
- Classes without any Python methods cannot use the fallback source lookup because there is no method code object to provide a filename and line anchor.

## Scenario: 1 Output Variable

Example:

```python
%%slurm_exec -o history
history = {"loss": [1.0, 0.8, 0.6]}
```

Slurm-side flow:

1. `driver.py` executes the notebook cell source in the remote namespace.
2. Because the user specified `-o history`, `driver.py` selects `history` from that namespace after execution.
3. It calls `serialize_variable("history", history, JOB_DIR, "outputs")`.
4. `serialize_variable()` uses the same record formats as input variables: pickle, save/load, or class source plus instance state.
5. The serialized output is written into `output.pkl` under `{"namespace": {"history": record}}`.
6. If output serialization fails, the error is written into the `errors` section of `output.pkl`.

Notebook-side flow:

1. `slurm_exec()` reads `output.pkl`.
2. For each record in `result_payload["namespace"]`, it calls `restore_from_record(record, job_dir, self.shell.user_ns)`.
3. `restore_from_record()` rebuilds the value from its record mode.
4. Restored values are collected in `namespace_update`.
5. `self.shell.push(namespace_update)` updates the notebook namespace.
6. The magic prints `Imported: history`.

Automatic output capture:

If the user does not specify `-o`, or uses `-o '*'`, then driver will attempt to capture all variables in remote namespace. It scans the remote namespace after the cell executes. It captures non-module, non-function, non-class values and skips callables/modules so helper objects do not flood the notebook namespace.

Important limitations:

- Explicit outputs are strict: if `history` was requested but cannot be serialized or restored, the magic raises.
- Automatic output capture is best effort: values that cannot be serialized or restored are reported, while successful outputs are still imported.
- Output variables are data values. Functions and classes created inside the Slurm cell are not imported back as notebook callables by automatic output capture.

## Job Files

A typical job directory contains:

- `payload.pkl`: notebook-to-job payload containing serialized inputs, module aliases, `sys.path`, output names, and cell source.
- `driver.py`: generated remote runner.
- `submit.sh`: generated Slurm submission script.
- `ipy_slurm_exec_runtime.py`: runtime helper copied beside the driver.
- `cell.py`: the executed cell source, written by the driver for clearer tracebacks.
- `input_sources/`: restored function and class source files.
- `slurm-<jobid>.out`: Slurm stdout, streamed back to the notebook while the job runs.
- `slurm-<jobid>.err`: Slurm stderr.
- `output.pkl`: job-to-notebook payload containing serialized outputs and export errors.
- `status.json`: final driver status.
- `traceback.log`: remote traceback when cell execution fails.

## Failure Handling

Input export failures are strict when the user explicitly names the failing input. When input capture is automatic, unsupported objects are skipped and reported as soft failures so unrelated notebook state does not prevent the job from running.

Remote execution failures write `traceback.log` and `status.json`. The notebook side displays the traceback when available and raises an error that points to the job directory.

Output import failures are strict for explicitly requested outputs. When output capture is automatic, failures are reported and the successfully restored outputs are still imported.

## Source Lookup References

- Python `inspect.getsource()` returns source text for objects such as functions and classes, and raises when source cannot be retrieved: <https://docs.python.org/3/library/inspect.html#inspect.getsource>.
- Python code objects expose `co_filename`, documented as the file name where the code object was created: <https://docs.python.org/3/library/inspect.html>.
- Python `linecache` retrieves source lines by filename and can use cached source rather than reading a normal file every time: <https://docs.python.org/3/library/linecache.html>.
- IPython provides a Jupyter kernel, input history, object introspection, and input caching facilities used by interactive execution: <https://ipython.readthedocs.io/en/stable/> and <https://ipython.readthedocs.io/en/9.4.0/interactive/reference.html#input-caching-system>.
- IPython's `CachingCompiler` API describes caching interactive statements and returning a generated name to pass as the compile filename: <https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.compilerop.html>.
- ipykernel's compiler API exposes the Jupyter-kernel-specific helpers `XCachingCompiler`, `get_file_name()`, and `get_tmp_directory()`: <https://ipykernel.readthedocs.io/en/stable/api/ipykernel.html>.
