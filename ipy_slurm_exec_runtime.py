import copy
import ast
import importlib
import inspect
import linecache
import pickle
import textwrap
from dataclasses import dataclass
from pathlib import Path

_SLURM_EXEC_SOURCE_RECORD_ATTR = "__slurm_exec_source_record__"


class SerializeFailure(RuntimeError):
    """Indicates an object cannot be serialized by :func:`serialize_variable`."""

    def __init__(self, obj_type, name):
        """Store the failed object's type and notebook variable name."""
        self.obj_type = obj_type
        self.name = name
        super().__init__(str(self))

    def __str__(self):
        """Return a concise description of the unsupported object."""
        module = ( "" if self.obj_type.__module__ == "builtins" else f"{self.obj_type.__module__}." )
        type_name = f"{module}{self.obj_type.__name__}"
        return f"{self.name} <{type_name}>"


@dataclass(frozen=True)
class SourceRef:
    mode: str
    name: str
    format: str
    payload: object

    def to_record(self):
        """Return this source reference as a pickle-friendly dictionary."""
        return { "mode": self.mode, "name": self.name, "format": self.format, "payload": self.payload }


def _is_defined_function(obj):
    """Return True when obj is a top-level function that can be restored from source."""
    return inspect.isfunction(obj) and getattr(obj, "__qualname__", "") == getattr(obj, "__name__", "")


def _is_defined_class(obj):
    """Return True when obj is a top-level class that can be restored from source."""
    return inspect.isclass(obj) and getattr(obj, "__qualname__", "") == getattr(obj, "__name__", "")


def is_source_restored_class_instance(obj):
    """Return True when obj was created from a class restored by this runtime."""
    return hasattr(type(obj), _SLURM_EXEC_SOURCE_RECORD_ATTR)


def _get_cached_source_block(filename, name, anchor_lineno, node_type):
    """Read a function or class source block from linecache using AST locations."""
    lines = linecache.getlines(filename)
    if not lines:
        raise OSError("could not get source code")
    source = "".join(lines)
    tree = ast.parse(source, filename=filename)
    for node in tree.body:
        if not isinstance(node, node_type) or node.name != name:
            continue
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            continue
        if anchor_lineno is not None and not (node.lineno <= anchor_lineno <= end_lineno):
            continue
        return textwrap.dedent("".join(lines[node.lineno - 1:end_lineno]))
    raise OSError("could not get source code")


def _get_source_for_function(func):
    """Return dedented source for a function, falling back to linecache if needed."""
    try:
        return inspect.getsource(func)
    except Exception:
        return _get_cached_source_block(
            filename=func.__code__.co_filename,
            name=func.__name__,
            anchor_lineno=func.__code__.co_firstlineno,
            node_type=ast.FunctionDef,
        )


def _get_source_for_class(cls):
    """Return source for a class by anchoring on one of its defined methods."""
    methods = [ value for value in cls.__dict__.values() if inspect.isfunction(value) ]
    if not methods:
        raise OSError("could not get source code")
    anchor = min(method.__code__.co_firstlineno for method in methods)
    filename = methods[0].__code__.co_filename
    try:
        return inspect.getsource(cls)
    except Exception:
        return _get_cached_source_block(
            filename=filename,
            name=cls.__name__,
            anchor_lineno=anchor,
            node_type=ast.ClassDef,
        )


def serialize_function(name, func):
    """Serialize a top-level function as source code that can be executed remotely."""
    if not _is_defined_function(func):
        raise SerializeFailure(type(func), name)
    try:
        source = _get_source_for_function(func)
    except Exception as exc:
        raise SerializeFailure(type(func), name) from exc
    return SourceRef(mode="function_ref", name=func.__name__, format="source", payload=textwrap.dedent(source)).to_record()


def serialize_class(name, cls):
    """Serialize a top-level class as source code that can be executed remotely."""
    if not _is_defined_class(cls):
        raise SerializeFailure(type(cls), name)
    try:
        source = _get_source_for_class(cls)
    except Exception as exc:
        raise SerializeFailure(type(cls), name) from exc
    return SourceRef(mode="class_ref", name=cls.__name__, format="source", payload=textwrap.dedent(source)).to_record()


def serialize_class_instance(name, value, protocol=pickle.HIGHEST_PROTOCOL):
    """Serialize an instance by pairing its class source with pickled instance state."""
    cls = type(value)
    if not _is_defined_class(cls):
        raise SerializeFailure(type(value), name)

    class_record = getattr(cls, _SLURM_EXEC_SOURCE_RECORD_ATTR, None)
    if class_record is None:
        class_record = serialize_class(cls.__name__, cls)
    state_kind = "dict"
    try:
        if hasattr(value, "__getstate__"):
            state = value.__getstate__()
            state_kind = "getstate"
        else:
            state = value.__dict__
        state_payload = pickle.dumps(state, protocol=protocol)
    except Exception as exc:
        raise SerializeFailure(type(value), name) from exc

    return {
        "mode": "class_instance",
        "class_record": class_record,
        "state_kind": state_kind,
        "state": state_payload,
    }


def _deep_signature(obj, _seen=None):
    """Build a lightweight recursive signature to detect mutation during pickling."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<cycle>"
    _seen.add(oid)

    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return obj
    if isinstance(obj, (tuple, list, set, frozenset)):
        if len(obj) > 10:
            return (type(obj).__name__, f"<len={len(obj)}>")
        return (type(obj).__name__, tuple(_deep_signature(x, _seen) for x in obj))
    if isinstance(obj, dict):
        if len(obj) > 10:
            return ("dict", f"<len={len(obj)}>")
        items = []
        for k, v in obj.items():
            items.append((_deep_signature(k, _seen), _deep_signature(v, _seen)))
        return ("dict", tuple(sorted(items)))

    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            items = []
            for k, v in d.items():
                items.append((k, _deep_signature(v, _seen)))
            return (type(obj).__name__, tuple(sorted(items)))
    except Exception:
        pass

    return (type(obj).__name__, repr(obj))


def pickle_safely(obj, protocol=pickle.HIGHEST_PROTOCOL):
    """Return pickled bytes only when copying and pickling leave obj unchanged."""
    try:
        probe = copy.copy(obj)
    except Exception:
        return None
    sig_before = _deep_signature(probe)
    try:
        pkl_obj = pickle.dumps(probe, protocol=protocol)
    except Exception:
        return None
    sig_after = _deep_signature(probe)
    if sig_before != sig_after:
        return None
    return pickle.dumps(obj, protocol=protocol)


def _has_single_path_param(fn, drop_first):
    """Return True when fn has exactly one required path-like positional argument."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    params = [ p for p in sig.parameters.values()
               if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
               and p.default is inspect._empty ]
    if drop_first and params and params[0].name in ("self", "cls"):
        params = params[1:]
    if len(params) != 1:
        return False
    name = params[0].name.lower()
    return any(tok in name for tok in ("path", "file", "dir"))


def detect_save_load_pair(obj):
    """Detect objects that expose compatible save(path) and type.load(path) methods."""
    save_fn = getattr(obj, "save", None)
    load_fn = getattr(type(obj), "load", None)
    if not (callable(save_fn) and callable(load_fn)):
        return None
    if not _has_single_path_param(save_fn, drop_first=False):
        return None
    if not _has_single_path_param(load_fn, drop_first=True):
        return None
    return {"save_fn": save_fn, "load_fn": load_fn, "cls": type(obj)}


def _import_class(module_name, qualname):
    """Import a class from its module name and qualified attribute path."""
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _is_importable_class(cls):
    """Return True when cls can be imported by its module and qualified name."""
    if cls.__module__ in {"__main__", "__mp_main__"}:
        return False
    try:
        return _import_class(cls.__module__, cls.__qualname__) is cls
    except Exception:
        return False


def serialize_variable(name, value, root_dir, rel_root, protocol=pickle.HIGHEST_PROTOCOL):
    """Serialize a variable using safe pickle, save/load hooks, or class source plus state."""
    value_cls = type(value)
    if (value_cls.__module__ != "builtins"
        and _is_defined_class(value_cls)
        and not _is_importable_class(value_cls)
    ):
        return serialize_class_instance(name, value, protocol=protocol)

    pkl_obj = pickle_safely(value, protocol=protocol)
    if pkl_obj is not None:
        # Then was safe
        return {"mode": "pickle", "data": pkl_obj}

    handler = detect_save_load_pair(value)
    if handler is None:
        return serialize_class_instance(name, value, protocol=protocol)

    # Some save functions accept boolean knobs such as save_anndata; enable any that look like save flags.
    save_kwargs = {}
    try:
        for pname, param in inspect.signature(handler["save_fn"]).parameters.items():
            if not pname.lower().startswith("save"):
                continue
            if param.default is inspect._empty:
                continue
            if isinstance(param.default, bool):
                save_kwargs[pname] = True
    except Exception:
        pass

    rel_path = Path(rel_root) / name
    abs_path = Path(root_dir) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    handler["save_fn"](abs_path, **save_kwargs)
    return {
        "mode": "save_load",
        "class_module": handler["cls"].__module__,
        "class_qualname": handler["cls"].__qualname__,
        "path": str(rel_path),
    }


def restore_from_record(record, job_dir, globals_ns=None):
    """Restore a serialized variable record in the remote or notebook namespace."""
    if globals_ns is None:
        globals_ns = {"__builtins__": __builtins__}
    mode = record.get("mode")
    if mode == "pickle":
        return pickle.loads(record["data"])
    if mode == "save_load":
        cls = _import_class(record["class_module"], record["class_qualname"])
        return cls.load(Path(job_dir) / record["path"])
    if mode == "class_instance":
        class_record = record["class_record"]
        class_name = class_record["name"]
        cls = globals_ns.get(class_name)
        if not inspect.isclass(cls):
            cls = restore_class_from_record(class_record, job_dir, globals_ns)
        state = pickle.loads(record["state"])
        obj = cls.__new__(cls)
        if record.get("state_kind") == "getstate" and hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
            return obj
        if hasattr(obj, "__dict__") and isinstance(state, dict):
            obj.__dict__.update(state)
            return obj
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
            return obj
        raise RuntimeError("Cannot restore class instance state.")
    raise RuntimeError("Unknown record mode when restoring variable.")


def restore_function_from_record(record, job_dir, globals_ns=None):
    """Restore a function record by executing its source in globals_ns."""
    if globals_ns is None:
        globals_ns = {"__builtins__": __builtins__}
    mode = record.get("mode")
    if mode != "function_ref":
        raise RuntimeError("Unknown record mode when restoring function.")
    fmt = record.get("format")
    if fmt == "source":
        payload = record.get("payload")
        filename = record.get("filename", f"<slurm_exec:function:{record['name']}>")
        exec(compile(payload, filename, "exec"), globals_ns)
        return globals_ns[record["name"]]
    raise RuntimeError("Unknown function payload format.")


def restore_class_from_record(record, job_dir, globals_ns=None):
    """Restore a class record by executing its source and tagging it for later export."""
    if globals_ns is None:
        globals_ns = {"__builtins__": __builtins__}
    mode = record.get("mode")
    if mode != "class_ref":
        raise RuntimeError("Unknown record mode when restoring class.")
    fmt = record.get("format")
    if fmt == "source":
        payload = record.get("payload")
        filename = record.get("filename", f"<slurm_exec:class:{record['name']}>")
        exec(compile(payload, filename, "exec"), globals_ns)
        cls = globals_ns[record["name"]]
        setattr(cls, _SLURM_EXEC_SOURCE_RECORD_ATTR, record)
        return cls
    raise RuntimeError("Unknown class payload format.")
