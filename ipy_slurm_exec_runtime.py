import copy
import importlib
import inspect
import pickle
from pathlib import Path


def _deep_signature(obj, _seen=None):
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


def is_pickle_safe(obj, protocol=pickle.HIGHEST_PROTOCOL):
    """Return True if object survives pickle round-trip without mutating state."""
    try:
        probe = copy.copy(obj)
    except Exception:
        return False
    sig_before = _deep_signature(probe)
    try:
        pickle.dumps(probe, protocol=protocol)
    except Exception:
        return False
    sig_after = _deep_signature(probe)
    if sig_before != sig_after:
        return False
    return True


def _has_single_path_param(fn, drop_first):
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty
    ]
    if drop_first and params and params[0].name in ("self", "cls"):
        params = params[1:]
    if len(params) != 1:
        return False
    name = params[0].name.lower()
    return any(tok in name for tok in ("path", "file", "dir"))


def detect_save_load_pair(obj):
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
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def serialize_variable(name, value, root_dir, rel_root, protocol=pickle.HIGHEST_PROTOCOL):
    if is_pickle_safe(value, protocol=protocol):
        return {"mode": "pickle", "data": pickle.dumps(value, protocol=protocol)}

    handler = detect_save_load_pair(value)
    if handler is None:
        raise RuntimeError("Object is not pickle-safe and lacks a save/load pair.")

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


def restore_from_record(record, job_dir):
    mode = record.get("mode")
    if mode == "pickle":
        return pickle.loads(record["data"])
    if mode == "save_load":
        cls = _import_class(record["class_module"], record["class_qualname"])
        return cls.load(Path(job_dir) / record["path"])
    raise RuntimeError("Unknown record mode when restoring variable.")
