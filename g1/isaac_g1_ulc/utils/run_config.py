"""Run configuration capture for ULC training scripts.

Every training run writes a ``run_config.json`` into its log directory recording
exactly what it was configured with: all CLI arguments, the fully resolved
curriculum (every level, not just a name), every reward weight, observation and
action dimensions, upstream checkpoints, the git commit and library versions.

The point is that two runs can be compared field by field afterwards instead of
being assumed comparable::

    python utils/diff_run_config.py logs/ulc/run_a/run_config.json \\
                                    logs/ulc/run_b/run_config.json

Wiring a new training script takes three lines - see the block comment at the
bottom of this file.
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

SCHEMA_VERSION = 1
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def set_global_seed(seed=DEFAULT_SEED, deterministic_torch=False):
    """Seed python, numpy and torch (CPU + all CUDA devices).

    Call this immediately after parsing arguments, before any environment or
    network is created. Returns a dict describing what was actually seeded, so
    the caller can record it.

    Note this does NOT make PhysX stepping bit-deterministic - it makes the
    python-side sampling (target poses, command sampling, weight init,
    exploration noise) reproducible. Set ``deterministic_torch=True`` to also
    request deterministic cuDNN kernels, at a speed cost.
    """
    applied = {"seed": int(seed), "python": False, "numpy": False,
               "torch": False, "cuda": False, "deterministic_torch": False}

    os.environ["PYTHONHASHSEED"] = str(seed)

    import random
    random.seed(seed)
    applied["python"] = True

    try:
        import numpy as np
        np.random.seed(seed)
        applied["numpy"] = True
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        applied["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["cuda"] = True
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                applied["deterministic_torch"] = True
            except Exception:
                pass
    except Exception:
        pass

    return applied


# ---------------------------------------------------------------------------
# environment capture
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Convert tuples/sets/numpy scalars to JSON-safe values, recursively."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            try:
                return _jsonable(getattr(obj, attr)())
            except Exception:
                pass
    return str(obj)


def _git_info(anchor_path):
    """Commit hash, branch and dirty flag for the repo containing anchor_path."""
    info = {"commit": None, "branch": None, "dirty": None, "describe": None, "error": None}
    try:
        d = os.path.dirname(os.path.abspath(anchor_path))

        def g(*a):
            r = subprocess.run(["git", "-C", d, *a], capture_output=True,
                               text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else ""

        commit = g("rev-parse", "HEAD")
        if not commit:
            info["error"] = "not a git repository (or git unavailable)"
            return info
        info["commit"] = commit
        info["branch"] = g("rev-parse", "--abbrev-ref", "HEAD") or None
        info["describe"] = g("describe", "--always", "--dirty") or None
        info["dirty"] = bool(g("status", "--porcelain"))
    except Exception as exc:  # never let bookkeeping kill a run
        info["error"] = str(exc)
    return info


def _versions():
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("torch", "numpy", "isaaclab", "isaacsim", "gymnasium", "tensorboard"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:
            out[name] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["cudnn"] = str(getattr(torch.backends.cudnn, "version", lambda: None)())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            out["gpu_vram_MiB"] = int(props.total_memory / (1024 * 1024))
        else:
            out["gpu"] = None
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------

def dump_run_config(log_dir, args, *, script=None, curriculum=None,
                    reward_weights=None, dims=None, upstream=None,
                    seed_applied=None, extra=None, filename="run_config.json",
                    quiet=False):
    """Write ``run_config.json`` into ``log_dir`` and return the dict written.

    Parameters
    ----------
    log_dir : str            destination directory (already created)
    args : argparse.Namespace  every CLI value is recorded verbatim
    script : str             ``__file__`` of the training script
    curriculum : sequence    the RESOLVED curriculum - every level dict
    reward_weights : dict    e.g. ``{"loco": LOCO_REWARD_WEIGHTS, "arm": ARM_REWARD_WEIGHTS}``
    dims : dict              e.g. ``{"loco_obs": 57, "arm_obs": 52, ...}``
    upstream : dict          upstream checkpoint paths actually used
    seed_applied : dict      return value of :func:`set_global_seed`
    extra : dict             anything else worth pinning

    Bookkeeping never aborts training: on failure this prints a warning and
    returns None.
    """
    try:
        args_dict = _jsonable(vars(args)) if args is not None else {}

        levels = None
        if curriculum is not None:
            levels = {str(i): _jsonable(lv) for i, lv in enumerate(curriculum)}

        cfg = {
            "schema": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": os.path.basename(script) if script else None,
            "log_dir": log_dir.replace("\\", "/") if log_dir else None,
            "command_line": " ".join(sys.argv),
            "seed": args_dict.get("seed"),
            "seed_applied": _jsonable(seed_applied) if seed_applied else None,
            "args": args_dict,
            "dims": _jsonable(dims) if dims else {},
            "reward_weights": _jsonable(reward_weights) if reward_weights else {},
            "curriculum": {
                "n_levels": len(curriculum) if curriculum is not None else None,
                "levels": levels,
            },
            "upstream_checkpoints": _jsonable(upstream) if upstream else {},
            "git": _git_info(script or __file__),
            "versions": _versions(),
            "determinism_note": (
                "Seeds cover python/numpy/torch sampling and weight init. PhysX "
                "stepping is not bit-deterministic, so runs are reproducible in "
                "configuration, not bit-exact in trajectory."
            ),
        }
        if extra:
            cfg["extra"] = _jsonable(extra)

        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)

        if not quiet:
            n = cfg["curriculum"]["n_levels"]
            print(f"\n[run_config] wrote {path}")
            print(f"[run_config] seed={cfg['seed']}  num_envs={args_dict.get('num_envs')}  "
                  f"max_iterations={args_dict.get('max_iterations')}  curriculum_levels={n}")
            g = cfg["git"]
            if g.get("commit"):
                print(f"[run_config] git {g['commit'][:10]}"
                      f"{' (dirty)' if g.get('dirty') else ''} on {g.get('branch')}")
        return cfg

    except Exception as exc:
        print(f"[run_config] WARNING: could not write run config: {exc}")
        return None


# ---------------------------------------------------------------------------
# Wiring a new training script (three edits)
# ---------------------------------------------------------------------------
#
# 1. near the top, after the stdlib imports:
#
#        import os, sys
#        sys.path.insert(0, os.path.join(
#            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#            "utils"))
#        from run_config import set_global_seed, dump_run_config, DEFAULT_SEED
#
# 2. in parse_args(), add the flag, and seed right after parsing:
#
#        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
#        ...
#        args_cli = parse_args()
#        SEED_APPLIED = set_global_seed(args_cli.seed)
#
# 3. right after the log directory is created:
#
#        dump_run_config(log_dir, args_cli, script=__file__,
#                        curriculum=CURRICULUM,
#                        reward_weights={"loco": LOCO_REWARD_WEIGHTS,
#                                        "arm": ARM_REWARD_WEIGHTS},
#                        dims={...}, upstream={...}, seed_applied=SEED_APPLIED)
