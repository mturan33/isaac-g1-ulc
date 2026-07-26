#!/usr/bin/env python3
"""Diff two ``run_config.json`` files and print every field that differs.

Answers one question: before calling two runs a comparison, what was actually
not held constant between them?

    python diff_run_config.py logs/ulc/run_a/run_config.json \
                              logs/ulc/run_b/run_config.json

Findings are grouped so the ones that invalidate a comparison come first.
Exit code is 1 when anything outside the always-differs set changed, 0 when the
two runs are configured identically - so it can gate a comparison in a script.

Options
    --all        also show incidental fields (timestamps, log dir, command line)
    --only PFX   restrict to fields under a prefix, e.g. --only curriculum
    --quiet      print the summary only
"""

import argparse
import json
import os
import sys

# Fields that differ between any two runs and say nothing about comparability.
INCIDENTAL = ("created_at", "log_dir", "command_line", "script_path",
              "git.describe", "git.dirty")

# Printed first: a difference here means the two runs are not a clean comparison.
CRITICAL_PREFIXES = (
    "args.num_envs", "args.max_iterations", "args.seed", "seed",
    "curriculum", "reward_weights", "dims", "upstream_checkpoints",
)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj                      # keep short scalar lists whole
        else:
            for i, v in enumerate(obj):
                out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def fmt(v, width=34):
    if v is None:
        s = "-"
    elif isinstance(v, list):
        s = "[" + ", ".join(str(x) for x in v) + "]"
    else:
        s = str(v)
    s = s.replace("\n", " ")
    return s if len(s) <= width else s[: width - 1] + "…"


def classify(key):
    for p in CRITICAL_PREFIXES:
        if key == p or key.startswith(p + ".") or key.startswith(p + "["):
            return "critical"
    if key.startswith("versions") or key.startswith("git"):
        return "environment"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Diff two run_config.json files.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--all", action="store_true", help="include incidental fields")
    ap.add_argument("--only", default=None, help="restrict to a field prefix")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not os.path.isfile(p):
            print(f"error: no such file: {p}")
            return 2

    with open(args.a, encoding="utf-8") as fh:
        A = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        B = json.load(fh)

    fa, fb = flatten(A), flatten(B)
    keys = sorted(set(fa) | set(fb))

    diffs = []
    for k in keys:
        if not args.all and any(k == i or k.startswith(i + ".") for i in INCIDENTAL):
            continue
        if args.only and not k.startswith(args.only):
            continue
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if va != vb:
            diffs.append((classify(k), k, va, vb))

    name_a = A.get("script") or os.path.basename(os.path.dirname(args.a))
    name_b = B.get("script") or os.path.basename(os.path.dirname(args.b))
    la = A.get("log_dir") or args.a
    lb = B.get("log_dir") or args.b

    print()
    print("=" * 100)
    print(f"A  {name_a}   {la}")
    print(f"B  {name_b}   {lb}")
    print("=" * 100)

    if not diffs:
        print("\nNo differences. These two runs were configured identically.\n")
        return 0

    if not args.quiet:
        for group, title in (
            ("critical", "COMPARISON-CRITICAL - these were NOT held constant"),
            ("environment", "ENVIRONMENT (git / library versions)"),
            ("other", "OTHER"),
        ):
            rows = [d for d in diffs if d[0] == group]
            if not rows:
                continue
            print(f"\n{title}")
            print("-" * 100)
            print(f"  {'FIELD':<40} {'A':<26} {'B':<26}")
            for _, k, va, vb in rows:
                print(f"  {fmt(k, 40):<40} {fmt(va, 26):<26} {fmt(vb, 26):<26}")

    n_crit = sum(1 for d in diffs if d[0] == "critical")
    print()
    print("=" * 100)
    print(f"{len(diffs)} field(s) differ - {n_crit} comparison-critical.")
    if n_crit:
        cl_a = (A.get("curriculum") or {}).get("n_levels")
        cl_b = (B.get("curriculum") or {}).get("n_levels")
        if cl_a != cl_b:
            print(f"NOTE: different curricula ({cl_a} vs {cl_b} levels) - the runs did not "
                  "train on the same task ladder.")
        if A.get("seed") != B.get("seed"):
            print(f"NOTE: different seeds ({A.get('seed')} vs {B.get('seed')}) - a single "
                  "pair cannot separate seed variance from a real effect.")
        print("Treat any performance gap between these runs as confounded until these match.")
    print("=" * 100)
    print()
    return 1 if n_crit else 0


if __name__ == "__main__":
    sys.exit(main())
