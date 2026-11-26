#!/usr/bin/env python3
"""Replay a pinned set of native source generators and count their outputs."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
ROLES = ("value", "response", "mixed", "support")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


def line_counts(text):
    lines = text.splitlines()
    return {
        "physical": len(lines),
        "nonblank": sum(bool(line.strip()) for line in lines),
    }


def describe(path, symbol="", *, content=None):
    if content is None:
        content = path.read_bytes()
    text = content.decode("utf-8")
    record = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if symbol:
        nodes = [
            node
            for node in ast.parse(text, filename=str(path)).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol
        ]
        if len(nodes) != 1:
            raise ValueError(f"{path}: expected one top-level {symbol}")
        node = nodes[0]
        first_line = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        record.update(symbol=symbol, first_line=first_line, last_line=node.end_lineno)
        text = "\n".join(text.splitlines()[first_line - 1 : node.end_lineno])
    return record | line_counts(text)


def authored_rows(repo, group, entries, tracked):
    rows = []
    for category, selectors in entries.items():
        for selector in selectors:
            path, _, symbol = selector.partition(":")
            if path not in tracked:
                raise ValueError(f"Counted input is outside the pinned tree: {path}")
            rows.append(
                {"group": group, "category": category, "path": path}
                | describe(repo / path, symbol)
            )
    return rows


def run_generator(repo, job, scratch, timeout, clang_format):
    directory = scratch / job["id"]
    directory.mkdir()
    command = [job["script"], *job["argv"]]
    script = repo / command[0]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, (repo / "python", repo))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    record = {
        "id": job["id"],
        "group": job["group"],
        "argv": ["python", *command],
        "generator_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "timeout_seconds": timeout,
    }
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(script),
                *(arg.replace("{source}", str(repo)) for arg in command[1:]),
            ],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        record["returncode"] = process.returncode
        status = "ok" if process.returncode == 0 else "failed"
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as error:
        status = "timeout"
        stdout = (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or b"").decode(errors="replace")

    for name, stream in (("stdout", stdout), ("stderr", stderr)):
        if stream.strip():
            record[name] = stream.replace(str(directory), "<output>").replace(
                str(repo), "<source>"
            )
    outputs = {}
    for name, role in job["outputs"].items():
        if "{df_class}" in name:
            for angular in itertools.product(range(4), range(4), range(5)):
                label = "".join(map(str, angular))
                outputs[name.format(df_class=label)] = role
        else:
            outputs[name] = role
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in NATIVE_SUFFIXES
    }
    expected = set(outputs)
    record["missing"] = sorted(expected - actual)
    record["unexpected"] = sorted(actual - expected)
    if status == "ok" and (record["missing"] or record["unexpected"]):
        status = "output_mismatch"
    record["status"] = status

    rows = []
    for name in sorted(actual):
        row = {
            "group": job["group"],
            "generator": job["id"],
            "path": name,
            "role": outputs.get(name, "unclassified"),
            **describe(directory / name),
        }
        if name in job.get("tracked", {}):
            tracked_path = job["tracked"][name]
            row["raw_replay"] = {
                key: row[key]
                for key in ("path", "sha256", "bytes", "physical", "nonblank")
            }
            formatted = subprocess.check_output(
                [str(clang_format), f"--assume-filename={repo / tracked_path}"],
                input=(directory / name).read_bytes(),
            )
            row["formatted_replay_sha256"] = hashlib.sha256(formatted).hexdigest()
            row.update(describe(repo / tracked_path), path=tracked_path)
            row["origin"] = "checked_in"
            if row["formatted_replay_sha256"] == row["sha256"]:
                row["comparison"] = "exact_after_formatting"
            elif (
                formatted.split(b"\n", 1)[1]
                == (repo / tracked_path).read_bytes().split(b"\n", 1)[1]
            ):
                row["comparison"] = "first_line_differs_after_formatting"
            else:
                row["comparison"] = "differs_after_formatting"
            if row["comparison"] != "exact_after_formatting" and row["role"] in ROLES:
                record["status"] = "tracked_mismatch"
        else:
            row["origin"] = "replayed"
        rows.append(row)
    return record, rows


def summarize(authored, generated, groups):
    tracked = [row for row in authored if row["group"] in groups]
    emitted = [
        row for row in generated if row["group"] in groups and row["role"] in ROLES
    ]
    return {
        "spec": sum(row["nonblank"] for row in tracked if row["category"] == "spec"),
        "runtime": sum(
            row["nonblank"] for row in tracked if row["category"] == "runtime"
        ),
        "compiler_support": sum(
            row["nonblank"] for row in tracked if row["category"] == "compiler_support"
        ),
        "generated_total": sum(row["nonblank"] for row in emitted),
        "generated_files": len(emitted),
        "generated_by_role": {
            role: sum(row["nonblank"] for row in emitted if row["role"] == role)
            for role in ROLES
        },
    }


GENERATED_HISTORY = {
    "include/vibeqc/generated_method_ids.h",
    "python/vibeqc/_generated_methods.py",
    "src/methods/generated_method_manifest.hpp",
}
IMPLEMENTATION_ROOTS = ("cmake/", "include/", "methods/", "python/", "src/", "tools/")
VALIDATION_ROOTS = ("tests/", ".github/")


def history_category(path):
    if path in GENERATED_HISTORY:
        return "generated"
    if path.startswith("external/"):
        return "vendored"
    if path.startswith(VALIDATION_ROOTS) or path in {
        "tools/generate_xc_references.py",
        "tools/validate_stationary_cuda_compile.py",
        "python/vibeqc_compiler/xc/fixtures.py",
    }:
        return "validation"
    if path.startswith(IMPLEMENTATION_ROOTS):
        return "implementation"
    return "other"


def count_history(repo, receipts):
    first = datetime.fromisoformat(receipts["initial_commit"]["authored_at"])
    for stage in receipts["stages"]:
        sha = stage["merge_commit"]
        git(repo, "merge-base", "--is-ancestor", sha, "HEAD")
        stage["first_parent"] = git(repo, "rev-parse", f"{sha}^1")
        changes = git(
            repo, "diff", "--no-renames", "--numstat", stage["first_parent"], sha
        )
        rows = []
        for line in changes.splitlines():
            added, deleted, path = line.split("\t", 2)
            rows.append(
                {
                    "path": path,
                    "category": history_category(path),
                    "added": None if added == "-" else int(added),
                    "deleted": None if deleted == "-" else int(deleted),
                }
            )
        stage["files"] = rows
        stage["line_changes"] = {
            category: {
                key: sum(row[key] or 0 for row in rows if row["category"] == category)
                for key in ("added", "deleted")
            }
            for category in (
                "implementation",
                "validation",
                "generated",
                "vendored",
                "other",
            )
        }
    for receipt in receipts["qualification"]:
        reported = datetime.fromisoformat(receipt["reported_at"])
        receipt["seconds_from_first_authored_commit_to_report"] = int(
            (reported - first).total_seconds()
        )
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="clean pinned checkout"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clang-format",
        type=Path,
        default=Path(sys.executable).with_name("clang-format"),
        help="formatter for checked-in generated MP2 headers",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel source generators"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per generator"
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")

    scope = json.loads((HERE / "scope.json").read_text())
    repo = args.source.resolve()
    if git(repo, "rev-parse", "HEAD") != scope["source"]:
        parser.error(f"--source must be checked out at {scope['source']}")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        parser.error("--source must be a clean source checkout")
    clang_format = args.clang_format.resolve(strict=True)
    clang_format_version = subprocess.check_output(
        [str(clang_format), "--version"], text=True
    ).strip()
    tracked = set(git(repo, "ls-files", "-z").rstrip("\0").split("\0"))
    authored = [
        row
        for group, entries in scope["groups"].items()
        for row in authored_rows(repo, group, entries, tracked)
    ]
    for spec in scope["generators"]:
        if (
            spec["script"] not in tracked
            or not set(spec.get("tracked", {}).values()) <= tracked
        ):
            raise ValueError(
                f"Generator or tracked output outside pinned tree: {spec['id']}"
            )
    with (
        tempfile.TemporaryDirectory(prefix="gqc-codegen-census-") as tmp,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        results = pool.map(
            lambda spec: run_generator(
                repo, spec, Path(tmp), args.timeout, clang_format
            ),
            scope["generators"],
        )
        runs, outputs = zip(*results, strict=True)
    generated = [row for rows in outputs for row in rows]
    complete = all(run["status"] == "ok" for run in runs)
    valid_ids = {run["id"] for run in runs if run["status"] == "ok"}
    valid_outputs = [row for row in generated if row["generator"] in valid_ids]
    report = {
        "source": scope["source"],
        "source_tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "environment": {
            "python": platform.python_version(),
            "git": git(repo, "--version"),
            "clang_format": clang_format_version,
            "system": platform.system(),
            "machine": platform.machine(),
            "os_release": platform.mac_ver()[0] or platform.release(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "array-api-compat", "typing-extensions")
            },
            "parallel_generators": args.jobs,
            "remote_allocations": [],
            "gpu_card_hours": 0,
            "remote_cost": 0,
        },
        "status": "complete" if complete else "partial",
        "authored_sources": authored,
        "generator_runs": runs,
        "generated_sources": generated,
        "r2scan_history": count_history(
            repo, json.loads((HERE / "r2scan-github.json").read_text())
        ),
        "groups": {
            group: summarize(authored, valid_outputs, [group])
            for group in scope["groups"]
        },
        "families": {
            family: {"groups": groups} | summarize(authored, valid_outputs, groups)
            for family, groups in scope["families"].items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not complete:
        sys.exit("Census incomplete. See generator_runs in the output record.")


if __name__ == "__main__":
    main()
