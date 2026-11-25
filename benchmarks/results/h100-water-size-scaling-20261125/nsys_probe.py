"""Run source-bound Nsight CUDA diagnostics for two observed Direct losses."""

import argparse
import json
import signal
import sqlite3
import subprocess
from pathlib import Path

from acquire import (
    METHODS,
    MODES,
    SIZES,
    digest,
    environment,
    interrupt,
    save,
    stop_process,
    verify,
)

CAPTURE_SHA256 = "67620bee0161e0296db2807ae8b5895fe433feb4dddc17543e712c744e2d4075"
CASES = ((240, "hf", "direct"), (96, "wb97m-v", "direct"))


def run(argv, logfile, cwd, env, timeout, result):
    result.update(argv=list(map(str, argv)), status="running", timeout=False)
    child = None
    try:
        with logfile.open("xb") as log:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timeout"] = True
            finally:
                stop_process(child)
    except (KeyboardInterrupt, SystemExit):
        result["status"] = "interrupted"
        raise
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["returncode"] = child.returncode if child is not None else None
    if result["timeout"]:
        result["status"] = "timeout"
    else:
        result["status"] = "complete" if result["returncode"] == 0 else "failed"


def kernel_census(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        fields = connection.execute(
            """SELECT n.value, COUNT(*), SUM(k.end-k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds n ON n.id=k.shortName
            GROUP BY n.value ORDER BY 3 DESC, 1"""
        )
        by_name = [
            {"name": name, "calls": count, "gpu_seconds_sum": ns * 1e-9}
            for name, count, ns in fields
        ]
        union, right = 0, 0
        for start, end in connection.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start, end"
        ):
            union += max(0, end - max(start, right))
            right = max(end, right)
    return {
        "kernel_union_seconds": union * 1e-9,
        "kernel_duration_sum_seconds": sum(row["gpu_seconds_sum"] for row in by_name),
        "kernels": by_name,
    }


def parse_case(value):
    try:
        atoms_text, method, mode = value.split(":")
        atoms = int(atoms_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected atoms:method:mode") from error
    if atoms not in SIZES or method not in METHODS or mode not in MODES:
        raise argparse.ArgumentTypeError("case is outside the measured protocol")
    return atoms, method, mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--case", action="append", type=parse_case)
    args = parser.parse_args()
    cases = args.case or CASES
    if len(set(cases)) != len(cases):
        parser.error("profile cases must be unique")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir()
    args.nsys = args.nsys.resolve()
    if digest(args.nsys) != CAPTURE_SHA256:
        raise ValueError("unexpected Nsight Systems binary")
    config = json.loads((root / "paper-config.json").read_text())
    env = environment(config, root, "standard")
    env["GENERATIVEQC_STATIONARY_CACHE"] = str(output / "runtime-cache")
    env["GENERATIVEQC_JIT_SCCACHE_ROOT"] = str(output / "jit-owners")
    _, hardware = verify(root, config, {"standard": env})
    version = subprocess.check_output([args.nsys, "--version"], text=True, timeout=30)
    report = {
        "diagnostic_only": True,
        "nsys": version.strip(),
        "nsys_sha256": CAPTURE_SHA256,
        "hardware": hardware,
        "cases": [],
    }
    destination = output / "nsys.json"
    save(destination, report)
    for atoms, method, mode in cases:
        folder = output / f"{atoms}-{method}-{mode}"
        folder.mkdir()
        prefix = folder / "capture"
        profile = [
            args.nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-event-trace=false",
            "--cuda-graph-trace=node",
            "-o",
            prefix,
            config["interpreters"]["standard"],
            "-u",
            root / "endpoint.py",
            "standard",
            method,
            mode,
            str(atoms),
            folder / "endpoint.json",
            "--root",
            root,
        ]
        entry = {"case": folder.name, "profile": {"status": "starting"}}
        report["cases"].append(entry)
        save(destination, report)
        try:
            run(
                profile,
                folder / "profile.txt",
                root / "source",
                env,
                720,
                entry["profile"],
            )
        finally:
            save(destination, report)
        capture = prefix.with_suffix(".nsys-rep")
        if capture.is_file():
            entry["capture"] = {
                "bytes": capture.stat().st_size,
                "sha256": digest(capture),
            }
            sqlite = prefix.with_suffix(".sqlite")
            export = [
                args.nsys,
                "export",
                "--type",
                "sqlite",
                "--output",
                sqlite,
                capture,
            ]
            entry["export"] = {"status": "starting"}
            save(destination, report)
            try:
                run(
                    export,
                    folder / "export.txt",
                    root / "source",
                    env,
                    240,
                    entry["export"],
                )
            finally:
                save(destination, report)
            if entry["export"]["status"] != "complete":
                continue
            if not sqlite.is_file():
                entry["census_error"] = "missing SQLite output"
                save(destination, report)
                continue
            entry["sqlite"] = {"bytes": sqlite.stat().st_size, "sha256": digest(sqlite)}
            save(destination, report)
            try:
                save(folder / "kernels.json", kernel_census(sqlite))
            except sqlite3.Error as error:
                entry["census_error"] = f"{type(error).__name__}: {error}"
                save(destination, report)


if __name__ == "__main__":
    main()
