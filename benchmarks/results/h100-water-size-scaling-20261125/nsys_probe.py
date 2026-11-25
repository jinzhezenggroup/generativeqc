"""Run source-bound Nsight CUDA diagnostics for two observed Direct losses."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from acquire import digest, environment, save, stop_process, verify

CAPTURE_SHA256 = "67620bee0161e0296db2807ae8b5895fe433feb4dddc17543e712c744e2d4075"
CASES = ((240, "hf"), (96, "wb97m-v"))


def run(argv, logfile, cwd, env, timeout):
    timed_out = False
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
            timed_out = True
        finally:
            if child.poll() is None:
                stop_process(child)
    return {
        "argv": list(map(str, argv)),
        "returncode": child.returncode,
        "timeout": timed_out,
    }


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir()
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
    for atoms, method in CASES:
        folder = output / f"{atoms}-{method}-direct"
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
            "direct",
            str(atoms),
            folder / "endpoint.json",
            "--root",
            root,
        ]
        entry = {"case": folder.name}
        report["cases"].append(entry)
        entry["profile"] = run(
            profile, folder / "profile.txt", root / "source", env, 720
        )
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
            entry["export"] = run(
                export, folder / "export.txt", root / "source", env, 240
            )
            if entry["export"]["returncode"] == 0:
                entry["sqlite"] = {
                    "bytes": sqlite.stat().st_size,
                    "sha256": digest(sqlite),
                }
                save(folder / "kernels.json", kernel_census(sqlite))
            save(destination, report)


if __name__ == "__main__":
    main()
