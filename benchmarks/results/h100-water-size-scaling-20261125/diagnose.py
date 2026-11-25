"""Collect native component data in a separate diagnostic GPU Job."""

import argparse
import json
import os
import signal
from pathlib import Path
import subprocess

from acquire import TIMEOUT, environment, interrupt, stop_process, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("cases", nargs="+", help="case as n-atoms:method:mode")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    root = args.acquisition.resolve()
    output = args.output.resolve()
    output.mkdir()
    config = json.loads((root / "paper-config.json").read_text())
    env = environment(config, root, "standard")
    # The diagnostic is a separate job and writes only to its own cache/output.
    env["GENERATIVEQC_STATIONARY_CACHE"] = str(output / "runtime-cache")
    env["GENERATIVEQC_JIT_SCCACHE_ROOT"] = str(output / "jit-owners")
    _, hardware = verify(root, config, {"standard": env})
    receipt = {"diagnostic_only": True, "hardware": hardware, "rows": []}
    for case in args.cases:
        atoms, method, mode = case.split(":")
        label = case.replace(":", "-")
        trace = output / f"{label}-trace.jsonl"
        argv = [
            config["interpreters"]["standard"],
            "-u",
            str(root / "endpoint.py"),
            "standard",
            method,
            mode,
            atoms,
            str(output / f"{label}.json"),
            "--root",
            str(root),
        ]
        with (output / f"{label}.txt").open("xb") as log:
            child = subprocess.Popen(
                argv,
                cwd=root / "source",
                env={**env, "GENERATIVEQC_DF_TRACE": str(trace)},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timeout = False
            try:
                child.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                timeout = True
            finally:
                stop_process(child)
        receipt["rows"].append(
            {"case": case, "timeout": timeout, "returncode": child.returncode}
        )
        (output / "diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
