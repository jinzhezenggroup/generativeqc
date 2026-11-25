"""Collect native component data in a separate diagnostic GPU Job."""

import argparse
import json
import signal
from pathlib import Path

from acquire import TIMEOUT, environment, interrupt, save, verify
from nsys_probe import parse_case, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "cases", nargs="+", type=parse_case, help="case as atoms:method:mode"
    )
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("diagnostic cases must be unique")
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
    destination = output / "diagnostic.json"
    for atoms, method, mode in args.cases:
        label = f"{atoms}-{method}-{mode}"
        trace = output / f"{label}-trace.jsonl"
        argv = [
            config["interpreters"]["standard"],
            "-u",
            str(root / "endpoint.py"),
            "standard",
            method,
            mode,
            str(atoms),
            str(output / f"{label}.json"),
            "--root",
            str(root),
        ]
        entry = {"case": f"{atoms}:{method}:{mode}", "status": "starting"}
        receipt["rows"].append(entry)
        save(destination, receipt)
        try:
            run(
                argv,
                output / f"{label}.txt",
                root / "source",
                {**env, "GENERATIVEQC_DF_TRACE": str(trace)},
                TIMEOUT,
                entry,
            )
        finally:
            save(destination, receipt)


if __name__ == "__main__":
    main()
