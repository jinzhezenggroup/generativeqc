"""Run independent PySCF oracles for retained #2054 native attempts.

Each reference is a fresh finite process. Unsupported/failed native attempts
remain explicit NOT_RUN dispositions; a timeout or crash gets its own receipt.
The summaries never infer a missing oracle or zero work counter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "tools/benchmark_hybrid_provider_crossover.py").is_file():
    raise RuntimeError("run #2054 oracle qualification from the frozen repository root")
sys.path.insert(0, str(ROOT))

from tools.benchmark_hybrid_provider_crossover import (
    SCHEMA,
    sha256,
    summarize,
    write,
)

CASES = ("water-48", "formaldehyde")
SUPPORTED = ("direct", "df-jk-occupied")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def qualify(args: argparse.Namespace) -> dict:
    root = args.run_directory.resolve()
    if not (root / "native/campaign.json").is_file():
        raise FileNotFoundError("native ABBA campaign is absent")
    campaign = read(root / "native/campaign.json")
    if campaign.get("schema") != SCHEMA or campaign.get("status") != "RECORDED":
        raise ValueError("native ABBA campaign is incomplete")
    if campaign.get("source", {}).get("revision") != args.expected_source:
        raise ValueError("native campaign source differs from the frozen build")
    if len(campaign.get("order", [])) != 10:
        raise ValueError("48-atom and holdout ABBA process inventory is incomplete")

    oracle_dir = root / "oracles"
    summary_dir = root / "summaries"
    oracle_dir.mkdir(exist_ok=True)
    summary_dir.mkdir(exist_ok=True)
    index_path = root / "oracle-index.json"
    index = {
        "schema": SCHEMA,
        "kind": "independent-oracle-index",
        "native_source": args.expected_source,
        "installed_library_sha256": args.expected_library_sha256,
        "oracle_runner_sha256": sha256(Path(__file__)),
        "python": sys.executable,
        "threads": args.threads,
        "per_reference_timeout_seconds": args.timeout,
        "attempts": [],
        "summaries": [],
        "status": "RUNNING",
    }
    write(index_path, index)

    for case in CASES:
        for arm in SUPPORTED:
            for repeat in (0, 1):
                native_path = root / "native" / f"{case}-{arm}-{repeat}.json"
                disposition = {
                    "case": case,
                    "arm": arm,
                    "repeat": repeat,
                    "native_path": str(native_path),
                    "oracle_status": "NOT_RUN",
                }
                index["attempts"].append(disposition)
                if not native_path.is_file():
                    disposition["reason"] = "native record absent"
                    write(index_path, index)
                    continue
                native = read(native_path)
                if native.get("status") != "MEASURED":
                    disposition["reason"] = f"native status {native.get('status')}"
                    write(index_path, index)
                    continue
                if (
                    native.get("source", {}).get("revision") != args.expected_source
                    or native.get("source", {}).get("dirty")
                    or native.get("library", {}).get("sha256")
                    != args.expected_library_sha256
                ):
                    disposition["reason"] = "native source/library identity mismatch"
                    write(index_path, index)
                    continue

                oracle_path = oracle_dir / f"{case}-{arm}-{repeat}.json"
                disposition["oracle_path"] = str(oracle_path)
                if oracle_path.exists():
                    previous = read(oracle_path)
                    if previous.get("status") == "MEASURED" and previous.get(
                        "native_record_sha256"
                    ) == sha256(native_path):
                        disposition["oracle_status"] = "MEASURED_REUSED"
                    else:
                        disposition["reason"] = (
                            "existing oracle is incomplete or mismatched"
                        )
                    write(index_path, index)
                    continue

                command = [
                    sys.executable,
                    str(ROOT / "tools/benchmark_hybrid_provider_crossover.py"),
                    "run-reference",
                    "--native",
                    str(native_path),
                    "--threads",
                    str(args.threads),
                    "--output",
                    str(oracle_path),
                ]
                started = time.monotonic()
                try:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                        check=False,
                    )
                    disposition["returncode"] = completed.returncode
                    (oracle_dir / f"{case}-{arm}-{repeat}.log").write_text(
                        completed.stdout + completed.stderr
                    )
                    if oracle_path.is_file():
                        disposition["oracle_status"] = read(oracle_path).get("status")
                    else:
                        disposition["reason"] = (
                            "oracle process returned without a record"
                        )
                except subprocess.TimeoutExpired as error:
                    disposition["oracle_status"] = "TIMEOUT"
                    disposition["reason"] = str(error)
                disposition["elapsed_seconds"] = time.monotonic() - started
                write(index_path, index)

        for repeat in (0, 1):
            native_records = [
                read(path)
                for path in (
                    root / "native" / f"{case}-{arm}-{repeat}.json" for arm in SUPPORTED
                )
                if path.is_file()
            ]
            unsupported = root / "native" / f"{case}-df-j-exact-k.json"
            if unsupported.is_file():
                native_records.append(read(unsupported))
            oracle_records = []
            for arm in SUPPORTED:
                path = oracle_dir / f"{case}-{arm}-{repeat}.json"
                if path.is_file():
                    value = read(path)
                    if value.get("status") == "MEASURED":
                        oracle_records.append(value)
            profile_records = [
                read(root / f"{case}-{arm}-profile.json")
                for arm in SUPPORTED
                if (root / f"{case}-{arm}-profile.json").is_file()
            ]
            result = (
                summarize(native_records, oracle_records, profile_records)
                if native_records
                else {
                    "schema": SCHEMA,
                    "case": case,
                    "status": "INCOMPLETE",
                    "failures": ["all native records absent"],
                    "crossover_claim_eligible": False,
                }
            )
            summary_path = summary_dir / f"{case}-pair{repeat}.json"
            if summary_path.exists():
                raise FileExistsError(summary_path)
            write(summary_path, result)
            index["summaries"].append(
                {"case": case, "repeat": repeat, "status": result["status"]}
            )
            write(index_path, index)

    index["status"] = "COMPLETE_WITH_LIMITATIONS"
    write(index_path, index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-library-sha256", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout < 1:
        parser.error("positive threads and timeout required")
    result = qualify(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "attempts": [
                    (a["case"], a["arm"], a["repeat"], a["oracle_status"])
                    for a in result["attempts"]
                ],
                "summaries": result["summaries"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
