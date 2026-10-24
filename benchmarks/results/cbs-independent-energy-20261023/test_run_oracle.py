"""Opt-in POSIX wrapper checks using a tiny fixture, never a scientific run.

Run on the retained CPU image with --scratch-root naming a fresh /inspire path.
No production oracle, dataset, checkpoint or measured output is executed/edited.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = """from pathlib import Path
import argparse
import sys
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path)
p.add_argument("--carbon")
a, _ = p.parse_known_args()
marker = Path(__file__).with_name("executed")
marker.write_bytes(marker.read_bytes() + b"x" if marker.exists() else b"x")
print("runner fixture only; no energies computed", flush=True)
if Path(__file__).with_name("fail").exists():
    raise SystemExit(19)
target = a.output / ("C" + a.carbon + ".json")
if target.exists():
    raise FileExistsError("Retain earlier attempts")
target.write_bytes(b'{"runner_fixture":true}\\n')
"""


def snapshot(root: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir())
        if p.is_file()
    }


def run(wrapper: Path, root: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(wrapper), str(root), "1", "1", "32", str(output)],
        capture_output=True,
        timeout=10,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    wrapper = args.wrapper.resolve(strict=True)
    scratch = args.scratch_root.resolve(strict=True)
    if not scratch.is_dir() or not str(scratch).startswith("/inspire/"):
        raise ValueError("Use a dedicated absolute /inspire scratch directory")
    report = {
        "schema": "generativeqc.oracle-runner-retry-check.v1",
        "scope": "wrapper lifecycle only; no scientific energies computed",
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.encode()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "cases": [],
    }
    cases = {
        "completed": {
            "C1.json": b'{"prior":"completed"}\n',
            "C1.stdout.log": b"original stdout\x00\n",
            "C1.outcome.json": b'{"exit_code":0}\n',
        },
        "stdout_only": {"C1.stdout.log": b"original interrupted stdout\n"},
        "outcome_only": {"C1.outcome.json": b'{"exit_code":124}\n'},
        "basis_log_only": {"C1-2.log": b"original basis log\n"},
        "checkpoint_only": {"C1-2.scf.chk": b"original checkpoint\x00\n"},
        "partial_json_write": {"C1.tmp": b'{"prior":"interrupted"'},
        "fresh_then_retry": {"C3.json": b'{"other":"molecule"}\n'},
        "failed_then_retry": {},
        "lease_contended": {"C1.stdout.log": b"original leased stdout\n"},
    }
    for name, files in cases.items():
        with tempfile.TemporaryDirectory(prefix="retry-", dir=scratch) as folder:
            root = Path(folder)
            (root / "env/bin").mkdir(parents=True)
            (root / "env/bin/python").symlink_to(Path(sys.executable).resolve())
            (root / "oracle.py").write_bytes(FIXTURE.encode())
            output = root / "attempt"
            output.mkdir()
            for relative, raw in files.items():
                (output / relative).write_bytes(raw)
            if name == "failed_then_retry":
                (root / "fail").touch()
            fresh_passed = True
            if name in {"fresh_then_retry", "failed_then_retry"}:
                first = run(wrapper, root, output)
                expected = 19 if name == "failed_then_retry" else 0
                outcome = json.loads((output / "C1.outcome.json").read_bytes())
                fresh_passed = first.returncode == expected == outcome["exit_code"]
                fresh_passed &= (root / "executed").is_file()
                fresh_passed &= (
                    b"runner fixture only" in (output / "C1.stdout.log").read_bytes()
                )
                if name == "fresh_then_retry":
                    fresh_passed &= (output / "C3.json").read_bytes() == files[
                        "C3.json"
                    ]
            before = snapshot(output)
            marker = root / "executed"
            executions = marker.read_bytes() if marker.exists() else b""
            if name == "lease_contended":
                with (root / "C1.lease").open("wb") as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    retry = run(wrapper, root, output)
            else:
                retry = run(wrapper, root, output)
            after = snapshot(output)
            untouched = before == after
            not_executed = (
                marker.read_bytes() if marker.exists() else b""
            ) == executions
            report["cases"].append(
                {
                    "name": name,
                    "retry_exit": retry.returncode,
                    "fresh_run_passed": fresh_passed,
                    "before": before,
                    "after": after,
                    "bytes_unchanged": untouched,
                    "oracle_not_executed": not_executed,
                    "passed": fresh_passed
                    and retry.returncode != 0
                    and untouched
                    and not_executed,
                }
            )
    report["passed"] = all(row["passed"] for row in report["cases"])
    args.report.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": [
                    {"name": row["name"], "passed": row["passed"]}
                    for row in report["cases"]
                ],
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
