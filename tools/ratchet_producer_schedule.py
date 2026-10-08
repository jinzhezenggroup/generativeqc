"""Advisory CPU-only PR ratchet for the production DF projection source schedule.

Runs source-bound static producer receipts in *separate* interpreters for the
base and candidate, under the same fixed scientific/resource regimes. There is
no native executable or GPU timing evidence in this report. An unknown domain
is INCOMPLETE, never PASS; extra semantic work is flagged, not called a bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from tools.audit_producer_work import ReceiptError, compare
except ModuleNotFoundError:
    # Invoked as "python tools/ratchet_producer_schedule.py" without installation.
    from audit_producer_work import ReceiptError, compare

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "tools/audit_producer_work.py"
SCHEDULE = "python/generativeqc_compiler/method/df_exchange_schedule.py"
DEPENDENCIES = (
    "python/generativeqc_compiler/__init__.py",
    "python/generativeqc_compiler/method/__init__.py",
    "python/generativeqc_compiler/method/df_occupied_gram_cuda.py",
)
# Frozen inputs, not elapsed time. Ratchet expansion requires review of the
# exact source-domain and available resource budget, not a numerical claim.
CASES = (
    ("triangular-once", 12, 5, 2, 120, 3, 3, True),
    ("triangular-repeat", 12, 5, 2, 48, 3, 3, True),
    ("full-repeat", 12, 5, 2, 48, 3, 3, False),
    ("triangular-tail", 13, 7, 3, 84, 4, 4, True),
    ("triangular-large", 96, 24, 8, 4096, 8, 8, True),
)
# The existing receipt contract requires a build digest even for static-only
# schedule evidence. This sentinel is NOT a digest of any compiled binary.
STATIC_NO_BUILD = hashlib.sha256(b"producer-work:static-only:no-build:v1").hexdigest()


def _git(root: Path, *args: str) -> bytes:
    command = ["git", "-C", str(root), *args]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ReceiptError(f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _baseline_tree(root: Path, sha: str, target: Path) -> None:
    for path in (*DEPENDENCIES, SCHEDULE):
        data = _git(root, "show", f"{sha}:{path}")
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _schedule(
    source_root: Path, case: tuple[str, int, int, int, int, int, int, bool],
    *, audit_script: Path = AUDIT_SCRIPT,
) -> dict[str, Any]:
    label, n, auxiliaries, rank, capacity, dense_rows, dense_outputs, triangular = case
    command = [
        sys.executable,
        "-I",
        "-S",
        str(audit_script),
        "schedule",
        "--root", str(source_root),
        "--n", str(n),
        "--auxiliaries", str(auxiliaries),
        "--rank", str(rank),
        "--capacity", str(capacity),
        "--dense-row-blocks", str(dense_rows),
        "--dense-output-blocks", str(dense_outputs),
        "--scientific-problem", f"ci:{label}:n={n}:a={auxiliaries}:rank={rank}",
        "--dependency-identity", "static:geometry+basis+occupied-coefficients:shape-only",
        "--build-sha256", STATIC_NO_BUILD,
    ]
    if triangular:
        command.append("--triangular")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReceiptError(f"{label} schedule unavailable: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"{label} schedule did not produce JSON") from exc


def audit(root: Path, base_sha: str, *, audit_script: Path = AUDIT_SCRIPT) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_sha):
        raise ReceiptError("base SHA must be a full 40-character commit digest")
    root = root.resolve()
    report: dict[str, Any] = {
        "schema": "generativeqc.producer-work-ci.v1",
        "base_commit": base_sha.lower(),
        "source": SCHEDULE,
        "scope": "source-bound static schedule only; not runtime or native binary evidence",
        "build_sha256": STATIC_NO_BUILD,
        "status": "PASS",
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="gqc-producer-base-") as temp:
        baseline = Path(temp)
        _baseline_tree(root, base_sha, baseline)
        # A changed imported helper changes the meaning of the schedule but is
        # outside the existing single-source receipt digest: do not certify it.
        for path in DEPENDENCIES:
            if (baseline / path).read_bytes() != (root / path).read_bytes():
                report.update(status="INCOMPLETE", reason=f"import dependency changed: {path}")
                return report
        for case in CASES:
            label = case[0]
            try:
                original = _schedule(baseline, case, audit_script=audit_script)
                candidate = _schedule(root, case, audit_script=audit_script)
                result = compare(
                    original, candidate, baseline / SCHEDULE, root / SCHEDULE
                )
                row = {"case": label, **result}
                row["baseline_executed"] = original["work"]["executed_elements"]
                row["candidate_executed"] = candidate["work"]["executed_elements"]
                row["baseline_callbacks"] = original["work"]["producer_callbacks"]
                row["candidate_callbacks"] = candidate["work"]["producer_callbacks"]
            except (ReceiptError, OSError) as exc:
                row = {"case": label, "status": "INCOMPLETE", "reason": str(exc)}
            report["cases"].append(row)
        outcomes = {row["status"] for row in report["cases"]}
        report["status"] = (
            "INCOMPLETE" if "INCOMPLETE" in outcomes else "FAIL" if "FAIL" in outcomes else "PASS"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root, args.base_sha)
    except (ReceiptError, OSError) as exc:
        report = {
            "schema": "generativeqc.producer-work-ci.v1",
            "status": "INCOMPLETE",
            "reason": str(exc),
            "cases": [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        print(f"::warning::Static DF producer-work ratchet {report['status']} (see artifact); not a numerical or runtime result")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
