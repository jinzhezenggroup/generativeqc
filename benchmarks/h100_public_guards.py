"""Replay frozen H100 public API guards, explicit fallbacks and retained failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks import h100_production_cohort as support
from tools.restore_retained_evidence import read_evidence_bytes

DIRECTORY = support.ROOT / "benchmarks/results/h100-sm90-cd2b-public-guards-20261103"
READERS = {
    "replay-public-budget-v5.py",
    "replay-public-budget-v7.py",
    "replay-public-budget-v9.py",
    "replay-facade-guards-v5.py",
    "replay-facade-correction-v2.py",
    "facade-correction-reference-replay-v1.py",
    "cohort-replay-atomic-v1.py",
    "unpack-notebook-guards-v1.py",
}
COHORTS = [
    "public-budget-v5",
    "public-budget-v7",
    "public-budget-v9",
    "facade-guards-v1",
    "notebook-public-guards-v1",
]
EXTRAS = {
    "facade-correction-contract-v1.json",
    "original-final-protocol-v1.json",
    "facade-scope-audit-sol-v1/audit-report.json",
    "facade-correction-result-audit-sol-v1/audit-report.json",
}


def check(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def artifact_bytes(name: str) -> bytes:
    """Read an original packet, falling back only to declared local Git history."""
    relative = Path(name)
    path = DIRECTORY / relative
    check(
        not relative.is_absolute()
        and ".." not in relative.parts
        and not path.is_symlink(),
        "Unsafe public guard evidence path",
    )
    if path.is_file():
        return path.read_bytes()
    check(
        not path.exists()
        and name in {cohort + "-completed.tar.xz" for cohort in COHORTS},
        "Missing original public guard evidence",
    )
    try:
        return read_evidence_bytes(
            support.ROOT
            / "benchmarks/results/h100-sm90-cd2b-public-guards-20261103"
            / relative,
            manifest=DIRECTORY / "historical-snapshot.json",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError("Missing historical public guard evidence: " + name) from error


def checked_publication() -> dict:
    publication = json.loads((DIRECTORY / "cohort.json").read_text())
    expected = (
        EXTRAS
        | {"readers/" + n for n in READERS}
        | {
            cohort + suffix
            for cohort in COHORTS
            for suffix in ["-completed.tar.xz", "-download-manifest.json"]
        }
    )
    entries = publication["files"]
    check(
        publication["schema"] == "generativeqc.h100-public-guards.v1"
        and publication["source_revision"] == support.SOURCE
        and publication["source_tree"] == "c76f5bd3cfa62416255c5d6914cf701d34e10a8d"
        and publication["clean_timing_votes"] == 0
        and publication["final_three_pair_acceptance"] is False,
        "Public guard scope differs",
    )
    check(
        len(entries) == len(expected) and {b["path"] for b in entries} == expected,
        "Mandatory public guard artifact inventory differs",
    )
    for binding in entries:
        relative = Path(binding["path"])
        data = artifact_bytes(binding["path"])
        check(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(data) == binding["bytes"]
            and hashlib.sha256(data).hexdigest() == binding["sha256"],
            "Public guard artifact differs: " + str(relative),
        )
    return publication


def retained_budget_failure(task: Path, cohort: str) -> dict:
    """Report authenticated failed runs without promoting them to qualification."""
    manifest = json.loads((task / (cohort + "-download-manifest.json")).read_text())
    campaign = json.loads((task / cohort / "campaign.json").read_text())
    check(
        manifest["platform_status"] == "SUCCEEDED"
        and manifest["recorded_provenance_errors"] == []
        and campaign["source_revision"] == support.SOURCE
        and len(campaign["runs"]) == 8,
        "Failed budget campaign identity differs",
    )
    returned = 0
    failures = []
    for row in campaign["runs"]:
        raw_path = task / cohort / (row["case"] + ".json")
        raw = json.loads(raw_path.read_text())
        check(
            support.sha(raw_path) == row["receipt_sha256"], "Failed raw output differs"
        )
        returned += len(raw["samples"])
        if row["returncode"]:
            check(
                raw["status"] == "FAIL_OR_INCOMPLETE" and raw.get("error"),
                "Failed attempt was relabelled",
            )
            failures.append({"case": row["case"], "error": raw["error"]})
    check(
        (len(failures), returned) == ((4, 0) if cohort.endswith("v5") else (3, 3)),
        "Original failed population or returned E/F count differs",
    )
    return {
        "cohort": cohort,
        "status": "FAILED_ORIGINAL_ATTEMPT_RETAINED",
        "returned_EF": returned,
        "failures": failures,
        "scientific_acceptance": False,
    }


def extend(task: Path, native: Path, reference: Path) -> list[dict]:
    publication = checked_publication()
    # Recompute proofs consumed by the later jobs rather than trusting their copies.
    support.PROOFS.update(
        {"public-budget-v9-local-replay.json", "facade-guards-v1-local-replay.json"}
    )
    for name in READERS:
        support.copy_exact(DIRECTORY / "readers" / name, task / name)
    for name in EXTRAS:
        support.copy_exact(DIRECTORY / name, task / name)
    outputs = []
    for cohort in COHORTS:
        for suffix in ["-completed.tar.xz", "-download-manifest.json"]:
            with (task / (cohort + suffix)).open("xb") as stream:
                stream.write(artifact_bytes(cohort + suffix))
        notebook = cohort.startswith("notebook-")
        unpacker = (
            task / "unpack-notebook-guards-v1.py"
            if notebook
            else support.artifact("unpack-cohort-v2.py")
        )
        support.reader(unpacker, task, native, reference, "--cohort", cohort)
        support.install(task, cohort)
        if notebook:
            for source in sorted((task / (cohort + "-available-producers")).iterdir()):
                support.copy_exact(source, task / source.name)
            continue
        if cohort in ["public-budget-v5", "public-budget-v7"]:
            outputs.append(retained_budget_failure(task, cohort))
            continue
        script = (
            "replay-public-budget-v9.py"
            if cohort.startswith("public-budget")
            else "replay-facade-guards-v5.py"
        )
        support.reader(task / script, task, native, reference)
        proof = task / (cohort + "-local-replay.json")
        result = json.loads(proof.read_text())
        check(
            result["status"] == publication["expected_statuses"][cohort]
            and support.sha(proof) == publication["expected_proof_sha256"][cohort],
            "Actual public guard replay changed",
        )
        support.verify_consumed_proofs(task)
        outputs.append(
            {
                "cohort": cohort,
                "status": result["status"],
                "proof_sha256": support.sha(proof),
            }
        )
    script = task / "replay-facade-correction-v2.py"
    support.reader(script, task, native, reference)
    proof = task / "facade-correction-v2-local-replay.json"
    result = json.loads(proof.read_text())
    check(
        result["status"] == publication["expected_statuses"]["facade-correction-v2"]
        and support.sha(proof)
        == publication["expected_proof_sha256"]["facade-correction-v2"],
        "Corrected public API/fallback replay changed",
    )
    outputs.append(
        {
            "cohort": "facade-correction-v2",
            "status": result["status"],
            "proof_sha256": support.sha(proof),
            "public_EF": result["public_returned_EF"],
            "closed_controls": result["closed_owner_controls"],
            "native_heterogeneous_df_batch_qualified": False,
            "all_fast_trials_native_force_qualified": False,
        }
    )
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    check(
        sys.version_info >= (3, 12) and not sys.flags.optimize,
        "Use unoptimized Python 3.12 or later",
    )
    print(
        json.dumps(
            {
                "source_revision": support.SOURCE,
                "phases": support.replay(extension=extend),
                "clean_timing_votes": 0,
                "final_three_pair_acceptance": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
