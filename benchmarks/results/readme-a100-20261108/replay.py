"""Replay every non-CBS README A100 population without executing GPU/CPU SCF."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks._exact_fp64_evidence import load


def require(condition: bool, detail: str) -> None:
    """Keep provenance checks active under optimized Python as well."""
    if not condition:
        raise ValueError(detail)


def load_evidence(directory: Path = DIRECTORY) -> dict:
    """Authenticate exact original text and every FP64 word before auditing."""
    bundle, files = load(directory)
    return {**bundle["experiment"], "files": files}


def replay(directory: Path = DIRECTORY, *, evidence: dict | None = None) -> dict:
    """Recompute timing, all paired E/F checks and independent qualifications.

    Failed scientific populations are observations, not an integrity failure.
    They must remain identical to the disclosed report and cannot become
    qualified wins merely because their timings beat the other engine.
    """
    evidence = load_evidence(directory) if evidence is None else evidence
    require(
        evidence["schema"] == "generativeqc.readme-a100-all-noncbs.v1", "wrong campaign"
    )
    files = evidence["files"]
    require(set(files) == set(evidence["sha256"]), "missing original receipts")
    for name, text in files.items():
        require(
            hashlib.sha256(text.encode()).hexdigest() == evidence["sha256"][name],
            "original bytes differ: " + name,
        )
    for name, digest in evidence["legacy_sha256"].items():
        require(
            hashlib.sha256(
                (ROOT / evidence["legacy_root"] / name).read_bytes()
            ).hexdigest()
            == digest,
            "independent reference changed: " + name,
        )
    image = json.loads(files["image-identity.json"])
    require(image == evidence["image"], "image mismatch")
    source = json.loads(files["source-receipt.json"])
    require(
        source["commit"] == evidence["source_revision"]
        and source["source_identity"] == image["source_identity"],
        "source mismatch",
    )
    require(files["build/exit-code.txt"].strip() == "0", "failed build")
    require(json.loads(files["build/identity.json"]) == image, "wrong compiled image")
    require(
        "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a"
        in files["build/restored-runtime.sha256"],
        "borrowed image not restored",
    )
    manifest = json.loads(files["campaign.json"])
    require(
        len(manifest["cases"]) == 64
        and len(
            {
                (case["method"], case["mode"], case["atoms"])
                for case in manifest["cases"]
            }
        )
        == 64,
        "incomplete matrix",
    )
    require(
        sum(case["order"][0] == "standard" for case in manifest["cases"]) == 32,
        "unbalanced first-arm order",
    )
    work = []
    for worker, job in enumerate(evidence["gpu_jobs"]):
        prefix = "raw/results-" + str(job) + "/"
        identity = json.loads(files[prefix + "identity.json"])
        require(
            all(identity.get(name) == value for name, value in image.items()),
            "wrong worker image",
        )
        require(
            str(identity["scheduler"]["LSB_JOBID"]) == str(job)
            and bool(identity["scheduler"]["CUDA_VISIBLE_DEVICES"]),
            "unallocated worker",
        )
        executions = json.loads(files[prefix + "executions.json"])
        expected = [
            (case["index"], arm)
            for case in manifest["cases"]
            if case["worker"] == worker
            for arm in case["order"]
        ]
        require(
            [(row["case"]["index"], row["arm"]) for row in executions] == expected,
            "lost or reordered engine population",
        )
        for row in executions:
            stem = (
                "{mode}-{method}-{atoms}-".format(**row["case"]) + row["arm"] + ".json"
            )
            if row["record_present"]:
                require(
                    hashlib.sha256(files[prefix + stem].encode()).hexdigest()
                    == row["record_sha256"],
                    "execution receipt differs",
                )
        work.extend(executions)
    require(len(work) == 128, "missing execution receipts")
    with tempfile.TemporaryDirectory(prefix="readme-a100-replay-") as temporary:
        trial = Path(temporary)
        for name, text in files.items():
            destination = trial / name
            require(destination.resolve().is_relative_to(trial), "unsafe retained path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(text.encode())
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + str(ROOT)
        result = subprocess.run(
            [
                sys.executable,
                str(trial / "reproduction/reduce.py"),
                "--repository-root",
                str(ROOT),
                "--campaign-root",
                str(trial),
                "--results",
                str(trial / "raw"),
                "--output",
                str(trial / "recomputed"),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        require(
            result.returncode == 0,
            "offline scientific audit failed: " + result.stdout + result.stderr,
        )
        for name in ("summary.json", "qualification.json", "reference-reuse.json"):
            require(
                json.loads((trial / "recomputed" / name).read_text())
                == json.loads(files["expected/" + name]),
                "recomputed report differs: " + name,
            )
    summary = json.loads(files["expected/summary.json"])
    summary["execution_counts"] = {
        "populations": len(work),
        "failed": sum(row["returncode"] != 0 for row in work),
        "complete_endpoints": sum(row.get("sample_count", 0) for row in work),
    }
    return summary


if __name__ == "__main__":
    summary = replay()
    print(
        json.dumps(
            {
                name: summary[name]
                for name in (
                    "source_revision",
                    "totals",
                    "qualification_counts",
                    "execution_counts",
                )
            },
            indent=2,
        )
    )
