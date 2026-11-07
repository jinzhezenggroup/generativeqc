"""Keep cold/force/provenance acceptance active, including with Python -O."""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/r2scan-density-gemm-20261107"
)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "valid",
        "missing_device",
        "image",
        "scheduler",
        "rollback",
        "lost_cold",
        "nonfinite_time",
        "warm_regression",
        "paired_force",
        "independent_force",
        "convergence",
        "precision",
        "work",
        "incomplete_inventory",
        "empty_inventory",
        "invalid_work_counters",
        "incomplete_work",
        "zero_work",
        "unexpected_exchange",
        "failed_reference",
        "repair_job",
        "memcheck",
        "cleanup",
        "raw_checksum",
        "source_patch",
        "trace_checksum",
    ],
)
def test_r2scan_density_gemm_complete_evidence(
    tmp_path: Path, corruption: str, optimized: bool
) -> None:
    """A faster number cannot compensate for missing science or failed calls."""
    evidence = runpy.run_path(str(DIRECTORY / "replay.py"))["load_evidence"]()
    files = evidence["files"]
    name = "a100-native/forward-candidate.json"
    candidate = json.loads(files[name])
    sample = candidate["samples"][0]
    if corruption == "missing_device":
        del evidence["devices"]["RTX5090"]
    elif corruption == "image":
        evidence["image"]["library_sha256"] = "unqualified"
    elif corruption == "scheduler":
        candidate["scheduler"]["CUDA_VISIBLE_DEVICES"] = ""
    elif corruption == "rollback":
        candidate["density_gemm_experiment"]["control"] = "1"
    elif corruption == "lost_cold":
        candidate["samples"].pop(0)
    elif corruption == "nonfinite_time":
        sample["seconds"] = float("inf")
    elif corruption == "warm_regression":
        candidate["samples"][1]["seconds"] *= 2
    elif corruption == "paired_force":
        sample["forces"][0][0] += 2e-7
    elif corruption == "independent_force":
        sample["forces"][0][0] += 1e-2
    elif corruption == "convergence":
        sample["settings"]["energy_tolerance"] = 1e-6
    elif corruption == "precision":
        sample["precision"]["operators"][0]["compute"] = "fp32"
    elif corruption == "work":
        sample["ks_diagnostic"]["coulomb_exchange_work"]["coulomb_applications"] += 1
    elif corruption in {
        "incomplete_inventory",
        "empty_inventory",
        "invalid_work_counters",
        "incomplete_work",
        "zero_work",
        "unexpected_exchange",
    }:
        # Matching corrupt receipts must not turn equality into qualification.
        for arm in ("baseline", "candidate"):
            name = f"a100-native/forward-{arm}.json"
            candidate = json.loads(files[name])
            for sample in candidate["samples"]:
                precision = sample["precision"]
                work = sample["ks_diagnostic"]["coulomb_exchange_work"]
                if corruption == "incomplete_inventory":
                    precision["operator_inventory_complete"] = False
                elif corruption == "empty_inventory":
                    precision["operators"] = []
                elif corruption == "invalid_work_counters":
                    precision["operator_work_counters_valid"] = False
                elif corruption == "incomplete_work":
                    work["complete"] = False
                elif corruption == "zero_work":
                    work["coulomb_applications"] = 0
                else:
                    work["exchange_applications"] = 1
            files[name] = json.dumps(candidate, indent=2) + "\n"
            evidence["sha256"][name] = hashlib.sha256(files[name].encode()).hexdigest()
    elif corruption == "failed_reference":
        name = "a100-native/reference.json"
        candidate = json.loads(files[name])
        candidate["status"] = "complete"
    elif corruption == "repair_job":
        evidence["devices"]["A100"]["reference_job"] = "wrong"
    elif corruption == "memcheck":
        name = "rtx5090/memcheck.log"
        files[name] = files[name].replace("0 errors", "1 errors")
    elif corruption == "cleanup":
        name = "cleanup.log"
        files[name] = "transfer still running"
    elif corruption == "source_patch":
        name = "source.patch"
        files[name] += "corrupted"
    if name.endswith(".json"):
        files[name] = json.dumps(candidate, indent=2) + "\n"
    evidence["sha256"][name] = hashlib.sha256(files[name].encode()).hexdigest()
    if corruption == "raw_checksum":
        files[name] += "corrupted"
    if corruption == "trace_checksum":
        for filename in ("receipts.json.xz", "fp64.xz", "sha256.txt"):
            (tmp_path / filename).write_bytes((DIRECTORY / filename).read_bytes())
        with (tmp_path / "fp64.xz").open("ab") as stream:
            stream.write(b"corrupted")
    (tmp_path / "evidence.json.xz").write_bytes(
        lzma.compress(json.dumps(evidence).encode(), preset=0)
    )
    command = [sys.executable] + (["-O"] if optimized else [])
    inject = (
        ""
        if corruption == "trace_checksum"
        else "import json,lzma; "
        f"payload=json.loads(lzma.decompress(Path({str(tmp_path / 'evidence.json.xz')!r}).read_bytes())); "
        "module['replay'].__globals__['load_evidence']=lambda directory: payload; "
    )
    command += [
        "-c",
        "import runpy; from pathlib import Path; "
        f"module=runpy.run_path({str(DIRECTORY / 'replay.py')!r}); "
        + inject
        + f"module['replay'](Path({str(tmp_path)!r}))",
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    assert (result.returncode == 0) == (corruption == "valid"), result.stderr
