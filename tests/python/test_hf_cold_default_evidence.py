"""Final HF defaults must not inherit qualifications from a rejected DFT image."""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/hf-cold-defaults-20261107"
)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "valid",
        "final_image",
        "missing_card",
        "decision",
        "failed_job",
        "identity",
        "scheduler",
        "visibility",
        "missing_warm",
        "time",
        "nan_time",
        "force",
        "partial_force",
        "dft_force",
        "settings",
        "work",
        "jk_work",
        "vv10_work",
        "sanitizer",
        "warm_protocol",
        "missing_history",
        "receipt",
        "oracle",
        "operator_inventory",
    ],
)
def test_cold_default_qualification_fails_closed(
    tmp_path: Path, corruption: str, optimized: bool
) -> None:
    """Validate immutable historical records, never production-source spelling."""
    for path in DIRECTORY.glob("*.xz"):
        shutil.copyfile(path, tmp_path / path.name)
    cohort = json.loads((DIRECTORY / "cohort.json").read_text())
    specification = cohort["cohorts"]["accepted"]["RTX5090"]
    path = tmp_path / specification["file"]
    payload = json.loads(lzma.decompress(path.read_bytes()))
    record = payload["records"]["hf"]["candidate"]
    sample = record["samples"][0]
    if corruption == "final_image":
        cohort["images"]["accepted"] = cohort["images"]["rejected"]
    elif corruption == "missing_card":
        del cohort["cohorts"]["accepted"]["A100"]
    elif corruption == "decision":
        cohort["rejected_dft"]["decision"] = "promoted"
    elif corruption == "failed_job":
        payload["exit_code"] = 1
    elif corruption == "identity":
        record["native_build"]["library_sha256"] = "wrong"
    elif corruption == "scheduler":
        record["scheduler"]["SLURM_JOB_ID"] = "wrong"
    elif corruption == "visibility":
        record["scheduler"]["CUDA_VISIBLE_DEVICES"] = None
    elif corruption == "missing_warm":
        record["samples"].pop()
    elif corruption == "time":
        sample["seconds"] *= 1.3
    elif corruption == "nan_time":
        record["samples"][1]["seconds"] = float("nan")
    elif corruption == "force":
        sample["forces"][0][0] += 1e-3
    elif corruption == "partial_force":
        sample["forces"].pop()
    elif corruption == "dft_force":
        payload["records"]["pbe0"]["candidate"]["samples"][0]["forces"][0][0] += 1e-3
    elif corruption == "settings":
        for arm in payload["records"]["hf"].values():
            arm["samples"][0]["settings"]["density_tolerance"] = 1e-3
    elif corruption == "work":
        sample["precision"]["strict_stage_fock_builds"] -= 1
    elif corruption == "oracle":
        payload["hf_targets"][0]["energy"] += 1e-3
    elif corruption == "operator_inventory":
        work = payload["records"]["pbe0"]["candidate"]["samples"][0]["precision"]
        work["operator_inventory_complete"] = False
    elif corruption == "jk_work":
        work = payload["records"]["pbe0"]["candidate"]["samples"][0]["ks_diagnostic"]
        work["coulomb_exchange_work"]["exchange_applications"] += 1
    elif corruption == "vv10_work":
        work = payload["records"]["wb97m-v"]["candidate"]["samples"][0]["ks_diagnostic"]
        work["nonlocal_work"]["pair_sweeps"] -= 1
    elif corruption == "sanitizer":
        payload["sanitizers"]["racecheck"] = "1 hazard"
    elif corruption == "warm_protocol":
        for arm in payload["records"]["hf"].values():
            arm["protocol"]["warm"] = "cross-owner density pool"
    elif corruption == "missing_history":
        cohort["history"] = {}
    raw = lzma.compress(json.dumps(payload).encode(), preset=0)
    path.write_bytes(raw)
    specification["sha256"] = hashlib.sha256(raw).hexdigest()
    if corruption == "receipt":
        specification["sha256"] = "wrong"
    (tmp_path / "cohort.json").write_text(json.dumps(cohort))
    command = (
        "import runpy; from pathlib import Path; "
        f"reader=runpy.run_path({str(DIRECTORY / 'replay.py')!r}); "
        f"reader['replay'](Path({str(tmp_path)!r}))"
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if corruption == "valid":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr
