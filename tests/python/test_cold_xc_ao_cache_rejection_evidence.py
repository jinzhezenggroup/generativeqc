"""A failed GPU endpoint must not become a qualified performance observation."""

from __future__ import annotations

import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/cold-xc-ao-cache-20261107"
)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "valid",
        "promotion",
        "missing_device",
        "failed_endpoint",
        "missing_oom",
        "handoff_jobs",
        "handoff_image",
        "overlay",
        "patch",
        "reference",
        "identity",
        "settings",
        "force",
        "incomplete_forces",
        "work",
        "source_work",
        "vv10",
        "time",
        "selector",
        "warm_protocol",
        "baseline_time",
        "failed_source",
        "diagnostic",
    ],
)
def test_rejected_cache_evidence_cannot_be_promoted(
    tmp_path: Path, corruption: str, optimized: bool
) -> None:
    """Corruptions must fail even when Python assertions are disabled."""
    evidence = json.loads(
        lzma.decompress((DIRECTORY / "evidence.json.xz").read_bytes())
    )
    candidate = evidence["devices"]["A100"]["records"]["default"]
    sample = candidate["samples"][0]
    failed = evidence["devices"]["RTX5090"]
    if corruption == "promotion":
        evidence["decision"] = "promoted"
    elif corruption == "missing_device":
        del evidence["devices"]["RTX5090"]
    elif corruption == "failed_endpoint":
        failed["records"]["default"]["status"] = "complete"
    elif corruption == "missing_oom":
        failed["logs"]["default"] = "success"
    elif corruption == "handoff_jobs":
        evidence["handoff_gpu_jobs"] = [16644]
    elif corruption == "handoff_image":
        evidence["handoff_compile_only"]["image"] = evidence["experiment"]["image"]
    elif corruption == "overlay":
        name = next(iter(evidence["experiment"]["overlays"]))
        evidence["experiment"]["overlays"][name] += "corrupted"
    elif corruption == "patch":
        evidence["experiment"]["patch"] += "corrupted"
    elif corruption == "reference":
        evidence["reference"]["sha256"] = "corrupted"
    elif corruption == "identity":
        candidate["native_build"]["library_sha256"] = "corrupted"
    elif corruption == "settings":
        for record in evidence["devices"]["A100"]["records"].values():
            record["samples"][0]["settings"]["screening_tolerance"] = 1e-6
    elif corruption == "force":
        sample["forces"][0][0] += 1e-3
    elif corruption == "incomplete_forces":
        sample["forces"].pop()
    elif corruption == "work":
        sample["ks_diagnostic"]["coulomb_exchange_work"]["exchange_applications"] *= 2
    elif corruption == "source_work":
        for operator in sample["precision"]["operators"]:
            if operator["name"] == "coulomb_recurrence":
                operator["count"] *= 2
    elif corruption == "vv10":
        sample["ks_diagnostic"]["nonlocal_work"]["pair_sweeps"] -= 1
    elif corruption == "time":
        sample["seconds"] *= 2
    elif corruption == "selector":
        candidate["geometry_reuse_experiment"]["combined_range_build"] = "0"
    elif corruption == "warm_protocol":
        candidate["protocol"]["warm"] = "pooled cross-owner density"
    elif corruption == "baseline_time":
        failed["records"]["baseline"]["samples"][0]["seconds"] = float("inf")
    elif corruption == "failed_source":
        failed["records"]["default"]["source_revision"] = "corrupted"
    elif corruption == "diagnostic":
        evidence["rtx5090_diagnostic"]["components"] = []
    (tmp_path / "evidence.json.xz").write_bytes(
        lzma.compress(json.dumps(evidence).encode(), preset=0)
    )
    command = (
        "import runpy; from pathlib import Path; "
        f"reader=runpy.run_path({str(DIRECTORY / 'replay.py')!r}); "
        f"reader['replay'](Path({str(tmp_path)!r}))"
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if corruption == "valid":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr
