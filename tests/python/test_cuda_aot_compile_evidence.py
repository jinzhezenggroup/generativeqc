"""Synthetic acquisition integrity tests, not CUDA numerical qualification."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
import shutil
from typing import TYPE_CHECKING

import pytest

from tools.generativeqc_validation.cuda_aot_evidence import apply_compile_evidence
from tools.generativeqc_validation.cuda_aot_policy import MANIFEST, ROOT

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = "a" * 64


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def acquisition(tmp_path: Path) -> tuple[dict, dict, Path]:
    """Use small raw inputs with no generated-source spelling expectations."""
    shutil.copyfile(ROOT / MANIFEST, tmp_path / "manifest.json")
    source = "synthetic acquisition input, not a production CUDA program"
    resource = (
        "arch = sm_80\n Function fixture_kernel:\n  REG:32 STACK:0 SHARED:64 LOCAL:0\n"
    )
    row = {
        "architecture": "sm_80",
        "shell_class": "fpps",
        "path": "input.cu",
        "source_sha256": digest(source),
        "object_sha256": "b" * 64,
        "consumers": ["force"],
        "mixed_fock": False,
        "status": "passed",
        "compile": {"returncode": 0, "timed_out": False, "duration_seconds": 1.0},
        "compile_log_sha256": digest("compile log"),
        "resource_file": "resources.txt",
        "resource_sha256": digest(resource),
    }
    bundle = {
        "schema": "generativeqc.cuda-aot-retained-compile/v1",
        "index": {
            "schema": "generativeqc.cuda-aot-audit-compile/v1",
            "source_commit": "1" * 40,
            "native_source_identity": SOURCE,
            "toolchain": dict.fromkeys(
                ("nvcc", "ptxas", "cuobjdump", "host_compiler"), "fixture"
            ),
            "cache_launcher": "fixture-only",
            "scope": "synthetic-primary-only",
            "rows": [row],
        },
        "files": {
            "scheduler.json": json.dumps(
                {"SLURM_JOB_ID": "test-only", "CUDA_VISIBLE_DEVICES": "0"}
            ),
            "sm_80-fpps.log": "compile log",
            "resources.txt": resource,
        },
    }
    report = {
        "rows": [
            {
                "source": "manifest.json",
                "feature": feature,
                "consumer": "force",
                "targets": {
                    architecture: {
                        "nvcc_compile_and_resources": "not-run-for-this-audit",
                        "independent_device_numerics": "not-run-for-this-audit",
                        "full_endpoint": {"cold": "not-run-for-this-audit"},
                    }
                    for architecture in ("sm_80", "sm_90")
                },
            }
            for feature in ("extra-force:fpps", "exact-derivative:fpps")
        ]
    }
    return bundle, report, tmp_path


def retain(bundle: dict, root: Path) -> dict:
    """Recompute only the container digest so raw integrity failures are tested."""
    path = root / "compile.json.xz"
    path.write_bytes(lzma.compress(json.dumps(bundle).encode()))
    return {
        "schema_version": 1,
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_compile_observation_does_not_qualify_science_or_exact_packages(
    acquisition: tuple[dict, dict, Path],
) -> None:
    bundle, report, root = acquisition
    before = copy.deepcopy(report)
    apply_compile_evidence(report, root, retain(bundle, root), SOURCE)
    target = report["rows"][0]["targets"]["sm_80"]
    assert target["nvcc_compile_and_resources"]["resource_maxima"]["REG"] == 32
    assert target["independent_device_numerics"] == "not-run-for-this-audit"
    assert target["full_endpoint"]["cold"] == "not-run-for-this-audit"
    assert report["rows"][1] == before["rows"][1]
    assert (
        report["rows"][0]["targets"]["sm_90"] == before["rows"][0]["targets"]["sm_90"]
    )


def test_old_source_remains_historical_instead_of_blocking_refactors(
    acquisition: tuple[dict, dict, Path],
) -> None:
    bundle, report, root = acquisition
    before = copy.deepcopy(report["rows"])
    apply_compile_evidence(report, root, retain(bundle, root), "c" * 64)
    assert report["compile_evidence"]["status"] == "historical-source-only"
    assert report["rows"] == before


@pytest.mark.parametrize(
    "field",
    (
        "source",
        "log",
        "resources",
        "architecture",
        "duplicate",
        "timeout",
        "exit",
        "nonfinite",
        "consumer",
        "scheduler",
        "missing-kernel",
    ),
)
def test_invalid_acquisition_is_rejected(
    acquisition: tuple[dict, dict, Path], field: str
) -> None:
    bundle, report, root = acquisition
    row = bundle["index"]["rows"][0]
    if field == "source":
        row["source_sha256"] = "invalid"
    elif field == "log":
        bundle["files"]["sm_80-fpps.log"] += "altered"
    elif field == "resources":
        bundle["files"]["resources.txt"] += "altered"
    elif field in ("architecture", "missing-kernel"):
        raw = (
            "arch = sm_90\n Function fixture_kernel:\n  REG:32 STACK:0 SHARED:64 LOCAL:0\n"
            if field == "architecture"
            else "arch = sm_80\n"
        )
        bundle["files"]["resources.txt"] = raw
        row["resource_sha256"] = digest(raw)
    elif field == "duplicate":
        bundle["index"]["rows"].append(copy.deepcopy(row))
    elif field == "timeout":
        row["compile"]["timed_out"] = True
    elif field == "exit":
        row["compile"]["returncode"] = 1
    elif field == "nonfinite":
        row["compile"]["duration_seconds"] = float("nan")
    elif field == "consumer":
        row["consumers"] = ["fock"]
    else:
        bundle["files"]["scheduler.json"] = "{}"
    with pytest.raises(ValueError):
        apply_compile_evidence(report, root, retain(bundle, root), SOURCE)


def test_bundle_digest_and_repository_boundary_are_enforced(
    acquisition: tuple[dict, dict, Path],
) -> None:
    bundle, report, root = acquisition
    declaration = retain(bundle, root)
    declaration["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_compile_evidence(report, root, declaration, SOURCE)
    declaration["path"] = "../outside.json.xz"
    with pytest.raises(ValueError, match="escapes repository"):
        apply_compile_evidence(report, root, declaration, SOURCE)
