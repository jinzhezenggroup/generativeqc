"""Host-only checks of the single-variable complete-endpoint experiment."""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.performance import assess_comparison

from benchmarks import pbe0_xc_tile_pairs as benchmark
from tools.generativeqc_validation.pbe0_xc_tiles import (
    verify_pairs,
    verify_point_family_pairs,
)
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

BUNDLE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/pbe0-xc-tiles-20261006"
)


def run_fake_campaign(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str | None = None,
    *,
    point_batch_tiles: int | None = None,
    point_specialization: bool = False,
    compact: bool = False,
    atoms: int = 48,
) -> tuple[dict[str, Any], list[Any]]:
    """Exercise orchestration without a CUDA library, device or reference solve."""
    scientific = {
        "atoms": atoms,
        "geometries_bohr": [[["H", [0.0, 0.0, 0.0]]]] * 2,
    }
    reference = {
        "protocol": scientific,
        "stage": "complete",
        "records": [
            {
                "geometry": geometry,
                "phase": phase,
                "energy": -1.0,
                "forces": [[0.0] * 3],
            }
            for geometry, phase in ((0, "cold"), (1, "moved"))
        ],
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    output = tmp_path / ".artifacts/pairs.json"
    owners = []

    class Owner:
        def __init__(self, tile: int) -> None:
            self.tile = tile
            self.batch_tiles = os.environ["GENERATIVEQC_CUDA_XC_BATCH_TILES"]
            self.point_specialization = os.environ.get(
                "GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION"
            )
            self.batch_policies = []
            self.point_policies = []
            self.compact_policy = os.environ["GENERATIVEQC_CUDA_XC_COMPACT_BATCH"]
            self.compact_policies = []
            self._warm_updates = True
            self.closed = False
            self.calls = []

        def set_warm_start_updates(self, enabled: bool) -> None:
            self._warm_updates = enabled

        def _public_dft_cuda_force(self) -> tuple[np.ndarray, dict[str, int]]:
            return np.zeros((1, 3)), {"tile_points": 256}

        def execute(self, coords: Any, *, strict: bool, properties: Any) -> Any:
            assert strict is False and properties == ("energy", "forces")
            self.calls.append((coords, self._warm_updates))
            self.batch_policies.append(os.environ["GENERATIVEQC_CUDA_XC_BATCH_TILES"])
            self.point_policies.append(
                os.environ.get("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION")
            )
            self.compact_policies.append(
                os.environ["GENERATIVEQC_CUDA_XC_COMPACT_BATCH"]
            )
            failed_force = (
                mutation == "force-status" and self.tile == 256 and len(self.calls) == 1
            )
            forces, _ = (None, None) if failed_force else self._public_dft_cuda_force()
            candidate = (
                self.tile == 512
                or self.point_specialization == "1"
                or self.compact_policy == "1"
            )
            replay = not self._warm_updates
            tile = 256 if candidate and mutation == "tile" else self.tile
            iterations = 2 if candidate and replay and mutation == "iterations" else 1
            fallback = candidate and replay and mutation == "fallback"
            if candidate and mutation == "forces":
                forces[0, 0] = 1e-4
            item = SimpleNamespace(
                energy=-1.0,
                forces=forces,
                status=3 if failed_force else 0,
                status_message="prepared derivative source unavailable"
                if failed_force
                else "",
                converged=True,
                iterations=iterations,
                fock_builds=iterations,
                warm_start_used=True,
                warm_start_fallback=fallback,
                ks_diagnostic=SimpleNamespace(
                    tile_points=tile,
                    to_payload=lambda: {
                        "tile_points": tile,
                        "history": [{}] * iterations,
                    },
                ),
            )
            return SimpleNamespace(items=[item])

        def close(self) -> None:
            self.closed = True

    class Calculator:
        def __init__(self, *, ks_options: Any, **options: Any) -> None:
            assert options["device"] == "cuda"
            self.tile = ks_options.tile_points

        def prepare_batch(self, systems: Any, *, warm_start: bool) -> Owner:
            assert warm_start
            owner = Owner(self.tile)
            owners.append(owner)
            return owner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "host-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-token")
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            )
        ),
    )
    monkeypatch.setattr(benchmark, "Calculator", Calculator)
    monkeypatch.setattr(benchmark, "scaling_cases", lambda: {f"water-{atoms}": {}})
    monkeypatch.setattr(
        benchmark, "load_comparison_basis", lambda *args, **kwargs: ({}, None)
    )
    monkeypatch.setattr(benchmark, "protocol", lambda *args, **kwargs: scientific)
    monkeypatch.setattr(benchmark, "source_hashes", dict)
    monkeypatch.setattr(benchmark, "cuda_accelerator_metadata", lambda *args: {})
    monkeypatch.setattr(benchmark, "environment_metadata", lambda **kwargs: {})
    monkeypatch.setattr(
        benchmark, "native_build_metadata", lambda *args: {"same_binary": True}
    )
    monkeypatch.setattr(
        benchmark, "require_tuned_native_build", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(benchmark, "normalize_force_work", lambda work: work)
    monkeypatch.setattr(
        benchmark, "read_ao_work", lambda owner: {"tiles": 256 // (owner.tile // 256)}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pbe0_xc_tile_pairs",
            "--atoms",
            str(atoms),
            "--basis-file",
            str(tmp_path / "basis.json"),
            "--reference",
            str(reference_path),
            "--output",
            str(output),
        ],
    )
    if point_batch_tiles is not None:
        sys.argv.extend(["--point-batch-tiles", str(point_batch_tiles)])
    if point_specialization:
        sys.argv.append("--point-specialization")
    if compact:
        sys.argv.append("--compact-xc-batches")
    try:
        benchmark.main()
    finally:
        assert all(owner.closed for owner in owners)
    return json.loads(output.read_text()), owners


@pytest.mark.parametrize("point_batch_tiles", [None, 32])
@pytest.mark.parametrize("compact", [False, True])
def test_batch_policy_overrides_default_during_owner_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    point_batch_tiles: int | None,
    compact: bool,
) -> None:
    """Promotion cannot contaminate baseline, moved owners or the caller's policy."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_BATCH_TILES", "99")
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_COMPACT_BATCH", "99")
    if compact and point_batch_tiles is None:
        with pytest.raises(SystemExit):
            run_fake_campaign(monkeypatch, tmp_path, compact=True)
        return
    record, owners = run_fake_campaign(
        monkeypatch, tmp_path, point_batch_tiles=point_batch_tiles, compact=compact
    )
    assert record["stage"] == "complete"
    assert os.environ["GENERATIVEQC_CUDA_XC_BATCH_TILES"] == "99"
    assert os.environ["GENERATIVEQC_CUDA_XC_COMPACT_BATCH"] == "99"
    assert record["compact_xc_batch_request"] is compact
    assert {owner.batch_tiles for owner in owners} == (
        {"32"} if compact else ({"1", "32"} if point_batch_tiles else {"1"})
    )
    for index, owner in enumerate(owners):
        assert len(owner.batch_policies) == 14
        assert set(owner.batch_policies) == {owner.batch_tiles}
        expected = "1" if compact and index % 2 == 1 else "0"
        assert owner.compact_policy == expected
        assert set(owner.compact_policies) == {expected}
    assert (
        "python/generativeqc_compiler/dft/xc_tile_batch_cuda.py"
        in record["source_file_sha256"]
    )
    assert "src/dft/cuda_xc_kernels.cuh" in record["source_file_sha256"]


@pytest.mark.parametrize("point_batch_tiles", [None, 32])
def test_point_specialization_preserves_work_and_restores_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, point_batch_tiles: int | None
) -> None:
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_BATCH_TILES", "99")
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "caller")
    record, owners = run_fake_campaign(
        monkeypatch,
        tmp_path,
        point_batch_tiles=point_batch_tiles,
        point_specialization=True,
    )
    assert record["schema"] == "generativeqc.pbe0-xc-point-family-pairs.v1"
    assert record["scf_tiles"] == {"baseline": 256, "candidate": 256}
    assert record["point_consumer"] == {
        "baseline": "generic",
        "candidate": "pbe-specialized",
    }
    assert record["point_batch_request"] == (
        {"tiles": point_batch_tiles, "device_bytes": 32 * 1024 * 1024}
        if point_batch_tiles
        else None
    )
    assert os.environ["GENERATIVEQC_CUDA_XC_BATCH_TILES"] == "99"
    assert os.environ["GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION"] == "caller"
    assert [owner.point_specialization for owner in owners] == ["0", "1"]
    assert {owner.tile for owner in owners} == {256}
    for owner in owners:
        assert owner.batch_tiles == str(point_batch_tiles or 1)
        assert set(owner.batch_policies) == {owner.batch_tiles}
        assert set(owner.point_policies) == {owner.point_specialization}
        assert len(owner.point_policies) == 14


def test_tiles_force_policy_and_frozen_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cold/prime work stays outside timing; both replay arms retain full SCF."""
    record, owners = run_fake_campaign(monkeypatch, tmp_path)
    assert record["stage"] == "complete"
    assert record["scf_tiles"] == {"baseline": 256, "candidate": 512}
    assert record["force_tile_points"] == 256
    assert len(record["setup"]) == len(record["priming"]) == 4
    assert len(record["samples"]) == 20
    assert {owner.tile for owner in owners} == {256, 512}
    for owner in owners:
        assert len(owner.calls) == 14
        assert sum(updates for _, updates in owner.calls) == 2
    for phase in ("warm", "moved-warm"):
        samples = [row for row in record["samples"] if row["phase"] == phase]
        assert record["assessments"][phase] == assess_comparison(samples)
        for row in samples:
            diagnostic = row["diagnostics"]
            tile = record["scf_tiles"][row["selection"]]
            assert diagnostic["native_ks_diagnostic"]["tile_points"] == tile
            assert diagnostic["native_force_work_raw"]["tile_points"] == 256
            assert diagnostic["native_scf_ao_work"]["tiles"] == 256 // (tile // 256)
            assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1


def test_compact_campaign_covers_small_ao_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Small mapped domains use complete E+F, not an isolated XC timing gate."""
    record, owners = run_fake_campaign(
        monkeypatch, tmp_path, atoms=12, point_batch_tiles=32, compact=True
    )
    assert record["protocol"]["atoms"] == 12
    assert record["scf_tiles"] == {"baseline": 256, "candidate": 256}
    assert len(record["samples"]) == 20
    assert {owner.batch_tiles for owner in owners} == {"32"}
    assert {owner.compact_policy for owner in owners} == {"0", "1"}


@pytest.mark.parametrize("mutation", ["tile", "iterations", "fallback", "forces"])
def test_invalid_controls_work_and_vectors_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    """A faster-looking invalid replay cannot become accepted timing evidence."""
    with pytest.raises(AssertionError):
        run_fake_campaign(monkeypatch, tmp_path, mutation)


def test_failed_force_reports_public_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(
        RuntimeError,
        match="baseline E\\+F failed: status=3, message=prepared derivative source unavailable",
    ):
        run_fake_campaign(monkeypatch, tmp_path, "force-status")


def test_retained_publication_and_complete_pairs() -> None:
    """Authenticate selected bytes and independently audit all four comparisons."""
    manifest = load_json(BUNDLE / "publication.json")
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    for atoms in (48, 96):
        record = load_publication_record(
            BUNDLE, role="samples", name=f"pairs-{atoms}.json"
        )
        reference = load_publication_record(
            BUNDLE, role="input", name=f"reference-{atoms}.json"
        )
        verify_pairs(record, reference)


@pytest.mark.parametrize(
    "mutation", ["forces", "iterations", "force-tile", "force-work", "scf-work"]
)
def test_edited_measured_vectors_and_work_are_rejected(mutation: str) -> None:
    """Stored success flags cannot hide altered physical vectors or semantic work."""
    record = deepcopy(
        load_publication_record(BUNDLE, role="samples", name="pairs-96.json")
    )
    reference = load_publication_record(BUNDLE, role="input", name="reference-96.json")
    diagnostic = record["samples"][0]["diagnostics"]
    if mutation == "forces":
        diagnostic["forces"][0][0] += 1e-3
    elif mutation == "iterations":
        diagnostic["iterations"] = 2
    elif mutation == "force-tile":
        diagnostic["native_force_work_raw"]["grid_tile_points_requested"] = 512
    elif mutation == "scf-work":
        diagnostic["native_scf_ao_work"]["xc_evaluations"] = 0
    else:
        diagnostic["native_force_components"]["becke_owners"]["stationary"][
            "work_counters"
        ]["becke_reverse_pair_visits"] += 1
    with pytest.raises(AssertionError):
        verify_pairs(record, reference)


def synthetic_point_family_evidence(candidate_seconds: float) -> tuple[dict, dict]:
    """Use retained baseline vectors as fixtures, not measured specialization data."""
    record = deepcopy(
        load_publication_record(BUNDLE, role="samples", name="pairs-96.json")
    )
    reference = load_publication_record(BUNDLE, role="input", name="reference-96.json")
    record.update(
        schema="generativeqc.pbe0-xc-point-family-pairs.v1",
        point_specialization=True,
        point_consumer={"baseline": "generic", "candidate": "pbe-specialized"},
        scf_tiles={"baseline": 256, "candidate": 256},
        point_batch_request=None,
    )
    for group in ("setup", "priming", "samples"):
        for row in record[group]:
            geometry = row.get("geometry", int(row.get("phase") == "moved-warm"))
            baseline = next(
                item
                for item in record[group]
                if item["selection"] == "baseline"
                and item.get("geometry", int(item.get("phase") == "moved-warm"))
                == geometry
            )
            row["diagnostics"] = deepcopy(baseline["diagnostics"])
            row["seconds"] = (
                10.0 if row["selection"] == "baseline" else candidate_seconds
            )
    for phase in ("warm", "moved-warm"):
        record["assessments"][phase] = assess_comparison(
            [row for row in record["samples"] if row["phase"] == phase]
        )
    return record, reference


@pytest.mark.parametrize("candidate_seconds", [9.9, 10.0, 11.0])
def test_point_family_negative_results_are_valid_but_cannot_promote(
    candidate_seconds: float,
) -> None:
    """Noisy, unchanged and slower endpoints remain auditable negative evidence."""
    record, reference = synthetic_point_family_evidence(candidate_seconds)
    verify_point_family_pairs(record, reference)
    assert all(row["status"] == "not-run" for row in record["assessments"].values())
    with pytest.raises(AssertionError):
        verify_point_family_pairs(record, reference, require_improvement=True)


@pytest.mark.parametrize("mutation", ["forces", "scf-work", "timing"])
def test_negative_point_family_results_still_reject_invalid_evidence(
    mutation: str,
) -> None:
    record, reference = synthetic_point_family_evidence(11.0)
    sample = record["samples"][0]
    if mutation == "forces":
        sample["diagnostics"]["forces"][0][0] += 1e-3
    elif mutation == "scf-work":
        sample["diagnostics"]["native_scf_ao_work"]["xc_evaluations"] = 0
    else:
        sample["seconds"] = -1.0
        record["assessments"]["warm"] = assess_comparison(
            [row for row in record["samples"] if row["phase"] == "warm"]
        )
    with pytest.raises(AssertionError):
        verify_point_family_pairs(record, reference)


def test_point_family_improvement_requires_valid_complete_evidence() -> None:
    record, reference = synthetic_point_family_evidence(9.0)
    verify_point_family_pairs(record, reference, require_improvement=True)
    record["samples"][0]["diagnostics"]["iterations"] = 2
    with pytest.raises(AssertionError):
        verify_point_family_pairs(record, reference, require_improvement=True)


@pytest.mark.parametrize("candidate_seconds", [9.0, 11.0])
def test_point_family_work_ignores_independent_discovery_timing(
    candidate_seconds: float,
) -> None:
    record, reference = synthetic_point_family_evidence(candidate_seconds)
    for row in record["samples"]:
        if row["selection"] == "candidate":
            row["diagnostics"]["native_scf_ao_work"]["discovery_seconds"] += 0.1
    verify_point_family_pairs(
        record, reference, require_improvement=candidate_seconds < 10.0
    )


@pytest.mark.parametrize(
    "counter", ["discovery_ao_jet_values", "point_ao_visits", "future_work_counter"]
)
def test_point_family_work_still_compares_every_nontiming_field(counter: str) -> None:
    record, reference = synthetic_point_family_evidence(9.0)
    sample = next(row for row in record["samples"] if row["selection"] == "candidate")
    work = sample["diagnostics"]["native_scf_ao_work"]
    work[counter] = work.get(counter, 0) + 1
    with pytest.raises(AssertionError):
        verify_point_family_pairs(record, reference, require_improvement=True)
