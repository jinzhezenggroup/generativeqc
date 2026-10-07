"""Host-only checks of the single-variable complete-endpoint experiment."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.performance import assess_comparison

from benchmarks import pbe0_xc_tile_pairs as benchmark
from tools.generativeqc_validation.pbe0_xc_tiles import verify_pairs
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

BUNDLE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/pbe0-xc-tiles-20261006"
)


def run_fake_campaign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str | None = None
) -> tuple[dict[str, Any], list[Any]]:
    """Exercise orchestration without a CUDA library, device or reference solve."""
    scientific = {
        "atoms": 48,
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
            forces, _ = self._public_dft_cuda_force()
            candidate = self.tile == 512
            replay = not self._warm_updates
            tile = 256 if candidate and mutation == "tile" else self.tile
            iterations = 2 if candidate and replay and mutation == "iterations" else 1
            fallback = candidate and replay and mutation == "fallback"
            if candidate and mutation == "forces":
                forces[0, 0] = 1e-4
            item = SimpleNamespace(
                energy=-1.0,
                forces=forces,
                status=0,
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
    monkeypatch.setattr(benchmark, "scaling_cases", lambda: {"water-48": {}})
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
    monkeypatch.setattr(benchmark, "require_tuned_native_build", lambda *args: None)
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
            "48",
            "--basis-file",
            str(tmp_path / "basis.json"),
            "--reference",
            str(reference_path),
            "--output",
            str(output),
        ],
    )
    try:
        benchmark.main()
    finally:
        assert all(owner.closed for owner in owners)
    return json.loads(output.read_text()), owners


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


@pytest.mark.parametrize("mutation", ["tile", "iterations", "fallback", "forces"])
def test_invalid_controls_work_and_vectors_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    """A faster-looking invalid replay cannot become accepted timing evidence."""
    with pytest.raises(AssertionError):
        run_fake_campaign(monkeypatch, tmp_path, mutation)


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
