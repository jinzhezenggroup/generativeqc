"""Grid sweeps must separate timing populations and quadrature accuracy gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import GridSpec

from benchmarks._gpu4pyscf_grid import reference_grid_order
from benchmarks._readme_grid import grid_directory, grid_work, parse_grid
from benchmarks.readme_dft import SCHEMA
from benchmarks.readme_omol25 import check_record
from tools import render_readme_grids
from tools.compare_readme_grids import compare
from tools.render_readme_dft import quadrature_label

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_grid_order_scope_covers_new_response_integrators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Integrator:
        def block_loop(
            self, *, deriv: int = 0, strict_grid_order: bool = False
        ) -> Iterator[tuple]:
            calls.append(strict_grid_order)
            yield (
                np.empty((4, 0, 2)),
                np.empty(0, dtype=np.int32),
                np.ones(2),
                np.zeros((2, 3)),
            )

    original = Integrator.block_loop
    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.dft",
        SimpleNamespace(numint=SimpleNamespace(NumInt=Integrator)),
    )
    with pytest.raises(RuntimeError, match="force failed"), reference_grid_order():
        ao, indices, weights, _ = next(Integrator().block_loop(deriv=1))
        assert calls and all(calls)
        assert ao.shape == (4, 1, 2) and not np.any(ao)
        assert indices.tolist() == [0]
        assert len(weights) == 2
        raise RuntimeError("force failed")
    assert Integrator.block_loop is original


@pytest.mark.parametrize(
    "value", ["", "1,2", "0,4,8", "513,4,8", "8,257,8", "8,4,2", "8,4,1025"]
)
def test_grid_size_validation(value: str) -> None:
    with pytest.raises(ValueError, match="invalid grid"):
        parse_grid(value)


def test_grid_work_is_explicitly_a_pre_screening_bound() -> None:
    work = grid_work(96, GridSpec(), has_vv10=True)
    assert work["points_per_atom"] == 24576
    assert work["semilocal_points"] == work["vv10_points"] == 2359296
    assert work["vv10_unscreened_ordered_pair_upper_bound"] == 2359296**2
    assert work["vv10_grid_policy"] == "same_as_semilocal"
    smaller = grid_work(96, GridSpec(24, 8, 16), has_vv10=True)
    assert (
        smaller["vv10_unscreened_ordered_pair_upper_bound"] * 64
        == work["vv10_unscreened_ordered_pair_upper_bound"]
    )
    assert grid_work(3, GridSpec(), has_vv10=False)["vv10_points"] == 0


def test_renderer_labels_grid_and_rejects_mixed_series() -> None:
    points = [{"protocol": {"grid": asdict(GridSpec())}}]
    assert quadrature_label(points) == "grid 48×16×32"
    points.append({"protocol": {"grid": asdict(GridSpec(24, 8, 16))}})
    with pytest.raises(ValueError, match="different grids"):
        quadrature_label(points)
    assert quadrature_label([{"protocol": {"grid": None}}]) is None


def test_grid_renderer_rejects_misfiled_populations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        render_readme_grids,
        "collect_point",
        lambda root, atoms: {
            "atoms": atoms,
            "protocol": {
                "atoms": atoms,
                "method": "WB97M-V/RKS",
                "grid": asdict(GridSpec()),
            },
        },
    )
    with pytest.raises(ValueError, match="wrong grid"):
        render_readme_grids.collect_grid(tmp_path, (48, 12, 24), [3])


def test_grid_summary_preserves_endpoint_samples_and_semantic_work() -> None:
    record = {
        "phase": "cold",
        "complete_seconds": 2.0,
        "iterations": 3,
        "fock_builds": 4,
        "gate": True,
        "native_scf_ao_work": {"point_ao_visits": 123},
        "native_ks_diagnostic": {
            "history": [{"energy_change": 0.01}],
            "components": {"xc": -1.0},
            "coulomb_exchange_work": {"quartets": 17},
            "nonlocal_work": {"pairs": 23},
            "nonlocal_history_scopes": ["full-target"],
        },
        "native_force_components": {
            "wall_seconds": {"xc": 0.2},
            "work_counts": {"pairs": 37},
            "becke_owners": {"semilocal": {"work_counters": {"pairs": 41}}},
            "traffic": {"bytes": 128},
        },
    }
    original = {
        "modes": {
            "none": [{"points": [{"engines": {"native": {"records": [record]}}}]}]
        }
    }
    compact = render_readme_grids.compact_summary(original)
    retained = compact["modes"]["none"][0]["points"][0]["engines"]["native"]["records"][
        0
    ]
    assert "history" in record["native_ks_diagnostic"]
    assert "wall_seconds" in record["native_force_components"]
    assert "history" not in retained["native_ks_diagnostic"]
    assert "components" not in retained["native_ks_diagnostic"]
    assert "wall_seconds" not in retained["native_force_components"]
    for field in (
        "complete_seconds",
        "iterations",
        "fock_builds",
        "gate",
        "native_scf_ao_work",
    ):
        assert retained[field] == record[field]
    for field in ("coulomb_exchange_work", "nonlocal_work", "nonlocal_history_scopes"):
        assert (
            retained["native_ks_diagnostic"][field]
            == record["native_ks_diagnostic"][field]
        )
    for field in ("work_counts", "becke_owners", "traffic"):
        assert (
            retained["native_force_components"][field]
            == record["native_force_components"][field]
        )
    assert render_readme_grids.compact_summary(compact) == compact


def test_grid_renderer_retains_separate_series_and_work(tmp_path: Path) -> None:
    entry = {
        "status": "measured",
        "records": [
            {
                "phase": phase,
                "complete_seconds": seconds,
                "iterations": 1,
                "fock_builds": 1,
            }
            for phase, seconds in (("cold", 2.0), ("warm", 0.5))
        ],
    }
    point = {"atoms": 3, "engines": {"native": entry, "reference": entry}}
    summary = {
        "sizes": [3],
        "modes": {
            "cuda": [{"dimensions": [48, 12, 24], "points": [point]}],
            "none": [
                {"dimensions": [48, 12, 24], "points": [point]},
                {"dimensions": [48, 16, 32], "points": [point]},
            ],
        },
    }
    render_readme_grids.render(summary, tmp_path)
    svg = (tmp_path / "wb97mv-grids.svg").read_text()
    assert "48×12×24" in svg and "48×16×32" in svg
    table = (tmp_path / "grid-timings.md").read_text()
    assert "2.000 | 0.500 | 1/1 | 1/1" in table
    assert table.index("| none |") < table.index("| cuda |")


@pytest.mark.parametrize("grids", ["24,8,16 48,16,32", "48,16,32", "24,8,16\n48,16,32"])
def test_runner_forwards_grid_and_isolates_references(
    tmp_path: Path, grids: str
) -> None:
    """Use real shell control flow without any CUDA calls or native library."""
    helper = tmp_path / "python-probe"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json,pathlib,sys\n"
        "if sys.argv[1] == '-':\n"
        "    pathlib.Path(sys.argv[2]).write_text('{}'); sys.exit(0)\n"
        "start=sys.argv.index('--grid')+1\n"
        "output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        "record={'grid':sys.argv[start:start+3]}\n"
        "if '--reference' in sys.argv:\n"
        "    record['reference']=sys.argv[sys.argv.index('--reference')+1]\n"
        "output.write_text(json.dumps(record))\n"
    )
    helper.chmod(0o700)
    result = subprocess.run(
        ["bash", "benchmarks/run_readme_dft_benchmarks.sh", "both"],
        env={
            **os.environ,
            "SLURM_JOB_ID": "host-only-fixture",
            "GENERATIVEQC_LIBRARY": "not-loaded",
            "README_BENCHMARK_PYTHON": str(helper),
            "README_BENCHMARK_OUTPUT": str(tmp_path / "records"),
            "README_BENCHMARK_METHODS": "wb97m-v",
            "README_BENCHMARK_SIZES": "3",
            "README_BENCHMARK_GRIDS": grids,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for value in grids.split():
        dimensions = parse_grid(value)
        point = tmp_path / "records" / grid_directory(dimensions) / "wb97m-v/3"
        native = json.loads((point / "native.json").read_text())
        reference = json.loads((point / "reference.json").read_text())
        assert native["grid"] == reference["grid"] == value.split(",")
        assert native["reference"] == str(point / "reference.json")


@pytest.mark.parametrize(
    "grids", ["", "0,4,8", "24,8,2", "24,8,16 24,8,16", "24,8,16 invalid"]
)
def test_invalid_sweep_fails_before_starting_endpoints(
    tmp_path: Path, grids: str
) -> None:
    result = subprocess.run(
        ["bash", "benchmarks/run_readme_dft_benchmarks.sh", "reference"],
        env={
            **os.environ,
            "SLURM_JOB_ID": "host-only-fixture",
            "README_BENCHMARK_GRIDS": grids,
            "README_BENCHMARK_OUTPUT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert not (tmp_path / "inputs").exists()


def retain_reference(
    root: Path, grid: tuple[int, int, int], *, energy: float = -76.0, force: float = 0.0
) -> Path:
    """A complete geometry oracle may survive a later warm replay timeout."""
    path = root / grid_directory(grid) / "wb97m-v/3/reference.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "engine": "reference",
                "status": "running",
                "protocol": {
                    "atoms": 3,
                    "method": "WB97M-V/RKS",
                    "grid": asdict(GridSpec(*grid)),
                    "basis_identity": "same-basis",
                },
                "records": [
                    {
                        "phase": phase,
                        "geometry": geometry,
                        "energy": energy,
                        "forces": [[force, 0, 0]] * 3,
                        "converged": True,
                        "status": 0,
                        "complete_seconds": 1.0,
                        "gate": True,
                    }
                    for phase, geometry in (("cold", 0), ("moved", 1))
                ],
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "energy,force,expected",
    [(-76.0, 0.0, True), (-75.99, 0.0, False), (-76.0, 1e-3, False)],
)
def test_grid_convergence_is_independent_of_same_grid_parity(
    tmp_path: Path, energy: float, force: float, expected: bool
) -> None:
    retain_reference(tmp_path, (48, 16, 32))
    retain_reference(tmp_path, (24, 8, 16), energy=energy, force=force)
    result = compare(
        tmp_path,
        (24, 8, 16),
        (48, 16, 32),
        "wb97m-v",
        3,
        energy_tolerance=1e-6,
        force_tolerance=1e-5,
    )
    assert result["convergence_gate"] is expected
    assert all(row["gate"] is expected for row in result["geometries"])
    assert result["reference"]["sha256"]


def test_missing_geometry_is_unknown_not_a_pass(tmp_path: Path) -> None:
    retain_reference(tmp_path, (48, 16, 32))
    path = retain_reference(tmp_path, (24, 8, 16))
    raw = json.loads(path.read_text())
    raw["records"].pop()
    path.write_text(json.dumps(raw))
    result = compare(
        tmp_path,
        (24, 8, 16),
        (48, 16, 32),
        "wb97m-v",
        3,
        energy_tolerance=1e-6,
        force_tolerance=1e-5,
    )
    assert result["convergence_gate"] is None
    assert result["geometries"][0]["gate"] is True
    assert result["geometries"][1]["gate"] is None


def test_cross_grid_comparison_rejects_other_model_changes(tmp_path: Path) -> None:
    retain_reference(tmp_path, (48, 16, 32))
    path = retain_reference(tmp_path, (24, 8, 16))
    raw = json.loads(path.read_text())
    raw["protocol"]["basis_identity"] = "different-basis"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="beyond quadrature"):
        compare(
            tmp_path,
            (24, 8, 16),
            (48, 16, 32),
            "wb97m-v",
            3,
            energy_tolerance=1e-6,
            force_tolerance=1e-5,
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_grid_convergence_rejects_invalid_tolerances(
    tmp_path: Path, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        compare(
            tmp_path,
            (48, 12, 24),
            (64, 16, 32),
            "wb97m-v",
            3,
            energy_tolerance=tolerance,
            force_tolerance=1e-4,
        )


def test_baseline_self_comparison_never_certifies_convergence(tmp_path: Path) -> None:
    retain_reference(tmp_path, (48, 16, 32))
    result = compare(
        tmp_path,
        (48, 16, 32),
        (48, 16, 32),
        "wb97m-v",
        3,
        energy_tolerance=1e-6,
        force_tolerance=1e-5,
    )
    assert result["is_baseline"] and result["convergence_gate"] is None


def test_retained_publication_replays_independent_energy_and_force_gates() -> None:
    """Protect measured oracle arrays, not production source spelling or timings."""
    publication = Path("benchmarks/results/readme-vv10-grids-20261018")
    manifest = json.loads((publication / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        assert (
            hashlib.sha256((publication / filename).read_bytes()).hexdigest() == digest
        )
    asset = manifest["homepage_asset"]
    assert (
        hashlib.sha256(Path(asset["file"]).read_bytes()).hexdigest() == asset["sha256"]
    )
    study = json.loads(
        lzma.decompress((publication / "grid-timings.json.xz").read_bytes())
    )
    assert study["raw_evidence"]["files"]
    native_calls = 0
    reference_calls = 0
    for populations in study["modes"].values():
        for population in populations:
            for point in population["points"]:
                reference = point["engines"]["reference"]["records"]
                reference_calls += len(reference)
                baselines = {
                    row["geometry"]: row
                    for row in reference
                    if row["phase"] in ("cold", "moved")
                }
                assert set(baselines) == {0, 1}
                assert (
                    point["native_source_identity"]
                    == manifest["native_source_identity"]
                )
                assert (
                    point["native_library_sha256"] == manifest["native_library_sha256"]
                )
                for row in point["engines"]["native"]["records"]:
                    checked = check_record(row, baselines[row["geometry"]])
                    assert checked["gate"] is True
                    assert checked["energy_error"] == pytest.approx(
                        row["energy_error"], abs=1e-14
                    )
                    assert checked["force_error"] == pytest.approx(
                        row["force_error"], abs=1e-14
                    )
                    native_calls += 1
    assert native_calls == manifest["native_endpoint_calls"]
    assert reference_calls == manifest["reference_endpoint_calls"]
