"""Observable endpoint/reduction contracts for the default-grid campaign."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np
import pytest

from benchmarks import readme_defaults as runner
from benchmarks.readme_defaults_report import reduce_record, render


def receipt(arm: str = "standard") -> dict:
    sample = {
        "phase": "cold",
        "repeat": 0,
        "seconds": 4.0,
        "energy": -75.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "converged": True,
        "iterations": 8,
        "precision_validation": None,
    }
    if arm == "fast":
        sample["precision_validation"] = {
            "profile": "fast",
            "status": "strict_fp64",
            "solve_count": 2,
            "solves": [{"role": "candidate"}, {"role": "fp64_fallback"}],
        }
    samples = [sample]
    for index, seconds in enumerate((3.0, 1.0, 2.0)):
        samples.append(
            dict(deepcopy(sample), phase="warm", repeat=index, seconds=seconds)
        )
    return {
        "schema": runner.SCHEMA,
        "arm": arm,
        "method": "pbe",
        "mode": "direct",
        "atoms": 3,
        "warm_repeats": 3,
        "samples": samples,
        "status": "complete",
        "source_revision": "a" * 40,
    }


def test_median_retains_all_samples_and_actual_fast_fallback() -> None:
    row = reduce_record(receipt("fast"))
    assert row["cold"]["median"] == 4
    assert row["warm"]["median"] == 2
    assert row["warm"]["seconds"] == [3, 1, 2]
    assert row["warm"]["fast_status"] == ["strict_fp64"] * 3
    assert row["warm"]["fast_solve_count"] == [2] * 3


def test_partial_warm_never_becomes_a_complete_median() -> None:
    record = receipt()
    record["samples"].pop()
    with pytest.raises(ValueError, match="incomplete population"):
        reduce_record(record)
    record.update(status="failed", error="external timeout")
    reduced = reduce_record(record)
    assert reduced["cold"]["median"] == 4
    assert reduced["warm"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["samples"][0].update(energy=float("nan")),
        lambda r: r["samples"][0].update(seconds=-1),
        lambda r: r["samples"][0].update(forces=[[0, 0, 0]]),
        lambda r: r["samples"][0].update(converged=False),
        lambda r: r["samples"][0].update(iterations=None),
        lambda r: r["samples"][2].update(repeat=0),
        lambda r: r["samples"][0].update(precision_validation=None),
    ],
)
def test_incomplete_or_invalid_endpoint_cannot_be_plotted(mutation: Callable) -> None:
    record = receipt("fast")
    mutation(record)
    with pytest.raises(ValueError):
        reduce_record(record)


def test_renderer_rejects_mixed_revisions_and_duplicate_populations(
    tmp_path: Path,
) -> None:
    one, two = receipt(), receipt("fast")
    two["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="revisions"):
        render([one, two], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        render([one, one], tmp_path)
    two["source_revision"] = one["source_revision"]
    two["warm_repeats"] = 2
    two["samples"].pop()
    with pytest.raises(ValueError, match="warm repeat"):
        render([one, two], tmp_path)


def test_native_public_mode_and_default_grid_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import generativeqc

    observations = []

    def constructor(**kwargs: Any) -> dict:
        observations.append(kwargs)
        return kwargs

    monkeypatch.setattr(generativeqc, "Calculator", constructor)
    monkeypatch.setattr(generativeqc, "PrecisionCalculator", constructor)
    for mode in ("direct", "df"):
        for method in runner.METHODS:
            standard = runner.native_calculator(method, "standard", mode, tmp_path)
            fast = runner.native_calculator(method, "fast", mode, tmp_path)
            assert "ks_options" not in standard and "ks_options" not in fast
            assert "precision" not in fast and fast["policy"] == "fast"
            assert "energy_tolerance" not in fast
            assert "density_tolerance" not in fast
            if method == "r2scan-3c":
                assert "basis" not in standard  # canonical method owns def2-mTZVPP
            if mode == "df":
                assert (
                    standard["auxiliary_basis"] == tmp_path / f"{method}-auxiliary.json"
                )


@pytest.mark.parametrize("method", ("hf", "pbe"))
def test_reference_defaults_and_frozen_warm_density(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warm call gets the same cold seed even if a previous solve changes it."""
    import sys

    engine = SimpleNamespace(
        converged=True,
        cycles=8,
        conv_tol=1e-9,
        conv_tol_grad=None,
        max_cycle=50,
        diis_space=8,
        init_guess="minao",
        direct_scf=True,
        direct_scf_tol=1e-13,
        small_rho_cutoff=0,
        get_veff=lambda: None,
    )
    seeds = []

    def kernel(dm0: Any = None) -> float:
        seeds.append(None if dm0 is None else dm0.copy())
        if dm0 is not None:
            dm0[:] = 99
        engine.get_veff()
        return -75.0

    engine.kernel = kernel
    engine.nuc_grad_method = lambda: SimpleNamespace(kernel=lambda: np.zeros((3, 3)))
    engine.make_rdm1 = lambda: np.eye(2)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(deviceSynchronize=lambda: None)
            ),
            asnumpy=np.asarray,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.dft",
        SimpleNamespace(rks=SimpleNamespace(RKS=lambda mol: engine)),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.scf",
        SimpleNamespace(hf=SimpleNamespace(RHF=lambda mol: engine)),
    )
    monkeypatch.setitem(
        sys.modules, "pyscf", SimpleNamespace(gto=SimpleNamespace(M=lambda **kw: kw))
    )
    args = SimpleNamespace(
        method=method, mode="direct", arm="gpu4pyscf", inputs=tmp_path, warm_repeats=3
    )
    rows = []
    runner.run_reference(args, [("H", (0, 0, 0))] * 3, {}, rows.append)
    assert seeds[0] is None
    for seed in seeds[1:]:
        np.testing.assert_array_equal(seed, np.eye(2))
    assert [row["phase"] for row in rows] == ["cold", "warm", "warm", "warm"]
    assert all(row["settings"]["energy_tolerance"] == 1e-9 for row in rows)
    assert all(row["settings"]["orbital_gradient_tolerance"] is None for row in rows)
    assert all(row["effective_potential_builds"] == 1 for row in rows)
    assert all(
        row["settings"]["force_grid_response"] == (None if method == "hf" else True)
        for row in rows
    )


def test_reference_variant_is_rejected_before_execution(tmp_path: Path) -> None:
    args = SimpleNamespace(arm="gpu4pyscf-fast")
    with pytest.raises(ValueError, match="default GPU4PySCF"):
        runner.run_reference(args, [], {}, lambda row: None)


def test_readme_excludes_archived_reference_variant(tmp_path: Path) -> None:
    render([receipt(arm) for arm in runner.ARMS], tmp_path)
    svg = (tmp_path / "direct.svg").read_text()
    assert "GPU4PySCF Fast" not in svg
    assert "GenerativeQC Fast" in svg
    assert "GPU4PySCF Default" in svg


def test_shared_observable_gates_keep_force_maximum_and_rms_separate() -> None:
    """One large force error cannot hide in a small RMS or a larger molecule."""
    from benchmarks.readme_defaults_accuracy import errors

    reference = {"energy": -75.0, "forces": np.zeros((96, 3))}
    observed = np.zeros((96, 3))
    assert errors(-75 + 5e-4, observed, reference, 96)["passed"]
    observed[0, 0] = 2e-4
    checked = errors(-75, observed, reference, 96)
    assert not checked["passed"]
    assert (
        checked["force_rms_hartree_per_bohr"]
        < runner.COMMON_GATES["force_rms_hartree_per_bohr"]
    )
    observed[0, 0] = np.nan
    assert not errors(-75, observed, reference, 96)["passed"]


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_offline_report_rejects_retained_output_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    """Replay must not overwrite accepted evidence, even through path aliases."""
    import sys

    from benchmarks import _retention, readme_defaults_report

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "summary.json"
    existing.write_text("original evidence")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid output reached receipt reading or rendering")

    monkeypatch.setattr(readme_defaults_report, "load_records", forbidden)
    monkeypatch.setattr(readme_defaults_report, "render", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "readme_defaults_report",
            "--receipts",
            "missing.json",
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as error:
        readme_defaults_report.main()
    assert error.value.code == 2
    assert existing.read_text() == "original evidence"
    assert not (tmp_path / ".artifacts").exists()
