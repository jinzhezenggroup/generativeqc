"""CPU-only contract tests for the cross-functional DFT force matrix runner."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import GridSpec

from benchmarks.dft_force_matrix import (
    QUALIFICATION_SYSTEMS,
    _atoms,
    _changed_atoms,
    _method_configuration,
)


def test_cam_b3lyp_matrix_case_uses_method_ir_range_exchange_not_name_dispatch() -> (
    None
):
    selector, options = _method_configuration(
        "cam-b3lyp-rks",
        GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
    )

    assert selector == "pbe-rks"
    assert options.composition is not None
    assert tuple(
        primitive.operator
        for primitive in options.composition.primitives
        if hasattr(primitive, "operator")
    ) == ("short-range", "long-range")


def test_matrix_systems_include_water_scaling_and_nonwater_holdout() -> None:
    assert len(_atoms("water-3")) == 3
    assert len(_atoms("water-6")) == 6
    assert len(_atoms("water-12")) == 12
    formaldehyde = _atoms("formaldehyde")
    assert [symbol for symbol, _ in formaldehyde] == ["C", "O", "H", "H"]


def test_changed_geometry_moves_only_last_atom_by_declared_vector() -> None:
    atoms = _atoms("formaldehyde")
    changed, coordinates = _changed_atoms(atoms)

    original = np.asarray([position for _, position in atoms])
    expected = original.copy()
    expected[-1] += np.asarray((0.0010, -0.0005, 0.0003))
    np.testing.assert_allclose(coordinates, expected, atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray([position for _, position in changed]),
        expected,
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(coordinates[:-1], original[:-1])


def test_unknown_matrix_method_and_system_fail_closed() -> None:
    grid = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
    with pytest.raises(ValueError, match="unknown benchmark method"):
        _method_configuration("unknown-rks", grid)
    with pytest.raises(ValueError, match="unknown benchmark system"):
        _atoms("unknown-system")


def test_qualification_system_set_covers_scaling_and_holdout() -> None:
    assert QUALIFICATION_SYSTEMS == (
        "water-3",
        "water-6",
        "water-12",
        "formaldehyde",
    )


def test_late_changed_geometry_failure_preserves_successful_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from types import SimpleNamespace

    import benchmarks.dft_force_matrix as matrix

    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"test")
    execution_plan = SimpleNamespace(to_payload=lambda: {"version": 1})
    options = SimpleNamespace(identity="ks-id", execution_plan=execution_plan)
    calculator = SimpleNamespace(
        _library=SimpleNamespace(_name=str(library)),
        _method_name="pbe-rks",
        method_ir=SimpleNamespace(identity="method-id"),
        ks_options=options,
    )

    class Batch:
        resource_diagnostics = {"ok": True}

        def set_warm_start_updates(self, _enabled):
            pass

        def close(self):
            pass

    batch = Batch()
    calculator.prepare_batch = lambda *_args, **_kwargs: batch
    monkeypatch.setattr(matrix, "_calculator", lambda *_args, **_kwargs: calculator)
    monkeypatch.setattr(matrix, "_exchange_operators", lambda _calculator: ())
    monkeypatch.setattr(matrix, "_has_nonlocal_correlation", lambda _calculator: False)

    calls = []

    def sample(_batch, _atoms, _cupy, *, scenario, coordinates=None):
        calls.append(scenario)
        if coordinates is not None:
            raise RuntimeError("changed replay failed")
        return {
            "scenario": scenario,
            "force_status": "ok",
            "force_components": {"schema": "vibeqc.dft-force-components.v1"},
        }

    monkeypatch.setattr(matrix, "_clean_sample", sample)
    monkeypatch.setattr(
        matrix,
        "_scf_trace_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )
    result = matrix.benchmark_case(
        method="pbe-rks",
        system="water-3",
        grid=object(),
        basis="def2-svp",
        density_fitting="none",
        repeats=2,
        trace_directory=tmp_path / "trace",
        cupy_module=object(),
    )

    assert result["status"] == "measured"
    assert result["cold"]["scenario"] == "cold"
    assert len(result["warm"]) == 2
    assert result["scf_profile"]["status"] == "failed"
    assert result["scf_profile"]["reason"] == "trace failed"
    assert result["changed_geometry"]["status"] == "failed"
    assert result["changed_geometry"]["error"] == "changed replay failed"
    assert calls[:4] == [
        "cold",
        "priming",
        "same_geometry_warm_0",
        "same_geometry_warm_1",
    ]
