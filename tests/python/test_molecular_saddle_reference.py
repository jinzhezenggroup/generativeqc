"""Independent full-matrix controls for the real public molecular saddle adapter."""

from __future__ import annotations

import hashlib
import json
import math
import typing
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    _molecular_saddle,
    composite_rks_hessian,
    rks_hessian_integrals,
    saddle,
)
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.response_solver import GMRESOptions
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc.vibrations import analyze_harmonic_modes
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import resolve_method
from test_dft_complete_cpu import independent_global_hybrid_gradient

MANIFEST = Path(__file__).resolve().parents[1] / "data/native_saddle/manifest.json"


def _json(value: typing.Any) -> typing.Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return dict(value)


@pytest.mark.ci_nightly
@pytest.mark.parametrize("case", ["saddle", "nonnegative", "minimum"])
def test_native_molecular_saddle_reference(case: str, tmp_path: Path) -> None:
    """Compare every final raw Hessian column at three independent stencil sizes."""
    pytest.importorskip("pyscf")
    started = perf_counter()
    manifest = json.loads(MANIFEST.read_text())
    gates = manifest["acceptance"]
    specification = manifest["cases"][case]
    if "coordinates_bohr" in specification:
        initial = tuple(
            (element, tuple(position))
            for element, position in zip(
                ("N", "H", "H", "H"), specification["coordinates_bohr"], strict=True
            )
        )
    else:
        radius = specification["hydrogen_radius_bohr"]
        initial = (
            ("N", (0, 0, specification["nitrogen_z_bohr"])),
            ("H", (radius, 0, 0)),
            ("H", (-radius / 2, radius * math.sqrt(3) / 2, 0)),
            ("H", (-radius / 2, -radius * math.sqrt(3) / 2, 0)),
        )
    method = resolve_method(manifest["method"])
    xc = "+".join(
        f"{float(weight)}*{name}"
        for name, weight in method.primitives[0].functional.components
    )
    xc += f"+{float(method.primitives[1].coefficient)}*HF"
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=manifest["basis"],
        basis_representation=manifest["basis_representation"],
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    source_files = (
        Path(__file__),
        MANIFEST,
        Path(calculator._library._name),
        Path(__file__).with_name("test_dft_complete_cpu.py"),
        Path(saddle.__file__),
        Path(_molecular_saddle.__file__),
        Path(composite_rks_hessian.__file__),
        Path(rks_hessian_integrals.__file__),
    )
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    receipt: dict[str, typing.Any] = {
        "accepted": False,
        "case": case,
        "source_hashes": source_hashes,
        "method_identity": method.identity,
        "method_manifest_identity": method.manifest_identity,
        "native_source_identity": calculator._library.generativeqc_get_source_identity().decode(),
        "production_callables_replaced": False,
        "production_source_rewriting": False,
        "installed_ts_cli_qualified": False,
        "requested_ni_domain_qualified": False,
        "stage": "public-endpoint-start",
        "independent_matrices": [],
    }

    def save(stage: str, **details: typing.Any) -> None:
        receipt.update(stage=stage, seconds=perf_counter() - started, **details)
        (tmp_path / "receipt.json").write_text(
            json.dumps(receipt, default=_json, indent=2, allow_nan=False) + "\n"
        )

    def observe(event: typing.Mapping[str, typing.Any]) -> None:
        with (tmp_path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event), default=_json, allow_nan=False) + "\n")

    def independent(atoms: tuple[Atom, ...]) -> typing.Any:
        grid_spec = calculator.ks_options.grid
        grid = MolecularGrid(atoms, grid_spec, 0, 1).explicit()
        weights = np.concatenate(
            [
                MolecularGrid((atom,), grid_spec, 0, 1).explicit().weights
                for atom in atoms
            ]
        )
        state = SimpleNamespace(
            grid=grid,
            _source=SimpleNamespace(grid_spec=grid_spec, atomic_weights=weights),
        )
        with NativeAO(
            atoms, basis=manifest["basis"], representation="spherical"
        ) as basis:
            return independent_global_hybrid_gradient(
                basis,
                state,
                "b3lyp-rks",
                xc=xc,
                return_mean_field=True,
                scf_gradient_tolerance=1e-11,
            )

    try:
        save("public-endpoint-start")
        options = SaddleOptions(**manifest["controls"])
        result = optimize_saddle(
            calculator,
            initial,
            masses=manifest["masses_dalton"],
            options=options,
            grid_budget_bytes=128 << 20,
            solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            evaluation_progress=observe,
        )
        native = result.optimizer
        save(
            "public-endpoint-returned",
            native_receipt=dict(native.receipt),
            native_failures=native.failures,
            diagnostics=result.diagnostics,
        )
        assert not native.failures and native.receipt["has_valid_state"]
        np.savez_compressed(
            tmp_path / "last-valid.npz",
            coordinates=native.coordinates,
            gradient=native.gradient,
            hessian=native.hessian,
        )
        assert native.receipt["status"] == specification["expected_status"]
        assert native.receipt["negative_curvatures"] == specification["expected_index"]
        assert not native.receipt["near_zero_curvatures"]
        assert result.converged == (case == "saddle")
        assert (
            native.receipt["stationary_verified"]
            == specification["expected_stationary_verified"]
        )
        if case == "saddle":
            assert (
                0
                < native.receipt["accepted_steps"]
                <= native.receipt["attempts"]
                <= 100
            )
        if native.receipt["stationary_verified"]:
            for name in (
                "energy_change",
                "gradient_rms",
                "gradient_max",
                "displacement_rms_angstrom",
                "displacement_max_angstrom",
            ):
                assert native.receipt[name] <= getattr(options, name)
            assert (
                result.harmonics.negative_mode_count == specification["expected_index"]
            )
            assert (
                result.harmonics.significant_imaginary_mode_count
                == specification["expected_index"]
            )
            assert result.harmonics.diagnostics["frequency_scale_factor"] == 1
            assert result.harmonics.frequencies_cm1.size == (
                native.coordinates.size - native.receipt["rigid_modes"]
            )
            save(
                "stationary-classification-checked",
                unscaled_frequencies_cm1=result.harmonics.frequencies_cm1,
                harmonic_negative_modes=result.harmonics.negative_mode_count,
                harmonic_rigid_modes=result.harmonics.rigid_mode_count,
            )
            if case == "minimum":
                assert native.receipt["attempts"] == 0
                assert (
                    native.receipt["gradient_max"]
                    < gates["stationary_input_gradient_max"]
                )
                assert np.all(result.harmonics.frequencies_cm1 > 0)
        else:
            assert native.receipt["attempts"] == 0 and result.harmonics is None
            assert native.receipt["gradient_max"] > options.gradient_max
        hessian_events = [
            event
            for event in result.diagnostics["evaluations"]
            if event["stage"] == "hessian" and event["status"] == "complete"
        ]
        assert len(hessian_events) == native.receipt["hessian_calls"]
        assert all(
            event["maximum_response_residual_norm"] < gates["response_residual_norm"]
            for event in hessian_events
        )
        assert result.diagnostics["work_counters_complete"]
        assert (
            result.diagnostics["nuclear_response_solves"]
            == native.receipt["hessian_calls"] * native.coordinates.size
        )
        matrix = native.hessian
        assert np.max(np.abs(matrix - matrix.T)) < gates["raw_symmetry_max"]
        assert (
            np.max(np.abs(matrix @ np.tile(np.eye(3), (4, 1))))
            < gates["raw_translation_max"]
        )
        energy, gradient, reference = independent(result.atoms)
        with (
            calculator.prepare_batch([result.atoms], warm_start=False) as batch,
            NativeAO(
                result.atoms, basis=manifest["basis"], representation="spherical"
            ) as basis,
        ):
            point = batch.execute(strict=True, properties=("energy",)).items[0]
            state = StationaryKsState.from_native(batch, basis)
            try:
                density_error = float(
                    np.max(np.abs(state.density[0] - reference.make_rdm1()))
                )
            finally:
                state._source.close()
        orbital_gradient = float(
            np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
        )
        assert (
            density_error < gates["density_max"]
            and orbital_gradient < gates["orbital_gradient_norm"]
        )
        assert abs(energy - native.receipt["energy"]) < gates["energy_abs_hartree"]
        assert abs(point.energy - energy) < gates["energy_abs_hartree"]
        assert np.max(np.abs(gradient - native.gradient)) < gates["gradient_max"]
        if case == "minimum":
            assert np.max(np.abs(gradient)) < gates["stationary_input_gradient_max"]
        save(
            "independent-central-state-matched",
            density_error=density_error,
            orbital_gradient_norm=orbital_gradient,
        )
        independent_matrices = []
        directions = np.eye(native.coordinates.size).reshape(
            native.coordinates.size, *native.coordinates.shape
        )
        for step in gates["difference_steps_bohr"]:
            reference_matrix = np.empty_like(matrix)
            for column, direction in enumerate(directions):
                gradients = {}
                for offset in (-2, -1, 1, 2):
                    moved = tuple(
                        Atom(
                            atom.atomic_number,
                            np.asarray(atom.position) + offset * step * motion,
                        )
                        for atom, motion in zip(result.atoms, direction, strict=True)
                    )
                    gradients[offset] = independent(moved)[1]
                reference_matrix[:, column] = (
                    (
                        8 * (gradients[1] - gradients[-1])
                        - (gradients[2] - gradients[-2])
                    )
                    / (12 * step)
                ).ravel()
            error = float(np.max(np.abs(matrix - reference_matrix)))
            assert error < gates["independent_matrix_max"]
            modes = analyze_harmonic_modes(
                reference_matrix,
                coordinates=native.coordinates,
                masses=manifest["masses_dalton"],
                hessian_units="hartree/bohr^2",
                coordinate_units="bohr",
                mass_units="dalton",
                symmetrize=True,
                raw_symmetry_tolerance=gates["independent_matrix_max"],
            )
            assert (
                modes.significant_imaginary_mode_count
                == specification["expected_index"]
            )
            frequency_error = None
            if result.harmonics is not None:
                frequency_error = float(
                    np.max(
                        np.abs(result.harmonics.frequencies_cm1 - modes.frequencies_cm1)
                    )
                )
                assert frequency_error < gates["independent_frequency_max_cm1"]
            receipt["independent_matrices"].append(
                {
                    "step_bohr": step,
                    "maximum_error": error,
                    "frequency_max_error_cm1": frequency_error,
                }
            )
            independent_matrices.append(reference_matrix)
            save("independent-matrix-step-complete")
        np.savez_compressed(
            tmp_path / "independent-hessians.npz", matrices=independent_matrices
        )
        assert all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
            for path, expected in source_hashes.items()
        )
        save("accepted", accepted=True)
    finally:
        save(receipt["stage"])
        calculator.clear_cache()
