"""Opt-in full UKS/D3 molecular saddle and independent final-curvature oracle."""

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Atom, Calculator
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.response_solver import GMRESOptions
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc.vibrations import analyze_harmonic_modes
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records

ecp_records = _canonical_records


def _json(value: object) -> object:
    """Serialize detached arrays and immutable observation mappings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return dict(value)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_UKS_SADDLE_ORACLE") != "1",
    reason="complete native-default def2-SVP UKS TS qualification is opt-in",
)
def test_h3_doublet_saddle_and_unscaled_frequencies(
    ecp_records: tuple, tmp_path: Path, record_property: object
) -> None:
    """Qualify a light-element control, not Ni/ECP or a Python-free CLI."""
    pytest.importorskip("pyscf")
    dispersion = pytest.importorskip("dftd3.interface")
    started = perf_counter()
    basis_manifest, _, table = ecp_records
    basis_record = replace(
        table,
        elements=tuple(
            element for element in table.elements if element.atomic_number == 1
        ),
    )
    electronic = resolve_method("B3LYP", spin="polarized")
    parameters = d3_parameters("B3LYP-D3(BJ)")
    method = replace(
        electronic,
        primitives=(
            *electronic.primitives,
            DispersionCorrectionPrimitive(D3Spec(**parameters)),
        ),
    )
    damping = dispersion.RationalDampingParam(
        **{name: parameters[name] for name in ("s6", "s8", "a1", "a2", "s9")}
    )
    atoms = (("H", (-1.9, 0, 0)), ("H", (0, 0, 0)), ("H", (1.9, 0, 0)))
    masses = [1.00782503223] * 3
    options = SaddleOptions(project_rigid_motions=True, raw_symmetry_tolerance=2e-8)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    gates = {
        "energy": 2e-8,
        "density": 2e-7,
        "gradient": 1e-5,
        "independent_orbital_gradient": 2e-9,
        "raw_matrix": 1e-5,
        "raw_symmetry": 2e-8,
        "translation": 2e-8,
        "hessian_hvp": 2e-8,
        "response_residual": 2e-10,
        "frequency_cm1": 1.0,
    }
    root = Path(__file__).resolve().parents[2]
    receipt = {
        "accepted": False,
        "installed_ts_cli_qualified": False,
        "requested_ni_ecp_domain_qualified": False,
        "production_callables_replaced": False,
        "production_displacements": 0,
        "independent_scf_cross_seeding": False,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                root / "python/generativeqc/_molecular_saddle.py",
                root / "python/generativeqc/saddle.py",
                root / "python/generativeqc/composite_rks_hessian.py",
                root / "python/generativeqc_compiler/xc/b3lyp_production_policy.py",
            )
        },
        "basis_manifest": basis_manifest,
        "basis_scope": "canonical def2-SVP H subset; not a modified Ni/ECP fixture",
        "method_identity": method.identity,
        "charge": 0,
        "multiplicity": 2,
        "grid": "native-default",
        "initial_atoms_bohr": atoms,
        "initial_mode_screen": "independent cold-SCF negative antisymmetric stretch at 1.9 Bohr; separate from retained target-not-negative 1.7-Bohr attempt",
        "gates": gates,
        "difference_steps_bohr": [0.003, 0.001, 0.0003],
        "independent_matrices": [],
    }

    def save(stage: str, **details: object) -> None:
        receipt.update(stage=stage, seconds=perf_counter() - started, **details)
        (tmp_path / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False, default=_json) + "\n"
        )

    def observe(event: object) -> None:
        with (tmp_path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event), allow_nan=False, default=_json) + "\n")

    calculator = Calculator(
        method=method,
        basis=basis_record,
        basis_representation="spherical",
        device="cpu",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    grid_spec = calculator.ks_options.grid
    cache = tmp_path / "compiler-cache"

    def independent(current: tuple[Atom, ...]) -> tuple:
        """Cold independent Libxc SCF and moving-grid gradients plus one D3."""
        grid = MolecularGrid(current, grid_spec, 0, 2).explicit()
        weights = np.concatenate(
            [
                MolecularGrid((atom,), grid_spec, 0, 1).explicit().weights
                for atom in current
            ]
        )
        state = SimpleNamespace(
            grid=grid,
            _source=SimpleNamespace(grid_spec=grid_spec, atomic_weights=weights),
        )
        with NativeAO(
            current, basis=basis_record, representation="spherical", multiplicity=2
        ) as basis:
            energy, gradient, reference = independent_global_hybrid_gradient(
                basis,
                state,
                "b3lyp-uks",
                xc="0.81*GGA_C_LYP+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.08*LDA_X+0.2*HF",
                return_mean_field=True,
                scf_gradient_tolerance=1e-11,
            )
        correction = dispersion.DispersionModel(
            np.ones(3, dtype=int), np.asarray([atom.position for atom in current])
        ).get_dispersion(damping, grad=True)
        return (
            energy + float(correction["energy"]),
            gradient + correction["gradient"],
            reference,
        )

    try:
        save("native-saddle-start")
        result = optimize_saddle(
            calculator,
            atoms,
            masses=masses,
            multiplicity=2,
            options=options,
            initial_mode=np.array([[1.0, 0, 0], [-2.0, 0, 0], [1.0, 0, 0]]),
            solver_options=solver,
            grid_budget_bytes=128 << 20,
            evaluation_progress=observe,
        )
        native = result.optimizer
        np.savez_compressed(
            tmp_path / "last-native.npz",
            coordinates=native.coordinates,
            gradient=native.gradient,
            matrix=native.hessian,
        )
        save(
            "native-saddle-returned",
            native_receipt=dict(native.receipt),
            failures=native.failures,
            diagnostics=dict(result.diagnostics),
        )
        assert result.converged and not native.failures
        assert native.receipt["status"] == "first_order_saddle"
        assert native.receipt["stationary_verified"]
        assert 0 < native.receipt["accepted_steps"] <= native.receipt["attempts"] <= 100
        for name in (
            "energy_change",
            "gradient_rms",
            "gradient_max",
            "displacement_rms_angstrom",
            "displacement_max_angstrom",
        ):
            assert native.receipt[name] <= getattr(options, name)
        assert native.receipt["negative_curvatures"] == 1
        assert result.harmonics.significant_imaginary_mode_count == 1
        assert result.harmonics.diagnostics["frequency_scale_factor"] == 1
        matrix = native.hessian
        assert np.max(abs(matrix - matrix.T)) < gates["raw_symmetry"]
        assert np.max(abs(matrix @ np.tile(np.eye(3), (3, 1)))) < gates["translation"]
        for event in result.diagnostics["evaluations"]:
            if event["stage"] == "energy_gradient" and event["status"] == "complete":
                assert tuple(event["occupied_counts"]) == (2, 1)
                assert min(event["minimum_occupied_overlaps_by_spin"]) >= 0.9
            if event["stage"] == "hessian" and event["status"] == "complete":
                assert (
                    event["maximum_response_residual_norm"] < gates["response_residual"]
                )
        energy, gradient, reference = independent(result.atoms)
        assert abs(energy - native.receipt["energy"]) < gates["energy"]
        assert np.max(abs(gradient - native.gradient)) < gates["gradient"]
        assert (
            np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
            < gates["independent_orbital_gradient"]
        )
        with (
            calculator.prepare_batch(
                [result.atoms], multiplicities=[2], warm_start=False
            ) as batch,
            NativeAO(
                result.atoms,
                basis=basis_record,
                representation="spherical",
                multiplicity=2,
            ) as basis,
        ):
            batch.execute(strict=True, properties=("energy",))
            state = StationaryKsState.from_native(batch, basis)
            try:
                density_error = float(
                    np.max(abs(state.density - reference.make_rdm1()))
                )
                assert density_error < gates["density"]
            finally:
                state._source.close()
        save("independent-central-state-matched", density_error=density_error)
        directions = np.eye(9).reshape(9, 3, 3)
        matrices = []
        for step in receipt["difference_steps_bohr"]:
            reference_matrix = np.empty_like(matrix)
            for column, direction in enumerate(directions):
                gradients = {}
                for offset in (-2, -1, 1, 2):
                    current = tuple(
                        Atom(
                            atom.atomic_number,
                            np.asarray(atom.position) + offset * step * motion,
                        )
                        for atom, motion in zip(result.atoms, direction, strict=True)
                    )
                    gradients[offset] = independent(current)[1]
                reference_matrix[:, column] = (
                    (
                        8 * (gradients[1] - gradients[-1])
                        - (gradients[2] - gradients[-2])
                    )
                    / (12 * step)
                ).ravel()
            error = float(np.max(abs(matrix - reference_matrix)))
            matrices.append(reference_matrix)
            np.savez_compressed(
                tmp_path / "independent-matrices.npz", matrices=matrices
            )
            assert error < gates["raw_matrix"]
            modes = analyze_harmonic_modes(
                reference_matrix,
                coordinates=native.coordinates,
                masses=masses,
                hessian_units="hartree/bohr^2",
                coordinate_units="bohr",
                mass_units="dalton",
                symmetrize=True,
                raw_symmetry_tolerance=gates["raw_matrix"],
            )
            assert modes.significant_imaginary_mode_count == 1
            frequency_error = float(
                np.max(abs(modes.frequencies_cm1 - result.harmonics.frequencies_cm1))
            )
            assert frequency_error < gates["frequency_cm1"]
            receipt["independent_matrices"].append(
                {
                    "step": step,
                    "raw_error": error,
                    "frequency_error_cm1": frequency_error,
                }
            )
            save("independent-matrix-complete")
        for index, direction in enumerate(
            (
                directions[0],
                np.array([[0.2, -0.3, 0.4], [-0.1, 0.5, 0.2], [0.3, 0.2, -0.4]]),
            )
        ):
            product = calculator.hessian_vector_product(
                result.atoms,
                direction,
                multiplicity=2,
                cache=cache,
                solver_options=solver,
                grid_budget_bytes=128 << 20,
            )
            np.savez_compressed(
                tmp_path / f"fresh-hvp-{index}.npz",
                direction=direction,
                value=product.value,
            )
            assert (
                np.max(abs(product.value.ravel() - matrix @ direction.ravel()))
                < gates["hessian_hvp"]
            )
        record_property(
            "unscaled_frequencies_cm1", result.harmonics.frequencies_cm1.tolist()
        )
        record_property("native_receipt", dict(native.receipt))
        save(
            "accepted",
            accepted=True,
            unscaled_frequencies_cm1=result.harmonics.frequencies_cm1.tolist(),
        )
    finally:
        save(receipt["stage"])
        calculator.clear_cache()
