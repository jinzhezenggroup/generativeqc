"""Original #2840 range-PBE state and stationary derivatives, with no VV10."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive

from benchmarks.ks_matrix_provider_qualification import final_state_gates

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_KS_STATE_CUDA_TEST") != "1",
    reason="requires explicitly scheduled original-input CUDA qualification",
)

_PRIMITIVES = (
    ("3.425250914", "0.1543289673"),
    ("0.6239137298", "0.5353281423"),
    ("0.168855404", "0.4446345422"),
)
_GRID = GridSpec(24, 12, 24)
_ECP_ATOMS = [("Na", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]


def _basis(representation: str, *, polarized_fit: bool = False) -> BasisSet:
    """Preserve the native fixture's decimals, including its auxiliary basis."""
    shells = (
        BasisShell(
            0,
            tuple(exponent for exponent, _ in _PRIMITIVES),
            (tuple(coefficient for _, coefficient in _PRIMITIVES),),
        ),
    )
    if polarized_fit:
        shells += (BasisShell(1, ("0.7",), (("1",),)),)
    return BasisSet(
        "original-2840-hydrogens",
        (ElementBasis(1, shells),),
        BasisProvenance(
            "tests/native/test_ks_cuda.cpp::hydrogens",
            "original-2840",
            "GPL-3.0-or-later",
            hashlib.sha256(repr((_PRIMITIVES, polarized_fit)).encode()).hexdigest(),
        ),
        representation=representation,
    )


def _atoms(unrestricted: bool, moved: bool) -> list[tuple[str, tuple[float, ...]]]:
    """Match hydrogens(count, restricted, shift=0.05), not a new geometry."""
    return [
        ("H", (0.15 * index * index, 0.13 * index, (1.55 if moved else 1.5) * index))
        for index in range(3 if unrestricted else 2)
    ]


def _range_method(unrestricted: bool, fractions: tuple[int, int] = (20, 65)) -> Any:
    spin = "polarized" if unrestricted else "unpolarized"
    pbe = resolve_method("PBE", spin=spin)
    return replace(
        pbe,
        identifier="original-2840-PBE-RSH",
        primitives=(
            *pbe.primitives,
            RangeSeparatedExchangePrimitive(
                Fraction(fractions[0], 100), Fraction(33, 100), "short-range"
            ),
            RangeSeparatedExchangePrimitive(
                Fraction(fractions[1], 100), Fraction(33, 100), "long-range"
            ),
        ),
    )


def _reference(
    atoms: Any,
    unrestricted: bool,
    *,
    fitted_primary: bool = True,
    fitted_lr: bool = False,
    fractions: tuple[int, int] = (20, 65),
    ecp: bool = False,
) -> Any:
    """Reconverge independently with the specified integral model.

    The fitted-LR fixture adds an auxiliary p shell and builds its own erf
    fit in Libcint. Exact LR cases must bypass PySCF's usual range DF path.
    """
    from pyscf import df, dft, gto, lib, scf

    lib.num_threads(1)
    orbital = (
        {"Na": [[0, (0.7, 1.0)]], "H": [[0, (0.9, 1.0)]]}
        if ecp
        else {
            "H": [
                [
                    0,
                    *[
                        (float(exponent), float(coefficient))
                        for exponent, coefficient in _PRIMITIVES
                    ],
                ]
            ]
        }
    )
    # Encode the test ECP in NWChem syntax independently of the production
    # ECP parser. Libcint supplies both the local and projector integrals.
    pseudopotential = (
        {"Na": gto.basis.parse_ecp("Na nelec 10\nNa ul\n2 0.8 -0.4\nNa s\n2 0.8 0.3")}
        if ecp
        else {}
    )
    molecule = gto.M(
        atom=atoms,
        basis=orbital,
        ecp=pseudopotential,
        charge=int(ecp and unrestricted),
        unit="Bohr",
        spin=int(unrestricted),
        verbose=0,
    )
    auxiliary = {
        symbol: [*shells, [1, (0.7, 1.0)]] if fitted_lr else shells
        for symbol, shells in orbital.items()
    }
    fitted = df.DF(molecule, auxbasis=auxiliary).build() if fitted_primary else None
    reference = dft.UKS(molecule) if unrestricted else dft.RKS(molecule)
    reference.xc = (
        f"PBE + {fractions[0] / 100}*SR_HF(0.33) + {fractions[1] / 100}*LR_HF(0.33)"
    )

    def get_jk(
        mol: Any,
        density: Any,
        hermi: int = 1,
        with_j: bool = True,
        with_k: bool = True,
        omega: float | None = None,
    ) -> Any:
        if fitted is None or (omega and not fitted_lr):
            return scf.hf.get_jk(
                mol, density, hermi, with_j=with_j, with_k=with_k, omega=omega
            )
        return fitted.get_jk(density, hermi, with_j, with_k, omega=omega or None)

    reference.get_jk = get_jk
    grid = MolecularGrid(atoms, _GRID, multiplicity=2 if unrestricted else 1).explicit()
    reference.grids.radii_adjust = None
    reference.grids.coords = np.asarray(grid.points)
    reference.grids.weights = np.asarray(grid.weights)
    reference.small_rho_cutoff = 0
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 200
    reference.kernel(dm0=reference.get_init_guess(key="1e"))
    assert reference.converged
    return reference


def _hvp_oracle(
    unrestricted: bool,
    fitted: bool,
    fractions: tuple[int, int],
    step: float,
    *,
    fitted_lr: bool = False,
    ecp: bool = False,
) -> np.ndarray:
    """Central mixed difference of independently converged moving-grid energies."""
    atoms = _ECP_ATOMS if ecp else _atoms(unrestricted, True)
    coordinates = np.array([position for _, position in atoms])
    direction = _hessian_direction(len(atoms))

    def energy(displaced: np.ndarray) -> float:
        geometry = [
            (atom[0], tuple(point))
            for atom, point in zip(atoms, displaced, strict=True)
        ]
        return float(
            _reference(
                geometry,
                unrestricted,
                fitted_primary=fitted,
                fitted_lr=fitted_lr,
                fractions=fractions,
                ecp=ecp,
            ).e_tot
        )

    result = np.empty(coordinates.size)
    for coordinate, unit in enumerate(np.eye(coordinates.size)):
        left, right = step * unit.reshape(coordinates.shape), step * direction
        result[coordinate] = (
            energy(coordinates + left + right)
            - energy(coordinates + left - right)
            - energy(coordinates - left + right)
            + energy(coordinates - left - right)
        ) / (4 * step**2)
    return result.reshape(coordinates.shape)


def _hessian_direction(atoms: int) -> np.ndarray:
    return (np.sin(np.arange(3 * atoms, dtype=np.float64) + 0.5) / 2).reshape(atoms, 3)


def _range_bases(
    ecp: bool, fitted: bool, fitted_lr: bool
) -> tuple[BasisSet, BasisSet | None]:
    """Build a separate fit basis on the same centers, with no orbital ECP."""
    if not ecp:
        return _basis("spherical"), (
            _basis("spherical", polarized_fit=fitted_lr) if fitted else None
        )
    from test_wb97mv_ecp_hessian import synthetic_basis

    basis = synthetic_basis()
    if not fitted:
        return basis, None
    return basis, replace(
        basis,
        name="ECP-free range-PBE fit fixture",
        elements=tuple(
            ElementBasis(
                element.atomic_number,
                (*element.shells, BasisShell(1, ("0.7",), (("1",),)))
                if fitted_lr
                else element.shells,
            )
            for element in basis.elements
        ),
    )


@pytest.mark.parametrize("ecp", (False, True), ids=("ae", "ecp"))
@pytest.mark.parametrize(
    ("unrestricted", "fitted", "fractions", "lr_policy"),
    [
        (False, False, (20, 65), "direct"),
        (True, False, (65, 20), "direct"),
        (False, True, (20, 65), None),
        (True, True, (20, 65), "direct"),
        (False, True, (20, 65), "auto"),
        (True, True, (65, 20), "df"),
    ],
    ids=(
        "direct-rks",
        "direct-uks-negative-lr",
        "df-rks",
        "df-uks",
        "ri-rks-auto",
        "ri-uks-negative-lr",
    ),
)
def test_range_pbe_native_hessian(
    unrestricted: bool,
    fitted: bool,
    fractions: tuple[int, int],
    lr_policy: str | None,
    ecp: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert os.environ.get("SLURM_JOB_ID")
    if lr_policy is None:
        monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", lr_policy)
    fitted_lr = lr_policy in ("df", "auto")
    atoms = _ECP_ATOMS if ecp else _atoms(unrestricted, True)
    basis, auxiliary = _range_bases(ecp, fitted, fitted_lr)
    settings = {
        "charge": int(ecp and unrestricted),
        "multiplicity": 2 if unrestricted else 1,
    }
    configuration = {
        "method": "pbe-uks" if unrestricted else "pbe-rks",
        "basis": basis,
        "device": "cuda",
        "precision": "fp64",
        "basis_representation": "spherical",
        "density_fitting": "cuda" if fitted else False,
        "auxiliary_basis": auxiliary,
        "ks_options": KsOptions(
            composition=_range_method(unrestricted, fractions), grid=_GRID
        ),
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "max_iterations": 200,
    }
    scf = (
        Calculator(**configuration)
        .estimate_resources(
            [atoms],
            charges=[settings["charge"]],
            multiplicities=[settings["multiplicity"]],
            properties=("energy",),
        )
        .require_feasible()
    )
    assert scf.requests[0].identity.observables == ("energy",)
    schedule = json.loads(scf.requests[0].identity.schedule)
    assert schedule["pbe_range_exchange"] == (lr_policy or "direct")
    if ecp:
        assert schedule["ecp_device_inventory"] == "native-ecp-v1"
    budget = ResourceBudget(
        host_bytes=scf.peak_bytes["host"] + (8 << 20),
        device_bytes=scf.peak_bytes["device"] + (96 << 20),
    )
    calculator = Calculator(**configuration, resource_budget=budget)
    assert calculator.second_order_capabilities == {"hessian", "hvp"}
    direction = _hessian_direction(len(atoms))
    dense = calculator.hessian(atoms, **settings)
    action = calculator.hessian_vector_product(atoms, direction, **settings)
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        dense.matrix @ direction.ravel(), action.value.ravel(), rtol=0, atol=2e-8
    )
    assert dense.diagnostics["completed_directions"] == direction.size
    assert bool(action.diagnostics["exact_primitive_components"]) != fitted_lr
    assert bool(action.diagnostics["df_value_passes"]) == fitted
    assert action.diagnostics["vv10_reference_pairs"] == 0
    assert action.diagnostics["ecp_ad_backend"] == ("cuda" if ecp else None)
    assert bool(action.diagnostics["ecp_workspace_bytes"]) == ecp
    ledger = action.diagnostics["resources"]["hessian_response"]["device_ledger"]
    assert ledger["peak_bytes"] <= budget.device_bytes
    coarse = _hvp_oracle(
        unrestricted, fitted, fractions, 8e-4, fitted_lr=fitted_lr, ecp=ecp
    )
    fine = _hvp_oracle(
        unrestricted, fitted, fractions, 4e-4, fitted_lr=fitted_lr, ecp=ecp
    )
    np.testing.assert_allclose(coarse, fine, rtol=0, atol=4e-6)
    np.testing.assert_allclose(action.value, (4 * fine - coarse) / 3, rtol=0, atol=2e-6)


@pytest.mark.skipif(
    not os.environ.get("GENERATIVEQC_WB97MV_CLI"), reason="requires compiled C++ CLI"
)
@pytest.mark.parametrize(
    ("unrestricted", "policy"),
    [(False, None), (False, "direct"), (True, "df")],
    ids=("direct-rks", "df-exact-rks", "df-erf-uks"),
)
def test_range_pbe_method_ir_cli(
    unrestricted: bool,
    policy: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip MethodIR and compare the actual accepted model to Libcint."""
    from test_wb97mv_hessian_cli_cuda import invoke_cli

    monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    if policy == "df":
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", policy)
    fitted = policy is not None
    fitted_lr = policy == "df"
    ecp = fitted
    atoms = _ECP_ATOMS if ecp else _atoms(unrestricted, True)
    basis, auxiliary = _range_bases(ecp, fitted, fitted_lr)
    basis_file = tmp_path / "Orbital Basis.json"
    basis_file.write_text(json.dumps(basis.to_payload()))
    fractions = (65, 20) if unrestricted else (20, 65)
    method = _range_method(unrestricted, fractions)
    ir_file = tmp_path / "Custom MethodIR.json"
    ir_file.write_text(json.dumps(method.to_payload()))
    common = (
        "--basis",
        str(basis_file),
        "--charge",
        str(int(ecp and unrestricted)),
        "--multiplicity",
        "2" if unrestricted else "1",
        "--grid-radial-points",
        str(_GRID.radial_points),
        "--grid-polar-points",
        str(_GRID.angular_polar),
        "--grid-azimuth-points",
        str(_GRID.angular_azimuth),
    )
    if auxiliary is not None:
        aux_file = tmp_path / "Fit Basis.json"
        aux_file.write_text(json.dumps(auxiliary.to_payload()))
        common += ("--density-fitting", "cuda", "--auxiliary-basis", str(aux_file))
    flags = (*common, "--method-ir", str(ir_file))
    direction = _hessian_direction(len(atoms))
    action = invoke_cli(
        tmp_path, atoms, direction=direction, extra=flags, default_grid=True
    )
    dense = invoke_cli(
        tmp_path, atoms, operation="hessian", extra=flags, default_grid=True
    )
    for result in (action, dense):
        assert result["method"] == method.identifier
        assert result["method_ir_identity"] == method.identity
        work = result["diagnostics"]
        assert bool(work["df_range_value_passes"]) == fitted_lr
        assert bool(work["df_value_passes"]) == fitted
        assert work["ecp_ad_backend"] == ("cuda" if ecp else None)
        assert work["vv10_reference_pairs"] == 0
    matrix = np.asarray(dense["matrix"])
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), np.ravel(action["value"]), rtol=0, atol=2e-8
    )
    coarse = _hvp_oracle(
        unrestricted, fitted, fractions, 8e-4, fitted_lr=fitted_lr, ecp=ecp
    )
    fine = _hvp_oracle(
        unrestricted, fitted, fractions, 4e-4, fitted_lr=fitted_lr, ecp=ecp
    )
    np.testing.assert_allclose(
        action["value"], (4 * fine - coarse) / 3, rtol=0, atol=2e-6
    )
    if not fitted:
        pure = invoke_cli(
            tmp_path,
            atoms,
            direction=direction,
            default_grid=True,
            extra=(*common, "--method", "pbe-rks"),
        )
        assert (
            pure["method_ir_identity"]
            == resolve_method("PBE", spin="unpolarized").identity
        )
        assert abs(pure["energy_hartree"] - action["energy_hartree"]) > 1e-4


@pytest.mark.parametrize("explicit", (False, True), ids=("missing", "ecp-bearing"))
def test_range_pbe_df_rejects_invalid_ecp_auxiliary(explicit: bool) -> None:
    from test_wb97mv_ecp_hessian import synthetic_basis

    basis = synthetic_basis()
    calculator = Calculator(
        "pbe-rks",
        basis,
        device="cuda",
        precision="fp64",
        density_fitting="cuda",
        auxiliary_basis=basis if explicit else None,
        ks_options=KsOptions(composition=_range_method(False), grid=_GRID),
    )
    message = "auxiliary basis must not carry ECPs" if explicit else "explicit ECP-free"
    with pytest.raises(NotImplementedError, match=message):
        calculator.estimate_resources([_ECP_ATOMS], properties=("energy",))
    # An unbudgeted caller must not bypass the native preparation guard.
    with pytest.raises(NotImplementedError, match="explicit ECP-free"):
        calculator.singlepoint(_ECP_ATOMS, properties=("energy",))


@pytest.mark.skipif(
    not os.environ.get("GENERATIVEQC_KS_HESSIAN_SDK_PROBE"),
    reason="requires the compiled public C++ SDK probe",
)
def test_range_pbe_public_sdk_hessian() -> None:
    """Compare the installed-header-only client to reconverged Libcint/Libxc."""
    assert os.environ.get("SLURM_JOB_ID")
    result = subprocess.run(
        [os.environ["GENERATIVEQC_KS_HESSIAN_SDK_PROBE"], "--cuda"],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    value = np.asarray(json.loads(result.stdout)).reshape((2, 3))
    coarse = _hvp_oracle(False, False, (20, 65), 8e-4)
    fine = _hvp_oracle(False, False, (20, 65), 4e-4)
    np.testing.assert_allclose(value, (4 * fine - coarse) / 3, rtol=0, atol=2e-6)


@lru_cache(maxsize=4)
def _independent_forces(atoms: tuple, unrestricted: bool) -> np.ndarray:
    """Differentiate independently reconverged, moving-grid mixed energies.

    A five-point stencil includes auxiliary-center and quadrature response.
    It never differentiates a frozen density or substitutes fitted LR-K.
    """
    positions = np.asarray([position for _, position in atoms])
    forces = np.zeros_like(positions)
    step = 2e-4
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for shift in (-2, -1, 1, 2):
                displaced = positions.copy()
                displaced[atom, axis] += shift * step
                current = [
                    (symbol, tuple(position))
                    for (symbol, _), position in zip(atoms, displaced, strict=True)
                ]
                energies.append(_reference(current, unrestricted).e_tot)
            forces[atom, axis] = -(
                energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]
            ) / (12 * step)
    return forces


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("unrestricted", (False, True), ids=("rks", "uks"))
def test_original_df_direct_lr_state_and_force_admission(
    representation: str,
    unrestricted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original mixed Hamiltonian owns all native integral and XC sources.

    This range-only case uses the compiled CUDA geometry owner, not the
    distinct resident range/VV10 controller. CPU derivatives cannot substitute
    for the native mixed integral bridge.
    """
    assert os.environ.get("SLURM_JOB_ID")
    pytest.importorskip("pyscf")
    from generativeqc.batch import PreparedBatch

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unqualified original RSH force entered a Python fallback")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    cuda_work = []
    cuda_force = PreparedBatch._public_dft_cuda_force

    def tracked_cuda_force(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = cuda_force(self, *args, **kwargs)
        cuda_work.append(result[1])
        return result

    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", tracked_cuda_force)
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
    method = _range_method(unrestricted)
    basis = _basis(representation)
    calculator = Calculator(
        "pbe-uks" if unrestricted else "pbe-rks",
        basis,
        device="cuda",
        precision="fp64",
        basis_representation=representation,
        density_fitting="cuda",
        auxiliary_basis=basis,
        ks_options=KsOptions(composition=method, grid=_GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=200,
    )
    observations: list[dict[str, Any]] = []
    provenance = {
        "source_identity": calculator._library.generativeqc_get_source_identity().decode(),
        "library_sha256": hashlib.sha256(
            Path(calculator._library._name).read_bytes()
        ).hexdigest(),
        "method_identity": method.identity,
        "basis_identity": basis.identity,
        "scheduler": {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
        },
        "force_disposition": "native mixed integral bridge plus compiled CUDA geometry",
    }
    references = {}
    try:
        with calculator.prepare_batch(
            [_atoms(unrestricted, False)], multiplicities=[2 if unrestricted else 1]
        ) as batch:
            assert not batch._native_dft_force_eligible(0)
            for phase in range(4):
                moved = phase >= 2
                atoms = _atoms(unrestricted, moved)
                item = batch.execute(
                    [[position for _, position in atoms]] if moved else None,
                    properties=("energy", "forces"),
                    strict=True,
                ).items[0]
                assert item.converged and item.executed_backend == "cuda"
                snapshot = NativeKsSnapshot(batch, 0)
                try:
                    gates = final_state_gates(
                        snapshot.metadata,
                        snapshot.values,
                        {"atomic_numbers": [1] * len(atoms), "charge": 0},
                    )
                    assert gates["accepted"]
                    assert gates["density_closure_error"] < 1e-8
                    assert gates["physical_residual_rms"] < 1e-9
                    dimension, spins = map(int, snapshot.metadata[1:3])
                    offset = 2 + 4 * len(atoms)
                    count = spins * dimension * dimension
                    density, fock = (
                        snapshot.values[begin : begin + count].reshape(
                            spins, dimension, dimension
                        )
                        for begin in (offset, offset + count)
                    )
                    if moved not in references:
                        references[moved] = _reference(atoms, unrestricted)
                    reference = references[moved]
                    reference_density = np.asarray(reference.make_rdm1()).reshape(
                        density.shape
                    )
                    reference_fock = np.asarray(
                        reference.get_fock(
                            dm=reference_density
                            if unrestricted
                            else reference_density[0]
                        )
                    ).reshape(fock.shape)
                    row = {
                        "phase": phase,
                        "energy": item.energy,
                        "reference_energy": float(reference.e_tot),
                        "energy_error": abs(item.energy - reference.e_tot),
                        "density_error": float(
                            np.max(np.abs(density - reference_density))
                        ),
                        "fock_error": float(np.max(np.abs(fock - reference_fock))),
                        "state": gates,
                        "precision": item.precision,
                        "history": item.ks_diagnostic.to_payload()["history"],
                    }
                    observations.append(row)
                    assert row["energy_error"] < 1e-8
                    assert row["density_error"] < 1e-6
                    assert row["fock_error"] < 1e-6
                    reference_forces = _independent_forces(tuple(atoms), unrestricted)
                    row["forces"] = np.asarray(item.forces).tolist()
                    row["reference_forces"] = reference_forces.tolist()
                    row["force_error"] = float(
                        np.max(np.abs(item.forces - reference_forces))
                    )
                    row["work"] = cuda_work[-1]
                    assert row["force_error"] < 3e-5
                    np.testing.assert_allclose(
                        np.asarray(item.forces).sum(axis=0), 0, atol=2e-8, rtol=0
                    )
                finally:
                    snapshot.close()
    finally:
        directory = os.environ.get("GENERATIVEQC_KS_STATE_EVIDENCE_DIR")
        if directory:
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"original-rsh-{representation}-{unrestricted}.json").write_text(
                json.dumps(
                    {"provenance": provenance, "observations": observations}, indent=2
                )
            )
        calculator.clear_cache()
