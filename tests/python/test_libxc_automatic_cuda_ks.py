"""Independent public automatic CUDA KS endpoints, with no CPU science staging.

An opt-in static native library and finite Slurm allocation are mandatory.
An explicit Hcore seed isolates SCF transport from the optional MINAO initializer.
PySCF/Libxc runs only after timed production endpoints, on their exported exact
grid and physical state. Receipts are qualification data, not a positive list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions, ResourceBudget
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.ks_diagnostics import read_ks_transport_diagnostic
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
from generativeqc_compiler.xc.automatic_semilocal import AUTOMATIC_SCF_DOMAIN
from test_libxc_automatic_point_native import NAMES

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_KS_TEST") != "1",
    reason="requires opt-in static CUDA providers and a finite Slurm allocation",
)
GRID = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
# Multiple occupied orbitals avoid an iso-orbital min/max branch boundary,
# where independently rounded AO products need not select the same derivative.
ATOMS = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.1, 0.2, 1.4)),
    ("H", (1.6, 0.4, 0.3)),
    ("H", (1.8, 0.3, 1.8)),
]
MOVED = [ATOMS[0], ("H", (0.1, 0.2, 1.47)), *ATOMS[2:]]
BASIS = {
    "H": [
        [
            0,
            [3.425250914, 0.1543289673],
            [0.6239137298, 0.5353281423],
            [0.168855404, 0.4446345422],
        ]
    ]
}


@pytest.mark.parametrize(
    "policy",
    ("host-unfused", "mixed-precision", "incremental-j"),
)
def test_automatic_cuda_policies_do_not_inherit_curated_qualification(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """The actual prepared native owner rejects unqualified products explicitly."""
    assert os.environ.get("SLURM_JOB_ID"), "real CUDA qualification requires Slurm"
    if policy == "incremental-j":
        monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "1")
    calculator = Calculator(
        method="libxc:GGA_X_APBE",
        basis="sto-3g",
        device="cuda",
        precision="auto" if policy == "mixed-precision" else "fp64",
        ks_options=KsOptions(
            grid=GRID,
            xc_schedule="host_unfused" if policy == "host-unfused" else "device_fused",
        ),
    )
    reason = (
        "incremental Direct-J qualification"
        if policy == "incremental-j"
        else "strict-FP64"
    )
    with (
        pytest.raises(RuntimeError, match=reason),
        calculator.prepare_batch([ATOMS]),
    ):
        pytest.fail("unqualified automatic CUDA policy prepared an execution owner")


def _physical_state(snapshot: NativeKsSnapshot) -> dict[str, np.ndarray]:
    """Read the existing snapshot transport without evaluating any XC or AO."""
    metadata, values = snapshot.metadata, snapshot.values
    _, nao, spins, atoms, packed, points = metadata[:6]
    offset = 2 + 4 * atoms
    result = {}
    for name in (
        "density",
        "fock",
        "coefficients",
        "orbital_energies",
        "occupations",
        "weighted_density",
        "overlap",
    ):
        shape = (
            (nao, nao)
            if name == "overlap"
            else (spins, nao)
            if name in ("orbital_energies", "occupations")
            else (spins, nao, nao)
        )
        count = int(np.prod(shape))
        result[name] = values[offset : offset + count].reshape(shape).copy()
        offset += count
    offset += packed
    result["points"] = values[offset : offset + 3 * points].reshape(points, 3).copy()
    offset += 3 * points
    result["weights"] = values[offset : offset + points].copy()
    return result


def _reference(
    name: str, spins: int, atoms: list[Any], state: dict[str, np.ndarray]
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Independent pinned SCF and fixed-density full-Fock directional oracle."""
    import pyscf
    from pyscf import dft, gto

    assert pyscf.__version__ == "2.14.0"
    assert dft.libxc.__version__ == "7.0.0"
    molecule = gto.M(
        atom=atoms,
        basis=BASIS,
        unit="Bohr",
        cart=True,
        charge=spins - 1,
        spin=spins - 1,
        verbose=0,
    )
    method = dft.RKS(molecule) if spins == 1 else dft.UKS(molecule)
    method.xc = name
    # Match the seed policy using independently computed PySCF Hcore, not the
    # native density. MINAO stalls on the pure LTA asymmetric H4 fixture.
    method.init_guess = "1e"
    method.small_rho_cutoff = 0
    method.grids.coords = state["points"]
    method.grids.weights = state["weights"]
    method.conv_tol = 1e-12
    method.conv_tol_grad = 1e-10
    # Near-flat tau-MGGA fixed points can need more independent DIIS cycles.
    # This finite oracle budget does not alter native limits or numeric gates.
    method.max_cycle = 1000
    energy = method.kernel()
    assert method.converged, (name, spins, atoms)
    np.testing.assert_array_equal(method.grids.coords, state["points"])
    np.testing.assert_array_equal(method.grids.weights, state["weights"])
    density = state["density"][0] if spins == 1 else state["density"]
    hcore = method.get_hcore()
    potential = method.get_veff(molecule, density)
    fock = hcore + potential
    fixed_energy = method.energy_tot(density, hcore, potential)
    nao = molecule.nao_nr()
    indices = np.arange(nao)
    perturbation = 0.02 / (1.0 + np.abs(indices[:, None] - indices))
    perturbation[np.diag_indices(nao)] += np.linspace(0.05, 0.11, nao)
    direction = perturbation @ density + density @ perturbation
    step = 2e-5
    # Congruence preserves PSD at integer-occupation SCF states; an arbitrary
    # linear density stencil can leave the domain along unoccupied directions.
    plus = np.eye(nao) + step * perturbation
    minus = np.eye(nao) - step * perturbation
    derivative = (
        method.energy_tot(plus @ density @ plus.T)
        - method.energy_tot(minus @ density @ minus.T)
    ) / (2 * step)
    contracted = float(np.sum(fock * direction))
    assert abs(derivative - contracted) < 5e-8
    native_contracted = float(np.sum(state["fock"] * direction))
    return (
        energy,
        fixed_energy,
        np.asarray(method.make_rdm1()),
        fock,
        abs(derivative - native_contracted),
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("input_form", ("selector", "method-ir"))
def test_public_automatic_cuda_ks_lifecycle_matches_independent_pyscf(
    name: str, spins: int, input_form: str
) -> None:
    """Cold, ordinary warm recomputation and changed geometry retain J-only work."""
    assert os.environ.get("SLURM_JOB_ID"), "real CUDA qualification requires Slurm"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "preserve scheduler GPU visibility"
    configured_at = perf_counter()
    options = {
        "method": (
            f"libxc-{'rks' if spins == 1 else 'uks'}:{name}"
            if input_form == "selector"
            else resolve_bulk_ks(
                name,
                spin="unpolarized" if spins == 1 else "polarized",
                backend="cuda",
            ).method
        ),
        "basis": "sto-3g",
        "device": "cuda",
        "precision": "fp64",
        "initial_guess": None,
        "ks_options": KsOptions(grid=GRID, tile_points=37),
        "max_iterations": 200,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
    }
    capacity = (
        Calculator(**options)
        .estimate_resources([ATOMS], charges=[spins - 1], multiplicities=[spins])
        .require_feasible()
    )
    assert capacity.requests[0].identity.observables == ("energy",)
    calculator = Calculator(
        **options,
        resource_budget=ResourceBudget(
            host_bytes=capacity.peak_bytes["host"],
            device_bytes=capacity.peak_bytes["device"],
        ),
    )
    configuration_resource_seconds = perf_counter() - configured_at
    rows = []
    started = perf_counter()
    with calculator.prepare_batch(
        [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
    ) as batch:
        preparation_seconds = perf_counter() - started
        previous = None
        exported_matrix_bytes = 0
        for phase, atoms in (
            ("cold", ATOMS),
            ("warm-recompute", ATOMS),
            ("changed-geometry", MOVED),
        ):
            started = perf_counter()
            result = batch.execute(
                coordinates=None
                if phase != "changed-geometry"
                else [[atom[1] for atom in atoms]],
                strict=True,
                properties=("energy",),
            ).items[0]
            endpoint_seconds = perf_counter() - started
            assert result.succeeded and result.converged
            assert result.executed_backend == "cuda"
            assert result.ks_diagnostic.scf_domain == AUTOMATIC_SCF_DOMAIN
            assert result.ks_diagnostic.physical_residual_max < 2e-9
            assert result.ks_diagnostic.fock_builds > 0
            assert result.warm_start_used == (phase != "cold")
            transport = read_ks_transport_diagnostic(
                batch._library, batch._batch, index=0
            )
            # Snapshot exports are explicit. Rebuilding also downloads one
            # last-good warm density before retiring its old device owner;
            # neither is iteration-time CPU XC staging.
            expected_matrix_bytes = (
                0
                if phase == "cold"
                else exported_matrix_bytes
                + (spins * len(ATOMS) ** 2 * 8 if phase == "changed-geometry" else 0)
            )
            assert transport.matrix_d2h_bytes == expected_matrix_bytes
            work = result.precision
            assert work["operator_inventory_complete"] and work["complete"]
            operators = {record["name"]: record for record in work["operators"]}
            assert operators["coulomb_j"]["count"] > 0
            assert operators["xc"]["count"] > 0
            assert not operators.get("exchange_k", {}).get("count", 0)
            assert not operators.get("exchange_recurrence", {}).get("count", 0)
            assert all(record["compute"] == "fp64" for record in work["operators"])
            if previous is not None:
                with pytest.raises(ValueError, match="stale"):
                    previous.check_current()
                previous.close()
            snapshot = NativeKsSnapshot(batch, 0)
            assert snapshot.backend == "cuda"
            assert snapshot.fock_provider_proof()[1] is None
            state = _physical_state(snapshot)
            expected, fixed, density, fock, directional_error = _reference(
                name, spins, atoms, state
            )
            np.testing.assert_allclose(result.energy, expected, rtol=0, atol=3e-9)
            np.testing.assert_allclose(result.energy, fixed, rtol=0, atol=3e-10)
            np.testing.assert_allclose(
                state["density"].squeeze(axis=0) if spins == 1 else state["density"],
                density,
                rtol=0,
                atol=2e-7,
            )
            np.testing.assert_allclose(
                state["fock"], fock.reshape(state["fock"].shape), rtol=0, atol=3e-9
            )
            assert directional_error < 5e-8
            rows.append(
                {
                    "phase": phase,
                    "atoms": atoms,
                    "endpoint_seconds": endpoint_seconds,
                    "complete_endpoint_seconds": endpoint_seconds
                    + (
                        preparation_seconds + configuration_resource_seconds
                        if phase == "cold"
                        else 0
                    ),
                    "energy_error": abs(result.energy - expected),
                    "fixed_energy_error": abs(result.energy - fixed),
                    "directional_error": directional_error,
                    "density_max_abs_error": float(
                        np.max(
                            np.abs(
                                state["density"]
                                - density.reshape(state["density"].shape)
                            )
                        )
                    ),
                    "fock_max_abs_error": float(
                        np.max(
                            np.abs(state["fock"] - fock.reshape(state["fock"].shape))
                        )
                    ),
                    "diagnostic": result.ks_diagnostic.to_payload(),
                    "transport": transport.to_payload(),
                    "semantic_work": work,
                }
            )
            previous = snapshot
            exported_matrix_bytes = read_ks_transport_diagnostic(
                batch._library, batch._batch, index=0
            ).matrix_d2h_bytes
        previous.close()
    if directory := os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_KS_EVIDENCE_DIR"):
        provider = emit_automatic_cuda_provider(name)
        payload = {
            "schema": "generativeqc.automatic-cuda-ks-qualification.v1",
            "name": name,
            "backend": "cuda",
            "product": "ks-energy",
            "domain": AUTOMATIC_SCF_DOMAIN,
            "spins": spins,
            "input_form": input_form,
            "basis": BASIS,
            "grid": {
                "radial_points": 12,
                "angular_polar": 6,
                "angular_azimuth": 12,
            },
            "tile_points": 37,
            "charge": spins - 1,
            "multiplicity": spins,
            "initial_guess": "hcore",
            "oracle_initial_guess": "independent-pyscf-hcore",
            "oracle_scf_limits": {
                "max_cycle": 1000,
                "conv_tol": 1e-12,
                "conv_tol_grad": 1e-10,
                "small_rho_cutoff": 0,
            },
            "precision": "fp64",
            "scf_limits": {
                "max_iterations": 200,
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "screening_tolerance": 1e-12,
            },
            "method_identity": calculator.method_ir.identity,
            "plan_identity": calculator.ks_options.execution_plan.identity,
            "expression_identity": provider.point.expression_identity,
            "source_identity": provider.source_identity,
            "native_library_sha256": file_hash(
                Path(os.environ["GENERATIVEQC_LIBRARY"])
            ),
            "preparation_seconds": preparation_seconds,
            "configuration_resource_seconds": configuration_resource_seconds,
            "resource_peak_bytes": capacity.peak_bytes,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "rows": rows,
        }
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}-{spins}-{input_form}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
