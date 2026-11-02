"""Full-rank FP64 density-fitting oracle and native endpoint observations.

The tuner initially admits the small-topology RHF resource owner (<=16 orbital
and <=128 auxiliary AOs). UHF lacks a complete native Fock census and composed
DFT/DF admission is not qualified. Independent oracle support is broader.
No metric cutoff, auxiliary basis or arithmetic precision is a tuning variable.
"""

from __future__ import annotations

import copy
from typing import Any

DF_CONTROLS = {
    "GENERATIVEQC_DF_EXCHANGE": ("auto", "dense", "occupied"),
    "GENERATIVEQC_DF_FINAL_EXCHANGE": ("auto", "dense", "occupied"),
    "GENERATIVEQC_DF_VALUE_STORAGE": ("auto", "dense", "packed"),
    "GENERATIVEQC_DF_JK_SHARED_SOURCE": ("auto", "0", "1"),
    "GENERATIVEQC_DF_DERIVATIVE_MAPPING": ("auto", "serial"),
    "GENERATIVEQC_DF_RESPONSE_STORAGE": ("auto", "panel", "jk-scratch"),
    "GENERATIVEQC_DF_VALUE_MAPPING": ("auto", "component", "primitive"),
}


def pyscf_basis(name: str, representation: str) -> dict:
    """Translate immutable input tables, not native integrals, into libcint input.

    A reference package's named basis can have different rounded constants.
    Strict fitted-HF gates must compare the same Gaussian model, not absorb
    those input differences into a looser numerical tolerance.
    """
    from pyscf.data.elements import ELEMENTS

    from ._model_resolution import snapshot_basis
    from .basis import BasisSet

    basis = snapshot_basis(name, representation)
    if not isinstance(basis, BasisSet):
        raise TypeError("DF reference requires an immutable element basis record")
    return {
        ELEMENTS[element.atomic_number]: [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell.coefficients],
                    ]
                    for index, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in basis.elements
    }


def candidate_policies(*, full: bool) -> list[dict]:
    """Reuse exact native storage/contraction mappings with bounded fallbacks."""
    policies = [
        {"GENERATIVEQC_DF_EXCHANGE": "dense"},
        {"GENERATIVEQC_DF_FINAL_EXCHANGE": "dense"},
        {"GENERATIVEQC_DF_VALUE_STORAGE": "dense"},
        {"GENERATIVEQC_DF_VALUE_STORAGE": "packed"},
        {"GENERATIVEQC_DF_JK_SHARED_SOURCE": "0"},
        {"GENERATIVEQC_DF_DERIVATIVE_MAPPING": "serial"},
    ]
    if full:
        policies += [
            {"GENERATIVEQC_DF_EXCHANGE": "occupied"},
            {"GENERATIVEQC_DF_RESPONSE_STORAGE": "panel"},
            {"GENERATIVEQC_DF_VALUE_MAPPING": "primitive"},
        ]
    return policies


def reference_workload(workload: dict) -> dict:
    """Independent libcint fitted HF E/F, including auxiliary-basis response.

    Refuse truncated/near-singular metrics: a different pseudoinverse convention
    must not masquerade as schedule error or silently change the Hamiltonian.
    The full-rank restriction makes PySCF's Cholesky representation equivalent
    to the native retained metric subspace without substituting native values.
    """
    import numpy as np
    import pyscf
    from pyscf import df, gto, lib, scf

    if workload.get("kind") != "df-hf" or workload["method"] not in ("rhf", "uhf"):
        raise ValueError("DF reference requires a supported fitted HF workload")
    lib.num_threads(1)
    orbital_basis = pyscf_basis(workload["basis"], workload["representation"])
    auxiliary_basis = pyscf_basis(
        workload["auxiliary_basis"], workload["representation"]
    )
    molecule = gto.M(
        atom=workload["atoms"],
        basis=orbital_basis,
        unit="Bohr",
        charge=workload["charge"],
        spin=workload["multiplicity"] - 1,
        cart=workload["representation"] == "cartesian",
        verbose=0,
    )
    auxiliary = df.addons.make_auxmol(molecule, auxiliary_basis)
    eigenvalues = np.linalg.eigvalsh(auxiliary.intor("int2c2e", hermi=1))
    cutoff = max(1e-7, workload["metric_relative_threshold"] * float(eigenvalues[-1]))
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[0]) <= cutoff:
        raise ValueError(
            "DF autotuning currently requires an independently full-rank metric"
        )
    solver = (scf.RHF if workload["method"] == "rhf" else scf.UHF)(
        molecule
    ).density_fit(auxbasis=auxiliary_basis)
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-12, 1e-10, 150
    solver.direct_scf_tol = 1e-14
    solver.kernel()
    if not solver.converged:
        solver = solver.newton()
        solver.max_cycle = 150
        solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent fitted HF reference did not converge")
    gradient = solver.nuc_grad_method()
    gradient.auxbasis_response = True
    forces = -gradient.kernel()
    return {
        "independent_reference": "PySCF/libcint fitted HF energy and analytic gradient with auxiliary response",
        "reference_family": "df-hf",
        "basis_contract": "exact immutable Gaussian input tables; independent libcint integrals",
        "pyscf_version": pyscf.__version__,
        "energies": [float(solver.e_tot)] * workload["batch"],
        "forces": [forces.tolist() for _ in range(workload["batch"])],
        "metric": {
            "naux": int(eigenvalues.size),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "full_rank_floor": cutoff,
        },
    }


def numerical_gate(samples: list[dict], reference: dict) -> dict:
    """Fitted HF has no XC-grid error allowance: use strict absolute E/F gates."""
    from ._dft_autotune import numerical_gate as ks_gate

    record = ks_gate(samples, reference)
    error = record["maximum_energy_error_per_atom"] * len(reference["forces"][0])
    record.update(
        maximum_energy_error=error,
        passed=error <= 1e-9
        and record["maximum_force_error"] <= 1e-8
        and record["maximum_force_rms"] <= 3e-9,
        tolerances={"energy_abs": 1e-9, "force_max": 1e-8, "force_rms": 3e-9},
    )
    return record


def execution_record(workload: dict, prepared: Any, result: Any) -> dict:
    """Observe complete native DF Fock work, retained metric rank and memory."""
    metrics = [
        item.to_dict() for item in prepared.last_density_fitting_metric_diagnostics()
    ]
    focks = [item.fock_builds for item in result.items]
    resources = prepared.resource_diagnostics
    if len(metrics) != workload["batch"] or any(
        type(count) is not int or count < 1 for count in focks
    ):
        raise ValueError(
            "DF tuning requires observed metric and complete Fock diagnostics"
        )
    if resources is None:
        raise ValueError("DF tuning requires composed native resource observations")
    return {
        "endpoint_kind": "energy_force",
        "synchronized": True,
        "source_hash": workload["source_identity"],
        "df_work": {"fock_builds": focks, "metrics": metrics},
        "resources": copy.deepcopy(resources[resources["phase"]]["device_ledger"]),
    }


def work_gate(samples: list[dict], reference: dict) -> dict:
    """Require the same full metric subspace and observed Fock census per sample."""
    signatures, complete = [], bool(samples)
    for sample in samples:
        work = sample.get("df_work", {})
        focks, metrics = work.get("fock_builds", []), work.get("metrics", [])
        complete = complete and len(focks) == len(reference["energies"]) == len(metrics)
        complete = complete and all(type(count) is int and count > 0 for count in focks)
        complete = complete and all(
            item["effective_rank"] == reference["metric"]["naux"] for item in metrics
        )
        signatures.append(
            (
                focks,
                [
                    (
                        item["system_index"],
                        item["effective_rank"],
                        item["absolute_threshold"],
                    )
                    for item in metrics
                ],
            )
        )
    return {
        "passed": complete and all(item == signatures[0] for item in signatures),
        "scope": "same full-rank auxiliary metric subspace and observed complete Fock census",
    }
