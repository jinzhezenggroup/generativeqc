"""Default HF force publication needs stronger stationarity than energy alone."""

from __future__ import annotations

import ctypes
import importlib.abc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU and independent PySCF",
)

_ORACLE = """
import json, sys
from dataclasses import replace
import numpy as np
import pyscf
from pyscf import df, gto, lib, scf
lib.num_threads(1)
request = json.load(sys.stdin)

def solve(atoms):
    molecule = gto.M(atom=atoms, unit='Bohr', basis=request['basis'],
                     spin=request['spin'], cart=True, verbose=0)
    solver = scf.UHF(molecule) if request['spin'] else scf.RHF(molecule)
    if request['mode'] == 'df':
        solver = solver.density_fit(auxbasis='def2-universal-jkfit')
        auxiliary = df.addons.make_auxmol(molecule, 'def2-universal-jkfit')
        # Relative metric truncation is gauge dependent. Native Cartesian
        # components have unit self-overlap; PySCF's raw Cartesian ones do not.
        scale = 1 / np.sqrt(np.diag(auxiliary.intor('int1e_ovlp_cart')))
        metric = scale[:,None] * auxiliary.intor('int2c2e_cart', hermi=1) * scale[None,:]
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        keep = eigenvalues > eigenvalues[-1] * 1e-10
        three = df.incore.aux_e2(molecule, auxiliary, intor='int3c2e_cart', aosym='s2ij')
        solver.with_df.auxmol = auxiliary
        solver.with_df._cderi = np.ascontiguousarray(
            (eigenvectors[:,keep] / np.sqrt(eigenvalues[keep])).T @ (scale[:,None] * three.T)
        )
        solver.oracle_metric_rank = int(np.sum(keep))
    solver.conv_tol = 1e-13 if request['mode'] == 'df' else 1e-12
    solver.conv_tol_grad = 1e-10 if request['mode'] == 'df' else 1e-9
    solver.max_cycle = 200
    solver.kernel()
    assert solver.converged
    return solver

def finite_difference_forces(atoms, step):
    forces = np.empty((len(atoms), 3))
    samples = []
    coordinates = np.array([position for _, position in atoms], dtype=float)
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = {}
            ranks = []
            for shift in (-2, -1, 1, 2):
                displaced = coordinates.copy()
                displaced[atom,axis] += shift * step
                geometry = [(symbol, position) for (symbol, _), position in zip(atoms, displaced)]
                displaced_solver = solve(geometry)
                energies[shift] = displaced_solver.e_tot
                ranks.append(displaced_solver.oracle_metric_rank)
            forces[atom,axis] = (energies[2] - 8*energies[1] + 8*energies[-1] - energies[-2]) / (12*step)
            samples.append({'atom': atom, 'axis': axis, 'energies': energies, 'ranks': ranks})
    return forces, samples

records = {}
for name, atoms in request['geometries'].items():
    solver = solve(atoms)
    records[name] = {'energy': solver.e_tot}
    if request['mode'] == 'df':
        # Reconverged displaced energies include both the metric projector and
        # auxiliary-center response; a full-rank analytic DF gradient does not.
        fine, fine_samples = finite_difference_forces(atoms, 0.01)
        coarse, coarse_samples = finite_difference_forces(atoms, 0.02)
        assert all(rank == solver.oracle_metric_rank
                   for sample in fine_samples + coarse_samples for rank in sample['ranks'])
        difference = float(np.max(np.abs(fine - coarse)))
        assert difference <= 5e-8, difference
        forces = (16*fine - coarse) / 15
        records[name].update(metric_rank=solver.oracle_metric_rank,
                             fd_steps=[0.01, 0.02], fd_step_difference=difference,
                             fd_samples=[fine_samples, coarse_samples])
    else:
        forces = -solver.nuc_grad_method().kernel()
    records[name]['forces'] = forces.tolist()
result = {'pyscf_version': pyscf.__version__, 'records': records}
if request['mode'] == 'df':
    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot
    # The shared exporter defaults to spherical, whereas PySCF inherits cart.
    auxiliary = replace(
        basis_snapshot('def2-universal-jkfit', sorted(set(solver.mol.atom_charges()))),
        representation='cartesian' if solver.with_df.auxmol.cart else 'spherical',
    )
    result['auxiliary'] = auxiliary.to_payload()
    result['auxiliary_representation'] = auxiliary.representation
    result['metric_relative_threshold'] = 1e-10
    result['metric_normalization'] = 'unit self-overlap Cartesian components'
json.dump(result, sys.stdout)
"""


class _NoProductionOracle(importlib.abc.MetaPathFinder):
    """Reference work belongs to the independent subprocess, never the endpoint."""

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise AssertionError("production HF must not import its independent oracle")


@pytest.mark.parametrize(
    "method,basis,mode",
    [
        (method, basis, mode)
        for basis in ("def2-svp", "def2-tzvp")
        for method, mode in (("rhf", "direct"), ("uhf", "direct"), ("rhf", "df"))
    ],
)
def test_default_force_stationarity_against_independent_oracle(
    method: str,
    basis: str,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the #3092 gates for cold/frozen/moved and energy-to-force reuse.

    Both AOT profiles run this same test. Public controls remain omitted, and
    oracle imports are forbidden during production. The fitted cases compare
    the same explicit auxiliary basis, including its analytic response.
    """
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    monkeypatch.delenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", raising=False)
    if method == "rhf":
        initial = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (1.45, 0.0, 1.1)),
            ("H", (-1.43, 0.1, 1.08)),
        ]
        moved = [initial[0], ("H", (1.45, 0.07, 1.1)), ("H", (-1.43, 0.1, 1.04))]
    else:
        initial = [("O", (0.0, 0.0, 0.0)), ("H", (0.2, 0.1, 1.8))]
        moved = [initial[0], ("H", (0.23, 0.08, 1.84))]
    request = {
        "geometries": {"initial": initial, "moved": moved},
        "basis": basis,
        "spin": int(method == "uhf"),
        "mode": mode,
    }
    oracle_process = subprocess.run(
        [sys.executable, "-c", _ORACLE],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    reference = json.loads(oracle_process.stdout)
    blocker = _NoProductionOracle()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    from generativeqc import Calculator, _native, profiles

    settings = {"density_fitting": "none"}
    if mode == "df":
        auxiliary = tmp_path / "auxiliary.json"
        auxiliary.write_text(json.dumps(reference["auxiliary"]) + "\n")
        settings = {"density_fitting": "cuda", "auxiliary_basis": auxiliary}
    calculator = Calculator(
        method=method,
        basis=basis,
        basis_representation="cartesian",
        device="cuda",
        **settings,
    )
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    device_uuid = (ctypes.c_ubyte * 16)()
    assert driver.cuInit(0) == 0
    assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0
    assert driver.cuDeviceGetUuid_v2(ctypes.byref(device_uuid), device) == 0
    assert calculator._energy_tolerance == 1e-9
    assert calculator._density_tolerance == 1e-5
    assert calculator._screening_tolerance == 1e-12
    receipt = {
        "method": method,
        "basis": basis,
        "mode": mode,
        "oracle": reference,
        "request": request,
        "public_controls": {"energy": 1e-9, "density": 1e-5, "screening": 1e-12},
        "native_probe": profiles.probe_device(
            _native.load_library(device="cuda", device_id=0)
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "device_uuid": bytes(device_uuid).hex(),
        "samples": [],
    }
    try:
        with calculator.prepare_batch(
            [initial], multiplicities=[2 if method == "uhf" else 1], warm_start=True
        ) as owner:
            for phase, geometry, forces, updates in (
                ("cold", "initial", True, True),
                ("warm", "initial", True, False),
                ("moved", "moved", True, True),
                ("moved_warm", "moved", True, False),
                ("energy_only", "initial", False, True),
                ("energy_to_forces", "initial", True, True),
            ):
                owner.set_warm_start_updates(updates)
                coordinates = [
                    position for _, position in request["geometries"][geometry]
                ]
                result = owner.execute(
                    [coordinates],
                    strict=True,
                    properties=("energy", "forces") if forces else ("energy",),
                ).items[0]
                expected = reference["records"][geometry]
                errors = {"energy": abs(result.energy - expected["energy"])}
                if forces:
                    errors["forces"] = float(
                        np.max(np.abs(result.forces - expected["forces"]))
                    )
                    errors["translation"] = float(
                        np.max(np.abs(np.sum(result.forces, axis=0)))
                    )
                receipt["samples"].append(
                    {
                        "phase": phase,
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "warm_start_used": result.warm_start_used,
                        "converged": result.converged,
                        "executed_backend": result.executed_backend,
                        "energy": result.energy,
                        "forces": result.forces.tolist() if forces else None,
                        "errors": errors,
                    }
                )
                assert result.converged and result.executed_backend == "cuda"
                if phase in ("warm", "moved_warm", "energy_to_forces"):
                    assert result.warm_start_used
                assert errors["energy"] <= 1e-9
                if forces:
                    assert errors["forces"] <= 1e-7
                    assert errors["translation"] <= 1e-7
                else:
                    assert result.forces is None
    finally:
        calculator.clear_cache()
        if destination := os.environ.get("GENERATIVEQC_SCF_ORACLE_RECEIPTS"):
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{method}-{mode}-{basis}.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
