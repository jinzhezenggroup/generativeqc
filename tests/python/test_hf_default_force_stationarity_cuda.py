"""Default HF force publication needs stronger stationarity than energy alone."""

from __future__ import annotations

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
import pyscf
from pyscf import gto, lib, scf
lib.num_threads(1)
request = json.load(sys.stdin)
records = {}
for name, atoms in request['geometries'].items():
    molecule = gto.M(atom=atoms, unit='Bohr', basis=request['basis'],
                     spin=request['spin'], cart=True, verbose=0)
    solver = scf.UHF(molecule) if request['spin'] else scf.RHF(molecule)
    if request['mode'] == 'df':
        solver = solver.density_fit(auxbasis='def2-universal-jkfit')
    solver.conv_tol = 1e-12
    solver.conv_tol_grad = 1e-9
    solver.max_cycle = 200
    energy = solver.kernel()
    assert solver.converged
    gradient = solver.nuc_grad_method()
    if request['mode'] == 'df':
        gradient.auxbasis_response = True
    records[name] = {'energy': energy, 'forces': (-gradient.kernel()).tolist()}
result = {'pyscf_version': pyscf.__version__, 'records': records}
if request['mode'] == 'df':
    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot
    result['auxiliary'] = basis_snapshot(
        'def2-universal-jkfit', sorted(set(molecule.atom_charges()))
    ).to_payload()
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
                        "energy": result.energy,
                        "forces": result.forces.tolist() if forces else None,
                        "errors": errors,
                    }
                )
                assert result.converged and result.executed_backend == "cuda"
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
