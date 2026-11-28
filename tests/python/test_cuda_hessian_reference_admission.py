"""Real-device admission, fallback and lifetime for the dense Hessian SCF source."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions, ResourceBudget
from generativeqc._hessian_resources import plan_cuda_hessian
from generativeqc._model_resolution import _named_basis_record
from generativeqc._native_hessian import _Options

from benchmarks.water_hessian_gpu4pyscf import water_cluster

if TYPE_CHECKING:
    from pathlib import Path


def test_cold_hessian_source_is_bounded_and_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only dense Hessian SCF may borrow the extra budget; release it before response."""
    if os.environ.get("GENERATIVEQC_TEST_CUDA_HESSIAN_REFERENCE") != "1":
        pytest.skip("requires a wide Hessian build and a Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite srun"
    monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_VALUES", raising=False)
    monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", raising=False)
    atoms = tuple(Atom.from_value(item) for item in water_cluster(32))
    budget = ResourceBudget(device_bytes=28 << 30, device_reserve_bytes=1 << 30)
    calc = Calculator(
        method="r2scan-rks",
        device="cuda",
        precision="fp64",
        basis=_named_basis_record("sto-3g", "cartesian"),
        basis_representation="cartesian",
        density_fitting="none",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
        resource_budget=budget,
        max_iterations=400,
        energy_tolerance=1e-13,
        density_tolerance=1e-9,
    )
    options = _Options(
        device_bytes=20 << 30, grid_bytes=8 << 30, integral_bytes=6 << 30
    )

    def solve(label: str, dense: bool) -> tuple[float, int, list[str]]:
        trace = tmp_path / f"{label}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
        controls = {}
        if dense:
            plan, request, _ = plan_cuda_hessian(
                calc,
                atoms,
                charge=0,
                multiplicity=1,
                dense=True,
                budget=budget,
                options=options,
            )
            controls.update(resource_plan=plan, _hessian_request=request)
        with calc.prepare_batch(
            [atoms], charges=[0], multiplicities=[1], warm_start=False, **controls
        ) as batch:
            result = batch.execute(strict=True, properties=("energy",))
            live_bytes = batch._resource_ledger.native_observation()["device_ledger"][
                "live_bytes"
            ]
            energy = result.items[0].energy
        statuses = [
            row["status"]
            for line in trace.read_text().splitlines()
            if (row := json.loads(line))["name"] == "cuda_ks_resident_values_prepare"
            and row["event"] == "END"
        ]
        return energy, live_bytes, statuses

    accelerated_energy, accelerated_live, admitted = solve("dense", True)
    monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1")
    fallback_energy, fallback_live, refused = solve("refused", True)
    monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES")
    ordinary_energy, _, ordinary_status = solve("ordinary", False)
    assert admitted == ["accepted"]
    assert refused == ["refused"]
    assert ordinary_status == []
    assert accelerated_live == fallback_live
    assert accelerated_energy == pytest.approx(fallback_energy, abs=1e-9, rel=0)
    assert ordinary_energy == pytest.approx(fallback_energy, abs=1e-9, rel=0)
