"""Independent current-domain force gates for native and compatibility owners."""

from __future__ import annotations

import gc
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module", params=[False, True], ids=["rks", "uks"])
def response_oracles(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Reconverge both geometries independently at the current VV10 cutoff."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly scheduled GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import cupy as cp
    from generativeqc import GridSpec
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )

    from benchmarks.readme_wb97mv import (
        reference_engine,
        reference_sample,
        reference_vv10_domain,
    )

    unrestricted = request.param
    atoms = (
        [
            ("N", (0.02, -0.03, 0.04)),
            ("H", (0.1, 1.51, 1.12)),
            ("H", (-0.15, -1.55, 1.13)),
        ]
        if unrestricted
        else [
            ("O", (0.02, -0.03, 0.04)),
            ("H", (0.1, 1.43, 1.1)),
            ("H", (-0.15, -1.45, 1.12)),
        ]
    )
    coordinates = np.asarray([position for _, position in atoms])
    moved = coordinates.copy()
    moved[-1, 0] += 0.001
    references = []
    for geometry in (coordinates, moved):
        geometry_atoms = [
            (symbol, tuple(position))
            for (symbol, _), position in zip(atoms, geometry, strict=True)
        ]
        with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
            references.append(
                reference_sample(
                    reference_engine(
                        geometry_atoms,
                        "def2-svp",
                        GridSpec(12, 4, 8),
                        spin=int(unrestricted),
                    ),
                    cp,
                )
            )
    # Oracle objects contain collectible engine cycles, not production state.
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "atoms": atoms,
        "unrestricted": unrestricted,
        "coordinates": coordinates,
        "moved": moved,
        "references": references,
        "density_threshold": float(MOLECULAR_VV10_DENSITY_THRESHOLD),
    }


@pytest.mark.parametrize("force_route", ["native", "compatibility"])
def test_response_selectors_preserve_independent_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response_oracles: dict[str, Any],
    force_route: str,
) -> None:
    """Qualify physical results without mistaking fallback hits for native work."""
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc.batch import PreparedBatch

    if force_route == "compatibility":
        monkeypatch.setattr(
            PreparedBatch, "_native_dft_force_eligible", lambda *_: False
        )
    else:

        def forbid_compatibility(*_: Any, **__: Any) -> Any:
            pytest.fail("native prepared-force capability routed through compatibility")

        monkeypatch.setattr(
            PreparedBatch, "_public_dft_cuda_force", forbid_compatibility
        )
    trace = tmp_path / "native-trace.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    evidence: list[dict[str, Any]] = []
    unrestricted = response_oracles["unrestricted"]
    try:
        for integral, geometry in (
            (None, None),
            ("0", "0"),
            ("1", "0"),
            ("0", "1"),
            ("1", "1"),
        ):
            for selector, value in (
                ("GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE", integral),
                ("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", geometry),
            ):
                if value is None:
                    monkeypatch.delenv(selector, raising=False)
                else:
                    monkeypatch.setenv(selector, value)
            calculator = Calculator(
                method="wb97m-v-uks" if unrestricted else "wb97m-v",
                basis="def2-svp",
                basis_representation="spherical",
                device="cuda",
                precision="fp64",
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-14,
                max_iterations=200,
                ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
            )
            try:
                with calculator.prepare_batch(
                    [response_oracles["atoms"]],
                    multiplicities=[2 if unrestricted else 1],
                    warm_start=True,
                ) as batch:
                    if force_route == "native":
                        assert batch._native_dft_force_eligible(0)
                    for geometry_index, coordinates in enumerate(
                        (response_oracles["coordinates"], response_oracles["moved"])
                    ):
                        reference = response_oracles["references"][geometry_index]
                        batch.set_warm_start_updates(True)
                        for repeat in range(3):
                            if repeat:
                                batch.set_warm_start_updates(False)
                            trace.write_text("")
                            result = batch.execute([coordinates], strict=True).items[0]
                            assert result.converged
                            if force_route == "native":
                                events = [
                                    json.loads(line)
                                    for line in trace.read_text().splitlines()
                                ]
                                work = [
                                    event["counters"]
                                    for event in events
                                    if event["operation"] == "stationary_force"
                                ][-1]
                                assert work["grid_evaluation_points"] > 0
                                assert (
                                    work["nonlocal_pair_cache_hits"]
                                    + work["nonlocal_pair_sweeps"]
                                    == 1
                                )
                            else:
                                work = batch._stationary_cuda_execution.last_work
                            evidence.append(
                                {
                                    "integral_policy": integral,
                                    "geometry_policy": geometry,
                                    "geometry": geometry_index,
                                    "repeat": repeat,
                                    "energy": result.energy,
                                    "forces": result.forces.tolist(),
                                    "energy_error": abs(
                                        result.energy - reference["energies_hartree"][0]
                                    ),
                                    "force_max_error": float(
                                        np.max(
                                            np.abs(
                                                result.forces
                                                - reference["forces_hartree_per_bohr"][
                                                    0
                                                ]
                                            )
                                        )
                                    ),
                                    "work": work,
                                }
                            )
                            assert evidence[-1]["energy_error"] < 1e-8
                            np.testing.assert_allclose(
                                result.forces,
                                reference["forces_hartree_per_bohr"][0],
                                atol=1e-7,
                                rtol=0,
                            )
                        if integral is None and geometry_index == 0:
                            direction = np.random.default_rng(13421389).normal(
                                size=coordinates.shape
                            )
                            direction -= direction.mean(axis=0)
                            direction /= np.linalg.norm(direction)
                            analytic = -np.vdot(result.forces, direction)
                            errors = []
                            for step in (1e-3, 3e-4, 1e-4):
                                energies = []
                                for sign in (-1, 1):
                                    displaced = coordinates + sign * step * direction
                                    displaced_result = batch.execute(
                                        [displaced], properties=("energy",), strict=True
                                    ).items[0]
                                    assert displaced_result.converged
                                    energies.append(displaced_result.energy)
                                errors.append(
                                    abs(
                                        (energies[1] - energies[0]) / (2 * step)
                                        - analytic
                                    )
                                )
                            assert errors[0] < 1e-5 and max(errors[1:]) < 2e-6, errors
                            evidence.append({"finite_difference_errors": errors})
            finally:
                calculator.clear_cache()
    finally:
        (tmp_path / "endpoint-evidence.json").write_text(
            json.dumps(
                {
                    "force_route": force_route,
                    "atoms": response_oracles["atoms"],
                    "reference_samples": response_oracles["references"],
                    "reference_scope": "independent accuracy-only, not benchmark populations",
                    "density_threshold": response_oracles["density_threshold"],
                    "slurm_job": os.environ["SLURM_JOB_ID"],
                    "observations": evidence,
                },
                indent=2,
            )
            + "\n"
        )
