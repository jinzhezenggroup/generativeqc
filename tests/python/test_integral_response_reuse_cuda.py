"""Complete Direct RSH endpoints with actual reuse and geometry/disable misses.

Independent molecular and reconverged energy-difference gates are supplied by
``test_wb97mv_complete_cuda.py``; this same-source comparison isolates retention.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("unrestricted", [False, True])
def test_exact_two_electron_response_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unrestricted: bool
) -> None:
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly scheduled GPU")
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator, GridSpec, KsOptions

    atoms = [("O", (0.02, -0.03, 0.04)), ("H", (0.1, 1.43, 1.1))]
    if not unrestricted:
        atoms.append(("H", (-0.15, -1.45, 1.12)))
    coordinates = np.array([[xyz for _, xyz in atoms]])
    moved = coordinates.copy()
    moved[0, -1, 0] += 0.001
    outputs = []
    for enabled in (False, True):
        monkeypatch.delenv(
            "GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK", raising=False
        )
        journal = tmp_path / f"response-{unrestricted}-{enabled}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
        monkeypatch.setenv("GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE", str(int(enabled)))
        calculator = Calculator(
            method="wb97m-v-uks" if unrestricted else "wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
            ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
        )
        results = []
        with calculator.prepare_batch(
            [atoms], multiplicities=[2 if unrestricted else 1], warm_start=True
        ) as batch:
            for geometry in (None, moved):
                batch.set_warm_start_updates(True)
                results.append(
                    batch.execute(
                        geometry, strict=True, properties=("energy", "forces")
                    ).items[0]
                )
                batch.set_warm_start_updates(False)
                for _ in range(2):
                    results.append(
                        batch.execute(
                            geometry, strict=True, properties=("energy", "forces")
                        ).items[0]
                    )
            # Explicit derivative routing keeps its own observable execution.
            monkeypatch.setenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK", "1")
            results.append(
                batch.execute(
                    moved, strict=True, properties=("energy", "forces")
                ).items[0]
            )
            monkeypatch.delenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK")
            # Policy disable must retire retention even for the same density.
            monkeypatch.setenv("GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE", "0")
            results.append(
                batch.execute(
                    moved, strict=True, properties=("energy", "forces")
                ).items[0]
            )
        assert all(item.converged for item in results)
        outputs.append(results)
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        work = [
            event["value"]
            for event in events
            if event.get("name") == "ks_integral_response"
            and event.get("key") == "two_electron_response_evaluations"
        ]
        assert len(work) == 8
        assert work[0] == work[3] == work[6] == work[-1] == 1
        if enabled:
            assert work[2] == work[5] == 0
        else:
            assert work == [1] * 8
    for original, reused in zip(*outputs, strict=True):
        assert abs(original.energy - reused.energy) <= 1e-8
        np.testing.assert_allclose(original.forces, reused.forces, rtol=0, atol=1e-7)
