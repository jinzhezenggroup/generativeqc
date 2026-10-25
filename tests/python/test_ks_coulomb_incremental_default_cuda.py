"""Independent complete endpoints for the bounded cold pure-J default."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU qualification and independent GPU4PySCF",
)


@pytest.mark.parametrize("method,count", [("r2scan-3c", 48), ("pbe", 96)])
def test_default_coulomb_cold_replay_moved_and_explicit_off(
    method: str, count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep full closure, current-geometry ownership and independent E/F gates."""
    from benchmarks.readme_defaults import native_calculator, run_reference
    from benchmarks.readme_defaults_accuracy import errors
    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID") and "CUDA_VISIBLE_DEVICES" in os.environ
    atoms = scaling_cases()[f"water-{count}"].atoms
    moved = [(z, list(r)) for z, r in atoms]
    moved[-1][1][0] += 0.003
    refs = {}
    for label, geometry in [("original", atoms), ("moved", moved)]:
        samples = []
        run_reference(
            SimpleNamespace(
                arm="gpu4pyscf",
                method=method,
                mode="direct",
                inputs=tmp_path,
                warm_repeats=0,
            ),
            geometry,
            {},
            samples.append,
        )
        refs[label] = samples[0]
    outcomes = {}
    for choice in ["unset", "off"]:
        with monkeypatch.context() as scope:
            scope.delenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", raising=False)
            scope.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
            if choice == "off":
                scope.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "off")
            calculator = native_calculator(method, "standard", "direct", tmp_path)
            try:
                with calculator.prepare_batch([atoms]) as owner:
                    for phase, geometry in [
                        ("cold", atoms),
                        ("warm", atoms),
                        ("moved", moved),
                    ]:
                        result = owner.execute(
                            [np.array([r for _, r in geometry])],
                            strict=True,
                            properties=("energy", "forces"),
                        ).items[0]
                        work = result.incremental_direct_jk
                        assert result.converged and result.executed_backend == "cuda"
                        assert work["requested"] == (choice == "unset")
                        assert work["active"] == (choice == "unset" and phase == "cold")
                        if work["active"]:
                            assert work["delta_builds"] > 0
                            assert work["anchor_full_builds"] > 0
                            assert work["post_scf_full_builds"] > 0
                        if phase == "warm":
                            assert result.warm_start_used and work["delta_builds"] == 0
                        reference = refs["moved" if phase == "moved" else "original"]
                        assert errors(result.energy, result.forces, reference, count)[
                            "passed"
                        ]
                        outcomes[choice, phase] = (result.energy, result.forces.copy())
                        if phase == "cold":
                            owner.set_warm_start_updates(False)
            finally:
                calculator.clear_cache()
    for phase in ["cold", "warm", "moved"]:
        before, after = outcomes["off", phase], outcomes["unset", phase]
        assert abs(before[0] - after[0]) < 1e-8
        np.testing.assert_allclose(before[1], after[1], atol=5e-6, rtol=0)
