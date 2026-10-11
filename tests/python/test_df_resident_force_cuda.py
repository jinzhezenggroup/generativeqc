"""Complete native DF E+F baseline/resident binding with independent physical FDs."""

from __future__ import annotations

import copy
import os
import typing

import numpy as np
import pytest
from test_df_complete_force import independent, run
from test_df_complete_force import probe as _native_probe

from tools.generativeqc_posthf.fixtures import load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_COMPLETE_FORCE_TEST") != "1",
    reason="requires finite Slurm allocation and compiled native complete DF endpoint",
)
probe = _native_probe


@pytest.mark.parametrize("budget", [512 << 20, 1024 << 20])
@pytest.mark.parametrize("triples", [False, True])
def test_complete_resident_df_endpoint_cold_warm_changed(
    probe: typing.Any,
    budget: int,
    triples: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm process calls still construct fresh logical primal/response owners."""
    assert os.environ.get("SLURM_JOB_ID")
    original, _ = load_fixture("h2")
    changed = copy.deepcopy(original)
    changed["inputs"]["coordinates"][1][2] += 0.007
    for metadata in (original, original, changed):
        monkeypatch.setenv("GENERATIVEQC_TEST_LAMBDA_RESIDENT_KRYLOV", "0")
        status, error, reference, baseline, base_work = run(
            probe, metadata, triples=triples, budget=budget
        )
        assert status == 0, error
        assert base_work[12] == 0
        monkeypatch.setenv("GENERATIVEQC_TEST_LAMBDA_RESIDENT_KRYLOV", "1")
        status, error, actual, values, work = run(
            probe, metadata, triples=triples, budget=budget
        )
        assert status == 0, error
        assert work[12] == 1 and 0 < work[13] <= budget
        assert 0 < work[18] <= work[17]
        assert work[14] > 0 and work[15] > 0 and work[16] > 0
        assert work[0] <= budget and np.max(values[4:8]) < 1e-8
        np.testing.assert_allclose(values[:4], baseline[:4], atol=3e-11, rtol=0)
        np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=0)
        np.testing.assert_allclose(
            values[:4], independent(metadata, triples), atol=3e-9, rtol=0
        )
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=3e-8, rtol=0)
        direction = np.random.default_rng(1764).normal(size=actual.shape)
        direction /= np.linalg.norm(direction)
        analytic = -float(np.sum(actual * direction))
        for step in (1e-4, 3e-5):
            energies = []
            for sign in (-1, 1):
                displaced = copy.deepcopy(metadata)
                displaced["inputs"]["coordinates"] = (
                    np.asarray(metadata["inputs"]["coordinates"])
                    + sign * step * direction
                ).tolist()
                energies.append(independent(displaced, triples)[0])
            np.testing.assert_allclose(
                (energies[1] - energies[0]) / (2 * step), analytic, atol=3e-7, rtol=3e-7
            )
