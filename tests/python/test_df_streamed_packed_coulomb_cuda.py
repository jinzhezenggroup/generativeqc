"""Lower-pair source replay preserves independent raw/J oracles and rank gates."""

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation.df_gradient import reference_df_matrices
from tools.validate_df_source import fixture_systems, write_input

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("mapping", ["auto", "auxiliary", "component", "primitive"])
def test_packed_streamed_raw_and_nonsymmetric_coulomb(
    tmp_path: Path, representation: str, dependent: bool, mapping: str
) -> None:
    """Different batch layouts, auxiliary g, ragged panels and signed D.

    The dense fallback is checked against the same independent libcint/NumPy
    result. Automatic metric and transformed-value schedules may differ, so
    qualify auto independently of every explicit control. Work counts measure
    raw values produced, not kernel call spelling.
    """
    assert os.environ.get("SLURM_JOB_ID")
    probe = os.environ.get("GENERATIVEQC_DF_SOURCE_PROBE")
    assert probe, "build tests/native/df_value_probe.cpp against the tested library"
    systems = fixture_systems(representation)
    for _, auxiliary in systems:
        auxiliary["shells"].append(
            {
                "atom_index": len(auxiliary["coordinates"]) - 1,
                "angular_momentum": 4,
                "primitives": [[0.71, 0.8], [1.92, -0.13]],
            }
        )
        if dependent:
            index = next(
                i
                for i, shell in enumerate(auxiliary["shells"])
                if shell["angular_momentum"] == 0
            )
            auxiliary["shells"].insert(index, copy.deepcopy(auxiliary["shells"][index]))
    inputs = tmp_path / "systems.txt"
    write_input(inputs, systems)
    raw, metrics, expected_j, ranks = [], [], [], []
    for item, (orbital, auxiliary) in enumerate(systems):
        values, metric, _, _ = reference_df_matrices(orbital, auxiliary)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        keep = eigenvalues > eigenvalues.max() * 1e-12
        n = values.shape[0]
        density = np.fromfunction(
            lambda i, j, item=item: (
                np.where(i == j, 0.3, 0.01 / (1 + abs(i - j))) * (item + 1)
                + np.sin(1 + 3 * i + 5 * j + item) / 50
            ),
            (n, n),
        )
        # Apply the pseudoinverse to raw charge independently of the native
        # full-rank eigen-factor and rank-reduced symmetric-root schedules.
        charge = np.einsum("ijp,ij->p", values, density)
        potential = eigenvectors[:, keep] @ (
            (eigenvectors[:, keep].T @ charge) / eigenvalues[keep]
        )
        raw.append(values)
        metrics.append(metric)
        expected_j.append(np.einsum("ijp,p->ij", values, potential))
        ranks.append(int(keep.sum()))
    n, a = raw[0].shape[0], raw[0].shape[-1]
    for packed in (False, True):
        prefix = tmp_path / str(packed)
        trace = tmp_path / f"{packed}.jsonl"
        run = subprocess.run(
            [probe, str(inputs), str(prefix), "7", "3", "--packed-values"],
            env={
                **os.environ,
                "GENERATIVEQC_DF_VALUE_MAPPING": mapping,
                "GENERATIVEQC_DF_STREAMED_COULOMB_PACKED": str(int(packed)),
                "GENERATIVEQC_DF_TRACE": str(trace),
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=240,
        )
        report = json.loads(run.stdout)
        assert [entry["rank"] for entry in report["metrics"]] == ranks
        assert all(rank < a for rank in ranks) == dependent
        for key, values in (
            ("raw", raw),
            ("metric", metrics),
            ("j", expected_j),
            ("uj", expected_j),
        ):
            expected = np.asarray(values)
            actual = np.fromfile(f"{prefix}-{key}.bin").reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        coulomb = [row for row in records if row["operation"] == "ri_j"]
        assert coulomb
        pairs = n * (n + 1) // 2 if packed else n * n
        for row in coulomb:
            assert row["streamed"]
            # Each independent J builds charge and potential from one raw pass
            # each, regardless of auxiliary tiling, metric rank or AO mapping.
            assert (
                row["counters"]["raw_value_bytes"] == 2 * len(systems) * pairs * a * 8
            )
