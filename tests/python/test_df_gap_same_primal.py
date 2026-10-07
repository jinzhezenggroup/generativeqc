"""Complete response controls on one native primal, with independent force FDs."""

from __future__ import annotations

import copy
import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_df_complete_force import independent

from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
from tools.generativeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_GAP_SAME_PRIMAL_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and same-primal native owner",
)


@pytest.fixture(scope="module")
def replay_probe() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    library = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_DF_COMPLETE_FORCE_PROBE"]).resolve())
    )
    count = library.df_gap_same_primal_fingerprint_count
    count.restype = ct.c_size_t
    assert count() == 35
    call = library.df_gap_same_primal_fingerprints_probe
    call.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_size_t),
        ct.POINTER(ct.c_uint64),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run_replay(probe: typing.Any, metadata: dict, budget: int = 1 << 30) -> tuple:
    """Sentinels expose partial publication after any comparison failure."""
    forces = np.full((4, len(metadata["inputs"]["atomic_numbers"]), 3), 12345.0)
    values = np.full((4, 7), 12345.0)
    shared = np.full(12, 12345, dtype=np.uint64)
    counts = np.full((4, 10), 12345, dtype=np.uintp)
    fingerprints = np.full((4, 35, 2), 12345, dtype=np.uint64)
    error = ct.create_string_buffer(2048)
    with NativeSource(**source_arguments(metadata)) as source:
        status = probe(
            source._handle,
            budget,
            forces.ctypes.data_as(ct.POINTER(ct.c_double)),
            values.ctypes.data_as(ct.POINTER(ct.c_double)),
            shared.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
            fingerprints.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            error,
            len(error),
        )
    return status, error.value.decode(), forces, values, shared, counts, fingerprints


def numeric_identity(values: np.ndarray) -> int:
    """Independently check the native length-prefixed, FP64-bit-pattern census."""
    identity = (14695981039346656037 ^ values.size) * 1099511628211 & ((1 << 64) - 1)
    for bits in np.asarray(values, dtype=np.float64).ravel().view(np.uint64):
        identity = (identity ^ int(bits)) * 1099511628211 & ((1 << 64) - 1)
    return identity


def test_same_primal_complete_force_and_copy_admission(
    replay_probe: typing.Any,
) -> None:
    metadata, _ = load_fixture("water")
    # Exercise parallel execution, not its v < 4 serial fallback.
    oxygen_end = next(
        index
        for index, shell in enumerate(metadata["inputs"]["shells"])
        if shell["atom_index"] != 0
    )
    metadata["inputs"]["shells"].insert(
        oxygen_end, {"atom_index": 0, "angular_momentum": 1, "primitives": [[0.4, 1.0]]}
    )
    status, error, forces, values, shared, counts, fingerprints = run_replay(
        replay_probe, metadata
    )
    assert status == 0, error
    assert shared[9] >= 4
    assert shared[0] > 0 and shared[1] > 0
    assert all(shared[index] != 0 for index in (5, 6, 7, 11))
    assert shared[3] == forces.nbytes
    assert shared[4] > shared[0] + shared[1] + shared[3]
    assert np.all(counts[:, 0] >= shared[4]) and np.all(counts[:, 0] <= 1 << 30)
    np.testing.assert_array_equal(counts[:, 1], [1, 1, 0, 1])
    np.testing.assert_array_equal(counts[:, 2], [0, 1, 0, 0])
    assert counts[0, 3] > counts[1, 3] > counts[2, 3] == 0
    assert counts[0, 5] > counts[1, 5] >= counts[2, 5] == 0
    np.testing.assert_array_equal(counts[:, 6:], np.broadcast_to(counts[0, 6:], (4, 4)))
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(forces))
    assert np.max(values[:, 2:4]) <= 1e-9
    assert np.max(values[:, 4]) <= 1e-8
    assert np.all(fingerprints[:, :, 0] != 0)
    assert np.all(fingerprints[:, :, 1] != 0)
    np.testing.assert_array_equal(
        fingerprints[:, :, 1], np.broadcast_to(fingerprints[0, :, 1], (4, 35))
    )
    for index, force in enumerate(forces):
        assert fingerprints[index, 34, 0] == numeric_identity(force)
        assert fingerprints[index, 34, 1] == force.size
    for serial in (0, 3):
        np.testing.assert_allclose(values[:, 0], values[serial, 0], atol=5e-11, rtol=0)
        np.testing.assert_allclose(
            forces, np.broadcast_to(forces[serial], forces.shape), atol=5e-10, rtol=0
        )
    expected = independent(metadata, True)
    np.testing.assert_allclose(values[:, 0], expected[0], atol=3e-9, rtol=0)
    np.testing.assert_allclose(values[:, 1], expected[3], atol=3e-10, rtol=0)
    direction = np.random.default_rng(1763).normal(size=forces.shape[1:])
    direction /= np.linalg.norm(direction)
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            displaced = copy.deepcopy(metadata)
            displaced["inputs"]["coordinates"] = (
                np.asarray(metadata["inputs"]["coordinates"]) + sign * step * direction
            ).tolist()
            energies.append(independent(displaced, True)[0])
        finite_difference = (energies[1] - energies[0]) / (2 * step)
        analytic = -np.einsum("cij,ij->c", forces, direction)
        np.testing.assert_allclose(analytic, finite_difference, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("budget", [0, 1])
def test_same_primal_refusal_does_not_publish(
    replay_probe: typing.Any, budget: int
) -> None:
    metadata, _ = load_fixture("water")
    status, error, forces, values, shared, counts, fingerprints = run_replay(
        replay_probe, metadata, budget
    )
    assert status != 0 and error
    assert np.all(forces == 12345) and np.all(values == 12345)
    assert np.all(shared == 12345) and np.all(counts == 12345)
    assert np.all(fingerprints == 12345)
