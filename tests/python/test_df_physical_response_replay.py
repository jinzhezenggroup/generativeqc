"""Bounded identical-seed physical branches, independently checked force FD."""

from __future__ import annotations

import copy
import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_df_complete_force import independent
from test_df_gap_same_primal import numeric_identity

from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
from tools.generativeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_GAP_SAME_PRIMAL_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and physical response replay owner",
)


@pytest.fixture(scope="module")
def physical_probe() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    library = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_DF_COMPLETE_FORCE_PROBE"]).resolve())
    )
    call = library.df_physical_response_replay_probe
    call.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        *([ct.POINTER(ct.c_double)] * 3),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_size_t),
        *([ct.POINTER(ct.c_uint64)] * 2),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run_physical(probe: typing.Any, metadata: dict, budget: int = 1 << 30) -> tuple:
    """Publish no buffers until all native response replays have succeeded."""
    shape = (len(metadata["inputs"]["atomic_numbers"]), 3)
    gradients = np.full((4, 2, *shape), 12345.0)
    forces = np.full(shape, 12345.0)
    values = np.full((4, 6), 12345.0)
    shared = np.full(9, 12345, dtype=np.uint64)
    counts = np.full((4, 8), 12345, dtype=np.uintp)
    weights = np.full((4, 2, 2), 12345, dtype=np.uint64)
    fingerprints = np.full((4, 35, 2), 12345, dtype=np.uint64)
    error = ct.create_string_buffer(2048)
    outputs = (gradients, forces, values, shared, counts, weights, fingerprints)
    with NativeSource(**source_arguments(metadata)) as source:
        status = probe(
            source._handle,
            budget,
            *[
                array.ctypes.data_as(pointer)
                for array, pointer in zip(outputs, probe.argtypes[2:9])
            ],
            error,
            len(error),
        )
    return status, error.value.decode(), outputs


def test_physical_replay_accounting_and_independent_force(
    physical_probe: typing.Any,
) -> None:
    metadata, _ = load_fixture("water")
    status, error, outputs = run_physical(physical_probe, metadata)
    assert status == 0, error
    gradients, forces, values, shared, counts, weights, fingerprints = outputs
    nocc, nvir, naux = map(int, shared[:3])
    nbf, coords = nocc + nvir, forces.size
    assert np.all(shared[3:7] != 0)
    assert shared[7] == gradients.nbytes
    assert shared[7] < shared[8] <= 1 << 30
    assert np.all((counts[:, 0] > shared[7]) & (counts[:, 0] <= shared[8]))
    assert np.all(counts[:, 1] == 8 * max(nbf * naux, naux * naux))
    assert np.all(counts[:, 2] == 8 * (nbf * nbf * naux + naux * naux))
    assert np.all(counts[:, 3] == nbf * nbf * naux)
    assert np.all(counts[:, 4] == naux * naux)
    np.testing.assert_array_equal(weights[:, :, 1], counts[:, 3:5])
    assert np.all(weights[:, :, 0] != 0)
    assert np.all(values[:, 1:4] > 0)
    assert np.all(values[:, 4] <= 1e-10)
    assert np.all(values[:, 5] <= 1e-9)
    assert np.all(fingerprints[:, :25] == 0)
    assert np.all(fingerprints[:, 34] == 0)
    expected_lengths = [
        nbf * nbf,
        coords,
        nocc * nvir,
        nocc * nvir,
        nbf * nbf,
        nbf * nbf,
        nbf * nbf,
        coords,
        nbf * nbf,
    ]
    np.testing.assert_array_equal(
        fingerprints[:, 25:34, 1], np.broadcast_to(expected_lengths, (4, 9))
    )
    assert np.all(counts[:, 5] == sum(expected_lengths))
    for index in range(4):
        assert fingerprints[index, 26, 0] == numeric_identity(gradients[index, 0])
        assert fingerprints[index, 32, 0] == numeric_identity(gradients[index, 1])
    for reference in (0, 3):
        np.testing.assert_allclose(
            gradients.sum(axis=1),
            np.broadcast_to(gradients[reference].sum(axis=0), (4, *forces.shape)),
            atol=5e-10,
            rtol=0,
        )
    expected = independent(metadata, True)
    np.testing.assert_allclose(values[:, 0], expected[0], atol=3e-9, rtol=0)
    direction = np.random.default_rng(1763).normal(size=forces.shape)
    direction /= np.linalg.norm(direction)
    energies = []
    for sign in (-1, 1):
        displaced = copy.deepcopy(metadata)
        displaced["inputs"]["coordinates"] = (
            np.asarray(metadata["inputs"]["coordinates"]) + sign * 3e-5 * direction
        ).tolist()
        energies.append(independent(displaced, True)[0])
    finite_difference = (energies[1] - energies[0]) / 6e-5
    np.testing.assert_allclose(
        -np.sum(forces * direction), finite_difference, atol=3e-7, rtol=3e-7
    )


@pytest.mark.parametrize("budget", [0, 1, 577])
def test_physical_replay_refusal_is_transactional(
    physical_probe: typing.Any, budget: int
) -> None:
    metadata, _ = load_fixture("water")
    status, error, outputs = run_physical(physical_probe, metadata, budget)
    assert status != 0 and error
    assert all(np.all(array == 12345) for array in outputs)
