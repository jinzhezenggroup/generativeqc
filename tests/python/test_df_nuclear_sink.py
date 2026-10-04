"""Device source-response to all-center nuclear sink, with independent energy FDs."""

from __future__ import annotations

import copy
import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Primitive, Shell
from test_df_native_g_derivatives import g_fixture
from test_df_source_metric_response import embed, root

from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_validation.df_gradient import reference_df_matrices

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_NUCLEAR_SINK_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and native sink probe",
)


@pytest.fixture(scope="module")
def probe() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    library = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_DF_NUCLEAR_SINK_PROBE"]).resolve())
    )
    function = library.df_nuclear_sink_probe
    dp = ct.POINTER(ct.c_double)
    function.argtypes = [
        ct.c_void_p,
        dp,
        ct.c_size_t,
        ct.POINTER(dp),
        ct.c_size_t,
        ct.c_int,
        dp,
        dp,
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    function.restype = ct.c_int
    return function


def arguments(inputs: dict) -> tuple:
    return tuple(
        Shell(
            s["atom_index"],
            s["angular_momentum"],
            tuple(Primitive(*p) for p in s["primitives"]),
        )
        for s in inputs["shells"]
    )


def run(
    probe: typing.Any,
    orbital: dict,
    auxiliary: dict,
    c: np.ndarray,
    seeds: list[np.ndarray],
    *,
    budget: int = 1 << 30,
    failure: int = 0,
) -> tuple:
    """Create only normalized metadata on the host; never call NativeSource.read."""
    dp = ct.POINTER(ct.c_double)
    pointers = (dp * 3)(*(s.ctypes.data_as(dp) for s in seeds))
    gradient = np.full((len(orbital["atomic_numbers"]), 3), 12345.0)
    frame = np.full_like(c, 12345.0)
    counts, error = (ct.c_size_t * 10)(), ct.create_string_buffer(1024)
    atoms = list(zip(orbital["atomic_numbers"], orbital["coordinates"], strict=True))
    with NativeSource(
        atoms,
        arguments(orbital),
        auxiliary_basis=arguments(auxiliary),
        representation=orbital["basis_representation"],
    ) as source:
        status = probe(
            source._handle,
            c.ctypes.data_as(dp),
            seeds[0].shape[1],
            pointers,
            budget,
            failure,
            gradient.ctypes.data_as(dp),
            frame.ctypes.data_as(dp),
            counts,
            error,
            len(error),
        )
    return status, error.value.decode(), gradient, frame, list(counts)


def fixture(representation: str, auxiliary_g: bool) -> tuple:
    orbital, auxiliary = g_fixture(representation, representation, False)
    if not auxiliary_g:
        for shell in auxiliary["shells"]:
            if shell["angular_momentum"] == 4:
                shell["angular_momentum"] = 3
    a, m, _, _ = reference_df_matrices(orbital, auxiliary)
    n, q, o = a.shape[0], m.shape[0], 2
    rng = np.random.default_rng(1765)
    c = np.ascontiguousarray(np.linalg.qr(rng.normal(size=(n, n)))[0])
    seeds = [
        rng.normal(scale=0.02, size=shape)
        for shape in ((q, o, o), (q, o, n - o), (q, n - o, n - o))
    ]
    return orbital, auxiliary, c, seeds, a, m


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("auxiliary_g", (False, True))
def test_source_sink_nuclear_directions_and_work_counts(
    probe: typing.Any,
    representation: str,
    auxiliary_g: bool,
) -> None:
    orbital, auxiliary, c, seeds, a, m = fixture(representation, auxiliary_g)
    status, error, gradient, frame, counts = run(
        probe, orbital, auxiliary, c, seeds, failure=4
    )
    assert status == 0, error
    n, q, o = c.shape[0], m.shape[0], seeds[0].shape[1]
    b = embed(seeds, o, n)
    w = root(m)
    expected_c = np.einsum("pqQ,nq,mnP,PQ->mp", b, c, a, w, optimize=True)
    expected_c += np.einsum("qpQ,nq,nmP,PQ->mp", b, c, a, w, optimize=True)
    np.testing.assert_allclose(frame, expected_c, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-10, rtol=0)
    assert counts[0] <= 1 << 30 and counts[1] < counts[0]
    assert counts[2:4] == [n * n * q, q * q]
    assert counts[4] > 0 and counts[5] == 0
    assert counts[6] == gradient.nbytes
    assert counts[7] == (n * n * q + q * q) * 8
    assert counts[8] == n + 1 and counts[9] == 2  # setup and final publication
    rng = np.random.default_rng(1805)
    for direction in (rng.normal(scale=0.2, size=gradient.shape), np.eye(3)):
        analytic = float(np.sum(direction * gradient))
        for step in (1e-4, 3e-5):
            values = []
            for sign in (-1, 1):
                oi, xi = copy.deepcopy(orbital), copy.deepcopy(auxiliary)
                for inputs in (oi, xi):
                    inputs["coordinates"] = (
                        np.asarray(inputs["coordinates"]) + sign * step * direction
                    ).tolist()
                av, mv, _, _ = reference_df_matrices(oi, xi)
                values.append(
                    np.einsum(
                        "pqQ,mp,nq,mnP,PQ->", b, c, c, av, root(mv), optimize=True
                    )
                )
            np.testing.assert_allclose(
                (values[1] - values[0]) / (2 * step), analytic, atol=3e-8, rtol=3e-7
            )


@pytest.mark.parametrize("failure", (1, 2, 3, 5))
def test_callback_failure_never_publishes_and_fresh_sink_recovers(
    probe: typing.Any, failure: int
) -> None:
    orbital, auxiliary, c, seeds, _, _ = fixture("spherical", True)
    status, error, gradient, frame, _ = run(
        probe, orbital, auxiliary, c, seeds, failure=failure
    )
    assert status != 0 and ("injected" in error or "stream changed" in error), error
    np.testing.assert_array_equal(gradient, 12345.0)
    np.testing.assert_array_equal(frame, 12345.0)
    status, error, _, _, _ = run(probe, orbital, auxiliary, c, seeds)
    assert status == 0, error


def test_sink_complete_budget_rejection_is_transactional(probe: typing.Any) -> None:
    orbital, auxiliary, c, seeds, _, _ = fixture("spherical", True)
    status, error, gradient, frame, _ = run(
        probe, orbital, auxiliary, c, seeds, budget=64
    )
    assert status != 0 and "budget" in error, error
    np.testing.assert_array_equal(gradient, 12345.0)
    np.testing.assert_array_equal(frame, 12345.0)


def test_reverse_budget_charges_sink_and_exact_limit_passes(probe: typing.Any) -> None:
    """The reverse-call budget includes the live sink, source and detached frame.

    Forward source creation has its separate allowance in this validation probe;
    these checks therefore qualify reverse admission, not a cold endpoint peak.
    """
    orbital, auxiliary, c, seeds, _, _ = fixture("spherical", True)
    status, error, first, _, counts = run(probe, orbital, auxiliary, c, seeds)
    assert status == 0, error
    status, error, exact, _, _ = run(
        probe, orbital, auxiliary, c, seeds, budget=counts[0]
    )
    assert status == 0, error
    np.testing.assert_allclose(exact, first, atol=2e-10, rtol=2e-11)
    status, error, gradient, frame, _ = run(
        probe, orbital, auxiliary, c, seeds, budget=counts[0] - 1
    )
    assert status != 0 and "budget" in error, error
    np.testing.assert_array_equal(gradient, 12345.0)
    np.testing.assert_array_equal(frame, 12345.0)
