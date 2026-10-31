"""Retained spectral factors preserve independent physical auxiliary adjoints."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm or LSF GPU allocation",
)


@pytest.fixture(scope="module")
def spectral_probe(tmp_path_factory: Any, native_cxx: NativeCxx) -> Any:
    assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    cuda = Path(os.environ["CUDA_PATH"])
    output = tmp_path_factory.mktemp("spectral-response") / "probe.so"
    native_cxx.build_shared(
        [ROOT / "tests/native/df_spectral_response_probe.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
        ),
        link_args=(
            str(library),
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-lcublas",
            "-Wl,-rpath," + str(library.parent),
            "-Wl,-rpath," + str(cuda / "lib64"),
        ),
    )
    call = ct.CDLL(str(output)).df_spectral_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = (
        [ct.c_size_t] * 5
        + [ct.c_uint, ct.c_bool, ct.c_bool]
        + [dp] * 11
        + [ct.c_void_p, ct.c_size_t]
    )
    call.restype = ct.c_int
    return call


@pytest.mark.parametrize("condition", (1e2, 1e6, 1e10))
@pytest.mark.parametrize("tile", (1, 3, 11))
@pytest.mark.parametrize(
    "mode,policy,compact",
    (
        (0, "auto", False),
        (1, "auto", False),
        (1, "spectral", False),
        (2, "auto", False),
        (1, "auto", True),
        (1, "spectral", True),
        (2, "auto", True),
    ),
    ids=(
        "coulomb",
        "occupied-root",
        "occupied-factor-order",
        "final-reuse",
        "compact-root",
        "compact-factor-order",
        "compact-final-reuse",
    ),
)
@pytest.mark.parametrize("gauge", ("symmetric", "spectral", "explicit"))
def test_rotated_metric_physical_adjoints(
    spectral_probe: Any,
    condition: float,
    tile: int,
    mode: int,
    policy: str,
    compact: bool,
    gauge: str,
    monkeypatch: Any,
) -> None:
    """Long-double independent J/K adjoints, signed terms and ragged panels.

    The random orthogonal Q is deliberately nonsymmetric: treating R as a
    symmetric root, or scaling the wrong auxiliary axis, fails these gates.
    The final-reuse arm consumes the same projected forward representation as K.
    """
    monkeypatch.setenv("GENERATIVEQC_DF_OCCUPIED_METRIC", policy)
    n, a, rank = 4, 11, 2
    terms = 1 if mode == 2 else 3
    rng = np.random.default_rng(2682)
    ld = np.longdouble
    q = np.linalg.qr(rng.normal(size=(a, a)))[0]
    eigenvalues = np.geomspace(1 / condition, 1, a)
    e = rng.normal(size=(n, n, a))
    e = (e + e.transpose(1, 0, 2)) / 2
    root = (q / np.sqrt(eigenvalues)) @ q.T
    spectral = gauge == "spectral"
    forward = e if spectral else e @ q.T
    explicit_root = None
    if gauge == "explicit":
        # Independently rotate the forward auxiliary gauge. The oracle below
        # still uses the original eigenfactors; it does not consume this W.
        rotation = np.linalg.qr(rng.normal(size=(a, a)))[0]
        forward = e @ rotation
        explicit_root = np.ascontiguousarray(
            ((rotation.T / np.sqrt(eigenvalues)) @ q.T).T
        )
    packed = np.ascontiguousarray(forward[np.tril_indices(n)])
    c = rng.normal(size=(terms, n, rank))
    density = 2 * np.einsum("tmi,tni->tmn", c, c)
    if mode == 0:
        # Pure J also accepts arbitrary nonsymmetric densities.
        density = rng.normal(size=(terms, n, n))
    coulomb = np.array([1.0, -0.37, 0.0])[:terms].copy()
    exchange = np.array([0.25, -0.13, 0.0])[:terms].copy()
    if mode == 0:
        exchange[:] = 0
    arrays = tuple(
        np.ascontiguousarray(x)
        for x in (
            packed,
            density,
            c.transpose(0, 2, 1),
            coulomb,
            exchange,
            q.T,
            eigenvalues,
            root.T,
        )
    )
    bar_a, bar_m = np.full((a, n, n), np.nan), np.full((a, a), np.nan)
    dp = ct.POINTER(ct.c_double)
    error = ct.create_string_buffer(1024)
    status = spectral_probe(
        n,
        a,
        terms,
        tile,
        rank,
        mode,
        spectral,
        compact,
        *(x.ctypes.data_as(dp) for x in arrays),
        explicit_root.ctypes.data_as(dp) if explicit_root is not None else None,
        bar_a.ctypes.data_as(dp),
        bar_m.ctypes.data_as(dp),
        error,
        len(error),
    )
    assert status == 0, error.value.decode()
    # A M^-1 from the independently chosen factor and metric, before conversion
    # to device double precision. This does not reuse the response implementation.
    fitted = (e.astype(ld) / np.sqrt(eigenvalues.astype(ld))) @ q.astype(ld).T
    potential = np.einsum("tij,ijp->tp", density.astype(ld), fitted)
    expected_a = np.einsum("t,tij,tp->pij", coulomb.astype(ld), density, potential)
    exchange_terms = np.einsum("tmi,mnp,tnj->tpij", density, fitted, density)
    expected_a -= 2 * np.einsum("t,tpij->pij", exchange, exchange_terms)
    expected_m = -np.einsum("pij,ijq->pq", expected_a, fitted) / 2
    np.testing.assert_allclose(bar_a, expected_a, atol=2e-8, rtol=3e-9)
    # Signed contributions may cancel; qualify against their absolute sum as
    # well as a small absolute floor, rather than the cancelled metric entry.
    scale = np.einsum("pij,ijq->pq", abs(expected_a), abs(fitted)) / 2
    assert np.all(abs(bar_m - expected_m) <= 2e-6 + 3e-9 * scale)
