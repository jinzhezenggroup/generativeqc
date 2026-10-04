"""Physical RHF/Z/Pulay CUDA closure against independent molecular derivatives."""

from __future__ import annotations

import copy
import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest

from tools.generate_validation_references import pyscf_molecule
from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
from tools.generativeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RHF_FRAME_CUDA_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and native response probe",
)


@pytest.fixture(scope="module")
def probe() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    library = ct.CDLL(str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve()))
    function = library.rhf_frame_response_probe
    dp = ct.POINTER(ct.c_double)
    function.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(dp),
        ct.c_bool,
        ct.c_bool,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(dp),
        ct.POINTER(ct.c_size_t),
        dp,
        ct.c_void_p,
        ct.c_size_t,
    ]
    function.restype = ct.c_int
    return function


def reference(metadata: dict) -> tuple:
    """Independent PySCF state in the native unit-normalized public AO frame."""
    from pyscf import scf

    mol, scale, _ = pyscf_molecule(metadata["inputs"])
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.direct_scf_tol = 0.0
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    c = mf.mo_coeff / scale[:, None]
    density = mf.make_rdm1() / (scale[:, None] * scale[None, :])
    matrices = [
        c,
        mf.get_hcore() * np.outer(scale, scale),
        mf.get_fock(dm=mf.make_rdm1()) * np.outer(scale, scale),
        mf.get_ovlp() * np.outer(scale, scale),
        density,
        mf.mo_energy,
    ]
    return [np.ascontiguousarray(a) for a in matrices], mf


def seeds(arrays: list[np.ndarray], o: int, strength: float) -> tuple:
    n = len(arrays[0])
    rng = np.random.default_rng(1807)
    w = rng.normal(scale=strength, size=(n, n))
    w = (w + w.T) / 2
    frame = np.zeros((n, n))
    frame[:, :o] = 4 * w @ arrays[0][:, :o]
    # Traces over entire occupied/virtual subspaces are gauge invariant, even
    # at exact internal degeneracy. Both Fock and frame seeds drive the Z solve.
    fseed = np.diag(np.r_[np.full(o, strength / 3), np.full(n - o, -strength / 5)])
    return [np.ascontiguousarray(fseed), np.ascontiguousarray(frame)], w


def run(
    probe: typing.Any,
    metadata: dict,
    arrays: list[np.ndarray],
    sources: list[np.ndarray],
    *,
    blas: bool = True,
    relax: bool = True,
    budget: int = 1 << 30,
    iterations: int = 200,
) -> tuple:
    n = len(arrays[0])
    o = metadata["records"]["conventional"]["electron_count"] // 2
    natom = len(metadata["inputs"]["atomic_numbers"])
    output = [
        np.full(shape, 12345.0)
        for shape in ((natom, 3), (n, n), (n, n), (n, n), (n, n), (o, n - o))
    ]
    dp = ct.POINTER(ct.c_double)
    feeds = (dp * 8)(*(a.ctypes.data_as(dp) for a in [*arrays, *sources]))
    destinations = (dp * 6)(*(a.ctypes.data_as(dp) for a in output))
    counts, values = np.zeros(12, dtype=np.uintp), np.zeros(3)
    error = ct.create_string_buffer(1024)
    with NativeSource(**source_arguments(metadata)) as source:
        status = probe(
            source._handle,
            o,
            feeds,
            blas,
            relax,
            budget,
            iterations,
            destinations,
            counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
            values.ctypes.data_as(dp),
            error,
            len(error),
        )
    return status, error.value.decode(), output, counts, values


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("blas", [False, True])
def test_complete_hf_limit(probe: typing.Any, name: str, blas: bool) -> None:
    metadata, _ = load_fixture(name)
    arrays, mf = reference(metadata)
    n, o = len(arrays[0]), mf.mol.nelectron // 2
    sources, _ = seeds(arrays, o, 0.0)
    status, error, output, counts, values = run(
        probe, metadata, arrays, sources, blas=blas
    )
    assert status == 0, error
    independent = mf.nuc_grad_method()
    np.testing.assert_allclose(
        output[0] + independent.grad_nuc(), independent.kernel(), atol=2e-8, rtol=2e-8
    )
    np.testing.assert_allclose(output[1], arrays[4], atol=3e-10, rtol=3e-10)
    expected_pulay = -2 * (arrays[0][:, :o] * arrays[5][:o]) @ arrays[0][:, :o].T
    np.testing.assert_allclose(output[2], expected_pulay, atol=3e-9, rtol=3e-10)
    assert counts[2] == 3 and counts[9] == 0 and counts[11] == 0
    assert bool(counts[4]) == blas
    assert max(values) < 1e-8
    assert n > o


@pytest.mark.parametrize("name", ["water", "lih"])
@pytest.mark.parametrize("blas", [False, True])
def test_nonzero_z_response_matches_complete_energy_directions(
    probe: typing.Any, name: str, blas: bool
) -> None:
    metadata, _ = load_fixture(name)
    arrays, mf = reference(metadata)
    o = mf.mol.nelectron // 2
    sources, w = seeds(arrays, o, 0.03)
    status, error, output, counts, values = run(
        probe, metadata, arrays, sources, blas=blas
    )
    assert status == 0, error
    assert counts[10] > 0 and counts[9] == 0
    assert values[0] < 1e-10 and values[1] < 1e-8
    gradient = output[0] + mf.nuc_grad_method().grad_nuc()
    direction = np.random.default_rng(1765).normal(size=gradient.shape)
    direction /= np.linalg.norm(direction)
    expected = float(np.sum(gradient * direction))
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            moved = copy.deepcopy(metadata)
            moved["inputs"]["coordinates"] = (
                np.asarray(metadata["inputs"]["coordinates"]) + sign * step * direction
            ).tolist()
            shifted, displaced = reference(moved)
            energies.append(
                float(
                    displaced.e_tot
                    + np.sum(w * shifted[4])
                    + np.diag(sources[0]) @ shifted[5]
                )
            )
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), expected, atol=3e-7, rtol=3e-7
        )
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-8, rtol=0)


def test_admission_reference_and_stationarity_fail_without_publication(
    probe: typing.Any,
) -> None:
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    status, error, expected, counts, _ = run(probe, metadata, arrays, sources)
    assert status == 0, error
    budget = int(counts[0])
    success = run(probe, metadata, arrays, sources, budget=budget)
    assert success[0] == 0, success[1]
    for actual, want in zip(success[2], expected, strict=True):
        np.testing.assert_allclose(actual, want, atol=1e-11, rtol=1e-11)
    changed = [a.copy() for a in arrays]
    changed[2][0, 0] += 1e-3
    bad_sources = [a.copy() for a in sources]
    bad_sources[1][:, 1] += arrays[3] @ arrays[0][:, 0]
    nan_sources = [a.copy() for a in sources]
    nan_sources[0][0, 0] = np.nan
    for data, seed, kwargs in (
        (arrays, sources, {"budget": budget - 1}),
        (changed, sources, {}),
        (arrays, bad_sources, {}),
        (arrays, nan_sources, {}),
        (arrays, sources, {"iterations": 1}),
    ):
        result = run(
            probe,
            metadata,
            data,
            seed,
            budget=kwargs.get("budget", 1 << 30),
            iterations=kwargs.get("iterations", 200),
        )
        assert result[0] != 0
        assert all(np.all(a == 12345.0) for a in result[2])


def test_same_operator_subspace_retains_exact_response_gates(
    probe: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    cold = run(probe, metadata, arrays, sources)
    assert cold[0] == 0, cold[1]
    monkeypatch.setenv("GENERATIVEQC_TEST_RHF_RECYCLE_REPEAT", "1")
    warm = run(probe, metadata, arrays, sources)
    assert warm[0] == 0, warm[1]
    assert warm[3][1] < cold[3][1]  # actual exact J/K calls, not a FLOP estimate
    assert warm[3][10] == 0  # the fresh physical residual accepts the projection
    for actual, expected in zip(warm[2], cold[2], strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert max(warm[4]) < 1e-8
