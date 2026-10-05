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
        ct.c_double,
        ct.c_uint,
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
    screening: float = 0.0,
    bilinear: bool = True,
    symmetric: bool = False,
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
    counts, values = np.zeros(20, dtype=np.uintp), np.zeros(5)
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
            screening,
            2 if symmetric else int(bilinear),
            destinations,
            counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
            values.ctypes.data_as(dp),
            error,
            len(error),
        )
    return status, error.value.decode(), output, counts, values


@pytest.mark.parametrize("name", ["h2", "water", "lih", "h2_f"])
@pytest.mark.parametrize("blas", [False, True])
def test_complete_hf_limit(probe: typing.Any, name: str, blas: bool) -> None:
    metadata, _ = load_fixture("h2" if name == "h2_f" else name)
    if name == "h2_f":
        metadata["inputs"]["shells"].append(
            {"atom_index": 1, "angular_momentum": 3, "primitives": [[0.8, 1.0]]}
        )
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
    # Specialized SPD leases are build/admission dependent. Without one, the
    # same canonical bilinear consumer is valid for these small bases too.
    assert counts[2] in (1, 3)
    assert counts[9] == 0 and counts[11] == 0
    if name == "h2_f":
        assert counts[2] == 1
        fallback = run(probe, metadata, arrays, sources, blas=blas, bilinear=False)
        assert fallback[0] == 0, fallback[1]
        assert fallback[3][2] == 3
        np.testing.assert_allclose(output[0], fallback[2][0], atol=3e-10, rtol=3e-10)
    assert bool(counts[4]) == blas
    assert bool(counts[17]) == blas
    assert bool(counts[18]) == blas and bool(counts[19]) == blas
    assert counts[19] <= counts[5]
    assert max(values) < 1e-8
    assert n > o
    symmetric = run(
        probe, metadata, arrays, sources, blas=blas, bilinear=False, symmetric=True
    )
    assert symmetric[0] == 0, symmetric[1]
    assert symmetric[3][2] == 2
    np.testing.assert_allclose(symmetric[2][0], output[0], atol=3e-10, rtol=3e-10)


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
    symmetric = run(
        probe, metadata, arrays, sources, blas=blas, bilinear=False, symmetric=True
    )
    assert symmetric[0] == 0, symmetric[1]
    assert symmetric[3][2] == 2
    np.testing.assert_allclose(symmetric[2][0], output[0], atol=3e-9, rtol=3e-10)
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
    assert success[3][17] == 1 and success[3][18] > 96 << 20
    for actual, want in zip(success[2], expected, strict=True):
        np.testing.assert_allclose(actual, want, atol=1e-11, rtol=1e-11)
    # One byte below the optional table/provider reservation keeps the complete
    # scalar response and its independent final residual; only its own floor fails.
    fallback = run(probe, metadata, arrays, sources, budget=budget - 1)
    assert fallback[0] == 0, fallback[1]
    assert fallback[3][4] == 0 and fallback[3][0] < budget - 1
    assert not np.any(fallback[3][17:20])
    assert counts[0] - fallback[3][0] == counts[18]
    for actual, want in zip(fallback[2], expected, strict=True):
        np.testing.assert_allclose(actual, want, atol=1e-11, rtol=1e-11)
    scalar_budget = int(fallback[3][0])
    exact_scalar = run(probe, metadata, arrays, sources, budget=scalar_budget)
    assert exact_scalar[0] == 0 and exact_scalar[3][4] == 0, exact_scalar[1]
    changed = [a.copy() for a in arrays]
    changed[2][0, 0] += 1e-3
    bad_sources = [a.copy() for a in sources]
    bad_sources[1][:, 1] += arrays[3] @ arrays[0][:, 0]
    nan_sources = [a.copy() for a in sources]
    nan_sources[0][0, 0] = np.nan
    for data, seed, kwargs in (
        (arrays, sources, {"budget": scalar_budget - 1}),
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


@pytest.mark.parametrize("threshold", [1e-12, 1e-4, 0.5])
def test_fixed_mask_response_is_qualified_against_exact_operator(
    probe: typing.Any, threshold: float
) -> None:
    """Even aggressive provisional masks must retain the original force gate."""
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    source, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    exact = run(probe, metadata, arrays, source)
    actual = run(probe, metadata, arrays, source, screening=threshold)
    assert exact[0] == actual[0] == 0, actual[1]
    assert actual[4][0] < 1e-10
    assert actual[4][3] == threshold
    assert actual[3][16] > 0
    if threshold >= 0.5:
        assert actual[3][15] > 0  # a failed exact audit requires correction
    for got, want in zip(actual[2], exact[2], strict=True):
        np.testing.assert_allclose(got, want, atol=3e-9, rtol=3e-10)
    assert actual[3][12] == actual[3][1]
    assert actual[3][14] <= actual[3][13]


@pytest.mark.parametrize("threshold", [0.0, 0.03])
def test_fixed_mask_signed_jk_matches_dense_oracle(threshold: float) -> None:
    """The same symmetric ERI mask must act linearly on arbitrary signed D."""
    from tools.generate_validation_references import pyscf_molecule

    metadata, _ = load_fixture("water")
    mol, scale, _ = pyscf_molecule(metadata["inputs"])
    eri = mol.intor("int2e") * np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)
    bounds = np.sqrt(np.maximum(np.einsum("ijij->ij", eri), 0))
    mask = bounds[:, :, None, None] * bounds[None, None, :, :] >= threshold
    screened = eri * mask
    n = len(scale)
    dp = ct.POINTER(ct.c_double)
    call = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve())
    ).rhf_linear_jk_probe
    call.argtypes = [
        ct.c_void_p,
        dp,
        ct.c_double,
        dp,
        ct.POINTER(ct.c_uint64),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    rng = np.random.default_rng(1830)
    a, b = (rng.normal(size=(n, n)) for _ in range(2))
    a, b = a + a.T, b + b.T
    potentials = []
    with NativeSource(**source_arguments(metadata)) as source:
        for d in (a, b, 2.3 * a - 0.7 * b):
            output = np.empty((2, n, n))
            counts = np.zeros(2, dtype=np.uint64)
            error = ct.create_string_buffer(2048)
            code = call(
                source._handle,
                d.ctypes.data_as(dp),
                threshold,
                output.ctypes.data_as(dp),
                counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                error,
                len(error),
            )
            assert code == 0, error.value.decode()
            reference = np.stack(
                [
                    np.einsum("ijkl,kl->ij", screened, d),
                    np.einsum("ikjl,kl->ij", screened, d),
                ]
            )
            np.testing.assert_allclose(output, reference, atol=2e-11, rtol=2e-12)
            assert counts[0] >= counts[1] > 0
            potentials.append(output[0] - 0.5 * output[1])
    np.testing.assert_allclose(
        potentials[2], 2.3 * potentials[0] - 0.7 * potentials[1], atol=2e-11, rtol=2e-12
    )
    np.testing.assert_allclose(
        np.sum(a * potentials[1]), np.sum(b * potentials[0]), atol=2e-11, rtol=2e-12
    )


@pytest.mark.parametrize("through_f", [False, True])
@pytest.mark.parametrize("representation", ["real_spherical", "cartesian"])
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("cache_budget", [1, 1 << 20])
def test_resident_exact_jk_signed_oracle_and_bounded_fallback(
    through_f: bool,
    representation: str,
    unrestricted: bool,
    cache_budget: int,
    allocation_refused: bool = False,
) -> None:
    """Cached sources retain the exact signed Hamiltonian and a real capacity fallback."""
    assert os.environ.get("SLURM_JOB_ID")
    metadata, _ = load_fixture("water")
    metadata["inputs"]["basis_representation"] = representation
    if through_f:
        metadata["inputs"]["shells"].append(
            {"atom_index": 1, "angular_momentum": 3, "primitives": [[0.8, 1.0]]}
        )
        # Libcint groups shells by atom; keep the native public AO order identical.
        metadata["inputs"]["shells"].sort(key=lambda shell: shell["atom_index"])
    mol, scale, _ = pyscf_molecule(metadata["inputs"])
    eri = mol.intor("int2e") * np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)
    dimension = len(scale)
    spin_count = 2 if unrestricted else 1
    rng = np.random.default_rng(1972)
    first, second = (
        rng.normal(size=(spin_count, dimension, dimension)) for _ in range(2)
    )
    first = first + first.swapaxes(-1, -2)
    second = second + second.swapaxes(-1, -2)
    densities = np.ascontiguousarray(
        [first, second, 2.3 * first - 0.7 * second, np.zeros_like(first), -first]
    )
    channel_count = 3 if unrestricted else 2
    output = np.full((len(densities), 2, channel_count, dimension, dimension), np.nan)
    counts = np.zeros(8, dtype=np.uint64)
    timings = np.zeros(5)
    error = ct.create_string_buffer(2048)
    double_pointer = ct.POINTER(ct.c_double)
    call = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve())
    ).rhf_resident_jk_probe
    call.argtypes = [
        ct.c_void_p,
        double_pointer,
        ct.c_size_t,
        ct.c_bool,
        ct.c_size_t,
        double_pointer,
        ct.POINTER(ct.c_uint64),
        double_pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    with NativeSource(**source_arguments(metadata)) as source:
        status = call(
            source._handle,
            densities.ctypes.data_as(double_pointer),
            len(densities),
            unrestricted,
            cache_budget,
            output.ctypes.data_as(double_pointer),
            counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            timings.ctypes.data_as(double_pointer),
            error,
            len(error),
        )
    assert status == 0, error.value.decode()
    assert counts[0] > 0
    if cache_budget >= counts[0] and not allocation_refused:
        assert counts[1] == counts[0] == 8 * counts[2]
        assert counts[6] == 0 and counts[7] == 1
        assert counts[4] == len(densities) * counts[2]
        assert counts[5] == 0
    else:
        assert counts[1] == counts[2] == 0
        assert counts[6] == 7
    expected = []
    for density in densities:
        channels = [np.einsum("ijkl,kl->ij", eri, density.sum(axis=0))]
        channels.extend(np.einsum("ikjl,kl->ij", eri, spin) for spin in density)
        expected.append(channels)
    expected = np.asarray(expected)
    for route in range(2):
        np.testing.assert_allclose(output[:, route], expected, atol=2e-11, rtol=2e-12)
    actual = output[:, 0]
    np.testing.assert_allclose(
        actual[2], 2.3 * actual[0] - 0.7 * actual[1], atol=3e-11, rtol=2e-12
    )
    np.testing.assert_array_equal(actual[3], np.zeros_like(actual[3]))
    np.testing.assert_allclose(actual[4], -actual[0], atol=2e-11, rtol=2e-12)
    exchange_scale = 1.0 if unrestricted else 0.5
    potentials = actual[:, :1] - exchange_scale * actual[:, 1:]
    np.testing.assert_allclose(
        np.sum(first * potentials[1]),
        np.sum(second * potentials[0]),
        atol=3e-11,
        rtol=2e-12,
    )
    assert np.isfinite(timings).all() and (timings > 0).all()


@pytest.mark.parametrize("representation", ["real_spherical", "cartesian"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_resident_allocation_refusal_keeps_exact_signed_actions(
    monkeypatch: pytest.MonkeyPatch, representation: str, unrestricted: bool
) -> None:
    """A rejected device lease must not poison later ordinary Direct actions."""
    monkeypatch.setenv("GENERATIVEQC_TEST_RHF_RESIDENT_ALLOCATION_REFUSAL", "1")
    test_resident_exact_jk_signed_oracle_and_bounded_fallback(
        True, representation, unrestricted, 1 << 20, allocation_refused=True
    )


@pytest.mark.parametrize("representation", ["real_spherical", "cartesian"])
def test_resident_quartic_inventory_cliff_refuses_without_allocating_values(
    representation: str,
) -> None:
    """A large metadata-only inventory cannot silently exceed the 8-GiB ceiling."""
    assert os.environ.get("SLURM_JOB_ID")
    metadata, _ = load_fixture("water")
    metadata["inputs"]["basis_representation"] = representation
    for _ in range(31):
        metadata["inputs"]["shells"].append(
            {"atom_index": 0, "angular_momentum": 3, "primitives": [[0.8, 1.0]]}
        )
    metadata["inputs"]["shells"].sort(key=lambda shell: shell["atom_index"])
    counts = np.zeros(8, dtype=np.uint64)
    timings = np.zeros(5)
    unused = np.zeros(1)
    error = ct.create_string_buffer(2048)
    double_pointer = ct.POINTER(ct.c_double)
    call = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve())
    ).rhf_resident_jk_probe
    call.argtypes = [
        ct.c_void_p,
        double_pointer,
        ct.c_size_t,
        ct.c_bool,
        ct.c_size_t,
        double_pointer,
        ct.POINTER(ct.c_uint64),
        double_pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    with NativeSource(**source_arguments(metadata)) as source:
        status = call(
            source._handle,
            unused.ctypes.data_as(double_pointer),
            0,
            False,
            8 << 30,
            unused.ctypes.data_as(double_pointer),
            counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            timings.ctypes.data_as(double_pointer),
            error,
            len(error),
        )
    assert status == 0, error.value.decode()
    cartesian_dimension = 7 + 31 * 10
    pair_count = cartesian_dimension * (cartesian_dimension + 1) // 2
    assert counts[0] == 8 * pair_count * (pair_count + 1) // 2 > 8 << 30
    np.testing.assert_array_equal(counts[1:6], 0)
    assert counts[6] == 7


@pytest.mark.parametrize("through_f", [False, True])
@pytest.mark.parametrize("representation", ["real_spherical", "cartesian"])
@pytest.mark.parametrize("centers", [2, 4])
def test_bilinear_derivative_signed_density_oracle(
    through_f: bool, representation: str, centers: int
) -> None:
    """Independent libcint nuclear derivatives and FD of signed P:G(D)."""
    metadata, _ = load_fixture("h2")
    metadata = copy.deepcopy(metadata)
    metadata["inputs"]["basis_representation"] = representation
    if centers == 4:
        # Nonplanar centers exercise all three explicit center jets and
        # translation reconstruction of the fourth atom.
        metadata["inputs"]["atomic_numbers"].extend([1, 1])
        metadata["inputs"]["coordinates"].extend([[0.7, 1.2, -0.3], [-1.1, 0.4, 0.8]])
        metadata["inputs"]["shells"].extend(
            {"atom_index": atom, "angular_momentum": 0, "primitives": [[0.9, 1.0]]}
            for atom in (2, 3)
        )
    if through_f:
        metadata["inputs"]["shells"].append(
            {
                "atom_index": centers - 1,
                "angular_momentum": 3,
                "primitives": [[0.8, 1.0]],
            }
        )
    mol, scale, _ = pyscf_molecule(metadata["inputs"])
    n = len(scale)
    rng = np.random.default_rng(1831)
    d, p = (rng.normal(scale=0.04, size=(n, n)) for _ in range(2))
    d, p = d + d.T, p + p.T
    weight = np.einsum("ij,kl->ijkl", p, d) - 0.5 * np.einsum("ik,jl->ijkl", p, d)
    normalization = np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)
    ip1 = mol.intor("int2e_ip1")
    oracle = np.zeros((mol.natm, 3))
    for permutation in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)):
        tensor = -ip1.transpose((0, *(1 + permutation.index(i) for i in range(4))))
        for atom, (_, _, begin, end) in enumerate(mol.aoslice_by_atom()):
            mask = np.zeros(n)
            mask[begin:end] = 1
            shape = [1] * 4
            shape[permutation[0]] = n
            oracle[atom] += np.einsum(
                "xijkl,ijkl->x", tensor, weight * normalization * mask.reshape(shape)
            )
    dp = ct.POINTER(ct.c_double)
    call = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve())
    ).rhf_bilinear_derivative_probe
    call.argtypes = [
        ct.c_void_p,
        dp,
        dp,
        dp,
        ct.POINTER(ct.c_uint64),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    with NativeSource(**source_arguments(metadata)) as source:
        for invalid in (False, True):
            data = np.full_like(d, 1e308) if invalid else d
            seed = np.full_like(p, 1e308) if invalid else p
            output = np.full_like(oracle, 12345.0)
            counts = np.zeros(2, dtype=np.uint64)
            error = ct.create_string_buffer(2048)
            code = call(
                source._handle,
                data.ctypes.data_as(dp),
                seed.ctypes.data_as(dp),
                output.ctypes.data_as(dp),
                counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                error,
                len(error),
            )
            if invalid:
                assert code != 0
                assert np.all(output == 12345.0)
                continue
            assert code == 0, error.value.decode()
            np.testing.assert_allclose(output, oracle, atol=2e-11, rtol=2e-10)
            np.testing.assert_allclose(output.sum(axis=0), 0, atol=2e-12)
            assert 0 < counts[1] <= (centers - 1) * counts[0]
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            displaced = copy.deepcopy(metadata["inputs"])
            displaced["coordinates"][0][2] += sign * step
            shifted, norms, _ = pyscf_molecule(displaced)
            eri = shifted.intor("int2e") * np.einsum(
                "i,j,k,l->ijkl", norms, norms, norms, norms
            )
            energies.append(np.sum(weight * eri))
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), oracle[0, 2], atol=3e-9
        )
