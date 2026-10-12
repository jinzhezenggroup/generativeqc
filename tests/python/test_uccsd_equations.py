"""Independent fermionic and molecular gates for the shared UCCSD graph."""

from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import execute, linearize, transpose_program


def _amplitudes(layout: UCCSDLayout, seed: int = 37) -> dict[str, np.ndarray]:
    return layout.unpack(
        np.random.default_rng(seed).normal(scale=0.04, size=layout.packed_size)
    )


def _spin_amplitudes(
    layout: UCCSDLayout, blocks: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Test-only embedding, independently enumerating fermionic exchange signs."""
    singles = np.zeros((layout.oa + layout.ob, layout.va + layout.vb))
    doubles = np.zeros(
        (*singles.shape[:1], singles.shape[0], singles.shape[1], singles.shape[1])
    )
    singles[: layout.oa, : layout.va] = blocks["t1a"]
    singles[layout.oa :, layout.va :] = blocks["t1b"]
    doubles[: layout.oa, : layout.oa, : layout.va, : layout.va] = blocks["t2aa"]
    doubles[layout.oa :, layout.oa :, layout.va :, layout.va :] = blocks["t2bb"]
    for indices in np.ndindex(blocks["t2ab"].shape):
        occ_a, occ_b, vir_a, vir_b = indices
        occ_b += layout.oa
        vir_b += layout.va
        value = blocks["t2ab"][indices]
        doubles[occ_a, occ_b, vir_a, vir_b] = value
        doubles[occ_b, occ_a, vir_a, vir_b] = -value
        doubles[occ_a, occ_b, vir_b, vir_a] = -value
        doubles[occ_b, occ_a, vir_b, vir_a] = value
    return singles, doubles


def _feeds(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    amplitudes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Slice test-oracle tensors; production never uses this full-spin embedding."""
    populations = (layout.oa, layout.ob, layout.va, layout.vb)
    offsets = np.cumsum((0, *populations))
    slices = dict(
        zip(
            ("oa", "ob", "va", "vb"),
            (slice(start, stop) for start, stop in itertools.pairwise(offsets)),
            strict=True,
        )
    )
    program = build_program(*populations)
    feeds = dict(amplitudes)
    for node in program.live_nodes:
        if node.op != "input" or node.attrs["name"] in feeds:
            continue
        name = node.attrs["name"]
        domains, spins = name[1:].split("_")
        source = fock if name.startswith("f") else integrals
        feeds[name] = source[
            tuple(
                slices[domain + spin]
                for domain, spin in zip(domains, spins, strict=True)
            )
        ].copy()
    return feeds


def _random_hamiltonian(layout: UCCSDLayout) -> tuple[np.ndarray, np.ndarray]:
    """Physical pair-factor Coulomb tensor with independent spin orbital bases."""
    rng = np.random.default_rng(19)
    sizes = (layout.oa, layout.ob, layout.va, layout.vb)
    count = sum(sizes)
    spins = np.repeat((0, 1, 0, 1), sizes)
    pair_factors = rng.normal(scale=0.1, size=(count, count, 5))
    pair_factors += pair_factors.swapaxes(0, 1)
    pair_factors *= (spins[:, None] == spins[None, :])[:, :, None]
    coulomb = np.einsum("prx,qsx->pqrs", pair_factors, pair_factors)
    integrals = coulomb - coulomb.swapaxes(2, 3)
    fock = rng.normal(scale=0.2, size=(count, count))
    fock += fock.T
    fock *= spins[:, None] == spins[None, :]
    return fock, integrals


def _apply(state: int, operations: tuple[tuple[int, bool], ...]) -> tuple[int, int]:
    """Apply right-to-left creation/annihilation actions using CAR bit signs."""
    sign = 1
    for orbital, create in operations:
        occupied = bool(state & (1 << orbital))
        if occupied == create:
            return state, 0
        sign *= (-1) ** (state & ((1 << orbital) - 1)).bit_count()
        state ^= 1 << orbital
    return state, sign


def _determinant_oracle(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    blocks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute projections of exp(-T) H exp(T), without CC intermediates."""
    nocc, nvir = layout.oa + layout.ob, layout.va + layout.vb
    norb = nocc + nvir
    states = [
        sum(1 << orbital for orbital in occupied)
        for occupied in itertools.combinations(range(norb), nocc)
    ]
    positions = {state: index for index, state in enumerate(states)}
    reference = (1 << nocc) - 1
    hcore = fock.copy()
    for occupied in range(nocc):
        hcore -= integrals[:, occupied, :, occupied]
    singles, doubles = _spin_amplitudes(layout, blocks)
    hterms = [
        (hcore[annihilate, create], ((create, False), (annihilate, True)))
        for annihilate, create in np.ndindex(hcore.shape)
    ]
    hterms += [
        (
            0.25 * integrals[first, second, third, fourth],
            ((third, False), (fourth, False), (second, True), (first, True)),
        )
        for first, second, third, fourth in np.ndindex(integrals.shape)
    ]
    tterms = [
        (singles[occupied, virtual], ((occupied, False), (nocc + virtual, True)))
        for occupied, virtual in np.ndindex(singles.shape)
    ]
    tterms += [
        (
            0.25 * doubles[first, second, third, fourth],
            (
                (first, False),
                (second, False),
                (nocc + fourth, True),
                (nocc + third, True),
            ),
        )
        for first, second, third, fourth in np.ndindex(doubles.shape)
    ]

    def matrix(terms: list) -> np.ndarray:
        result = np.zeros((len(states), len(states)))
        for column, source in enumerate(states):
            for value, operations in terms:
                target, sign = _apply(source, operations)
                if value and sign:
                    result[positions[target], column] += sign * value
        return result

    hamiltonian, cluster = matrix(hterms), matrix(tterms)
    positive = negative = power = np.eye(len(states))
    for order in range(1, nocc + 1):
        power = power @ cluster / order
        positive = positive + power
        negative = negative + (-1) ** order * power
    column = positions[reference]
    transformed = (negative @ hamiltonian @ positive)[:, column]
    result = {
        "correlation_energy": np.array(
            transformed[column] - hamiltonian[column, column]
        )
    }
    for block in layout.blocks:
        residual = np.zeros(block.spec.shape)
        for indices in np.ndindex(residual.shape):
            orbitals = [
                index
                + (
                    layout.oa
                    if axis.space.name == "ob"
                    else layout.va
                    if axis.space.name == "vb"
                    else 0
                )
                for index, axis in zip(indices, block.spec.indices, strict=True)
            ]
            if block.name.startswith("t1"):
                occupied, virtual = orbitals
                operations = ((occupied, False), (nocc + virtual, True))
            else:
                first, second, third, fourth = orbitals
                operations = (
                    (first, False),
                    (second, False),
                    (nocc + fourth, True),
                    (nocc + third, True),
                )
            target, sign = _apply(reference, operations)
            if sign:
                residual[indices] = sign * transformed[positions[target]]
        result[block.name.replace("t", "r", 1)] = residual
    return result


def _assert_outputs(actual: dict, expected: dict, tolerance: float = 3e-12) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        np.testing.assert_allclose(
            value, expected[name], atol=tolerance, rtol=tolerance, err_msg=name
        )


@pytest.mark.parametrize(
    "populations",
    [
        (2, 1, 1, 2),
        (1, 2, 2, 1),
        (1, 1, 2, 2),
        (2, 2, 2, 2),
        (2, 0, 2, 0),
        (0, 2, 0, 2),
        (0, 0, 0, 0),
        (1, 1, 0, 0),
    ],
)
def test_residuals_against_complete_determinant_space(
    populations: tuple[int, ...],
) -> None:
    layout = UCCSDLayout(*populations)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    outputs = execute(
        build_program(*populations), _feeds(layout, fock, integrals, amplitudes)
    ).outputs
    _assert_outputs(outputs, _determinant_oracle(layout, fock, integrals, amplitudes))
    for name in ("r2aa", "r2bb"):
        np.testing.assert_allclose(
            outputs[name], -outputs[name].swapaxes(0, 1), atol=1e-14
        )
        np.testing.assert_allclose(
            outputs[name], -outputs[name].swapaxes(2, 3), atol=1e-14
        )


def _molecular_case(
    atom: str, spin: int, *, restricted: bool = False, broken: bool = False
) -> tuple[Any, Any, UCCSDLayout, np.ndarray, np.ndarray]:
    pyscf = pytest.importorskip("pyscf")
    cc = pytest.importorskip("pyscf.cc")
    molecule = pyscf.gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    reference = pyscf.scf.RHF(molecule) if restricted else pyscf.scf.UHF(molecule)
    reference.conv_tol = 1e-13
    if broken:
        density = np.zeros((2, molecule.nao_nr(), molecule.nao_nr()))
        density[0, 0, 0] = density[1, 1, 1] = 1
        reference.kernel(dm0=density)
    else:
        reference.kernel()
    assert reference.converged
    coupled_cluster = cc.CCSD(reference)
    eris = coupled_cluster.ao2mo()
    if restricted:
        occ_a = occ_b = molecule.nelectron // 2
        coeff_a = coeff_b = reference.mo_coeff
        fock_a = fock_b = eris.fock
    else:
        occ_a, occ_b = molecule.nelec
        coeff_a, coeff_b = reference.mo_coeff
        fock_a, fock_b = eris.focka, eris.fockb
    layout = UCCSDLayout(
        occ_a, occ_b, coeff_a.shape[1] - occ_a, coeff_b.shape[1] - occ_b
    )
    coefficients = np.concatenate(
        (
            coeff_a[:, :occ_a],
            coeff_b[:, :occ_b],
            coeff_a[:, occ_a:],
            coeff_b[:, occ_b:],
        ),
        axis=1,
    )
    spins = np.repeat((0, 1, 0, 1), (layout.oa, layout.ob, layout.va, layout.vb))
    coulomb = np.einsum(
        "uvwx,up,vr,wq,xs->pqrs",
        molecule.intor("int2e"),
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        optimize=True,
    )
    coulomb *= (spins[:, None, None, None] == spins[None, None, :, None]) & (
        spins[None, :, None, None] == spins[None, None, None, :]
    )
    integrals = coulomb - coulomb.swapaxes(2, 3)
    count = len(spins)
    fock = np.zeros((count, count))
    for spin_value, source in ((0, fock_a), (1, fock_b)):
        positions = np.flatnonzero(spins == spin_value)
        fock[np.ix_(positions, positions)] = source
    if broken:
        assert np.linalg.norm(reference.make_rdm1()[0] - reference.make_rdm1()[1]) > 0.1
    return coupled_cluster, eris, layout, fock, integrals


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("O 0 0 0; H 0 0 0.97", 1, False),
        ("H 0 0 0; H 0 0 2.5", 0, True),
    ],
)
def test_random_amplitudes_against_pyscf_uccsd(
    atom: str, spin: int, broken: bool
) -> None:
    coupled_cluster, eris, layout, fock, integrals = _molecular_case(
        atom, spin, broken=broken
    )
    amplitudes = _amplitudes(layout)
    t1 = (amplitudes["t1a"], amplitudes["t1b"])
    t2 = (amplitudes["t2aa"], amplitudes["t2ab"], amplitudes["t2bb"])
    new_t1, new_t2 = coupled_cluster.update_amps(t1, t2, eris)
    diagonal = np.diag(fock)
    nocc = layout.oa + layout.ob
    expected = {"correlation_energy": np.array(coupled_cluster.energy(t1, t2, eris))}
    for block, new in zip(layout.blocks, (*new_t1, *new_t2), strict=True):
        denominator = np.zeros(block.spec.shape)
        for axis, index in enumerate(block.spec.indices):
            start = {"oa": 0, "ob": layout.oa, "va": nocc, "vb": nocc + layout.va}[
                index.space.name
            ]
            shape = [1] * len(block.spec.shape)
            shape[axis] = index.extent
            denominator += (1 if index.space.kind == "occupied" else -1) * diagonal[
                start : start + index.extent
            ].reshape(shape)
        expected[block.name.replace("t", "r", 1)] = denominator * (
            new - amplitudes[block.name]
        )
    actual = execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, fock, integrals, amplitudes),
    ).outputs
    _assert_outputs(actual, expected, tolerance=3e-11)


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("N 0 0 0; H 0 -0.8 0.6; H 0 0.8 0.6", 1, False),
        ("H 0 0 0; H 0 0 1.6", 0, True),
    ],
)
def test_graph_at_independently_converged_molecular_solution(
    atom: str, spin: int, broken: bool
) -> None:
    coupled_cluster, eris, layout, fock, integrals = _molecular_case(
        atom, spin, broken=broken
    )
    coupled_cluster.conv_tol = 1e-13
    coupled_cluster.conv_tol_normt = 1e-11
    coupled_cluster.max_cycle = 400
    energy, singles, doubles = coupled_cluster.kernel(eris=eris)
    assert coupled_cluster.converged
    amplitudes = dict(
        zip(("t1a", "t1b", "t2aa", "t2ab", "t2bb"), (*singles, *doubles), strict=True)
    )
    actual = execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, fock, integrals, amplitudes),
    ).outputs
    np.testing.assert_allclose(actual["correlation_energy"], energy, atol=2e-11, rtol=0)
    for name, value in actual.items():
        if name != "correlation_energy":
            assert np.max(np.abs(value), initial=0) < 2e-9, name


def test_restricted_limit_matches_rccsd_random_amplitudes() -> None:
    coupled_cluster, eris, layout, fock, integrals = _molecular_case(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", 0, restricted=True
    )
    rng = np.random.default_rng(91)
    singles = rng.normal(scale=0.04, size=(layout.oa, layout.va))
    doubles = rng.normal(scale=0.04, size=(layout.oa, layout.oa, layout.va, layout.va))
    doubles = (doubles + doubles.transpose(1, 0, 3, 2)) / 2
    amplitudes = {
        "t1a": singles,
        "t1b": singles,
        "t2aa": doubles - doubles.swapaxes(2, 3),
        "t2ab": doubles,
        "t2bb": doubles - doubles.swapaxes(2, 3),
    }
    new_singles, new_doubles = coupled_cluster.update_amps(singles, doubles, eris)
    denominator = eris.mo_energy[: layout.oa, None] - eris.mo_energy[None, layout.oa :]
    r1 = denominator * (new_singles - singles)
    r2 = (denominator[:, None, :, None] + denominator[None, :, None, :]) * (
        new_doubles - doubles
    )
    expected = {
        "correlation_energy": np.array(coupled_cluster.energy(singles, doubles, eris)),
        "r1a": r1,
        "r1b": r1,
        "r2aa": r2 - r2.swapaxes(2, 3),
        "r2ab": r2,
        "r2bb": r2 - r2.swapaxes(2, 3),
    }
    actual = execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, fock, integrals, amplitudes),
    ).outputs
    _assert_outputs(actual, expected, tolerance=3e-11)
    coefficients = coupled_cluster._scf.mo_coeff
    spatial = np.einsum(
        "uvwx,up,vq,wr,xs->pqrs",
        coupled_cluster._scf.mol.intor("int2e"),
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        optimize=True,
    )
    restricted = build_ccsd_program(layout.oa, layout.va, diagnostics=False)
    restricted_feeds = {
        "t1": singles,
        "t2": doubles,
        "foo": eris.fock[: layout.oa, : layout.oa],
        "fov": eris.fock[: layout.oa, layout.oa :],
        "fvv": eris.fock[layout.oa :, layout.oa :],
    }
    domains = {"o": slice(0, layout.oa), "v": slice(layout.oa, None)}
    for node in restricted.live_nodes:
        if node.op == "input" and node.attrs["name"] not in restricted_feeds:
            name = node.attrs["name"]
            restricted_feeds[name] = spatial[tuple(domains[axis] for axis in name)]
    restricted_outputs = execute(restricted, restricted_feeds).outputs
    np.testing.assert_allclose(
        actual["correlation_energy"],
        restricted_outputs["correlation_energy"],
        atol=3e-12,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        actual["r1a"], restricted_outputs["singles_residual"], atol=3e-12, rtol=3e-12
    )
    np.testing.assert_allclose(
        actual["r2ab"], restricted_outputs["doubles_residual"], atol=3e-12, rtol=3e-12
    )


def test_shared_graph_jvp_vjp_and_finite_difference() -> None:
    """Protect the differentiable graph without claiming solved Lambda forces."""
    layout = UCCSDLayout(2, 1, 1, 2)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    program = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    feeds = _feeds(layout, fock, integrals, amplitudes)
    rng = np.random.default_rng(63)
    directions = {
        name: rng.normal(scale=0.1, size=value.shape) for name, value in feeds.items()
    }
    directions.update(_amplitudes(layout, seed=62))
    seeds = {
        name: rng.normal(size=node.spec.shape) for name, node in program.outputs.items()
    }
    jvp = linearize(program, tuple(feeds))
    vjp = transpose_program(program, tuple(program.outputs), inputs=tuple(feeds))
    forward = execute(
        jvp.program,
        {**feeds, **{f"d_{name}": value for name, value in directions.items()}},
    ).outputs
    reverse = execute(
        vjp.program,
        {**feeds, **{f"bar_{name}": value for name, value in seeds.items()}},
    ).outputs
    forward_dot = sum(np.vdot(seeds[name], forward[f"d_{name}"]) for name in seeds)
    reverse_dot = sum(
        np.vdot(directions[name], reverse[f"bar_{name}"]) for name in directions
    )
    np.testing.assert_allclose(forward_dot, reverse_dot, atol=3e-12, rtol=3e-12)
    step = 1e-5
    plus = execute(
        program,
        {name: value + step * directions[name] for name, value in feeds.items()},
    ).outputs
    minus = execute(
        program,
        {name: value - step * directions[name] for name, value in feeds.items()},
    ).outputs
    for name in program.outputs:
        np.testing.assert_allclose(
            forward[f"d_{name}"],
            (plus[name] - minus[name]) / (2 * step),
            atol=3e-10,
            rtol=3e-9,
        )


@pytest.mark.parametrize(
    "populations", [(-1, 1, 1, 1), (True, 1, 1, 1), (1.0, 1, 1, 1), (1 << 63, 0, 0, 0)]
)
def test_invalid_populations_fail_closed(populations: tuple) -> None:
    with pytest.raises((ValueError, OverflowError)):
        build_program(*populations)


def test_fixed_amplitude_admission_and_budget_fail_closed() -> None:
    layout = UCCSDLayout(2, 1, 1, 2)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    program = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    feeds = _feeds(layout, fock, integrals, amplitudes)
    originals = {name: value.copy() for name, value in feeds.items()}
    with pytest.raises((ValueError, MemoryError), match="budget"):
        execute(program, feeds, max_bytes=1)
    for invalid in (
        np.array([[float("nan")], [0.0]]),
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((3, 1)),
    ):
        with pytest.raises((ValueError, TypeError)):
            execute(program, {**feeds, "t1a": invalid})
    for name, value in originals.items():
        np.testing.assert_array_equal(feeds[name], value)


@pytest.mark.parametrize("populations", [(2, 1, 1, 2), (2, 2, 2, 2), (2, 0, 2, 0)])
def test_native_cpu_matches_determinant_oracle(
    tmp_path: Path, populations: tuple[int, ...]
) -> None:
    from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor

    layout = UCCSDLayout(*populations)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    executor = NativeCCTensorExecutor(max_bytes=256 << 20, cache=tmp_path)
    actual = executor.execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, fock, integrals, amplitudes),
    )
    _assert_outputs(actual, _determinant_oracle(layout, fock, integrals, amplitudes))


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
@pytest.mark.parametrize("populations", [(2, 1, 1, 2), (2, 2, 2, 2), (2, 0, 2, 0)])
def test_real_cuda_matches_determinant_oracle(
    tmp_path: Path, populations: tuple[int, ...]
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor

    nvcc = os.environ.get("GENERATIVEQC_NVCC") or shutil.which("nvcc")
    assert nvcc is not None, "allocated CUDA test requires nvcc"
    compiler = CudaCompilerAdapter(
        Path(nvcc),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    layout = UCCSDLayout(*populations)
    amplitudes = _amplitudes(layout)
    fock, integrals = _random_hamiltonian(layout)
    executor = CudaCCTensorExecutor(
        max_bytes=256 << 20, compiler=compiler, cache=tmp_path
    )
    actual = executor.execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, fock, integrals, amplitudes),
    )
    _assert_outputs(actual, _determinant_oracle(layout, fock, integrals, amplitudes))
