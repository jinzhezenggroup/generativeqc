"""Independent covariant-resolvent and complete bounded native response gates."""

from __future__ import annotations

import os
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_triples import CHANNELS, channel_shape
from generativeqc_compiler.cc.uccsd_triples_fock import (
    build_axis_moment_program,
    build_tile_resolvent_program,
)
from test_uccsd_equations import _amplitudes, _random_hamiltonian, _spin_amplitudes
from test_uccsd_t_owner import (
    _ASYMMETRIC_H6,
    _BROKEN_H4,
    _molecule,
    _noninteracting,
)
from test_uccsd_t_owner import small_parent as _small_parent
from test_uccsd_triples import _tile_feeds
from test_uccsd_triples_response import _snapshot_hamiltonian, _source_feeds

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_cc.uccsd_t import UCCSDTOptions
from tools.generativeqc_cc.uccsd_triples_fock_response import (
    PreparedUCCSDTriplesFockResponse,
    UCCSDTriplesFockOptions,
)
from tools.generativeqc_cc.uccsd_triples_response import PreparedUCCSDTriplesResponse
from tools.generativeqc_posthf.providers import ConventionalProvider

small_parent = _small_parent
_MOLECULAR_FOCK_CASES = (
    (_BROKEN_H4, 0, True, (2, 8, 8, 8, 8, 8)),
    (_ASYMMETRIC_H6, 0, False, (2, 8, 8, 8, 8, 8)),
    ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False, (2, 8, 8, 8, 8, 8)),
    (_BROKEN_H4, 0, True, (1, 2, 2, 1, 2, 2)),
)


def _oracle_vectors(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    amplitudes: dict,
    energies: np.ndarray,
) -> tuple:
    """Expand the independent triangular GCCSD(T) oracle into test-only full X/Y.

    Production does not use these molecular tensors, NumPy contractions or
    oracle equations. Occupied cyclic symmetrization of the oracle's right
    numerator yields its complete antisymmetrized right vector; six signed
    virtual permutations expand each separately implemented triangular term.
    """
    gccsd, reference = (
        pytest.importorskip("pyscf.cc.gccsd"),
        pytest.importorskip("pyscf.cc.gccsd_t"),
    )
    occupied, virtual = layout.oa + layout.ob, layout.va + layout.vb
    eris = gccsd._PhysicistsERIs()
    eris.fock, eris.mo_energy = fock, energies
    eris.ovvv = integrals[:occupied, occupied:, occupied:, occupied:]
    eris.ooov = integrals[:occupied, :occupied, :occupied, occupied:]
    eris.oovv = integrals[:occupied, :occupied, occupied:, occupied:]
    captured = []

    def contraction(equation: str, *values: Any, **kwargs: Any) -> Any:
        if equation == "ijk,ijk":
            captured.append(tuple(value.copy() for value in values))
        return np.einsum(equation, *values, **kwargs)

    singles, doubles = _spin_amplitudes(layout, amplitudes)
    with patch.object(
        reference, "numpy", SimpleNamespace(asarray=np.asarray, einsum=contraction)
    ):
        energy = reference.kernel(None, eris, singles, doubles)
    left = np.zeros((occupied,) * 3 + (virtual,) * 3)
    right = np.zeros_like(left)
    triangular = [
        (first, second, third)
        for first in range(virtual)
        for second in range(first)
        for third in range(second)
    ]
    occupied_energy = energies[:occupied]
    occupied_sum = (
        occupied_energy[:, None, None]
        + occupied_energy[None, :, None]
        + occupied_energy[None, None, :]
    )
    for coordinates, (resolved_left, numerator_right) in zip(
        triangular, captured, strict=True
    ):
        denominator = occupied_sum - sum(
            energies[occupied + coordinate] for coordinate in coordinates
        )
        resolved_right = (
            numerator_right
            + numerator_right.transpose(2, 0, 1)
            + numerator_right.transpose(1, 2, 0)
        ) / denominator
        for permutation in permutations(range(3)):
            inversions = sum(
                permutation[first] > permutation[second]
                for first in range(3)
                for second in range(first + 1, 3)
            )
            sign = (-1) ** inversions
            reordered = tuple(coordinates[position] for position in permutation)
            left[(slice(None),) * 3 + reordered] = sign * resolved_left
            right[(slice(None),) * 3 + reordered] = sign * resolved_right
    denominator = occupied_sum[:, :, :, None, None, None] - (
        energies[occupied:][None, None, None, :, None, None]
        + energies[occupied:][None, None, None, None, :, None]
        + energies[occupied:][None, None, None, None, None, :]
    )
    np.testing.assert_allclose(
        np.sum(left * right * denominator) / 36, energy, atol=3e-13, rtol=0
    )
    return left * denominator, right * denominator, left, right, energy


def _resolved_energy(
    connected: np.ndarray,
    complete: np.ndarray,
    occupied_fock: np.ndarray,
    virtual_fock: np.ndarray,
) -> float:
    """Independent dense-test Fock diagonalization, including exact degeneracy."""
    occupied_energies, occupied_vectors = np.linalg.eigh(occupied_fock)
    virtual_energies, virtual_vectors = np.linalg.eigh(virtual_fock)
    rotated = []
    for original in (connected, complete):
        values = original
        for axis, vectors in enumerate(
            (occupied_vectors,) * 3 + (virtual_vectors,) * 3
        ):
            values = np.moveaxis(
                np.tensordot(vectors.T, values, axes=((1,), (axis,))), 0, axis
            )
        rotated.append(values)
    denominator = (
        occupied_energies[:, None, None, None, None, None]
        + occupied_energies[None, :, None, None, None, None]
        + occupied_energies[None, None, :, None, None, None]
        - virtual_energies[None, None, None, :, None, None]
        - virtual_energies[None, None, None, None, :, None]
        - virtual_energies[None, None, None, None, None, :]
    )
    return float(np.sum(rotated[0] * rotated[1] / denominator) / 36)


def _oracle_moments(left: np.ndarray, right: np.ndarray) -> tuple:
    """Full-spin test-only marginal contractions, never production storage."""
    occupied, virtual = left.shape[0], left.shape[3]
    moments = [np.zeros((occupied, occupied)), np.zeros((virtual, virtual))]
    for axis in range(6):
        left_matrix = np.moveaxis(left, axis, 0).reshape(left.shape[axis], -1)
        right_matrix = np.moveaxis(right, axis, 0).reshape(right.shape[axis], -1)
        moment = left_matrix @ right_matrix.T
        moments[axis // 3] += (-1 if axis < 3 else 1) * (moment + moment.T) / 72
    return tuple(moments)


def _positions(layout: UCCSDLayout) -> dict:
    return {
        ("o", "a"): tuple(range(layout.oa)),
        ("o", "b"): tuple(range(layout.oa, layout.oa + layout.ob)),
        ("v", "a"): tuple(range(layout.va)),
        ("v", "b"): tuple(range(layout.va, layout.va + layout.vb)),
    }


def _check_native_vectors(executor: Any, degenerate: bool) -> None:
    layout = UCCSDLayout(3, 4, 4, 3)
    fock, integrals = _random_hamiltonian(layout)
    amplitudes = _amplitudes(layout)
    occupied, virtual = layout.oa + layout.ob, layout.va + layout.vb
    energies = np.concatenate(
        (
            -np.ones(occupied) if degenerate else -np.linspace(0.5, 2, occupied),
            np.ones(virtual) if degenerate else np.linspace(0.6, 2.1, virtual),
        )
    )
    connected, complete, left, right, _ = _oracle_vectors(
        layout, fock, integrals, amplitudes, energies
    )
    sources = _source_feeds(layout, fock, integrals, amplitudes, energies)
    expected = _oracle_moments(left, right)
    actual = [np.zeros_like(values) for values in expected]
    positions = _positions(layout)
    for channel in CHANNELS:
        shape = channel_shape(layout, channel)
        tile = build_tile_resolvent_program(layout, channel, shape)
        resolved = executor.execute(
            tile.program,
            _tile_feeds(tile, sources, tuple(tuple(range(size)) for size in shape)),
        )
        selection = np.ix_(
            *(
                positions[domain, spin]
                for domain, spin in zip("ooovvv", channel * 2, strict=True)
            )
        )
        np.testing.assert_allclose(resolved["x"], left[selection], atol=3e-13, rtol=0)
        np.testing.assert_allclose(resolved["y"], right[selection], atol=3e-13, rtol=0)
        for axis in range(6):
            program = build_axis_moment_program(layout, channel, shape, shape, axis)
            outputs = executor.execute(
                program,
                {
                    "x_left": resolved["x"],
                    "y_left": resolved["y"],
                    "x_right": resolved["x"],
                    "y_right": resolved["y"],
                },
            )
            domain, spin = ("o" if axis < 3 else "v"), (channel * 2)[axis]
            name = f"f{domain * 2}_{spin * 2}"
            target = positions[domain, spin]
            actual[axis // 3][np.ix_(target, target)] += outputs[name]
            if axis == 0:
                paired = build_axis_moment_program(
                    layout, channel, (1, *shape[1:]), (shape[0] - 1, *shape[1:]), axis
                )
                edge = executor.execute(
                    paired,
                    {
                        "x_left": np.ascontiguousarray(resolved["x"][:1]),
                        "y_left": np.ascontiguousarray(resolved["y"][:1]),
                        "x_right": np.ascontiguousarray(resolved["x"][1:]),
                        "y_right": np.ascontiguousarray(resolved["y"][1:]),
                    },
                )
                np.testing.assert_allclose(
                    edge[name], outputs[name][:1, 1:], atol=3e-13, rtol=0
                )
    generator = np.random.default_rng(451)
    matrices = [np.diag(energies[:occupied]), np.diag(energies[occupied:])]
    for domain, size in enumerate((occupied, virtual)):
        np.testing.assert_allclose(actual[domain], expected[domain], atol=3e-13, rtol=0)
        direction = generator.normal(scale=0.1, size=(size, size))
        direction = (direction + direction.T) / 2
        split = layout.oa if domain == 0 else layout.va
        direction[:split, split:], direction[split:, :split] = 0, 0
        np.fill_diagonal(direction, 0)
        step = 1e-5
        plus, minus = list(matrices), list(matrices)
        plus[domain], minus[domain] = (
            matrices[domain] + step * direction,
            matrices[domain] - step * direction,
        )
        difference = (
            _resolved_energy(connected, complete, *plus)
            - _resolved_energy(connected, complete, *minus)
        ) / (2 * step)
        np.testing.assert_allclose(
            np.sum(actual[domain] * direction), difference, atol=2e-9, rtol=2e-7
        )


@pytest.mark.parametrize("degenerate", [False, True])
def test_native_cpu_vectors_and_all_axis_moments_have_independent_resolvent_closure(
    degenerate: bool,
) -> None:
    _check_native_vectors(NativeCCTensorExecutor(max_bytes=64 << 20), degenerate)


@pytest.mark.parametrize("atom,spin,broken,tile_shape", _MOLECULAR_FOCK_CASES)
def test_complete_cpu_fock_response_matches_independent_molecular_oracle(
    atom: str, spin: int, broken: bool, tile_shape: tuple[int, ...]
) -> None:
    snapshot, source, _, _ = _molecule(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDTriplesResponse(
                parent, state, options=UCCSDTOptions(tile_shape=tile_shape)
            ) as triples,
            PreparedUCCSDTriplesFockResponse(triples) as owner,
        ):
            result = owner.response()
            assert result.converged, result.triples_response.energy.reason
            fock, integrals, energies = _snapshot_hamiltonian(parent, source)
            _, _, left, right, _ = _oracle_vectors(
                parent.layout, fock, integrals, dict(state.amplitudes), energies
            )
            expected = _oracle_moments(left, right)
            positions = _positions(parent.layout)
            full, diagonal = (
                result.hamiltonian_cotangents(),
                result.triples_response.hamiltonian_cotangents(),
            )
            for name, moment in result.fock_moments.items():
                domain, spin_label = name[1], name[-1]
                coordinates = positions[domain, spin_label]
                np.testing.assert_allclose(
                    moment,
                    expected[0 if domain == "o" else 1][
                        np.ix_(coordinates, coordinates)
                    ],
                    atol=2e-10,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    np.diag(full[name]), np.diag(diagonal[name]), atol=3e-13, rtol=0
                )
            assert (
                result.provenance["planned_semantic_scalar_work"]
                == result.provenance["executed_semantic_scalar_work"]
            )


@pytest.fixture
def small_fock(small_parent: tuple) -> Any:
    parent, state, _ = small_parent
    with (
        PreparedUCCSDTriplesResponse(
            parent, state, options=UCCSDTOptions(tile_shape=(2,) * 6)
        ) as triples,
        PreparedUCCSDTriplesFockResponse(triples) as owner,
    ):
        yield triples, owner


@pytest.mark.parametrize(
    "options,match",
    [
        (UCCSDTriplesFockOptions(max_bytes=1), "numeric"),
        (UCCSDTriplesFockOptions(max_pairs=1), "pair-count"),
        (UCCSDTriplesFockOptions(max_total_work=1), "semantic-work"),
    ],
)
def test_fock_admission_rejects_before_native_execution(
    small_fock: tuple,
    options: UCCSDTriplesFockOptions,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triples, _ = small_fock

    def forbidden(*_: Any) -> None:
        pytest.fail("rejected Fock admission executed native work")

    monkeypatch.setattr(triples.prepared.executor, "execute", forbidden)
    with pytest.raises(MemoryError, match=match):
        PreparedUCCSDTriplesFockResponse(triples, options=options)


@pytest.mark.parametrize("phase", ["vectors", "moments", "nonfinite"])
def test_late_fock_failure_never_publishes_partial_correction(
    small_fock: tuple, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    triples, owner = small_fock
    original = triples.prepared.executor.execute
    crossed = False

    def failing(program: Any, feeds: Any) -> Any:
        nonlocal crossed
        selected = (
            "x" in program.outputs
            if phase == "vectors"
            else set(program.outputs).issubset({"foo_aa", "foo_bb", "fvv_aa", "fvv_bb"})
        )
        if selected and crossed:
            if phase == "nonfinite":
                outputs = original(program, feeds)
                next(iter(outputs.values()))[0, 0] = np.nan
                return outputs
            raise RuntimeError("injected late Fock failure")
        if selected:
            crossed = True
        return original(program, feeds)

    monkeypatch.setattr(triples.prepared.executor, "execute", failing)
    result = owner.response()
    assert not result.converged
    assert result.triples_response.energy.triples_energy is None
    assert not result.triples_response.cotangents and not result.fock_moments
    with pytest.raises(ValueError):
        result.hamiltonian_cotangents()


def test_fock_lifetimes_and_owned_immutable_records(small_fock: tuple) -> None:
    triples, owner = small_fock
    result = owner.response()
    assert result.converged
    copied = {name: values.copy() for name, values in result.fock_moments.items()}
    detached = replace(result, fock_moments=copied)
    copied["foo_aa"][0, 0] = 1
    assert detached.fock_moments["foo_aa"][0, 0] == 0
    with pytest.raises(ValueError):
        result.fock_moments["foo_aa"].setflags(write=True)
    original = owner.options
    owner.options = replace(original, max_total_work=1)
    with pytest.raises(RuntimeError, match="configuration"):
        owner.response()
    owner.options = original
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.response()
    assert triples.response().converged


def test_numeric_fallback_keeps_complete_paired_response(small_fock: tuple) -> None:
    triples, initial = small_fock
    baseline = initial.response()
    with PreparedUCCSDTriplesFockResponse(
        triples, options=UCCSDTriplesFockOptions(max_bytes=initial.peak_bytes - 1)
    ) as reduced:
        result = reduced.response()
        assert result.converged, result.triples_response.energy.reason
        assert reduced.fallback_steps > 0
        assert reduced.peak_bytes <= reduced.options.max_bytes
        for name, values in baseline.fock_moments.items():
            np.testing.assert_allclose(
                result.fock_moments[name], values, atol=3e-13, rtol=0
            )


def test_full_fock_diagonal_must_match_fresh_eps_bars(
    small_fock: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    triples, owner = small_fock
    actual = triples.response()
    bars = {name: values.copy() for name, values in actual.cotangents.items()}
    bars["eps_o_a"][0] += 1e-4
    forged = replace(actual, cotangents=bars)
    monkeypatch.setattr(triples, "response", lambda: forged)
    result = owner.response()
    assert not result.converged
    assert "diagonal differs" in result.triples_response.energy.reason
    assert not result.fock_moments and not result.triples_response.cotangents


def test_closed_borrowed_triples_owner_fails_closed(small_fock: tuple) -> None:
    triples, owner = small_fock
    triples.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.response()


@pytest.mark.parametrize("nmo,oa,ob", [(3, 3, 0), (3, 3, 3), (4, 1, 0)])
def test_empty_fock_response_retains_complete_zero_spin_moments(
    nmo: int, oa: int, ob: int
) -> None:
    snapshot, source = _noninteracting(nmo, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDTriplesResponse(parent, state) as triples,
            PreparedUCCSDTriplesFockResponse(triples) as owner,
        ):
            result = owner.response()
            assert result.converged
            assert all(not np.any(values) for values in result.fock_moments.values())
            assert not owner.pair_count and not owner.vector_count


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires finite Slurm allocated GPU",
)
@pytest.mark.parametrize("degenerate", [False, True])
def test_native_cuda_all_sectors_and_degenerate_resolvent_closure(
    degenerate: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor

    assert os.environ.get("SLURM_JOB_ID")
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    cache = Path(os.environ.get("GENERATIVEQC_UCCSD_T_FOCK_TEST_CACHE", str(tmp_path)))

    def forbidden(*_: Any, **__: Any) -> None:
        pytest.fail("CUDA Fock resolvent executed CPU equations")

    monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
    executor = CudaCCTensorExecutor(
        max_bytes=512 << 20, compiler=compiler, cache=cache / "tensor"
    )
    _check_native_vectors(executor, degenerate)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires finite Slurm allocated GPU",
)
@pytest.mark.parametrize("atom,spin,broken,tile_shape", _MOLECULAR_FOCK_CASES)
def test_complete_cuda_molecular_fock_response_has_independent_full_spin_oracle(
    atom: str,
    spin: int,
    broken: bool,
    tile_shape: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID")
    snapshot, source, _, _ = _molecule(atom, spin, broken)
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    cache = Path(os.environ.get("GENERATIVEQC_UCCSD_T_FOCK_TEST_CACHE", str(tmp_path)))
    artifact = compile_cuda(compiler, cache / "transform")
    executor = CudaCCTensorExecutor(
        max_bytes=512 << 20, compiler=compiler, cache=cache / "tensor"
    )
    with (
        ConventionalProvider(
            snapshot,
            source,
            backend="cuda",
            cuda_artifact=artifact,
            budget_bytes=512 << 20,
            cache_policy="lru",
        ) as provider,
        PreparedUCCSD(
            snapshot,
            provider,
            executor=executor,
            options=SolverOptions(max_iterations=400),
        ) as parent,
    ):

        def forbidden(*_: Any, **__: Any) -> None:
            pytest.fail("complete CUDA Fock response executed CPU equations")

        monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDTriplesResponse(
                parent,
                state,
                options=UCCSDTOptions(tile_shape=tile_shape),
            ) as triples,
            PreparedUCCSDTriplesFockResponse(triples) as owner,
        ):
            result = owner.response()
            assert result.converged, result.triples_response.energy.reason
            fock, integrals, energies = _snapshot_hamiltonian(parent, source)
            _, _, left, right, _ = _oracle_vectors(
                parent.layout, fock, integrals, dict(state.amplitudes), energies
            )
            expected, positions = (
                _oracle_moments(left, right),
                _positions(parent.layout),
            )
            for name, moment in result.fock_moments.items():
                domain, spin_label = name[1], name[-1]
                coordinates = positions[domain, spin_label]
                np.testing.assert_allclose(
                    moment,
                    expected[0 if domain == "o" else 1][
                        np.ix_(coordinates, coordinates)
                    ],
                    atol=2e-10,
                    rtol=0,
                )
            assert result.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
