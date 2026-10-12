"""Independent all-spin derivatives and complete bounded response contracts."""

from __future__ import annotations

import os
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_triples import (
    CHANNELS,
    build_tile_response_program,
    channel_shape,
)
from test_uccsd_equations import (
    _amplitudes,
    _feeds,
    _random_hamiltonian,
    _spin_amplitudes,
)
from test_uccsd_t_owner import (
    _ASYMMETRIC_H6,
    _BROKEN_H4,
    _molecule,
    _noninteracting,
)
from test_uccsd_t_owner import small_parent as _small_parent
from test_uccsd_triples import _tile_feeds

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_cc.uccsd_t import PreparedUCCSDT, UCCSDTOptions
from tools.generativeqc_cc.uccsd_triples_response import PreparedUCCSDTriplesResponse
from tools.generativeqc_posthf.providers import ConventionalProvider

small_parent = _small_parent


def _source_feeds(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    amplitudes: dict,
    energies: np.ndarray,
) -> dict:
    feeds = _feeds(layout, fock, integrals, amplitudes)
    segments = (layout.oa, layout.ob, layout.va, layout.vb)
    offset = 0
    for name, size in zip(
        ("eps_o_a", "eps_o_b", "eps_v_a", "eps_v_b"), segments, strict=True
    ):
        feeds[name] = energies[offset : offset + size]
        offset += size
    return feeds


def _independent_channels(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    amplitudes: dict,
    energies: np.ndarray,
) -> dict:
    """Use PySCF's triangular GCCSD(T), not the production ordered tile graph.

    Each scalar reduction owns one distinct virtual triple. Group those
    independent reductions by its actual virtual spins; occupied spin is
    conserved by the physical test Hamiltonian. The oracle enumerates only
    distinct triangular virtual triples, unlike production's ordered grid.
    """
    reference = pytest.importorskip("pyscf.cc.gccsd_t")
    gccsd = pytest.importorskip("pyscf.cc.gccsd")
    eris = gccsd._PhysicistsERIs()
    occupied = layout.oa + layout.ob
    eris.fock, eris.mo_energy = fock, energies
    eris.ovvv = integrals[:occupied, occupied:, occupied:, occupied:]
    eris.ooov = integrals[:occupied, :occupied, :occupied, occupied:]
    eris.oovv = integrals[:occupied, :occupied, occupied:, occupied:]
    singles, doubles = _spin_amplitudes(layout, amplitudes)
    contributions = []

    def contraction(equation: str, *values: Any, **kwargs: Any) -> Any:
        result = np.einsum(equation, *values, **kwargs)
        if equation == "ijk,ijk":
            contributions.append(float(result) / 2)
        return result

    class OracleArrays:
        asarray = staticmethod(np.asarray)
        einsum = staticmethod(contraction)

    with patch.object(reference, "numpy", OracleArrays):
        total = reference.kernel(None, eris, singles, doubles)
    virtual_spins = "a" * layout.va + "b" * layout.vb
    channels = dict.fromkeys(CHANNELS, 0.0)
    triples = (
        (first, second, third)
        for first in range(len(virtual_spins))
        for second in range(first)
        for third in range(second)
    )
    for coordinates, value in zip(triples, contributions, strict=True):
        channel = "".join(sorted(virtual_spins[index] for index in coordinates))
        channels[channel] += value
    np.testing.assert_allclose(sum(channels.values()), total, atol=3e-13, rtol=3e-13)
    return channels


def _directions(layout: UCCSDLayout) -> tuple:
    fock, integrals = _random_hamiltonian(layout)
    sizes = (layout.oa, layout.ob, layout.va, layout.vb)
    boundaries = np.cumsum((0, *sizes))
    permutation = np.concatenate(
        [np.arange(start, stop)[::-1] for start, stop in pairwise(boundaries)]
    )
    return (
        _amplitudes(layout, seed=87),
        0.1 * fock[np.ix_(permutation, permutation)],
        0.1 * integrals[np.ix_(permutation, permutation, permutation, permutation)],
        np.random.default_rng(89).normal(scale=0.05, size=sum(sizes)),
    )


def _check_directional_derivatives(
    layout: UCCSDLayout,
    fock: np.ndarray,
    integrals: np.ndarray,
    amplitudes: dict,
    energies: np.ndarray,
    bars: dict,
    channel: str | None = None,
) -> None:
    directions = _directions(layout)
    step = 1e-5
    for family in range(4):
        delta_amplitudes, delta_fock, delta_integrals, delta_energies = (
            directions[0]
            if family == 0
            else {name: np.zeros_like(value) for name, value in amplitudes.items()},
            directions[1] if family == 1 else np.zeros_like(fock),
            directions[2] if family == 2 else np.zeros_like(integrals),
            directions[3] if family == 3 else np.zeros_like(energies),
        )
        delta = _source_feeds(
            layout, delta_fock, delta_integrals, delta_amplitudes, delta_energies
        )
        actual = sum(np.vdot(values, delta[name]) for name, values in bars.items())
        expected = []
        for sign in (1, -1):
            result = _independent_channels(
                layout,
                fock + sign * step * delta_fock,
                integrals + sign * step * delta_integrals,
                {
                    name: value + sign * step * delta_amplitudes[name]
                    for name, value in amplitudes.items()
                },
                energies + sign * step * delta_energies,
            )
            expected.append(result[channel] if channel else sum(result.values()))
        np.testing.assert_allclose(
            actual, (expected[0] - expected[1]) / (2 * step), atol=3e-10, rtol=3e-9
        )
        assert abs(actual) > 1e-13, (channel, family)


def _snapshot_hamiltonian(parent: PreparedUCCSD, source: Any) -> tuple:
    """Test-only independent full-spin AO transform for the pinned oracle."""
    snapshot, layout = parent.reference, parent.layout
    coefficients = np.concatenate(
        (
            snapshot.coefficients_alpha[:, : layout.oa],
            snapshot.coefficients_beta[:, : layout.ob],
            snapshot.coefficients_alpha[:, layout.oa :],
            snapshot.coefficients_beta[:, layout.ob :],
        ),
        axis=1,
    )
    spins = np.repeat((0, 1, 0, 1), (layout.oa, layout.ob, layout.va, layout.vb))
    coulomb = np.einsum(
        "uvwx,up,vr,wq,xs->pqrs",
        source.ao,
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        optimize=True,
    )
    coulomb *= (spins[:, None, None, None] == spins[None, None, :, None]) & (
        spins[None, :, None, None] == spins[None, None, None, :]
    )
    fock = np.zeros((len(spins), len(spins)))
    for spin, suffix in enumerate(("alpha", "beta")):
        frame = getattr(snapshot, f"coefficients_{suffix}")
        positions = np.flatnonzero(spins == spin)
        fock[np.ix_(positions, positions)] = (
            frame.T @ getattr(snapshot, f"fock_{suffix}") @ frame
        )
    energies = np.concatenate(
        (
            snapshot.orbital_energies_alpha[: layout.oa],
            snapshot.orbital_energies_beta[: layout.ob],
            snapshot.orbital_energies_alpha[layout.oa :],
            snapshot.orbital_energies_beta[layout.ob :],
        )
    )
    return fock, coulomb - coulomb.swapaxes(2, 3), energies


def _check_tile_response(executor: Any, channel: str) -> None:
    layout = UCCSDLayout(3, 4, 4, 3)
    fock, integrals = _random_hamiltonian(layout)
    amplitudes = _amplitudes(layout)
    energies = np.r_[
        np.linspace(-2, -1, layout.oa + layout.ob),
        np.linspace(1, 2, layout.va + layout.vb),
    ]
    sources = _source_feeds(layout, fock, integrals, amplitudes, energies)
    tile = build_tile_response_program(layout, channel, channel_shape(layout, channel))
    coordinates = tuple(tuple(range(size)) for size in tile.shape)
    outputs = executor.execute(
        tile.program,
        {
            **_tile_feeds(tile, sources, coordinates),
            "bar_triples_energy": np.array(1.0),
        },
    )
    expected = _independent_channels(layout, fock, integrals, amplitudes, energies)
    np.testing.assert_allclose(
        outputs["triples_energy"], expected[channel], atol=3e-13, rtol=3e-13
    )
    bars = {name: np.zeros_like(value) for name, value in sources.items()}
    for binding in tile.operands:
        bars[binding.source] += np.asarray(outputs[f"bar_{binding.name}"]).transpose(
            tuple(np.argsort(binding.axes))
        )
    _check_directional_derivatives(
        layout, fock, integrals, amplitudes, energies, bars, channel
    )


@pytest.mark.parametrize("channel", CHANNELS)
def test_native_all_source_derivatives_against_independent_triangular_oracle(
    channel: str, tmp_path: Path
) -> None:
    _check_tile_response(
        NativeCCTensorExecutor(max_bytes=256 << 20, cache=tmp_path), channel
    )


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        (_ASYMMETRIC_H6, 0, False),
        (_BROKEN_H4, 0, True),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
    ],
)
def test_complete_cpu_response_and_tiled_scatter(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, _, channels = _molecule(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDTriplesResponse(
            parent, state, options=UCCSDTOptions(tile_shape=(2, 8, 8, 8, 8, 8))
        ) as response:
            actual = response.response()
            assert actual.converged, actual.energy.reason
            assert actual.provenance["completed_tiles"] == response.tile_count
            for channel, expected in channels.items():
                np.testing.assert_allclose(
                    actual.energy.channel_energies[channel],
                    expected,
                    atol=5e-10,
                    rtol=0,
                )
            fock, integrals, energies = _snapshot_hamiltonian(parent, source)
            _check_directional_derivatives(
                parent.layout,
                fock,
                integrals,
                dict(state.amplitudes),
                energies,
                dict(actual.cotangents),
            )
            hamiltonian = actual.hamiltonian_cotangents()
            delta_amplitudes, delta_fock, delta_integrals, _ = _directions(
                parent.layout
            )
            physical_direction = _source_feeds(
                parent.layout,
                delta_fock,
                delta_integrals,
                delta_amplitudes,
                np.diag(delta_fock),
            )
            np.testing.assert_allclose(
                sum(
                    np.vdot(values, physical_direction[name])
                    for name, values in hamiltonian.items()
                ),
                sum(
                    np.vdot(values, physical_direction[name])
                    for name, values in actual.cotangents.items()
                    if name not in BLOCK_NAMES
                ),
                atol=3e-13,
                rtol=3e-13,
            )
            assert not any(
                name.startswith("eps_") or name in BLOCK_NAMES for name in hamiltonian
            )
            direction = _directions(parent.layout)[0]
            np.testing.assert_allclose(
                np.vdot(
                    actual.packed_amplitude_cotangent, parent.layout.pack(direction)
                ),
                sum(
                    np.vdot(actual.cotangents[name], direction[name])
                    for name in BLOCK_NAMES
                ),
                atol=3e-13,
            )
        with PreparedUCCSDTriplesResponse(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
        ) as full:
            baseline = full.response()
            assert baseline.converged, baseline.energy.reason
            for name in actual.cotangents:
                np.testing.assert_allclose(
                    actual.cotangents[name],
                    baseline.cotangents[name],
                    atol=3e-12,
                    rtol=3e-12,
                )
        with pytest.raises(ValueError):
            actual.cotangents["t1a"].setflags(write=True)
        with pytest.raises(TypeError):
            actual.provenance["options"]["max_bytes"] = 1


@pytest.mark.parametrize(
    "options,match",
    [
        (UCCSDTOptions(max_bytes=1), "replay"),
        (UCCSDTOptions(tile_shape=(1,) * 6, max_tiles=1), "tile-count"),
        (UCCSDTOptions(max_total_work=1), "semantic-work"),
        (UCCSDTOptions(denominator_threshold=1e6), "denominator"),
    ],
)
def test_response_admission_before_execution(
    small_parent: tuple,
    options: UCCSDTOptions,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, state, _ = small_parent

    def forbidden(*_: Any) -> None:
        pytest.fail("rejected derivative admission executed native equations")

    monkeypatch.setattr(parent.executor, "execute", forbidden)
    with pytest.raises((MemoryError, ValueError), match=match):
        PreparedUCCSDTriplesResponse(parent, state, options=options)


@pytest.mark.parametrize(
    "fault", ["nonfinite", "float32", "shape", "missing", "native"]
)
def test_late_tile_failure_never_publishes_partial_bars(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    parent, state, _ = small_parent
    with PreparedUCCSDTriplesResponse(
        parent, state, options=UCCSDTOptions(tile_shape=(1,) * 6)
    ) as response:
        execute = parent.executor.execute
        completed = 0

        def failing(program: Any, feeds: Any) -> Any:
            nonlocal completed
            outputs = execute(program, feeds)
            if "triples_energy" not in outputs:
                return outputs
            completed += 1
            if completed == 2:
                name = next(name for name in outputs if name.startswith("bar_"))
                if fault == "native":
                    raise RuntimeError("injected late native failure")
                if fault == "missing":
                    del outputs[name]
                else:
                    values = np.asarray(outputs[name])
                    outputs[name] = (
                        np.full_like(values, np.nan)
                        if fault == "nonfinite"
                        else values.astype(np.float32)
                        if fault == "float32"
                        else np.zeros((999,))
                    )
            return outputs

        monkeypatch.setattr(parent.executor, "execute", failing)
        result = response.response()
        assert not result.converged
        assert result.energy.provenance["completed_tiles"] > 0
        assert not result.cotangents and not result.energy.channel_energies
        assert result.packed_amplitude_cotangent is result.response_identity is None
        assert result.energy.triples_energy is result.energy.total_energy is None
        with pytest.raises(ValueError, match="failed"):
            result.hamiltonian_cotangents()


def test_response_memory_fallback_and_borrowed_lifetime(small_parent: tuple) -> None:
    parent, state, _ = small_parent
    options = UCCSDTOptions(tile_shape=(8,) * 6)
    with PreparedUCCSDTriplesResponse(parent, state, options=options) as full:
        baseline = full.response()
        assert baseline.converged
        assert full.peak_bytes > full.replay_peak_bytes
        bound = (full.peak_bytes + full.replay_peak_bytes) // 2
    with PreparedUCCSDTriplesResponse(
        parent, state, options=replace(options, max_bytes=bound)
    ) as bounded:
        actual = bounded.response()
        assert actual.converged and bounded.fallback_steps > 0
        assert actual.provenance["numeric_bound_bytes"] <= bound
        for name, values in actual.cotangents.items():
            np.testing.assert_array_equal(values, baseline.cotangents[name])
        original = bounded.options
        bounded.options = replace(original, max_tiles=1)
        with pytest.raises(RuntimeError, match="configuration"):
            bounded.response()
        bounded.options = original
    with pytest.raises(RuntimeError, match="closed"):
        bounded.response()
    assert not parent._closed
    with PreparedUCCSDT(parent, state) as energy:
        assert energy.energy().converged


def test_fresh_response_rejects_nonstationary_parent(small_parent: tuple) -> None:
    parent, state, _ = small_parent
    amplitudes = {name: value.copy() for name, value in state.amplitudes.items()}
    amplitudes["t1a"][0, 0] = 0.01
    stale = replace(
        state, amplitudes=amplitudes, packed_amplitudes=parent.layout.pack(amplitudes)
    )
    with (
        PreparedUCCSDTriplesResponse(parent, stale) as response,
        pytest.raises(ValueError, match="fresh physical"),
    ):
        response.response()


def test_response_record_owns_and_validates_cotangents(small_parent: tuple) -> None:
    parent, state, source = small_parent
    with PreparedUCCSDTriplesResponse(parent, state) as response:
        source.closed = True
        result = response.response()
        assert result.converged
        copied = {name: value.copy() for name, value in result.cotangents.items()}
        detached = replace(result, cotangents=copied)
        copied["t1a"][0, 0] = 1.0
        assert detached.cotangents["t1a"][0, 0] == 0
        with pytest.raises(ValueError, match="packed cotangents"):
            replace(result, cotangents=copied)
        missing = dict(result.cotangents)
        del missing["eps_o_a"]
        with pytest.raises(ValueError, match="denominators"):
            replace(result, cotangents=missing)
        with pytest.raises(ValueError, match="finite base FP64"):
            replace(
                result,
                packed_amplitude_cotangent=result.packed_amplitude_cotangent.astype(
                    np.float32
                ),
            )
        for values in result.hamiltonian_cotangents().values():
            with pytest.raises(ValueError):
                values.setflags(write=True)
        with pytest.raises(RuntimeError, match="closed"):
            parent.close()
            response.response()


@pytest.mark.parametrize("nmo,oa,ob", [(3, 3, 0), (3, 3, 3), (4, 1, 0)])
def test_empty_response_domains_publish_exact_zero(nmo: int, oa: int, ob: int) -> None:
    snapshot, source = _noninteracting(nmo, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        with PreparedUCCSDTriplesResponse(parent, state) as response:
            result = response.response()
            assert result.converged and result.energy.triples_energy == 0
            assert all(not np.any(values) for values in result.cotangents.values())
            assert not np.any(result.packed_amplitude_cotangent)
            assert result.provenance["completed_tiles"] == 0
            assert all(
                not np.any(values)
                for values in result.hamiltonian_cotangents().values()
            )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU opt-in",
)
@pytest.mark.parametrize("channel", CHANNELS)
def test_real_cuda_all_source_derivatives_against_independent_triangular_oracle(
    channel: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor

    assert os.environ.get("SLURM_JOB_ID")
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    executor = CudaCCTensorExecutor(
        max_bytes=256 << 20, compiler=compiler, cache=tmp_path
    )

    def forbidden(*_: Any, **__: Any) -> None:
        pytest.fail("CUDA triples derivatives executed CPU equations")

    monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
    _check_tile_response(executor, channel)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU opt-in",
)
@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        (_ASYMMETRIC_H6, 0, False),
        (_BROKEN_H4, 0, True),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
    ],
)
def test_complete_cuda_response_has_no_cpu_equation_fallback(
    atom: str, spin: int, broken: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID")
    snapshot, source, _, channels = _molecule(atom, spin, broken)
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    artifact = compile_cuda(compiler, tmp_path / "transform")
    executor = CudaCCTensorExecutor(
        max_bytes=512 << 20, compiler=compiler, cache=tmp_path / "tensor"
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
            options=SolverOptions(max_iterations=400),
            executor=executor,
        ) as parent,
    ):

        def forbidden(*_: Any, **__: Any) -> None:
            pytest.fail("CUDA response executed CPU equations")

        monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDTriplesResponse(
            parent, state, options=UCCSDTOptions(tile_shape=(2, 8, 8, 8, 8, 8))
        ) as response:
            result = response.response()
            assert result.converged, result.energy.reason
            assert result.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
            assert result.provenance["integral_backend"] == "cuda"
            assert result.provenance["completed_tiles"] == response.tile_count
            for channel, expected in channels.items():
                np.testing.assert_allclose(
                    result.energy.channel_energies[channel],
                    expected,
                    atol=5e-10,
                    rtol=0,
                )
            fock, integrals, energies = _snapshot_hamiltonian(parent, source)
            _check_directional_derivatives(
                parent.layout,
                fock,
                integrals,
                dict(state.amplitudes),
                energies,
                dict(result.cotangents),
            )
            assert not any(
                name.startswith("eps_") for name in result.hamiltonian_cotangents()
            )
