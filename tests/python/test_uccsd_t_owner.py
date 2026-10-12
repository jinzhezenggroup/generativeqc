"""Complete UCCSD(T) molecular, resource and transactional-failure gates."""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from test_uccsd_equations import _molecular_case
from test_uccsd_solver import _snapshot
from test_uccsd_triples import _oracle_channels
from test_ump2_energy import DenseSource

from tools.generativeqc_cc import rccsd_t_energy
from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_cc.uccsd_t import PreparedUCCSDT, UCCSDTOptions
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.reference import ReferenceSnapshot
from tools.generativeqc_response.uhf import UHFReferenceSnapshot

_BROKEN_H4 = "; ".join(f"H 0 0 {1.3 * position}" for position in range(4))
_ASYMMETRIC_H6 = "H 0 0 0; H 0 0 1; H .1 .05 2; H -.1 .02 3; H .1 -.04 4; H 0 .1 5"


@lru_cache(maxsize=8)
def _molecule(atom: str, spin: int, broken: bool = False) -> tuple:
    restricted = spin == 0 and not broken
    coupled_cluster, _, _, _, _ = _molecular_case(
        atom, spin, broken=broken, restricted=restricted
    )
    reference = coupled_cluster._scf
    reference.direct_scf = False
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 300
    reference.kernel(dm0=reference.make_rdm1())
    if not reference.converged:
        reference = reference.newton()
        reference.conv_tol = 1e-13
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 300
        reference.kernel(dm0=reference.make_rdm1())
    assert reference.converged
    if broken:
        density = reference.make_rdm1()
        assert np.linalg.norm(density[0] - density[1]) > 0.1
    if restricted:
        # Keep the collapsed-spin reference stationary at the strict gate; UHF
        # DIIS can stall above it even when the corresponding RHF converges.
        reference = reference.to_uhf()
    cc = pytest.importorskip("pyscf.cc")
    coupled_cluster = cc.UCCSD(reference)
    eris = coupled_cluster.ao2mo()
    coupled_cluster.conv_tol = 1e-13
    coupled_cluster.conv_tol_normt = 1e-10
    coupled_cluster.max_cycle = 400
    correction, _, _ = coupled_cluster.kernel(eris=eris)
    assert coupled_cluster.converged
    amplitudes = dict(
        zip(
            ("t1a", "t1b", "t2aa", "t2ab", "t2bb"),
            (*coupled_cluster.t1, *coupled_cluster.t2),
            strict=True,
        )
    )
    channels = _oracle_channels(coupled_cluster, eris, amplitudes)
    snapshot = _snapshot(coupled_cluster)
    source = DenseSource(reference.mol.intor("int2e"), snapshot)
    return snapshot, source, correction, channels


def _noninteracting(nmo: int, oa: int, ob: int, energies: Any = None) -> tuple:
    """A canonical exact one-body Hamiltonian for resource/fault tests only."""
    energies = np.arange(nmo, dtype=float) - oa if energies is None else energies
    identity = np.eye(nmo)
    fock = np.diag(energies)
    snapshot = UHFReferenceSnapshot(
        overlap=identity,
        hcore=fock,
        fock_alpha=fock,
        fock_beta=fock,
        coefficients_alpha=identity,
        coefficients_beta=identity,
        orbital_energies_alpha=energies,
        orbital_energies_beta=energies,
        occupations_alpha=np.r_[np.ones(oa), np.zeros(nmo - oa)],
        occupations_beta=np.r_[np.ones(ob), np.zeros(nmo - ob)],
        reference_energy=float(sum(energies[:oa]) + sum(energies[:ob])),
        scf_residual=0.0,
        geometry_hash="one-body-test-geometry",
        basis_hash="one-body-test-basis",
        generation_id="one-body-test-generation",
        hf_backend="exact-one-body-test",
    )
    return snapshot, DenseSource(np.zeros((nmo,) * 4), snapshot)


@pytest.fixture
def small_parent() -> Any:
    snapshot, source = _noninteracting(4, 2, 2)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        yield parent, state, source


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("O 0 0 0; H 0 0 .97", 1, False),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
    ],
)
def test_complete_cpu_energy_and_channels_against_pyscf(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, correction, channels = _molecule(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source, cache_policy="lru") as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged, state.history[-1]
        with PreparedUCCSDT(parent, state) as prepared:
            result = prepared.energy()
            assert result.converged, result.reason
            assert result.correlation_energy is not None
            assert result.total_energy is not None
            for channel, expected in channels.items():
                np.testing.assert_allclose(
                    result.channel_energies[channel], expected, atol=5e-10, rtol=0
                )
            np.testing.assert_allclose(
                result.correlation_energy,
                correction + sum(channels.values()),
                atol=3e-9,
                rtol=0,
            )
            np.testing.assert_allclose(
                result.total_energy,
                snapshot.reference_energy + correction + sum(channels.values()),
                atol=3e-9,
                rtol=0,
            )
            assert (
                result.provenance["completed_tiles"]
                == result.provenance["planned_tiles"]
            )
            assert (
                result.provenance["numeric_bound_bytes"] <= prepared.options.max_bytes
            )
            assert (
                result.provenance["semantic_scalar_work"]
                <= prepared.options.max_total_work
            )
            assert result.provenance["tensor_backend"] == "native-cpu-tensorir"
            assert (
                result.provenance["fresh_parent_residual_max"]
                <= parent.options.residual_tolerance
            )
            assert (
                result.provenance["fresh_parent_energy_drift"]
                <= parent.options.energy_tolerance
            )
            if atom == _ASYMMETRIC_H6:
                assert all(
                    abs(value) > 1e-14 for value in result.channel_energies.values()
                )


def test_memory_fallback_preserves_complete_grid_and_energy() -> None:
    snapshot, source = _noninteracting(8, 4, 3)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDT(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
        ) as full:
            baseline = full.energy()
            assert full.peak_bytes > full.replay_peak_bytes
            bound = (full.peak_bytes + full.replay_peak_bytes) // 2
        with PreparedUCCSDT(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6, max_bytes=bound)
        ) as bounded:
            actual = bounded.energy()
            assert actual.converged and baseline.converged
            assert actual.total_energy is not None and baseline.total_energy is not None
            assert bounded.fallback_steps > 0
            assert bounded.peak_bytes <= bound
            assert bounded.tile_count > baseline.provenance["planned_tiles"]
            np.testing.assert_allclose(
                actual.total_energy, baseline.total_energy, atol=1e-13
            )
        with pytest.raises(MemoryError, match="tile-count"):
            PreparedUCCSDT(
                parent,
                state,
                options=UCCSDTOptions(
                    tile_shape=(8,) * 6,
                    max_bytes=bound,
                    max_tiles=baseline.provenance["planned_tiles"],
                ),
            )
        with pytest.raises(MemoryError, match="semantic-work"):
            PreparedUCCSDT(
                parent,
                state,
                options=UCCSDTOptions(
                    tile_shape=(8,) * 6,
                    max_bytes=bound,
                    max_total_work=actual.provenance["semantic_scalar_work"] - 1,
                ),
            )


def test_complete_restricted_limit_matches_existing_rccsdt_owner() -> None:
    snapshot, source, _, channels = _molecule(_ASYMMETRIC_H6, 0)
    restricted = ReferenceSnapshot(
        overlap=snapshot.overlap,
        hcore=snapshot.hcore,
        fock=snapshot.fock_alpha,
        coefficients=snapshot.coefficients_alpha,
        orbital_energies=snapshot.orbital_energies_alpha,
        occupations=2 * snapshot.occupations_alpha,
        electron_count=2 * snapshot.nocc("alpha"),
        reference_energy=snapshot.reference_energy,
        scf_residual=snapshot.scf_residual,
        geometry_hash=snapshot.geometry_hash,
        basis_hash=snapshot.basis_hash,
        generation_id=snapshot.generation_id,
    )
    with ConventionalProvider(restricted, source) as provider:
        expected = rccsd_t_energy(
            restricted, provider, options=SolverOptions(max_iterations=400)
        )
    assert expected.converged, expected.reason
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged, state.reason
        with PreparedUCCSDT(parent, state) as prepared:
            actual = prepared.energy()
    assert actual.converged, actual.reason
    assert actual.triples_energy is not None and expected.triples_energy is not None
    assert actual.total_energy is not None and expected.total_energy is not None
    assert abs(sum(channels.values())) > 1e-8
    np.testing.assert_allclose(
        actual.triples_energy, expected.triples_energy, atol=5e-10, rtol=0
    )
    np.testing.assert_allclose(
        actual.total_energy, expected.total_energy, atol=3e-9, rtol=0
    )


def test_nonzero_molecular_memory_fallback_preserves_oracle_energy() -> None:
    snapshot, source, _, channels = _molecule(_ASYMMETRIC_H6, 0)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged, state.reason
        with PreparedUCCSDT(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
        ) as full:
            baseline = full.energy()
            assert full.peak_bytes > full.replay_peak_bytes
            bound = (full.peak_bytes + full.replay_peak_bytes) // 2
        with PreparedUCCSDT(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6, max_bytes=bound)
        ) as bounded:
            actual = bounded.energy()
            assert bounded.fallback_steps > 0
            assert bounded.peak_bytes <= bound
            assert actual.provenance["completed_tiles"] == bounded.tile_count
        assert baseline.converged and actual.converged
        assert baseline.triples_energy is not None and actual.triples_energy is not None
        assert baseline.total_energy is not None and actual.total_energy is not None
        assert abs(baseline.triples_energy) > 1e-8
        np.testing.assert_allclose(
            actual.triples_energy, sum(channels.values()), atol=5e-10, rtol=0
        )
        np.testing.assert_allclose(
            actual.total_energy, baseline.total_energy, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("nmo,oa,ob", [(3, 3, 0), (3, 3, 3), (4, 1, 0)])
def test_empty_excitation_or_triples_domain_publishes_exact_zero(
    nmo: int, oa: int, ob: int
) -> None:
    snapshot, source = _noninteracting(nmo, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDT(parent, state) as prepared:
            actual = prepared.energy()
            assert not prepared.channels and not prepared.templates
        assert actual.converged
        assert actual.triples_energy == actual.correlation_energy == 0
        assert actual.total_energy == snapshot.reference_energy
        assert actual.channel_energies == dict.fromkeys(("aaa", "aab", "abb", "bbb"), 0)
        assert actual.provenance["completed_tiles"] == 0
        assert actual.provenance["minimum_absolute_executed_denominator"] is None


@pytest.mark.parametrize(
    "options,match",
    [
        (UCCSDTOptions(max_bytes=1), "replay"),
        (UCCSDTOptions(tile_shape=(1,) * 6, max_tiles=1), "tile-count"),
        (UCCSDTOptions(max_total_work=1), "semantic-work"),
        (UCCSDTOptions(denominator_threshold=1e6), "denominator"),
    ],
)
def test_complete_admission_precedes_native_execution(
    small_parent: tuple,
    options: UCCSDTOptions,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, state, source = small_parent
    reads = source.reads

    def forbidden(*_: Any) -> None:
        pytest.fail("rejected complete admission executed a native program")

    monkeypatch.setattr(parent.executor, "execute", forbidden)
    with pytest.raises((MemoryError, ValueError), match=match):
        PreparedUCCSDT(parent, state, options=options)
    assert source.reads == reads


def test_denominator_extrema_overflow_fail_closed() -> None:
    energies = np.r_[np.full(3, -3.01e307), np.full(3, 3.01e307)]
    snapshot, source = _noninteracting(6, 3, 0, energies)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with pytest.raises(ValueError, match="nonfinite.*denominator"):
            PreparedUCCSDT(parent, state)


@pytest.mark.parametrize(
    "energies,threshold",
    [
        (np.r_[np.full(3, 1e16), np.full(3, 1e16 + 2)], 5),
        (np.array([-1, -1 + 5e-9, -1, 0, -5e-9, 0]), 3 - 1e-8),
    ],
)
def test_denominator_gate_uses_executed_fp64_extrema(
    energies: np.ndarray, threshold: float
) -> None:
    snapshot, source = _noninteracting(6, 3, 0, energies)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        # Respect ordered FP64 cancellation and reference-tolerated disorder.
        with pytest.raises(ValueError, match="denominator"):
            PreparedUCCSDT(
                parent, state, options=UCCSDTOptions(denominator_threshold=threshold)
            )


def test_false_or_foreign_converged_state_is_rejected(small_parent: tuple) -> None:
    parent, state, _ = small_parent
    with pytest.raises(ValueError, match="converged"):
        PreparedUCCSDT(parent, replace(state, status="iteration_limit"))
    with pytest.raises(ValueError, match="identities"):
        PreparedUCCSDT(
            parent,
            replace(state, provenance={**state.provenance, "reference_id": "other"}),
        )
    packed = state.packed_amplitudes.copy()
    packed[0] += 0.01
    with pytest.raises(ValueError, match="packed"):
        PreparedUCCSDT(parent, replace(state, packed_amplitudes=packed))
    forged = replace(
        state, packed_amplitudes=packed, amplitudes=parent.layout.unpack(packed)
    )
    with (
        PreparedUCCSDT(parent, forged) as prepared,
        pytest.raises(ValueError, match="fresh physical"),
    ):
        prepared.energy()


def test_lifetimes_immutable_publication_and_borrowed_configuration(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, state, source = small_parent
    with PreparedUCCSDT(parent, state) as prepared:
        source.closed = True
        result = prepared.energy()
        assert result.converged
        for value in (prepared.packed_amplitudes, *prepared.amplitudes.values()):
            with pytest.raises(ValueError):
                value.setflags(write=True)
        with pytest.raises(TypeError):
            cast("dict[str, float]", result.channel_energies)["aab"] = 0.0
        with monkeypatch.context() as context:
            context.setattr(parent.executor, "max_bytes", parent.executor.max_bytes * 2)
            with pytest.raises(RuntimeError, match="configuration"):
                prepared.energy()
        with monkeypatch.context() as context:
            context.setattr(prepared, "options", UCCSDTOptions(max_bytes=1))
            with pytest.raises(RuntimeError, match="configuration"):
                prepared.energy()
        parent.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared.energy()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.energy()


def test_late_nonfinite_and_execution_failure_publish_no_partial_energy(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, state, _ = small_parent
    with PreparedUCCSDT(
        parent, state, options=UCCSDTOptions(tile_shape=(1,) * 6)
    ) as prepared:
        original = parent.executor.execute
        seen = 0

        def failing(program: Any, feeds: Any) -> dict:
            nonlocal seen
            if set(program.outputs) == {"triples_energy"}:
                if seen:
                    return {"triples_energy": np.array(float("nan"))}
                seen += 1
            return original(program, feeds)

        monkeypatch.setattr(parent.executor, "execute", failing)
        failed = prepared.energy()
        assert failed.status == "nonfinite"
        assert 0 < failed.provenance["completed_tiles"] < prepared.tile_count
        assert failed.triples_energy is None and failed.total_energy is None
        assert not failed.channel_energies

        def native_failure(program: Any, feeds: Any) -> dict:
            if set(program.outputs) == {"triples_energy"}:
                raise ValueError("injected native tensor evaluation failure")
            return original(program, feeds)

        monkeypatch.setattr(parent.executor, "execute", native_failure)
        failed = prepared.energy()
        assert failed.status == "execution_error"
        assert failed.correlation_energy is None and failed.total_energy is None
        assert not failed.channel_energies


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_shape": ()},
        {"tile_shape": (True,) * 6},
        {"tile_shape": (0,) * 6},
        {"max_tiles": 0},
        {"max_total_work": True},
        {"denominator_threshold": float("nan")},
    ],
)
def test_invalid_options_fail_closed(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        UCCSDTOptions(**kwargs)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
    ],
)
def test_complete_cuda_energy_matches_independent_pyscf(
    atom: str, spin: int, broken: bool, tmp_path: Path
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID"), (
        "real CUDA gates require finite Slurm allocation"
    )
    snapshot, source, correction, channels = _molecule(atom, spin, broken)
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
        state = parent.solve()
        assert state.converged, state.history[-1]
        with PreparedUCCSDT(parent, state) as prepared:
            result = prepared.energy()
            assert result.converged, result.reason
            assert result.correlation_energy is not None
            assert result.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
            assert result.provenance["integral_backend"] == "cuda"
            assert result.provenance["completed_tiles"] == prepared.tile_count
            assert (
                result.provenance["numeric_bound_bytes"] <= prepared.options.max_bytes
            )
            assert (
                result.provenance["semantic_scalar_work"]
                <= prepared.options.max_total_work
            )
            for channel, expected in channels.items():
                np.testing.assert_allclose(
                    result.channel_energies[channel], expected, atol=5e-10, rtol=0
                )
            np.testing.assert_allclose(
                result.correlation_energy,
                correction + sum(channels.values()),
                atol=3e-9,
                rtol=0,
            )
