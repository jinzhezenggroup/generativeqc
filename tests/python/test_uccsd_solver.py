"""Complete conventional UCCSD iteration and spin-provider qualification."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from test_uccsd_equations import (
    _amplitudes,
    _assert_outputs,
    _determinant_oracle,
    _feeds,
    _molecular_case,
)
from test_ump2_energy import DenseSource, _eri, _reference

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
from tools.generativeqc_cc.solver import SolverOptions, solve
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_posthf.conventions import SpinMOBlock
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.reference import ReferenceSnapshot
from tools.generativeqc_response.uhf import UHFReferenceSnapshot


def _snapshot(
    coupled_cluster: Any, *, restricted: bool = False
) -> UHFReferenceSnapshot:
    reference = coupled_cluster._scf
    fock = reference.get_fock()
    coefficients = reference.mo_coeff
    energies = reference.mo_energy
    occupations = reference.mo_occ
    if restricted:
        fock = (fock, fock)
        coefficients = (coefficients, coefficients)
        energies = (energies, energies)
        occupations = (occupations / 2, occupations / 2)
    return UHFReferenceSnapshot(
        overlap=reference.get_ovlp(),
        hcore=reference.get_hcore(),
        fock_alpha=fock[0],
        fock_beta=fock[1],
        coefficients_alpha=coefficients[0],
        coefficients_beta=coefficients[1],
        orbital_energies_alpha=energies[0],
        orbital_energies_beta=energies[1],
        occupations_alpha=occupations[0],
        occupations_beta=occupations[1],
        reference_energy=reference.e_tot,
        scf_residual=1e-10,
        geometry_hash="uccsd-test-geometry",
        basis_hash="uccsd-test-basis",
        generation_id="uccsd-test-generation",
        hf_backend="pyscf-test-oracle",
    )


@pytest.mark.parametrize("policy", ["pin", "lru"])
def test_mo_feeds_match_independent_spin_transformation(policy: str) -> None:
    coupled_cluster, _, layout, fock, integrals = _molecular_case("Li 0 0 0", 1)
    snapshot = _snapshot(coupled_cluster)
    source = DenseSource(coupled_cluster._scf.mol.intor("int2e"), snapshot)
    with (
        ConventionalProvider(snapshot, source, cache_policy=policy) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        expected = _feeds(layout, fock, integrals, _amplitudes(layout))
        for name, value in prepared.feeds.items():
            np.testing.assert_allclose(
                value, expected[name], atol=3e-12, rtol=3e-12, err_msg=name
            )
            with pytest.raises(ValueError):
                value.setflags(write=True)
        initial = _amplitudes(layout)
        actual = prepared.evaluate(layout.pack(initial))
        assert (
            abs(
                float(actual["correlation_energy"])
                - coupled_cluster.energy(
                    (initial["t1a"], initial["t1b"]),
                    (initial["t2aa"], initial["t2ab"], initial["t2bb"]),
                )
            )
            < 3e-12
        )


def test_equal_populations_preserve_independent_spin_coefficients() -> None:
    snapshot = _reference(beta_occupied=2)
    eri = _eri()
    source = DenseSource(eri, snapshot)
    coefficients = np.concatenate(
        (
            snapshot.coefficients_alpha[:, :2],
            snapshot.coefficients_beta[:, :2],
            snapshot.coefficients_alpha[:, 2:],
            snapshot.coefficients_beta[:, 2:],
        ),
        axis=1,
    )
    spins = np.repeat((0, 1, 0, 1), 2)
    coulomb = np.einsum(
        "uvwx,up,vr,wq,xs->pqrs",
        eri,
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
    fock = np.diag(
        np.concatenate(
            (
                snapshot.orbital_energies_alpha[:2],
                snapshot.orbital_energies_beta[:2],
                snapshot.orbital_energies_alpha[2:],
                snapshot.orbital_energies_beta[2:],
            )
        )
    )
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        initial = _amplitudes(prepared.layout)
        actual = prepared.evaluate(prepared.layout.pack(initial))
        expected = _determinant_oracle(prepared.layout, fock, integrals, initial)
        _assert_outputs(actual, expected)


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("O 0 0 0; H 0 0 0.97", 1, False),
        ("N 0 0 0; H 0 -0.8 0.6; H 0 0.8 0.6", 1, False),
        ("H 0 0 0; H 0 0 1.6", 0, True),
    ],
)
def test_complete_cpu_solver_against_independent_pyscf(
    atom: str, spin: int, broken: bool
) -> None:
    coupled_cluster, eris, _, _, _ = _molecular_case(atom, spin, broken=broken)
    coupled_cluster.conv_tol = 1e-13
    coupled_cluster.conv_tol_normt = 1e-10
    coupled_cluster.max_cycle = 400
    expected, _, _ = coupled_cluster.kernel(eris=eris)
    assert coupled_cluster.converged
    snapshot = _snapshot(coupled_cluster)
    source = DenseSource(coupled_cluster._scf.mol.intor("int2e"), snapshot)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=300)
        ) as prepared,
    ):
        result = prepared.solve()
        assert result.converged, (result.status, result.reason, result.history[-1])
        np.testing.assert_allclose(
            result.correlation_energy, expected, atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(
            result.total_energy,
            snapshot.reference_energy + expected,
            atol=2e-9,
            rtol=0,
        )
        assert result.history[-1].residual_max <= prepared.options.residual_tolerance
        assert result.history[-1].energy_change <= prepared.options.energy_tolerance
        assert result.provenance["tensor_backend"] == "native-cpu-tensorir"
        assert result.provenance["iteration_backend"] == "host-packed-excitation-diis"
        assert (
            result.provenance["solver_numeric_bound_bytes"]
            <= prepared.options.max_bytes
        )
        assert (
            result.provenance["complete_endpoint_seconds"]
            >= result.provenance["solve_seconds"]
        )
        for value in (result.packed_amplitudes, *result.amplitudes.values()):
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_complete_restricted_limit_matches_existing_rccsd_solver() -> None:
    coupled_cluster, _, _, _, _ = _molecular_case(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", 0, restricted=True
    )
    snapshot = _snapshot(coupled_cluster, restricted=True)
    source = DenseSource(coupled_cluster._scf.mol.intor("int2e"), snapshot)
    restricted = ReferenceSnapshot(
        overlap=snapshot.overlap,
        hcore=snapshot.hcore,
        fock=snapshot.fock_alpha,
        coefficients=snapshot.coefficients_alpha,
        orbital_energies=snapshot.orbital_energies_alpha,
        occupations=2 * snapshot.occupations_alpha,
        electron_count=coupled_cluster._scf.mol.nelectron,
        reference_energy=snapshot.reference_energy,
        scf_residual=snapshot.scf_residual,
        geometry_hash=snapshot.geometry_hash,
        basis_hash=snapshot.basis_hash,
        generation_id=snapshot.generation_id,
    )
    with ConventionalProvider(restricted, source) as provider:
        expected = solve(restricted, provider, options=SolverOptions())
    assert expected.converged
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        actual = prepared.solve()
    assert actual.converged, actual.history[-1]
    np.testing.assert_allclose(
        actual.correlation_energy, expected.correlation_energy, atol=2e-9, rtol=0
    )
    for name in ("t1a", "t1b"):
        np.testing.assert_allclose(
            actual.amplitudes[name], expected.t1, atol=2e-8, rtol=0
        )
    np.testing.assert_allclose(
        actual.amplitudes["t2ab"], expected.t2, atol=2e-8, rtol=0
    )
    for name in ("t2aa", "t2bb"):
        np.testing.assert_allclose(
            actual.amplitudes[name],
            expected.t2 - expected.t2.swapaxes(2, 3),
            atol=2e-8,
            rtol=0,
        )


@pytest.mark.parametrize(
    "options",
    [
        SolverOptions(diis_size=0, damping=0.2),
        SolverOptions(level_shift=0.3, damping=0.1),
    ],
)
def test_proposal_controls_preserve_physical_solution(options: SolverOptions) -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    with ConventionalProvider(snapshot, source) as provider:
        baseline = PreparedUCCSD(snapshot, provider).solve()
        prepared = PreparedUCCSD(snapshot, provider, options=options)
        actual = prepared.solve()
    assert baseline.converged and actual.converged
    np.testing.assert_allclose(
        actual.correlation_energy, baseline.correlation_energy, atol=2e-10, rtol=0
    )
    assert actual.history[-1].residual_max <= options.residual_tolerance


def test_solver_and_denominator_admission_precede_all_ao_reads() -> None:
    snapshot = _reference()
    source = DenseSource(_eri(), snapshot)
    with ConventionalProvider(snapshot, source) as provider:
        with pytest.raises((ValueError, MemoryError), match="budget"):
            PreparedUCCSD(snapshot, provider, options=SolverOptions(max_bytes=1))
        assert source.reads == 0
        with pytest.raises(TypeError, match="reference"):
            PreparedUCCSD(
                replace(
                    snapshot,
                    algorithm="UKS",
                    functional_identity="test",
                    grid_identity="test",
                ),
                provider,
            )
        assert source.reads == 0
    near_zero = _reference(alpha_energies=np.array([-1.25, -0.43, -0.43 + 1e-12, 0.88]))
    source = DenseSource(_eri(), near_zero)
    with ConventionalProvider(near_zero, source) as provider:
        with pytest.raises(ValueError, match="denominator"):
            PreparedUCCSD(near_zero, provider)
        assert source.reads == 0
    source = DenseSource(_eri(), snapshot)
    with ConventionalProvider(snapshot, source) as provider:
        with pytest.raises(ValueError, match="nonfinite.*denominator"):
            PreparedUCCSD(snapshot, provider, options=SolverOptions(level_shift=1e308))
        assert source.reads == 0


@pytest.mark.parametrize("occupied_energy", [0.0, 2e-9])
def test_denominator_extrema_allow_snapshot_ordering_tolerance(
    occupied_energy: float,
) -> None:
    # Snapshot admission permits ordering noise below its validation tolerance.
    # Check actual extrema, including a non-boundary occupied orbital, before
    # producing any integrals or a zero/non-Aufbau proposal denominator.
    snapshot = _reference(
        alpha_coefficients=np.eye(4),
        alpha_energies=np.array([occupied_energy, -5e-9, 0.0, 1.0]),
    )
    source = DenseSource(_eri(), snapshot)
    with ConventionalProvider(snapshot, source) as provider:
        with pytest.raises(ValueError, match="denominator"):
            PreparedUCCSD(snapshot, provider)
        assert source.reads == 0


def test_denominator_minimum_uses_actual_unsorted_extrema() -> None:
    snapshot = _reference(
        alpha_coefficients=np.eye(4),
        alpha_energies=np.array([-1e-8, -1.5e-8, 1e-8, 5e-9]),
    )
    source = DenseSource(_eri(), snapshot)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        assert prepared.minimum_absolute_denominator == np.min(
            np.abs(prepared.denominators)
        )


def test_provider_sequence_admission_is_spin_aware_and_transactional() -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    slots = ((0, 1),) * 4
    alpha = SpinMOBlock(slots, ("alpha",) * 4)
    beta = SpinMOBlock(slots, ("beta",) * 4)
    provider = ConventionalProvider(snapshot, source)
    peak = provider.plan(alpha).peak_bytes
    provider.budget_bytes = peak + int(8 * np.prod(alpha.shape)) - 1
    first = provider.get(alpha).to_host().copy()
    reads = source.reads
    with pytest.raises(MemoryError, match="sequence"):
        provider.admit((alpha, beta))
    assert source.reads == reads
    np.testing.assert_array_equal(provider.get(alpha).to_host(), first)
    provider.close()


def test_cuda_cache_admission_includes_library_allowance_without_gpu_access() -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    alpha = SpinMOBlock(((0, 1),) * 4, ("alpha",) * 4)
    beta = SpinMOBlock(((0, 1),) * 4, ("beta",) * 4)
    with ConventionalProvider(snapshot, source, backend="cuda") as provider:
        plan = provider.plan(alpha)
        assert plan.retained_bytes > plan.allocation_bytes
        provider.budget_bytes = plan.peak_bytes + plan.retained_bytes - 1
        with pytest.raises(MemoryError, match="sequence"):
            provider.admit((alpha, beta))
        assert source.reads == 0


def test_lru_fallback_and_owned_feeds_survive_provider_close() -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    with ConventionalProvider(snapshot, source) as provider:
        baseline = PreparedUCCSD(snapshot, provider).solve()
    provider = ConventionalProvider(snapshot, source, cache_policy="lru")
    largest = SpinMOBlock(((0, 1),) * 4, ("alpha",) * 4)
    provider.budget_bytes = provider.plan(largest).peak_bytes + int(
        8 * np.prod(largest.shape)
    )
    prepared = PreparedUCCSD(snapshot, provider)
    assert provider.statistics["evictions"] > 0
    provider.close()
    source.closed = True
    actual = prepared.solve()
    assert actual.converged and baseline.converged
    np.testing.assert_allclose(
        actual.correlation_energy, baseline.correlation_energy, atol=2e-12
    )
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.solve()


def test_iteration_limit_and_nonfinite_proposal_keep_last_finite_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    executor = NativeCCTensorExecutor(max_bytes=256 << 20, cache=tmp_path)
    with ConventionalProvider(snapshot, source) as provider:
        prepared = PreparedUCCSD(
            snapshot,
            provider,
            options=SolverOptions(max_iterations=1),
            executor=executor,
        )
        limited = prepared.solve()
        assert not limited.converged and limited.status == "iteration_limit"
        assert np.isfinite(limited.packed_amplitudes).all()
        original = executor.execute

        def failing(program: Any, feeds: Any) -> dict:
            result = original(program, feeds)
            if any(
                np.any(feeds[name]) for name in ("t1a", "t1b", "t2aa", "t2ab", "t2bb")
            ):
                result["correlation_energy"] = np.array(float("nan"))
            return result

        monkeypatch.setattr(executor, "execute", failing)
        failed = prepared.solve(np.zeros(prepared.layout.packed_size))
    assert not failed.converged and failed.status == "nonfinite"
    assert failed.correlation_energy == 0
    np.testing.assert_array_equal(
        failed.packed_amplitudes, np.zeros(prepared.layout.packed_size)
    )


def test_no_excitation_domain_and_invalid_initial_amplitudes() -> None:
    snapshot = _reference(alpha_occupied=4, beta_occupied=0)
    source = DenseSource(_eri(), snapshot)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        result = prepared.solve()
        assert result.converged and result.correlation_energy == 0
        assert result.packed_amplitudes.size == 0
        with pytest.raises(ValueError):
            prepared.solve(np.ones(1))
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as prepared,
    ):
        initial = _amplitudes(prepared.layout)
        initial["t2aa"][0, 0, 0, 0] = 0.1
        with pytest.raises(ValueError, match="antisymmetr|repeated"):
            prepared.solve(initial)


def test_backend_and_mutated_executor_configuration_fail_closed(tmp_path: Path) -> None:
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    compiler = CudaCompilerAdapter(Path("unprobed-nvcc"), cuda_target_info("sm_120"))
    cuda = CudaCCTensorExecutor(max_bytes=256 << 20, compiler=compiler, cache=tmp_path)
    with ConventionalProvider(snapshot, source) as provider:
        with pytest.raises(ValueError, match="backends.*no CPU fallback"):
            PreparedUCCSD(snapshot, provider, executor=cuda)
        assert source.reads == 0
        cpu = NativeCCTensorExecutor(max_bytes=256 << 20, cache=tmp_path)
        prepared = PreparedUCCSD(snapshot, provider, executor=cpu)
        cpu.max_bytes *= 2
        with pytest.raises(RuntimeError, match="configuration changed"):
            prepared.solve()
        prepared.close()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("Li 0 0 0", 1, False),
        ("N 0 0 0; H 0 -0.8 0.6; H 0 0.8 0.6", 1, False),
        ("H 0 0 0; H 0 0 1.6", 0, True),
    ],
)
def test_complete_cuda_solver_against_cpu_and_pyscf(
    tmp_path: Path, atom: str, spin: int, broken: bool
) -> None:
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID"), (
        "real CUDA gates require a finite Slurm allocation"
    )
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    coupled_cluster, eris, _, _, _ = _molecular_case(atom, spin, broken=broken)
    (
        coupled_cluster.conv_tol,
        coupled_cluster.conv_tol_normt,
        coupled_cluster.max_cycle,
    ) = 1e-13, 1e-10, 400
    expected, _, _ = coupled_cluster.kernel(eris=eris)
    assert coupled_cluster.converged
    snapshot = _snapshot(coupled_cluster)
    source = DenseSource(coupled_cluster._scf.mol.intor("int2e"), snapshot)
    options = SolverOptions(max_iterations=300, max_bytes=512 << 20)
    with ConventionalProvider(snapshot, source) as provider:
        cpu = PreparedUCCSD(snapshot, provider, options=options).solve()
    artifact = compile_cuda(compiler, tmp_path / "transform")
    executor = CudaCCTensorExecutor(
        max_bytes=options.max_bytes, compiler=compiler, cache=tmp_path / "tensor"
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
            snapshot, provider, options=options, executor=executor
        ) as prepared,
    ):
        actual = prepared.solve()
    assert cpu.converged and actual.converged, actual.history[-1]
    np.testing.assert_allclose(actual.correlation_energy, expected, atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        actual.correlation_energy, cpu.correlation_energy, atol=2e-9, rtol=0
    )
    assert actual.provenance["integral_backend"] == "cuda"
    assert actual.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
    for name in actual.amplitudes:
        np.testing.assert_allclose(
            actual.amplitudes[name], cpu.amplitudes[name], atol=2e-8, rtol=0
        )


def test_nonfinite_diis_extrapolation_keeps_latest_finite_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite trial remains the checkpoint if the following DIIS state fails."""
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    executor = NativeCCTensorExecutor(max_bytes=256 << 20)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot,
            provider,
            options=SolverOptions(diis_size=2, max_iterations=6),
            executor=executor,
        ) as prepared,
    ):
        evaluations = []

        def finite_trial_oracle(program: Any, feeds: Any) -> dict:
            coordinate = float(feeds["t1a"][0, 0])
            evaluations.append(coordinate)
            if 0.49 < coordinate < 0.51:
                raise FloatingPointError("nonfinite DIIS extrapolation")
            residuals = prepared.layout.unpack(
                prepared.denominators * (1 + coordinate**2)
            )
            return {
                "correlation_energy": np.array(coordinate),
                **{
                    name.replace("t", "r", 1): value
                    for name, value in residuals.items()
                },
            }

        monkeypatch.setattr(executor, "execute", finite_trial_oracle)
        result = prepared.solve()
        np.testing.assert_allclose(evaluations, [0, 1, 1, 3, 0.5])
        assert result.status == "nonfinite"
        assert result.correlation_energy == 3
        assert result.total_energy == snapshot.reference_energy + 3
        np.testing.assert_array_equal(
            result.packed_amplitudes, np.full(prepared.layout.packed_size, 3.0)
        )
        np.testing.assert_array_equal(
            prepared.layout.pack(result.amplitudes), result.packed_amplitudes
        )


def test_cached_provider_admission_includes_live_reference_and_source() -> None:
    """An all-hit sequence still charges the reference and integral producer."""
    nmo = 12
    coefficients = np.eye(nmo)
    energies = np.arange(nmo, dtype=np.float64) - 5.5
    occupations = np.r_[np.ones(6), np.zeros(6)]
    snapshot = replace(
        _reference(),
        overlap=coefficients,
        hcore=np.zeros((nmo, nmo)),
        coefficients_alpha=coefficients,
        coefficients_beta=coefficients,
        fock_alpha=np.diag(energies),
        fock_beta=np.diag(energies),
        orbital_energies_alpha=energies,
        orbital_energies_beta=energies,
        occupations_alpha=occupations,
        occupations_beta=occupations,
    )
    source = DenseSource(np.zeros((nmo,) * 4), snapshot)
    slots = tuple(range(nmo))
    blocks = [
        SpinMOBlock(
            (
                slots[first:] + slots[:first],
                slots[second:] + slots[:second],
                slots,
                slots,
            ),
            ("alpha",) * 4,
        )
        for first, second in ((index % nmo, index // nmo) for index in range(64))
    ]
    with ConventionalProvider(
        snapshot, source, axis_tile=nmo, budget_bytes=64 << 20
    ) as provider:
        for block in blocks:
            provider.get(block)
        baseline = provider._retained + snapshot.numeric_bytes + source.numeric_bytes
        assert provider._retained > provider.plan(blocks[0]).peak_bytes
        before = (source.reads, dict(provider.statistics), tuple(provider._cache))
        assert provider.admit(blocks) >= baseline
        assert before == (
            source.reads,
            dict(provider.statistics),
            tuple(provider._cache),
        )
        provider.budget_bytes = baseline - 1
        with pytest.raises(MemoryError, match="budget"):
            provider.admit(blocks)
        assert source.reads == before[0]


@pytest.mark.parametrize("replay_failure", ["nonfinite", "finite_rejected"])
def test_failed_final_replay_preserves_latest_finite_owned_checkpoint(
    monkeypatch: pytest.MonkeyPatch, replay_failure: str
) -> None:
    """Replay failure cannot promote success or discard its latest finite energy."""
    snapshot = _reference(beta_occupied=2)
    source = DenseSource(_eri(), snapshot)
    executor = NativeCCTensorExecutor(max_bytes=256 << 20)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot,
            provider,
            options=SolverOptions(diis_size=0, max_iterations=1),
            executor=executor,
        ) as prepared,
    ):
        evaluations = []
        borrowed_amplitudes = []

        def replay_oracle(program: Any, feeds: Any) -> dict:
            evaluations.append(float(feeds["t1a"][0, 0]))
            borrowed_amplitudes.extend(
                feeds[block.name] for block in prepared.layout.blocks
            )
            final_replay = len(evaluations) == 4
            energy = (
                float("nan")
                if final_replay and replay_failure == "nonfinite"
                else 7.5
                if final_replay
                else 7.0
            )
            return {
                "correlation_energy": np.array(energy),
                **{
                    block.name.replace("t", "r", 1): np.full(
                        block.spec.shape, 1.0 if final_replay else 0.0
                    )
                    for block in prepared.layout.blocks
                },
            }

        monkeypatch.setattr(executor, "execute", replay_oracle)
        initial = np.full(prepared.layout.packed_size, 0.2)
        result = prepared.solve(initial)
        assert len(evaluations) == 4
        assert not result.converged
        assert result.status == (
            "nonfinite" if replay_failure == "nonfinite" else "iteration_limit"
        )
        expected_energy = 7.0 if replay_failure == "nonfinite" else 7.5
        assert result.correlation_energy == expected_energy
        assert result.total_energy == snapshot.reference_energy + expected_energy
        initial.fill(9)
        for borrowed in borrowed_amplitudes:
            borrowed.fill(9)
        np.testing.assert_array_equal(
            result.packed_amplitudes, np.full(prepared.layout.packed_size, 0.2)
        )
        np.testing.assert_array_equal(
            prepared.layout.pack(result.amplitudes), result.packed_amplitudes
        )
        for value in (result.packed_amplitudes, *result.amplitudes.values()):
            with pytest.raises(ValueError):
                value.setflags(write=True)
