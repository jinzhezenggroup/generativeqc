"""Corrected Lambda with independent spin-orbital finite-difference equations."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_uccsd_equations import _spin_amplitudes
from test_uccsd_lambda import _problem
from test_uccsd_t_owner import _ASYMMETRIC_H6, _BROKEN_H4, _noninteracting
from test_uccsd_triples_response import _snapshot_hamiltonian

from tools.generativeqc_cc.lambda_solver import BoundCCSDLambda
from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
from tools.generativeqc_cc.solver import SolverOptions, solve
from tools.generativeqc_cc.triples_lambda_response import solve_corrected_lambda
from tools.generativeqc_cc.uccsd_lambda import PreparedUCCSDLambda, UCCSDLambdaResult
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_cc.uccsd_t import UCCSDTOptions
from tools.generativeqc_cc.uccsd_t_lambda import (
    PreparedUCCSDTLambda,
    UCCSDTLambdaOptions,
)
from tools.generativeqc_cc.uccsd_triples_response import PreparedUCCSDTriplesResponse
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.reference import ReferenceSnapshot
from tools.generativeqc_response.implicit import ImplicitSolveError


def _independent_energy_residual(
    parent: PreparedUCCSD,
    packed: np.ndarray,
    fock: np.ndarray,
    integrals: np.ndarray,
    energies: np.ndarray,
) -> tuple:
    """Only tests embed full spin and use PySCF's separately implemented equations."""
    gccsd = pytest.importorskip("pyscf.cc.gccsd")
    triples = pytest.importorskip("pyscf.cc.gccsd_t")
    layout = parent.layout
    occupied = layout.oa + layout.ob
    eris = gccsd._PhysicistsERIs()
    eris.fock, eris.mo_energy = fock, energies
    for domains in ("oooo", "ooov", "oovv", "ovov", "ovvo", "ovvv", "vvvv"):
        setattr(
            eris,
            domains,
            integrals[
                tuple(
                    slice(0, occupied) if label == "o" else slice(occupied, None)
                    for label in domains
                )
            ],
        )
    amplitudes = layout.unpack(packed)
    singles, doubles = _spin_amplitudes(layout, amplitudes)
    updated_singles, updated_doubles = gccsd.update_amps(
        SimpleNamespace(level_shift=0.0, cc2=False), singles, doubles, eris
    )
    gaps = energies[:occupied, None] - energies[None, occupied:]
    residual_singles = gaps * (updated_singles - singles)
    residual_doubles = (gaps[:, None, :, None] + gaps[None, :, None, :]) * (
        updated_doubles - doubles
    )
    residual = {
        "t1a": residual_singles[: layout.oa, : layout.va],
        "t1b": residual_singles[layout.oa :, layout.va :],
        "t2aa": residual_doubles[: layout.oa, : layout.oa, : layout.va, : layout.va],
        "t2ab": residual_doubles[: layout.oa, layout.oa :, : layout.va, layout.va :],
        "t2bb": residual_doubles[layout.oa :, layout.oa :, layout.va :, layout.va :],
    }
    coordinates = np.concatenate(
        [
            np.array(
                [
                    residual[block.name][block.representative(slot)]
                    for slot in range(block.packed_layout.storage_elements)
                ]
            )
            for block in layout.blocks
        ]
    )
    return float(
        gccsd.energy(None, singles, doubles, eris)
        + triples.kernel(None, eris, singles, doubles)
    ), coordinates


def _independent_lambda(parent: PreparedUCCSD, state: Any, source: Any) -> np.ndarray:
    """Dense test-only J and corrected RHS from independent central differences."""
    fock, integrals, energies = _snapshot_hamiltonian(parent, source)
    packed = np.asarray(state.packed_amplitudes)
    dimension, step = packed.size, 2e-5
    jacobian, gradient = np.empty((dimension, dimension)), np.empty(dimension)
    for slot in range(dimension):
        direction = np.zeros(dimension)
        direction[slot] = step
        plus = _independent_energy_residual(
            parent, packed + direction, fock, integrals, energies
        )
        minus = _independent_energy_residual(
            parent, packed - direction, fock, integrals, energies
        )
        gradient[slot] = (plus[0] - minus[0]) / (2 * step)
        jacobian[:, slot] = (plus[1] - minus[1]) / (2 * step)
    return np.linalg.solve(jacobian.T, -gradient)


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
    ],
)
def test_corrected_cpu_lambda_against_independent_full_spin_finite_differences(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, _, _, _ = _problem(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDLambda(parent, state) as standard,
            PreparedUCCSDTriplesResponse(
                parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
            ) as triples,
            PreparedUCCSDTLambda(standard, triples) as owner,
        ):
            result = owner.solve()
            assert result.converged and not isinstance(result, UCCSDLambdaResult)
            expected = _independent_lambda(parent, state, source)
            np.testing.assert_allclose(
                result.packed_multipliers, expected, atol=2e-8, rtol=0
            )
            _, _, diagnostics = owner.qualify_with_correction(result)
            assert diagnostics["stationarity_norm"] < 1e-9
            assert (
                diagnostics["executed_semantic_scalar_work"]
                <= owner.limits.max_total_work
            )
            plain = standard.solve()
            assert (
                np.linalg.norm(result.packed_multipliers - plain.packed_multipliers)
                > 1e-7
            )
            with pytest.raises(ValueError):
                standard.qualify(result)
            with pytest.raises(ValueError):
                owner.qualify(plain)
            forged = replace(
                plain,
                packed_multipliers=result.packed_multipliers,
                multipliers=result.multipliers,
                lambda_identity=standard._lambda_identity(result.packed_multipliers),
            )
            with pytest.raises(ImplicitSolveError, match="stationarity"):
                standard.qualify(forged)
            weight = owner.parameter_cotangent(result, "fov_aa")
            fock, integrals, energies = _snapshot_hamiltonian(parent, source)
            positions = np.r_[
                np.arange(parent.layout.oa),
                np.arange(
                    parent.layout.oa + parent.layout.ob,
                    parent.layout.oa + parent.layout.ob + parent.layout.va,
                ),
            ]
            direction = np.zeros_like(fock)
            perturbation = np.random.default_rng(71).normal(
                scale=0.05, size=weight.values.shape
            )
            direction[
                np.ix_(positions[: parent.layout.oa], positions[parent.layout.oa :])
            ] = perturbation
            direction += direction.T
            step, values = 1e-5, []
            for sign in (-1, 1):
                energy, residual = _independent_energy_residual(
                    parent,
                    state.packed_amplitudes,
                    fock + sign * step * direction,
                    integrals,
                    energies,
                )
                values.append(energy + np.vdot(result.packed_multipliers, residual))
            np.testing.assert_allclose(
                weight.contract(perturbation),
                (values[1] - values[0]) / (2 * step),
                atol=3e-9,
                rtol=0,
            )


@pytest.fixture
def small_corrected() -> Any:
    snapshot, source = _noninteracting(4, 2, 2)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        with (
            PreparedUCCSDLambda(parent, state) as standard,
            PreparedUCCSDTriplesResponse(parent, state) as triples,
            PreparedUCCSDTLambda(standard, triples) as corrected,
        ):
            yield standard, triples, corrected


def test_corrected_restricted_limit_matches_existing_rccsdt_dual() -> None:
    snapshot, source, _, _, _ = _problem(_ASYMMETRIC_H6, 0)
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
        state = solve(restricted, provider, options=SolverOptions(max_iterations=400))
    assert state.converged
    bound = BoundCCSDLambda(restricted, state)
    baseline = bound.solve(reference_identity=restricted.identity)
    expected = solve_corrected_lambda(bound, baseline, vir_chunk_size=2)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDLambda(parent, state) as standard,
            PreparedUCCSDTriplesResponse(
                parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
            ) as triples,
            PreparedUCCSDTLambda(standard, triples) as owner,
        ):
            actual = owner.solve()
            np.testing.assert_allclose(
                actual.multipliers["l1a"] + actual.multipliers["l1b"],
                expected.lambda1,
                atol=2e-8,
                rtol=0,
            )
            np.testing.assert_allclose(
                actual.multipliers["l2ab"] + actual.multipliers["l2aa"],
                expected.lambda2,
                atol=2e-8,
                rtol=0,
            )


def test_corrected_parameter_storage_admission_precedes_replay(
    small_corrected: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, triples, owner = small_corrected
    result = owner.solve()
    with PreparedUCCSDTLambda(
        standard, triples, options=UCCSDTLambdaOptions(max_bytes=owner.peak_bytes)
    ) as bounded:

        def forbidden(*_: Any) -> None:
            pytest.fail("rejected parameter admission executed native equations")

        monkeypatch.setattr(standard.prepared.executor, "execute", forbidden)
        with pytest.raises(MemoryError, match="parameter numeric"):
            bounded.parameter_cotangent(result, "fov_aa")


@pytest.mark.parametrize("phase", ["triples", "stationarity"])
def test_late_corrected_native_failure_never_publishes(
    small_corrected: tuple, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, _, owner = small_corrected
    original = standard.prepared.executor.execute

    def failing(program: Any, feeds: Any) -> Any:
        if (phase == "triples" and "triples_energy" in program.outputs) or (
            phase == "stationarity"
            and program.logical_hash
            == standard.programs.stationarity_vjp.derivative_hash
        ):
            raise RuntimeError("injected late corrected native failure")
        return original(program, feeds)

    monkeypatch.setattr(standard.prepared.executor, "execute", failing)
    with pytest.raises((ImplicitSolveError, RuntimeError), match="failure"):
        owner.solve()


@pytest.mark.parametrize(
    "options,match",
    [
        (UCCSDTLambdaOptions(max_bytes=1), "numeric"),
        (UCCSDTLambdaOptions(max_total_work=1), "semantic-work"),
    ],
)
def test_corrected_admission_precedes_execution(
    small_corrected: tuple,
    options: UCCSDTLambdaOptions,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standard, triples, _ = small_corrected

    def forbidden(*_: Any) -> None:
        pytest.fail("rejected corrected admission executed native equations")

    monkeypatch.setattr(standard.prepared.executor, "execute", forbidden)
    with pytest.raises(MemoryError, match=match):
        PreparedUCCSDTLambda(standard, triples, options=options)


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "correction",
        "packed",
        "bars",
        "energies",
        "missing_triples_energy",
        "missing_correlation_energy",
        "backend",
        "channel",
    ],
)
def test_corrected_qualification_does_not_trust_records(
    small_corrected: tuple, fault: str
) -> None:
    _, _, owner = small_corrected
    result = owner.solve()
    if fault == "state":
        forged = replace(
            result,
            state_identity="other",
            triples_response=replace(
                result.triples_response,
                energy=replace(
                    result.triples_response.energy,
                    provenance={
                        **result.triples_response.energy.provenance,
                        "state_identity": "other",
                    },
                ),
            ),
        )
    elif fault == "correction":
        forged = replace(result, correction_identity="other")
    elif fault == "packed":
        packed = result.packed_multipliers.copy()
        packed[0] += 0.01
        forged = replace(
            result,
            packed_multipliers=packed,
            multipliers={
                name.replace("t", "l", 1): value
                for name, value in owner.layout.unpack(packed).items()
            },
        )
    else:
        response = result.triples_response
        if fault == "bars":
            bars = {name: value.copy() for name, value in response.cotangents.items()}
            bars["fov_aa"][0, 0] += 0.01
            changed = replace(response, cotangents=bars)
        elif fault == "energies":
            changed = replace(
                response,
                energy=replace(
                    response.energy, total_energy=response.energy.total_energy + 0.01
                ),
            )
        elif fault.startswith("missing_"):
            changed = replace(
                response,
                energy=replace(
                    response.energy, **{fault.removeprefix("missing_"): None}
                ),
            )
        elif fault == "channel":
            channels = dict(response.energy.channel_energies)
            channels["aab"], channels["abb"] = 0.01, -0.01
            changed = replace(
                response, energy=replace(response.energy, channel_energies=channels)
            )
        else:
            changed = replace(
                response,
                energy=replace(
                    response.energy,
                    provenance={
                        **response.energy.provenance,
                        "tensor_backend": "other",
                    },
                ),
            )
        forged = replace(result, triples_response=changed)
    with pytest.raises((ValueError, ImplicitSolveError)):
        owner.qualify(forged)


def test_corrected_lifetimes_and_detached_records(small_corrected: tuple) -> None:
    standard, triples, owner = small_corrected
    result = owner.solve()
    copied = {name: value.copy() for name, value in result.multipliers.items()}
    detached = replace(result, multipliers=copied)
    copied["l1a"][0, 0] = 1.0
    assert detached.multipliers["l1a"][0, 0] == 0
    channels = dict(result.triples_response.energy.channel_energies)
    nested = replace(
        result.triples_response,
        energy=replace(result.triples_response.energy, channel_energies=channels),
    )
    detached_response = replace(result, triples_response=nested)
    channels["aab"] = 1.0
    assert detached_response.triples_response.energy.channel_energies["aab"] == 0
    with pytest.raises(ValueError):
        result.packed_multipliers.setflags(write=True)
    with pytest.raises(TypeError):
        result.provenance["limits"]["max_bytes"] = 1
    original = owner.limits
    owner.limits = replace(original, max_total_work=1)
    with pytest.raises(RuntimeError, match="configuration"):
        owner.solve()
    owner.limits = original
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.qualify(result)
    assert not standard._closed and not triples._closed
    assert standard.solve().converged


@pytest.mark.parametrize("nmo,oa,ob", [(3, 3, 0), (3, 3, 3), (4, 1, 0)])
def test_empty_corrected_domain_has_distinct_zero_stationarity(
    nmo: int, oa: int, ob: int
) -> None:
    snapshot, source = _noninteracting(nmo, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        with (
            PreparedUCCSDLambda(parent, state) as standard,
            PreparedUCCSDTriplesResponse(parent, state) as triples,
            PreparedUCCSDTLambda(standard, triples) as owner,
        ):
            result = owner.solve()
            assert (
                result.stationarity_norm
                == result.triples_response.energy.triples_energy
                == 0
            )
            assert owner.qualify(result)[1]["stationarity_norm"] == 0
            with pytest.raises(ValueError):
                standard.qualify(result)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires finite Slurm allocated GPU",
)
@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
    ],
)
def test_complete_cuda_corrected_lambda_against_independent_equations(
    atom: str, spin: int, broken: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID")
    snapshot, source, _, _, _ = _problem(atom, spin, broken)
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    cache = Path(
        os.environ.get("GENERATIVEQC_UCCSD_T_LAMBDA_TEST_CACHE", str(tmp_path))
    )
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
            pytest.fail("CUDA corrected Lambda executed CPU equations")

        monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
        state = parent.solve()
        assert state.converged
        with (
            PreparedUCCSDLambda(parent, state) as standard,
            PreparedUCCSDTriplesResponse(
                parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
            ) as triples,
            PreparedUCCSDTLambda(standard, triples) as owner,
        ):
            result = owner.solve()
            np.testing.assert_allclose(
                result.packed_multipliers,
                _independent_lambda(parent, state, source),
                atol=2e-8,
                rtol=0,
            )
            _, diagnostics = owner.qualify(result)
            assert diagnostics["stationarity_norm"] < 1e-9
            assert result.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
