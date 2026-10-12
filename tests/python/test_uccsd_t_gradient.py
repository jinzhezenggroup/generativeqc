"""Corrected physical/nuclear composition against independent complete energies."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_uccsd_gradient import _NH2, _RESTRICTED_H4, _native_oracle
from test_uccsd_lambda import _problem
from test_uccsd_physical_response import _ao_derivative_oracle
from test_uccsd_solver import _snapshot
from test_uccsd_t_owner import _ASYMMETRIC_H6, _BROKEN_H4, _noninteracting

from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_gradient import (
    PreparedUCCSDGradient,
    UCCSDGradientOptions,
    UCCSDGradientResult,
)
from tools.generativeqc_cc.uccsd_lambda import PreparedUCCSDLambda, UCCSDLambdaOptions
from tools.generativeqc_cc.uccsd_physical_response import (
    PreparedUCCSDPhysicalResponse,
    UCCSDPhysicalResponseOptions,
    UCCSDRelaxedCotangents,
)
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_cc.uccsd_t import UCCSDTOptions
from tools.generativeqc_cc.uccsd_t_gradient import PreparedUCCSDTGradient
from tools.generativeqc_cc.uccsd_t_lambda import (
    PreparedUCCSDTLambda,
    UCCSDTLambdaOptions,
)
from tools.generativeqc_cc.uccsd_t_physical_response import (
    PreparedUCCSDTPhysicalResponse,
)
from tools.generativeqc_cc.uccsd_triples_fock_response import (
    PreparedUCCSDTriplesFockResponse,
)
from tools.generativeqc_cc.uccsd_triples_response import PreparedUCCSDTriplesResponse
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_response.implicit import ImplicitSolveError

_NATIVE_H6 = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.9)),
    ("H", (0.19, 0.1, 3.8)),
    ("H", (-0.19, 0.04, 5.7)),
    ("H", (0.19, -0.08, 7.6)),
    ("H", (0.0, 0.19, 9.5)),
]


@contextmanager
def _response(
    parent: Any,
    state: Any,
    provider: Any,
    *,
    lambda_options: Any = None,
    correction_options: Any = None,
    **kwargs: Any,
) -> Any:
    """Every test owns live consumers; no detached replay capability is assumed."""
    with (
        PreparedUCCSDLambda(parent, state, options=lambda_options) as standard,
        PreparedUCCSDTriplesResponse(
            parent, state, options=UCCSDTOptions(tile_shape=(8,) * 6)
        ) as triples,
        PreparedUCCSDTLambda(
            standard, triples, options=correction_options
        ) as corrected,
        PreparedUCCSDTriplesFockResponse(triples) as fock,
        PreparedUCCSDTPhysicalResponse(
            corrected, corrected.solve(), fock, provider, **kwargs
        ) as physical,
    ):
        yield physical, corrected, fock


def _independent_fd(oracle: Any, atom: int, axis: int) -> float:
    """Re-converge independent UHF/CCSD/(T) at both displaced nuclear geometries."""
    from pyscf import cc, scf

    step, energies = 2e-4, []
    for sign in (-1, 1):
        coordinates = oracle.mol.atom_coords().copy()
        coordinates[atom, axis] += sign * step
        molecule = oracle.mol.copy().set_geom_(coordinates, unit="Bohr")
        reference = scf.UHF(molecule)
        reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = (
            1e-13,
            1e-10,
            300,
        )
        reference.kernel(dm0=oracle._scf.make_rdm1())
        assert reference.converged
        correlated = cc.UCCSD(reference)
        correlated.conv_tol, correlated.conv_tol_normt, correlated.max_cycle = (
            1e-13,
            1e-11,
            400,
        )
        correlated.kernel()
        assert correlated.converged
        energies.append(correlated.e_tot + correlated.ccsd_t())
    return (energies[1] - energies[0]) / (2 * step)


def _independent_gradient_fd(oracle: Any) -> np.ndarray:
    """Displace every physical coordinate; do not infer rows from symmetries."""
    gradient = np.empty((oracle.mol.natm, 3))
    for selected in product(range(oracle.mol.natm), range(3)):
        gradient[selected] = _independent_fd(oracle, *selected)
    return gradient


def _independent_analytic(oracle: Any) -> np.ndarray:
    """Use a restricted analytic oracle only on a collapsed stationary branch.

    The installed unrestricted analytic endpoint fails its own reconverged
    energy derivative on these fixtures even with explicit corrected Lambda.
    All unrestricted coordinates instead retain the tight independent FD gate.
    """
    from pyscf import cc, scf
    from pyscf.cc import ccsd_t_lambda_slow
    from pyscf.grad import ccsd_t

    assert oracle.mol.spin == 0
    density = oracle._scf.make_rdm1()
    np.testing.assert_allclose(density[0], density[1], atol=1e-7, rtol=0)
    reference = scf.RHF(oracle.mol)
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-13, 1e-10, 300
    reference.kernel()
    assert reference.converged
    restricted = cc.CCSD(reference)
    restricted.conv_tol, restricted.conv_tol_normt, restricted.max_cycle = (
        1e-13,
        1e-11,
        400,
    )
    restricted.kernel()
    assert restricted.converged
    np.testing.assert_allclose(
        restricted.e_tot + restricted.ccsd_t(),
        oracle.e_tot + oracle.ccsd_t(),
        atol=3e-9,
        rtol=0,
    )
    integrals = restricted.ao2mo()
    converged, singles, doubles = ccsd_t_lambda_slow.kernel(
        restricted, integrals, max_cycle=400, tol=1e-11, verbose=0
    )
    assert converged
    return ccsd_t.Gradients(restricted).kernel(
        restricted.t1, restricted.t2, singles, doubles, eris=integrals
    )


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
    ],
)
def test_corrected_physical_weights_close_reconverged_total_energy_fd(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, oracle, _, _ = _problem(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with _response(
            parent, state, provider, nuclear_repulsion=oracle.mol.energy_nuc()
        ) as (physical, _, _):
            result = physical.solve()
            gradient = _ao_derivative_oracle(snapshot, oracle, result.weights)
            np.testing.assert_allclose(
                result.total_energy, oracle.e_tot + oracle.ccsd_t(), atol=3e-9, rtol=0
            )
            np.testing.assert_allclose(
                gradient, _independent_gradient_fd(oracle), atol=2e-7, rtol=0
            )
            np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=2e-9, rtol=0)
            assert not isinstance(result, UCCSDRelaxedCotangents)
            assert result.orbital_stationarity_max <= 1e-8
            assert (
                result.provenance["executed_semantic_scalar_work"]
                <= physical.planned_work
            )
            for value in (*result.weights.values(), result.orbital_adjoint):
                with pytest.raises(ValueError):
                    value.setflags(write=True)


@pytest.mark.parametrize(
    "atoms,spin,broken,representation",
    [
        (_RESTRICTED_H4, 0, False, "cartesian"),
        (
            [("H", (0.0, 0.0, 2.45665 * index)) for index in range(4)],
            0,
            True,
            "cartesian",
        ),
        (_NH2, 1, False, "spherical"),
        (_NATIVE_H6, 0, False, "cartesian"),
    ],
)
def test_complete_corrected_native_cpu_force_has_independent_oracles(
    atoms: list, spin: int, broken: bool, representation: str
) -> None:
    with NativeSource(
        atoms, multiplicity=spin + 1, representation=representation
    ) as source:
        oracle = _native_oracle(source, broken=broken)
        snapshot = replace(
            _snapshot(oracle),
            geometry_hash=source.geometry_hash,
            basis_hash=source.basis_hash,
            representation=source.representation,
        )
        with (
            ConventionalProvider(snapshot, source) as provider,
            PreparedUCCSD(
                snapshot, provider, options=SolverOptions(max_iterations=400)
            ) as parent,
        ):
            state = parent.solve()
            assert state.converged
            with (
                _response(parent, state, provider) as (physical, _, _),
                PreparedUCCSDTGradient(physical) as owner,
            ):
                result = owner.solve()
                np.testing.assert_allclose(
                    result.total_energy,
                    oracle.e_tot + oracle.ccsd_t(),
                    atol=3e-9,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    result.gradient, _independent_gradient_fd(oracle), atol=2e-7, rtol=0
                )
                if spin == 0 and not broken:
                    np.testing.assert_allclose(
                        result.gradient,
                        _independent_analytic(oracle),
                        atol=2e-8,
                        rtol=0,
                    )
                np.testing.assert_allclose(
                    result.gradient.sum(axis=0), 0.0, atol=2e-9, rtol=0
                )
                if atoms == _RESTRICTED_H4:
                    from tools.generativeqc_cc.triples_complete_gradient import (
                        complete_ccsdt_gradient_validation,
                    )

                    restricted = complete_ccsdt_gradient_validation(source)
                    np.testing.assert_allclose(
                        result.gradient, restricted.gradient, atol=2e-8, rtol=0
                    )
                assert not isinstance(result, UCCSDGradientResult)
                assert result.provenance["method"] == "UCCSD(T)"
                assert result.provenance["nuclear_repulsion_terms"] == 1
                assert (
                    result.provenance["executed_generated_scalar_work"]
                    <= owner.planned_work
                )
                for value in (result.gradient, result.forces):
                    with pytest.raises(ValueError):
                        value.setflags(write=True)


@pytest.fixture
def small_response() -> Any:
    snapshot, source = _noninteracting(4, 2, 2)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
        _response(parent, parent.solve(), provider, nuclear_repulsion=0.0) as owners,
    ):
        yield owners


def test_standard_and_corrected_type_boundaries_remain_distinct(
    small_response: tuple,
) -> None:
    physical, corrected, _ = small_response
    with pytest.raises(TypeError, match="PreparedUCCSDLambda"):
        PreparedUCCSDPhysicalResponse(
            corrected, physical.result, physical.provider, nuclear_repulsion=0.0
        )
    with pytest.raises(TypeError, match="physical response owner"):
        PreparedUCCSDGradient(physical)
    standard = corrected.standard
    with (
        PreparedUCCSDPhysicalResponse(
            standard, standard.solve(), physical.provider, nuclear_repulsion=0.0
        ) as uncorrected,
        pytest.raises(TypeError, match="corrected physical owner"),
    ):
        PreparedUCCSDTGradient(uncorrected)
    result = physical.solve()
    assert result.total_energy == physical.reference.reference_energy
    assert result.triples_energy == 0.0


@pytest.mark.parametrize(
    "options",
    [
        UCCSDPhysicalResponseOptions(max_bytes=1),
        UCCSDPhysicalResponseOptions(max_total_work=1),
    ],
)
def test_corrected_admission_precedes_raw_producer_reads(
    small_response: tuple, options: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, corrected, fock = small_response

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("rejected corrected physical admission read raw integrals")

    monkeypatch.setattr(physical.provider, "get", forbidden)
    with pytest.raises(MemoryError):
        PreparedUCCSDTPhysicalResponse(
            corrected,
            physical.result,
            fock,
            physical.provider,
            options=options,
            nuclear_repulsion=0.0,
        )


def test_corrected_requires_the_same_live_triples_owner(small_response: tuple) -> None:
    physical, corrected, _ = small_response
    parent = corrected.prepared
    with (
        PreparedUCCSDTriplesResponse(parent, parent.solve()) as other_triples,
        PreparedUCCSDTriplesFockResponse(other_triples) as other_fock,
        pytest.raises(ValueError, match="triples/parent/state"),
    ):
        PreparedUCCSDTPhysicalResponse(
            corrected,
            physical.result,
            other_fock,
            physical.provider,
            nuclear_repulsion=0.0,
        )


def test_complete_second_replay_is_admitted_before_raw_reads(
    small_response: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, corrected, fock = small_response

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("under-admitted correction replay read raw integrals")

    monkeypatch.setattr(physical.provider, "get", forbidden)
    options = replace(
        physical.options,
        max_total_work=physical.planned_work - fock.total_scalar_work,
    )
    with pytest.raises(MemoryError, match="complete work budget"):
        PreparedUCCSDTPhysicalResponse(
            corrected,
            physical.result,
            fock,
            physical.provider,
            options=options,
            nuclear_repulsion=0.0,
        )


def test_late_correction_transport_failure_never_publishes(
    small_response: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, _, _ = small_response
    executor = physical.lambdas.prepared.executor
    original = executor.execute

    def nonfinite(program: Any, feeds: Any) -> Any:
        values = original(program, feeds)
        if program is physical.correction_program:
            values["overlap_a"] = np.full_like(values["overlap_a"], np.nan)
        return values

    monkeypatch.setattr(executor, "execute", nonfinite)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        physical.solve()


def test_fresh_correction_comparison_rejects_detached_bar_drift(
    small_response: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, _, fock = small_response
    original = fock.response

    def drifted() -> Any:
        result = original()
        replay = result.triples_response
        bars = dict(replay.cotangents)
        bars["t1a"] = bars["t1a"] + 0.01
        packed = fock.layout.pack_dual(
            {name: bars[name] for name in ("t1a", "t1b", "t2aa", "t2ab", "t2bb")}
        )
        return replace(
            result,
            triples_response=replace(
                replay, cotangents=bars, packed_amplitude_cotangent=packed
            ),
        )

    monkeypatch.setattr(fock, "response", drifted)
    with pytest.raises(ImplicitSolveError, match="energy/bar drift"):
        physical.solve()


def test_final_same_occupancy_stationarity_is_not_an_ov_only_gate(
    small_response: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, _, _ = small_response
    original = physical._run

    def changed(program: Any, feeds: Any, counter: dict) -> Any:
        values = original(program, feeds, counter)
        if (
            program is physical.programs.weights
            and float(feeds["bar_reference_electronic_energy"]) == 1.0
        ):
            values["stationarity_a"] = values["stationarity_a"].copy()
            values["stationarity_a"][0, 1] += 0.1
        return values

    monkeypatch.setattr(physical, "_run", changed)
    with pytest.raises(ImplicitSolveError, match="full orbital stationarity"):
        physical.solve()


def test_closed_full_fock_owner_never_publishes(small_response: tuple) -> None:
    physical, corrected, fock = small_response
    fock.close()
    with pytest.raises(RuntimeError, match="closed"):
        physical.solve()
    assert not physical.provider._closed
    corrected._check_open()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1", reason="Slurm CUDA opt-in"
)
@pytest.mark.parametrize(
    "atoms,spin,broken",
    [
        (_RESTRICTED_H4, 0, False),
        ([("H", (0.0, 0.0, 2.45665 * index)) for index in range(4)], 0, True),
        (_NH2, 1, False),
        (_NATIVE_H6, 0, False),
    ],
)
def test_complete_corrected_cuda_forces_forbid_cpu_fallback(
    atoms: list,
    spin: int,
    broken: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda
    from tools.generativeqc_response.spin_cuda import CudaSpinJKBackend

    assert os.environ.get("SLURM_JOB_ID")
    cache = Path(
        os.environ.get("GENERATIVEQC_UCCSD_T_GRADIENT_TEST_CACHE", str(tmp_path))
    )
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    artifact = compile_cuda(compiler, cache / "transform")
    executor = CudaCCTensorExecutor(
        max_bytes=1 << 30, compiler=compiler, cache=cache / "tensor"
    )
    with NativeSource(atoms, multiplicity=spin + 1) as source:
        oracle = _native_oracle(source, broken=broken)
        snapshot = replace(
            _snapshot(oracle),
            geometry_hash=source.geometry_hash,
            basis_hash=source.basis_hash,
        )
        with (
            CudaSpinJKBackend(source) as backend,
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

            def forbidden(*args: Any, **kwargs: Any) -> None:
                pytest.fail("corrected CUDA forces executed CPU equations/derivatives")

            monkeypatch.setattr(NativeCCTensorExecutor, "execute", forbidden)
            monkeypatch.setattr(NativeSource, "one_electron_gradient_cpu", forbidden)
            monkeypatch.setattr(
                NativeSource, "weighted_eri_shell_gradient_cpu", forbidden
            )
            state = parent.solve()
            assert state.converged
            with (
                _response(
                    parent,
                    state,
                    provider,
                    lambda_options=UCCSDLambdaOptions(max_bytes=2 << 30),
                    correction_options=UCCSDTLambdaOptions(max_bytes=3 << 30),
                    options=UCCSDPhysicalResponseOptions(max_bytes=3 << 30),
                    response_backend=backend,
                ) as (physical, _, _),
                PreparedUCCSDTGradient(
                    physical, options=UCCSDGradientOptions(max_bytes=4 << 30)
                ) as owner,
            ):
                result = owner.solve()
                np.testing.assert_allclose(
                    result.total_energy,
                    oracle.e_tot + oracle.ccsd_t(),
                    atol=3e-9,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    result.gradient, _independent_gradient_fd(oracle), atol=2e-7, rtol=0
                )
                if spin == 0 and not broken:
                    np.testing.assert_allclose(
                        result.gradient,
                        _independent_analytic(oracle),
                        atol=2e-8,
                        rtol=0,
                    )
                assert result.provenance["derivative_backend"] == "cuda"
                assert result.provenance["one_electron_policy"] == "generated-cuda"
                assert result.provenance["eri_policy"] == "generated-cuda"
                assert (
                    result.provenance["executed_generated_scalar_work"]
                    <= owner.planned_work
                )
                backend.close()
                with pytest.raises(RuntimeError, match="closed"):
                    owner.solve()
