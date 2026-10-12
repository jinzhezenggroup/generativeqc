"""Converged unrestricted Lambda, fixed-Hamiltonian response and ownership gates."""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from generativeqc.response_solver import GMRESOptions
from test_uccsd_equations import _molecular_case
from test_uccsd_solver import _snapshot
from test_uccsd_t_owner import _ASYMMETRIC_H6, _BROKEN_H4, _noninteracting
from test_ump2_energy import DenseSource

from tools.generativeqc_cc.lambda_solver import BoundCCSDLambda, LambdaOptions
from tools.generativeqc_cc.solver import SolverOptions, solve
from tools.generativeqc_cc.uccsd_lambda import PreparedUCCSDLambda, UCCSDLambdaOptions
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.reference import ReferenceSnapshot
from tools.generativeqc_response.implicit import ImplicitSolveError, ResponseGMRES


@lru_cache(maxsize=8)
def _problem(atom: str, spin: int, broken: bool = False) -> tuple:
    """Polish one stationary reference, then rebuild independent CC/Lambda objects."""
    cc_module = pytest.importorskip("pyscf.cc")
    restricted = spin == 0 and not broken
    initial, _, _, _, _ = _molecular_case(
        atom, spin, restricted=restricted, broken=broken
    )
    reference = initial._scf
    reference.direct_scf = False
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 300
    reference.kernel(dm0=reference.make_rdm1())
    if not reference.converged:
        reference = reference.newton()
        reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = (
            1e-13,
            1e-10,
            300,
        )
        reference.kernel(dm0=reference.make_rdm1())
    assert reference.converged
    if broken:
        density = reference.make_rdm1()
        assert np.linalg.norm(density[0] - density[1]) > 0.1
    if restricted:
        reference = reference.to_uhf()
    coupled_cluster = cc_module.UCCSD(reference)
    coupled_cluster.conv_tol = 1e-13
    coupled_cluster.conv_tol_normt = 1e-11
    coupled_cluster.max_cycle = 400
    eris = coupled_cluster.ao2mo()
    coupled_cluster.kernel(eris=eris)
    assert coupled_cluster.converged
    coupled_cluster.solve_lambda(eris=eris)
    assert coupled_cluster.converged_lambda
    expected = dict(
        zip(
            ("l1a", "l1b", "l2aa", "l2ab", "l2bb"),
            (*coupled_cluster.l1, *coupled_cluster.l2),
            strict=True,
        )
    )
    snapshot = _snapshot(coupled_cluster)
    return (
        snapshot,
        DenseSource(reference.mol.intor("int2e"), snapshot),
        coupled_cluster,
        eris,
        expected,
    )


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
def test_complete_cpu_lambda_against_independent_pyscf(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, _, _, expected = _problem(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged, state.reason
        with PreparedUCCSDLambda(parent, state) as owner:
            result = owner.solve()
            assert result.converged
            for name, values in expected.items():
                np.testing.assert_allclose(
                    result.multipliers[name], values, atol=2e-7, rtol=0
                )
            assert result.primal_residual_max <= owner.options.cc_tolerance
            assert (
                max(
                    result.solver_residual_norm,
                    result.stationarity_norm,
                    result.stationarity_max,
                )
                <= owner.options.lambda_tolerance
            )
            assert result.provenance["tensor_backend"] == "native-cpu-tensorir"
            assert (
                result.provenance["executed_semantic_scalar_work"]
                <= owner.planned_scalar_work
                <= owner.options.max_total_work
            )
            assert result.provenance["numeric_bound_bytes"] <= owner.options.max_bytes


def test_restricted_limit_matches_existing_rccsd_lambda_dual() -> None:
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
    assert state.converged, state.reason
    bound = BoundCCSDLambda(
        restricted,
        state,
        options=LambdaOptions(
            gmres=GMRESOptions(rtol=0.0, atol=1e-11, max_iterations=400)
        ),
    )
    expected = bound.solve(reference_identity=restricted.identity)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDLambda(parent, state) as owner:
            actual = owner.solve()
    np.testing.assert_allclose(
        actual.multipliers["l1a"] + actual.multipliers["l1b"],
        expected.lambda1,
        atol=2e-7,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.multipliers["l2ab"] + actual.multipliers["l2aa"],
        expected.lambda2,
        atol=2e-7,
        rtol=0,
    )


def test_fixed_hamiltonian_cotangents_match_reconverged_pyscf_energy() -> None:
    cc_module = pytest.importorskip("pyscf.cc")
    snapshot, source, oracle, eris, _ = _problem("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1)
    rng = np.random.default_rng(1824)
    directions = {}
    for spin in ("a", "b"):
        direction = rng.normal(scale=0.1, size=(snapshot.nmo,) * 2)
        directions[spin] = direction + direction.T
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDLambda(parent, state) as owner:
            result = owner.solve()
            derivative = 0.0
            for spin, name in (("alpha", "a"), ("beta", "b")):
                occupied = snapshot.nocc(spin)
                for domain, selected in (
                    ("oo", (slice(0, occupied), slice(0, occupied))),
                    ("ov", (slice(0, occupied), slice(occupied, None))),
                    ("vv", (slice(occupied, None), slice(occupied, None))),
                ):
                    cotangent = owner.parameter_cotangent(
                        result, f"f{domain}_{name * 2}"
                    )
                    derivative += cotangent.contract(directions[name][selected].copy())
    energies = []
    for sign in (-1, 1):
        perturbed = cc_module.UCCSD(oracle._scf)
        perturbed.conv_tol, perturbed.conv_tol_normt, perturbed.max_cycle = (
            1e-13,
            1e-11,
            400,
        )
        perturbed_eris = perturbed.ao2mo()
        perturbed_eris.focka = eris.focka + sign * 1e-4 * directions["a"]
        perturbed_eris.fockb = eris.fockb + sign * 1e-4 * directions["b"]
        energies.append(perturbed.kernel(eris=perturbed_eris)[0])
        assert perturbed.converged
    np.testing.assert_allclose(
        derivative, (energies[1] - energies[0]) / 2e-4, atol=2e-7, rtol=0
    )


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
    "options",
    [
        UCCSDLambdaOptions(max_bytes=1),
        UCCSDLambdaOptions(max_total_work=1),
        UCCSDLambdaOptions(
            gmres=GMRESOptions(rtol=0.0, atol=1e-11, max_workspace_bytes=1)
        ),
    ],
)
def test_admission_precedes_native_execution(
    small_parent: tuple, options: UCCSDLambdaOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, state, source = small_parent
    reads = source.reads

    def forbidden(*_: Any) -> None:
        pytest.fail("rejected Lambda admission executed a native program")

    monkeypatch.setattr(parent.executor, "execute", forbidden)
    with pytest.raises(MemoryError):
        PreparedUCCSDLambda(parent, state, options=options)
    assert source.reads == reads


def test_forged_state_and_lambda_cannot_bypass_fresh_stationarity(
    small_parent: tuple,
) -> None:
    parent, state, _ = small_parent
    with pytest.raises(ValueError, match="converged"):
        PreparedUCCSDLambda(parent, replace(state, status="iteration_limit"))
    with pytest.raises(ValueError, match="identities"):
        PreparedUCCSDLambda(
            parent,
            replace(state, provenance={**state.provenance, "reference_id": "other"}),
        )
    forged = state.packed_amplitudes.copy()
    forged[0] += 0.01
    with (
        PreparedUCCSDLambda(
            parent,
            replace(
                state, packed_amplitudes=forged, amplitudes=parent.layout.unpack(forged)
            ),
        ) as owner,
        pytest.raises(ImplicitSolveError, match="fresh physical"),
    ):
        owner.solve()
    with PreparedUCCSDLambda(parent, state) as owner:
        result = owner.solve()
        packed = result.packed_multipliers.copy()
        packed[0] += 0.01
        multipliers = {
            name.replace("t", "l", 1): value
            for name, value in owner.layout.unpack(packed).items()
        }
        forged = replace(
            result,
            packed_multipliers=packed,
            multipliers=multipliers,
            lambda_identity=owner._lambda_identity(packed),
        )
        with pytest.raises(ImplicitSolveError, match="stationarity"):
            owner.parameter_cotangent(forged, "foo_aa")


def test_immutable_publication_close_and_configuration_guards(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, state, source = small_parent
    with PreparedUCCSDLambda(parent, state) as owner:
        source.closed = True
        result = owner.solve()
        for value in (result.packed_multipliers, *result.multipliers.values()):
            with pytest.raises(ValueError):
                value.setflags(write=True)
        with pytest.raises(TypeError):
            cast("dict[str, Any]", result.provenance["options"]["gmres"])["atol"] = 1.0
        weight = owner.parameter_cotangent(result, "foo_aa")
        with pytest.raises(ValueError):
            weight.values.setflags(write=True)
        with monkeypatch.context() as context:
            context.setattr(owner, "options", UCCSDLambdaOptions(max_bytes=1))
            with pytest.raises(RuntimeError, match="configuration"):
                owner.solve()
        with monkeypatch.context() as context:
            context.setattr(parent.executor, "max_bytes", parent.executor.max_bytes * 2)
            with pytest.raises(RuntimeError, match="configuration"):
                owner.solve()
        parent.close()
        with pytest.raises(RuntimeError, match="closed"):
            owner.solve()
    with pytest.raises(RuntimeError, match="closed"):
        owner.solve()


def test_false_success_nonfinite_and_native_failures_never_publish_lambda(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, state, _ = small_parent

    class FalseSuccess:
        identity, backend = "injected-solver", "test-false-success"
        workspace_bytes, rtol, atol = 1024, 0.0, 1e-11

        def solve(self, operator: Any, rhs: np.ndarray) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                converged=True,
                solution=np.ones_like(rhs),
                residual_norm=0.0,
                iterations=1,
                workspace_bytes=1024,
            )

    class ExhaustActions(FalseSuccess):
        """Exercise the action bound even when the physical RHS is exactly zero."""

        def solve(self, operator: Any, rhs: np.ndarray) -> Any:
            operator.apply(np.zeros_like(rhs))
            operator.apply(np.zeros_like(rhs))
            return super().solve(operator, rhs)

    with (
        PreparedUCCSDLambda(parent, state, solver=FalseSuccess()) as owner,
        pytest.raises(ImplicitSolveError, match="true adjoint"),
    ):
        owner.solve()
    with (
        PreparedUCCSDLambda(
            parent,
            state,
            options=UCCSDLambdaOptions(max_operator_actions=1),
            solver=ExhaustActions(),
        ) as owner,
        pytest.raises(MemoryError, match="operator-action"),
    ):
        owner.solve()
    with pytest.raises(ValueError, match="dimension"):
        PreparedUCCSDLambda(
            parent, state, solver=ResponseGMRES(1, GMRESOptions(rtol=0.0, atol=1e-11))
        )
    with PreparedUCCSDLambda(parent, state) as owner:
        original = parent.executor.execute

        def nonfinite(program: Any, feeds: Any) -> dict:
            outputs = original(program, feeds)
            if program.logical_hash == owner.programs.stationarity_vjp.derivative_hash:
                outputs["bar_t1a"] = np.full_like(outputs["bar_t1a"], np.nan)
            return outputs

        monkeypatch.setattr(parent.executor, "execute", nonfinite)
        with pytest.raises(FloatingPointError, match="nonfinite"):
            owner.solve()

        def failure(program: Any, feeds: Any) -> dict:
            if program.logical_hash == owner.programs.residual_vjp.derivative_hash:
                raise ValueError("injected native derivative failure")
            return original(program, feeds)

        monkeypatch.setattr(parent.executor, "execute", failure)
        with pytest.raises(ValueError, match="native derivative failure"):
            owner.solve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cc_tolerance": 1e-8},
        {"lambda_tolerance": np.nan},
        {"lambda_tolerance": True},
        {"max_bytes": 0},
        {"max_operator_actions": 0},
        {"max_total_work": -1},
        {"gmres": object()},
    ],
)
def test_invalid_lambda_options_fail_closed(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        UCCSDLambdaOptions(**kwargs)


def test_parameter_failures_are_transactional(
    small_parent: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late parameter failure cannot publish a bar or poison valid requests."""
    parent, state, _ = small_parent
    with PreparedUCCSDLambda(parent, state) as owner:
        result = owner.solve()
        original = parent.executor.execute
        for fault in ("nonfinite", "dtype", "shape", "missing"):
            with monkeypatch.context() as context:

                def broken(
                    program: Any, feeds: Any, selected_fault: str = fault
                ) -> dict:
                    outputs = original(program, feeds)
                    if "bar_foo_aa" not in outputs:
                        return outputs
                    if selected_fault == "missing":
                        return {}
                    value = outputs["bar_foo_aa"]
                    if selected_fault == "nonfinite":
                        outputs["bar_foo_aa"] = np.full_like(value, np.nan)
                    elif selected_fault == "dtype":
                        outputs["bar_foo_aa"] = value.astype(np.float32)
                    else:
                        outputs["bar_foo_aa"] = value.reshape(-1)
                    return outputs

                context.setattr(parent.executor, "execute", broken)
                with pytest.raises((ValueError, FloatingPointError)):
                    owner.parameter_cotangent(result, "foo_aa")
        valid = owner.parameter_cotangent(result, "foo_aa")
        np.testing.assert_array_equal(valid.values, np.zeros_like(valid.values))
        for direction in (
            np.zeros_like(valid.values, dtype=np.float32),
            np.full_like(valid.values, np.nan),
            np.ma.array(np.zeros_like(valid.values)),
        ):
            with pytest.raises(ValueError, match="direction"):
                valid.contract(direction)


@pytest.mark.parametrize("nmo,oa,ob", [(3, 3, 0), (3, 3, 3)])
def test_empty_excitation_lambda_and_correlation_cotangents_are_exact_zero(
    nmo: int, oa: int, ob: int
) -> None:
    snapshot, source = _noninteracting(nmo, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDLambda(parent, state) as owner:
            result = owner.solve()
            assert not result.packed_multipliers.size
            assert result.operator_actions == result.stationarity_norm == 0
            cotangent = owner.parameter_cotangent(result, "foo_aa")
            np.testing.assert_array_equal(cotangent.values, np.zeros((oa, oa)))


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
def test_complete_cuda_lambda_and_parameter_vjp_match_independent_oracle(
    atom: str, spin: int, broken: bool, tmp_path: Path
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda

    assert os.environ.get("SLURM_JOB_ID"), (
        "real CUDA gates require finite Slurm allocation"
    )
    snapshot, source, _, _, expected = _problem(atom, spin, broken)
    cache = Path(os.environ.get("GENERATIVEQC_UCCSD_LAMBDA_TEST_CACHE", str(tmp_path)))
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
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
            options=SolverOptions(max_iterations=400),
            executor=executor,
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged, state.reason
        with PreparedUCCSDLambda(parent, state) as owner:
            actual = owner.solve()
            assert actual.provenance["tensor_backend"] == "cuda-fp64-ordinary-stream"
            assert actual.provenance["integral_backend"] == "cuda"
            for name, values in expected.items():
                np.testing.assert_allclose(
                    actual.multipliers[name], values, atol=2e-7, rtol=0
                )
            parameters = ("fov_aa", "fov_bb", "goovv_aaaa", "goovv_abab", "goovv_bbbb")
            weights = {
                name: owner.parameter_cotangent(actual, name) for name in parameters
            }
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(
            snapshot, provider, options=SolverOptions(max_iterations=400)
        ) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDLambda(parent, state) as owner:
            reference = owner.solve()
            for name in parameters:
                expected_weight = owner.parameter_cotangent(reference, name)
                np.testing.assert_allclose(
                    weights[name].values, expected_weight.values, atol=2e-7, rtol=0
                )
