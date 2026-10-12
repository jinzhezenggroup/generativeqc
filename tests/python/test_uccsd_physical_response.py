"""Relaxed unrestricted weights against independent AO derivatives and failures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from test_uccsd_lambda import _problem
from test_uccsd_t_owner import _ASYMMETRIC_H6, _BROKEN_H4, _noninteracting

from tools.generativeqc_cc.uccsd_lambda import PreparedUCCSDLambda
from tools.generativeqc_cc.uccsd_physical_response import (
    PreparedUCCSDPhysicalResponse,
    UCCSDPhysicalResponseOptions,
)
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_response.implicit import ImplicitSolveError


def _independent_gradient(oracle: Any) -> np.ndarray:
    """Tighten only PySCF's default CPHF tolerance; keep its independent algebra."""
    from pyscf.scf import ucphf

    original = ucphf.solve

    def tight(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **{**kwargs, "tol": 1e-12, "max_cycle": 200})

    with patch("pyscf.grad.uccsd.ucphf.solve", side_effect=tight):
        return oracle.nuc_grad_method().kernel()


def _independent_fd(oracle: Any, atom: int = 0, axis: int = 2) -> float:
    """Re-converge both UHF and UCCSD at displaced physical coordinates.

    The installed PySCF open-shell analytic gradient has microhartree-scale
    differences from this stricter independent complete-energy derivative.
    Keep this gate separate rather than weakening the force FD tolerance.
    """
    from pyscf import cc, scf

    step = 2e-4
    energies = []
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
        energies.append(correlated.e_tot)
    return (energies[1] - energies[0]) / (2 * step)


def _ao_derivative_oracle(snapshot: Any, oracle: Any, weights: Any) -> np.ndarray:
    """Dense independent test-only contraction, with four separately owned slots."""
    coefficients = {"a": snapshot.coefficients_alpha, "b": snapshot.coefficients_beta}
    hcore = sum(
        coefficients[spin] @ weights[f"h_{spin}"] @ coefficients[spin].T
        for spin in "ab"
    )
    overlap = sum(
        coefficients[spin] @ weights[f"overlap_{spin}"] @ coefficients[spin].T
        for spin in "ab"
    )
    eri_weights = sum(
        np.einsum(
            "up,vq,wr,xs,pqrs->uvwx",
            coefficients[pair[0]],
            coefficients[pair[0]],
            coefficients[pair[1]],
            coefficients[pair[1]],
            weights[f"g_{pair}"],
            optimize=True,
        )
        for pair in ("aa", "ab", "bb")
    )
    molecule = oracle.mol
    hf_gradient = oracle._scf.nuc_grad_method()
    hcore_generator = hf_gradient.hcore_generator(molecule)
    overlap_ip = molecule.intor("int1e_ipovlp", comp=3)
    eri_ip = molecule.intor("int2e_ip1", comp=3).reshape(3, *(snapshot.nbf,) * 4)
    gradient = hf_gradient.grad_nuc()
    for atom, (_, _, ao_start, ao_end) in enumerate(molecule.aoslice_by_atom()):
        derivative = np.zeros_like(overlap_ip)
        derivative[:, ao_start:ao_end, :] -= overlap_ip[:, ao_start:ao_end, :]
        derivative[:, :, ao_start:ao_end] -= overlap_ip[
            :, ao_start:ao_end, :
        ].transpose(0, 2, 1)
        gradient[atom] += np.einsum("xpq,pq->x", hcore_generator(atom), hcore)
        gradient[atom] += np.einsum("xpq,pq->x", derivative, overlap)
        for slot, permutation in enumerate(
            ((0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 3, 4, 1, 2), (0, 3, 4, 2, 1))
        ):
            selected = tuple(
                slice(ao_start, ao_end) if axis == slot else slice(None)
                for axis in range(4)
            )
            gradient[atom] -= np.einsum(
                "xpqrs,pqrs->x",
                eri_ip.transpose(permutation)[(slice(None), *selected)],
                eri_weights[selected],
            )
    return gradient


@pytest.mark.parametrize(
    "atom,spin,broken",
    [
        ("N 0 0 0; H 0 -.8 .6; H 0 .8 .6", 1, False),
        (_BROKEN_H4, 0, True),
        (_ASYMMETRIC_H6, 0, False),
    ],
)
def test_relaxed_weights_close_independent_complete_molecular_gradient(
    atom: str, spin: int, broken: bool
) -> None:
    snapshot, source, oracle, _, _ = _problem(atom, spin, broken)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        assert state.converged
        with PreparedUCCSDLambda(parent, state) as lambdas:
            multipliers = lambdas.solve()
            with PreparedUCCSDPhysicalResponse(
                lambdas,
                multipliers,
                provider,
                nuclear_repulsion=oracle.mol.energy_nuc(),
            ) as physical:
                result = physical.solve()
                gradient = _ao_derivative_oracle(snapshot, oracle, result.weights)
                np.testing.assert_allclose(
                    gradient, _independent_gradient(oracle), atol=2e-6, rtol=0
                )
                np.testing.assert_allclose(
                    gradient[0, 2], _independent_fd(oracle), atol=2e-7, rtol=0
                )
                assert result.orbital_residual_norm <= 1e-8
                assert result.orbital_stationarity_max <= 1e-8
                assert (
                    result.provenance["executed_semantic_scalar_work"]
                    <= physical.planned_work
                )
                for value in (*result.weights.values(), result.orbital_adjoint):
                    with pytest.raises(ValueError):
                        value.setflags(write=True)


@pytest.fixture
def small_physical() -> Any:
    snapshot, source = _noninteracting(3, 1, 1)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
    ):
        state = parent.solve()
        with PreparedUCCSDLambda(parent, state) as lambdas:
            yield lambdas, lambdas.solve(), provider, source


@pytest.mark.parametrize(
    "options",
    [
        UCCSDPhysicalResponseOptions(max_bytes=1),
        UCCSDPhysicalResponseOptions(max_total_work=1),
        UCCSDPhysicalResponseOptions(max_producer_tiles=1),
    ],
)
def test_physical_admission_precedes_producer_reads(
    small_physical: tuple, options: Any
) -> None:
    lambdas, result, provider, source = small_physical
    reads = source.reads
    with pytest.raises(MemoryError):
        PreparedUCCSDPhysicalResponse(
            lambdas, result, provider, options=options, nuclear_repulsion=0.0
        )
    assert source.reads == reads


def test_forged_lambda_closed_resources_and_mutated_binding_fail_closed(
    small_physical: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    lambdas, result, provider, source = small_physical
    packed = result.packed_multipliers.copy()
    packed[0] += 0.01
    forged = replace(
        result,
        packed_multipliers=packed,
        multipliers={
            name.replace("t", "l", 1): value
            for name, value in lambdas.layout.unpack(packed).items()
        },
        lambda_identity=lambdas._lambda_identity(packed),
    )
    with (
        PreparedUCCSDPhysicalResponse(
            lambdas, forged, provider, nuclear_repulsion=0.0
        ) as physical,
        pytest.raises(ImplicitSolveError, match="stationarity"),
    ):
        physical.solve()
    with PreparedUCCSDPhysicalResponse(
        lambdas, result, provider, nuclear_repulsion=0.0
    ) as physical:
        with monkeypatch.context() as context:
            context.setattr(physical, "nuclear_repulsion", 1.0)
            with pytest.raises(RuntimeError, match="configuration"):
                physical.solve()
        source.closed = True
        with pytest.raises(RuntimeError, match="closed"):
            physical.solve()
        source.closed = False
        provider.close()
        with pytest.raises(RuntimeError, match="closed"):
            physical.solve()
    with pytest.raises(RuntimeError, match="closed"):
        physical.solve()


def test_false_success_and_late_nonfinite_never_publish(
    small_physical: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    lambdas, result, provider, _ = small_physical

    class FalseSuccess:
        identity, backend = "injected-physical", "test-false-success"
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

    with (
        PreparedUCCSDPhysicalResponse(
            lambdas, result, provider, nuclear_repulsion=0.0, solver=FalseSuccess()
        ) as physical,
        pytest.raises(ImplicitSolveError, match="true adjoint"),
    ):
        physical.solve()
    with PreparedUCCSDPhysicalResponse(
        lambdas, result, provider, nuclear_repulsion=0.0
    ) as physical:
        original = lambdas.prepared.executor.execute

        def nonfinite(program: Any, feeds: Any) -> dict:
            outputs = original(program, feeds)
            if (
                program.logical_hash == physical.programs.weights.logical_hash
                and float(feeds["bar_reference_electronic_energy"]) == 1
            ):
                outputs["g_ab"] = np.full_like(outputs["g_ab"], np.nan)
            return outputs

        monkeypatch.setattr(lambdas.prepared.executor, "execute", nonfinite)
        with pytest.raises(FloatingPointError, match="nonfinite"):
            physical.solve()


@pytest.mark.parametrize("oa,ob", [(3, 0), (3, 3)])
def test_empty_response_domains_preserve_exact_reference_weights(
    oa: int, ob: int
) -> None:
    snapshot, source = _noninteracting(3, oa, ob)
    with (
        ConventionalProvider(snapshot, source) as provider,
        PreparedUCCSD(snapshot, provider) as parent,
        PreparedUCCSDLambda(parent, parent.solve()) as lambdas,
        PreparedUCCSDPhysicalResponse(
            lambdas, lambdas.solve(), provider, nuclear_repulsion=0.0
        ) as physical,
    ):
        result = physical.solve()
        assert not result.orbital_adjoint.size
        assert result.orbital_stationarity_max == result.orbital_residual_norm == 0
        np.testing.assert_array_equal(result.weights["h_a"], np.eye(3))
        np.testing.assert_array_equal(
            result.weights["h_b"], np.eye(3) if ob else np.zeros((3, 3))
        )
