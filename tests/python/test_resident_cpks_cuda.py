"""Complete native resident CPKS solves against independent libcint/Libxc."""

import os
import typing
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions
from generativeqc.response_problem import ResponseUnsupported
from generativeqc.response_solver import GMRESOptions
from generativeqc.rks_response import NativeRKSResponse
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_response_native_rks import GRID, H2, _independent_mf

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)

H4 = [
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.1, 0.0, 0.7)),
    ("H", (3.0, 0.2, -0.65)),
    ("H", (3.2, 0.1, 0.65)),
]


def _calculator(method: str, *, renamed: bool = False) -> Calculator:
    composition = (
        replace(
            resolve_method(method.rsplit("-", 1)[0].upper(), spin="unpolarized"),
            identifier="resident-graph-not-a-dispatch-name",
        )
        if renamed
        else None
    )
    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID, composition=composition),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


@pytest.fixture(params=("lda-rks", "pbe-rks", "pbe0-rks"), scope="module")
def native(request: pytest.FixtureRequest) -> typing.Iterator[typing.Any]:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with _calculator(request.param).prepare_batch([H4]) as batch, NativeAO(H4) as basis:
        batch.execute(strict=True, properties=("energy",))
        with NativeRKSResponse.from_native(batch, basis, tile_points=127) as response:
            yield request.param, response, basis


def _independent_operator(
    method: str, response: typing.Any, basis: typing.Any
) -> np.ndarray:
    """Build a tiny oracle Jacobian with independent integrals and fxc/K."""
    reference, layout = response.problem.reference, response.problem.layout
    oracle = _independent_mf(response, basis)
    oracle.xc = {"lda-rks": "LDA_X,LDA_C_PW", "pbe-rks": "PBE", "pbe0-rks": "PBE0"}[
        method
    ]
    oracle.mo_coeff = reference.coefficients
    oracle.mo_occ = reference.occupations
    oracle.mo_energy = reference.orbital_energies
    action = oracle.gen_response(hermi=1)
    coefficients = reference.coefficients
    gap = (
        reference.orbital_energies[list(layout.virtual)][None, :]
        - reference.orbital_energies[list(layout.occupied)][:, None]
    )
    columns = []
    for direction in np.eye(response.dimension):
        density = coefficients @ layout.density_matrix(direction) @ coefficients.T
        potential = coefficients.T @ action(density) @ coefficients
        columns.append(
            (
                gap * layout.as_ia(direction)
                + potential[np.ix_(layout.virtual, layout.occupied)].T
            ).ravel()
        )
    return np.column_stack(columns)


def _forbid_host(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    raise AssertionError("resident CPKS used a Python action/transform/Krylov fallback")


def test_complete_resident_solve_and_transfer_contract(
    native: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, response, basis = native
    oracle = _independent_operator(method, response, basis)
    rhs = np.random.default_rng(302).normal(size=response.dimension) * 0.03
    expected = np.linalg.solve(oracle, rhs)
    before = response.xc_kernel._cuda.diagnostics
    monkeypatch.setattr(response, "apply", _forbid_host)
    monkeypatch.setattr(response.backend, "coulomb_exchange", _forbid_host)
    monkeypatch.setattr(response.xc_kernel, "apply", _forbid_host)
    monkeypatch.setattr(basis, "evaluate", _forbid_host)
    monkeypatch.setattr("generativeqc.rks_response.solve", _forbid_host)
    result = response.solve(
        rhs, options=GMRESOptions(rtol=1e-11, atol=1e-12, restart=2)
    )
    assert result.converged, result.reason
    np.testing.assert_allclose(result.solution, expected, atol=2e-9, rtol=2e-9)
    assert np.linalg.norm(oracle @ result.solution - rhs) < 2e-9
    after = response.xc_kernel._cuda.diagnostics
    assert after["action_h2d_bytes"] == before["action_h2d_bytes"]
    info = response.diagnostics["last_resident_solve"]
    reference = response.problem.reference
    setup = (
        reference.nmo**2 + reference.nocc**2 + (reference.nmo - reference.nocc) ** 2
    ) * 8
    assert info["h2d_bytes"] == setup + rhs.nbytes
    assert info["owned_device_bytes"] <= response._device_budget_bytes
    assert response.diagnostics["krylov_execution"] == "native-cuda"


def test_native_initial_guess_zero_rhs_and_workspace_refusal(
    native: typing.Any,
) -> None:
    _, response, _ = native
    zero = response.solve(np.zeros(response.dimension))
    assert zero.converged and zero.operator_actions == 0
    np.testing.assert_array_equal(zero.solution, 0)
    rhs = np.random.default_rng(303).normal(size=response.dimension) * 0.01
    baseline = response.solve(rhs, options=GMRESOptions(rtol=1e-11, atol=1e-12))
    assert baseline.converged
    warm = response.solve(
        rhs,
        initial_guess=baseline.solution,
        options=GMRESOptions(rtol=1e-10, atol=1e-11),
    )
    assert warm.converged and warm.iterations == 0
    refused = response.solve(rhs, options=GMRESOptions(max_workspace_bytes=1))
    assert refused.reason == "workspace_limit" and not refused.converged
    info = response.diagnostics["last_resident_solve"]
    assert info["h2d_bytes"] == info["d2h_bytes"] == info["owned_device_bytes"] == 0
    assert info["operator_actions"] == 0
    budget = response._device_budget_bytes
    response._device_budget_bytes = response.diagnostics["owned_device_bytes"] + 1
    try:
        with pytest.raises(
            (MemoryError, RuntimeError), match="memory|capacity|admission"
        ):
            response.solve(rhs)
    finally:
        response._device_budget_bytes = budget
    replay = response.solve(rhs)
    assert replay.converged
    np.testing.assert_allclose(replay.solution, baseline.solution, atol=2e-9)


def test_native_stale_generation_fails_before_even_zero_rhs() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    with _calculator("pbe-rks").prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True, properties=("energy",))
        with NativeRKSResponse.from_native(batch, basis) as response:
            batch.execute(strict=True, properties=("energy",))
            with pytest.raises(ValueError, match="stale"):
                response.solve(np.zeros(response.dimension))


def test_renamed_graph_keeps_resident_admission() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    with (
        _calculator("pbe-rks", renamed=True).prepare_batch([H2]) as batch,
        NativeAO(H2) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        with NativeRKSResponse.from_native(batch, basis) as response:
            result = response.solve(np.ones(response.dimension) * 0.01)
            assert result.converged


def test_wb97mv_scf_does_not_claim_complete_resident_response() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    with _calculator("wb97m-v-rks").prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True, properties=("energy",))
        with pytest.raises(
            ResponseUnsupported, match="tau, range exchange and nonlocal"
        ):
            NativeRKSResponse.from_native(batch, basis)
