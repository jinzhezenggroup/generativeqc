"""Independent adjoints for source-first, full-rank pure Coulomb response."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


@pytest.fixture(scope="module")
def response_probe(tmp_path_factory: Any, native_cxx: NativeCxx) -> Any:
    assert os.environ.get("SLURM_JOB_ID")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    cuda = Path(os.environ["CUDA_PATH"])
    output = tmp_path_factory.mktemp("raw-coulomb-response") / "probe.so"
    native_cxx.build_shared(
        [ROOT / "tests/native/df_raw_coulomb_response_probe.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
        ),
        link_args=(
            str(library),
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-lcublas",
            "-Wl,-rpath," + str(library.parent),
            "-Wl,-rpath," + str(cuda / "lib64"),
        ),
    )
    call = ct.CDLL(str(output)).df_raw_coulomb_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = (
        [ct.c_size_t] * 4
        + [dp] * 5
        + [ct.c_bool] * 3
        + [dp] * 2
        + [ct.c_void_p, ct.c_size_t]
    )
    call.restype = ct.c_int
    return call


@pytest.mark.parametrize("condition", (1e2, 1e6, 1e10))
@pytest.mark.parametrize("tile", (1, 3, 11))
@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("batched", (False, True), ids=("slices", "panels"))
def test_signed_multidensity_adjoint_and_rank_refusal(
    response_probe: Any,
    condition: float,
    tile: int,
    signed: bool,
    batched: bool,
) -> None:
    """Ragged/full panels and ill-conditioned metrics retain signed terms.

    Extended-precision host contractions independently check both A and M
    adjoints. Unsupported exchange and rank truncation must refuse the route.
    """
    n, a, count = 4, 11, 3
    rng = np.random.default_rng(2657)
    q, _ = np.linalg.qr(rng.normal(size=(a, a)))
    eigenvalues = np.geomspace(1 / condition, 1, a)
    # A physical raw source becomes small in weak metric directions. Retain
    # this scaling while rotating the eigensystem away from coordinate axes.
    raw = (q @ (np.sqrt(eigenvalues)[:, None] * rng.normal(size=(a, n * n)))).reshape(
        a, n, n
    )
    raw = np.ascontiguousarray((raw + raw.transpose(0, 2, 1)) / 2)
    density = rng.normal(size=(count, n, n))
    density = np.ascontiguousarray((density + density.transpose(0, 2, 1)) / 2)
    coefficients = np.array([1.0, -0.37 if signed else 0.37, 0.0])
    q_column = np.ascontiguousarray(q.T)
    bar_a, bar_m = np.full_like(raw, np.nan), np.full((a, a), np.nan)
    arrays = (raw, density, coefficients, q_column, eigenvalues)
    dp = ct.POINTER(ct.c_double)
    pointers = [array.ctypes.data_as(dp) for array in arrays]
    error = ct.create_string_buffer(1024)
    status = response_probe(
        n,
        a,
        count,
        tile,
        *pointers,
        True,
        False,
        batched,
        bar_a.ctypes.data_as(dp),
        bar_m.ctypes.data_as(dp),
        error,
        len(error),
    )
    assert status == 0, error.value.decode()
    ld = np.longdouble
    charge = np.einsum("pij,tij->tp", raw.astype(ld), density.astype(ld))
    potential = ((charge @ q.astype(ld)) / eigenvalues.astype(ld)) @ q.astype(ld).T
    expected_a = np.einsum(
        "t,tij,tp->pij", coefficients.astype(ld), density.astype(ld), potential
    )
    expected_m = (
        -np.einsum("t,tp,tq->pq", coefficients.astype(ld), potential, potential) / 2
    )
    np.testing.assert_allclose(bar_a, expected_a, atol=2e-8, rtol=3e-9)
    # Signed outer products can cancel a large adjoint almost completely.
    # Bound their sum against the magnitudes of the contributing terms, not
    # the cancelled result. Complete molecular force gates remain absolute.
    metric_scale = (
        np.einsum(
            "t,tp,tq->pq", abs(coefficients.astype(ld)), abs(potential), abs(potential)
        )
        / 2
    )
    assert np.all(abs(bar_m - expected_m) <= 2e-6 + 3e-9 * metric_scale)
    for full_rank, exchange in ((False, False), (True, True)):
        status = response_probe(
            n,
            a,
            count,
            tile,
            *pointers,
            full_rank,
            exchange,
            batched,
            bar_a.ctypes.data_as(dp),
            bar_m.ctypes.data_as(dp),
            error,
            len(error),
        )
        assert status > 0


@pytest.mark.parametrize("method,xc", (("pbe-rks", "PBE"), ("r2scan-3c", "R2SCAN")))
@pytest.mark.parametrize("execution", ("auto", "generic"))
def test_streamed_pure_coulomb_complete_endpoint(
    method: str, xc: str, execution: str, monkeypatch: Any, tmp_path: Path
) -> None:
    """Default auxiliary-g grids/forces agree with independent PySCF endpoints.

    A constrained dense value owner ensures this gate exercises source replay;
    the unchanged physical gates include moving grids and auxiliary response.
    """
    import test_scf_df_auxiliary_g_cuda as endpoint
    from generativeqc import Calculator

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    monkeypatch.setenv("GENERATIVEQC_DF_COULOMB_RESPONSE", "auto")
    if execution == "auto":
        monkeypatch.delenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", execution)
    trace = tmp_path / "response.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    monkeypatch.setattr(
        endpoint,
        "Calculator",
        lambda **options: Calculator(
            **options,
            density_fitting_memory_budget_bytes=(256 if method == "r2scan-3c" else 64)
            << 20,
        ),
    )
    endpoint.check_endpoint(method, xc, False, "water-tetramer-def2-svp-spherical")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    responses = [row for row in records if row["operation"] == "force_response"]
    assert responses
    for row in responses:
        counts = row["counters"]
        assert counts.get("response_raw_coulomb_source_passes") == 1
        assert counts["raw_value_bytes"] == row["nbf"] ** 2 * row["naux"] * 8
        assert not counts.get("response_fitted_projection_flops", 0)
        # Confirm bounded source batching was exercised by this molecular
        # owner, without pinning a planner-specific tile or exact call count.
        assert 0 < counts["response_raw_coulomb_source_panels"] < row["naux"]
        # Source-only pure J has no resident tensor prerequisite for generated
        # derivatives. The explicit generic rollback still reaches the same
        # independent force/energy gates, without enabling extra screening.
        assert counts["response_source_coulomb_shell_promoted"] == (execution == "auto")
        assert counts["response_derivative_shell_execution"] == (execution == "auto")
        assert counts["response_derivative_profile_promoted"] == 0


def test_raw_coulomb_geometry_rebuild(monkeypatch: Any, tmp_path: Path) -> None:
    """Changing centers invalidates raw values and metric potentials together.

    This is an independent force/lifetime gate, not a moved-geometry timing.
    Displace one atom so a stale but translationally invariant tensor cannot
    accidentally pass by reproducing the old forces.
    """
    from generativeqc import Calculator
    from pyscf import dft, gto, lib
    from test_dft_complete_cuda import no_cpu_derivatives
    from test_scf_df_auxiliary_g_cuda import auxiliary_snapshot

    from benchmarks._cases import benchmark_cases

    lib.num_threads(1)
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = [
        (symbol, tuple(np.asarray(xyz) + (0, 0, 0.025) if i == 1 else xyz))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    molecule = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", verbose=0)
    auxiliary = auxiliary_snapshot(molecule)
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    monkeypatch.setenv("GENERATIVEQC_DF_COULOMB_RESPONSE", "auto")
    calculator = Calculator(
        method="pbe-rks",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        auxiliary_basis=auxiliary,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=64 << 20,
    )
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as batch:
            for phase, geometry in enumerate((atoms, moved)):
                mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", verbose=0)
                oracle = dft.RKS(mol).density_fit()
                oracle.xc = "PBE"
                oracle.conv_tol, oracle.conv_tol_grad = 1e-12, 1e-9
                energy = oracle.kernel()
                assert oracle.converged
                gradient = oracle.nuc_grad_method()
                gradient.grid_response = True
                gradient.auxbasis_response = True
                forces = -gradient.kernel()
                trace = tmp_path / f"geometry-{phase}.jsonl"
                monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                positions = (
                    [np.asarray([xyz for _, xyz in geometry])] if phase else None
                )
                with no_cpu_derivatives():
                    result = batch.execute(
                        positions, strict=True, properties=("energy", "forces")
                    ).items[0]
                assert result.converged and result.executed_backend == "cuda"
                assert abs(result.energy - energy) < 2e-7
                delta = result.forces - forces
                assert np.max(abs(delta)) < 2e-6
                assert np.sqrt(np.mean(delta**2)) < 7e-7
                np.testing.assert_allclose(
                    result.forces.sum(axis=0), 0, atol=2e-8, rtol=0
                )
                records = [json.loads(line) for line in trace.read_text().splitlines()]
                assert any(
                    row["counters"].get("response_raw_coulomb_source_passes")
                    for row in records
                )
    finally:
        calculator.clear_cache()
