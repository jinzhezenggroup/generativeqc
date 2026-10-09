"""Internal native DF solver against dense and determinant-space endpoints."""

from __future__ import annotations

import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest

from tools import generate_df_ccsd_core as core
from tools import generate_df_ccsd_hoisted as hoisted
from tools import generate_df_ccsd_native as actions
from tools import generate_rccsd_native as conventional
from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "consumer", ["test_df_auxiliary_accumulation.cu", "test_cuda_pair_history.cu"]
)
def test_generated_auxiliary_accumulation_preserves_order(
    tmp_path: Path, consumer: str
) -> None:
    """Execute generated/storage consumers against independent full tensors."""
    if (
        os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1"
        or os.environ.get("GENERATIVEQC_DF_CC_USE_LIBRARY") != "1"
    ):
        pytest.skip("requires the complete CUDA library in a finite Slurm allocation")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    cache, compiler = shutil.which("ccache"), shutil.which("nvcc")
    assert cache and compiler
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    obj, executable = tmp_path / "probe.o", tmp_path / "probe"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=sm_120",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(library.parent / "generated"),
            "-c",
            str(ROOT / "tests/native" / consumer),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [
            compiler,
            str(obj),
            "-L" + str(library.parent),
            "-lgenerativeqc",
            "-lcublas",
            "-arch=sm_120",
            "-Xlinker",
            "-rpath=" + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=30)


FIELDS = (
    "foo",
    "fov",
    "fvv",
    "ovov",
    "ovvo",
    "oovv",
    "ovvv",
    "ovoo",
    "oooo",
    "vvvv",
    "d1",
    "d2",
    "t1",
    "t2",
    "bov",
    "bvv",
)
COLUMNS = (
    "status",
    "energy",
    "iterations",
    "r1",
    "r2",
    "capacity",
    "device_bytes",
    "h2d",
    "scalar_d2h",
    "amplitude_d2h",
    "iterations_called",
    "replays_called",
    "q_calls",
    "q_operations",
    "accumulations",
    "seconds",
    "hoisted_evaluations",
    "preparation_calls",
    "contraction_terms",
    "matrix_gemm",
    "gemm_calls",
    "gemm_summands",
    "packing_bytes",
    "provider_capacity",
    "batch_size",
    "q_tiles",
    "accumulation_bytes",
    "denominator_identity",
    "derived_d2_iteration_evaluations",
    "conventional_prepared",
    "conventional_calls",
    "conventional_summands",
    "conventional_provider_capacity",
    "conventional_binding_host",
    "packed_diis",
    "packed_diis_refused",
    "diis_disabled_after_packing_refusal",
    "diis_history_capacity_bytes",
    "diis_conversion_bytes",
    "diis_metric_weight_terms",
    "diis_pack_calls",
    "diis_maximum_pair_asymmetry",
    "replay_matrix",
)


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def solver_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, bool]:
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicit Slurm real-device qualification")
    cache = shutil.which("ccache")
    cxx = shutil.which("c++")
    nvcc = shutil.which("nvcc")
    if not cache or not cxx or (cuda and not nvcc):
        pytest.skip("requires ccache and selected compilers")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-solver-" + request.param)
    # Endpoint qualification can exercise the frozen complete library instead
    # of recompiling a standalone solver. The default keeps codegen coverage.
    if cuda and os.environ.get("GENERATIVEQC_DF_CC_USE_LIBRARY") == "1":
        library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
        obj, executable = directory / "probe.o", directory / "solver-probe"
        subprocess.run(
            [
                cache,
                cxx,
                "-std=c++20",
                "-O2",
                "-I" + str(ROOT / "src"),
                "-c",
                str(ROOT / "tests/native/df_cc_solver_probe.cpp"),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        subprocess.run(
            [
                cxx,
                str(obj),
                str(library),
                "-Wl,-rpath," + str(library.parent),
                "-o",
                str(executable),
            ],
            check=True,
            capture_output=True,
        )
        return executable, cuda
    for name, producer in (
        ("generated_rccsd_cpu.hpp", conventional.cpu_header),
        ("generated_rccsd_cuda.cu", conventional.cuda_source),
        ("generated_df_ccsd_cpu.hpp", actions.cpu_header),
        ("generated_df_ccsd_cuda.cuh", actions.cuda_header),
        ("generated_df_ccsd_cuda.cu", actions.cuda_source),
        ("generated_df_ccsd_core_cpu.hpp", core.cpu_header),
        ("generated_df_ccsd_core_cuda.cuh", core.cuda_header),
        ("generated_df_ccsd_core_cuda.cu", core.cuda_source),
        ("generated_df_ccsd_hoisted_cpu.hpp", hoisted.cpu_header),
        ("generated_df_ccsd_hoisted_cuda.cuh", hoisted.cuda_header),
        ("generated_df_ccsd_hoisted_cuda.cu", hoisted.cuda_source),
    ):
        (directory / name).write_text(producer())
    sources = [ROOT / "src/cc/solver.cpp", ROOT / "tests/native/df_cc_solver_probe.cpp"]
    if cuda:
        sources += [
            ROOT / "src/cc/cuda_solver.cu",
            *[
                directory / name
                for name in (
                    "generated_rccsd_cuda.cu",
                    "generated_df_ccsd_cuda.cu",
                    "generated_df_ccsd_core_cuda.cu",
                    "generated_df_ccsd_hoisted_cuda.cu",
                )
            ],
        ]
    objects = []
    for source in sources:
        device = source.suffix == ".cu"
        compiler = nvcc if device else cxx
        assert compiler is not None
        obj = directory / (source.name + ".o")
        command = [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            f"-DGENERATIVEQC_HAS_CUDA={int(cuda)}",
            "-I" + str(directory),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ]
        if device:
            command += ["-arch=sm_120"]
        subprocess.run(
            [*command, "-c", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(str(obj))
    executable = directory / "solver-probe"
    compiler = nvcc if cuda else cxx
    assert compiler is not None
    subprocess.run(
        [compiler, *objects, *(["-lcublas"] if cuda else []), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return executable, cuda


def _case(
    o: int = 2, v: int = 3, q: int = 4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(1759 + o + v)
    factors = rng.normal(scale=0.025, size=(q, o + v, o + v))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    g = np.einsum("Qpq,Qrs->pqrs", factors, factors)
    eps = np.r_[np.linspace(-1.3, -0.7, o), np.linspace(0.4, 1.1, v)]
    fock = np.diag(eps)
    fock[:o, o:] = rng.normal(scale=0.002, size=(o, v))
    fock[o:, :o] = fock[:o, o:].T
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    arrays = dense_feeds(
        fock, g, fock[:o, o:] / d1, g[:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2
    )
    arrays.update(d1=d1, d2=d2, bov=factors[:, :o, o:], bvv=factors[:, o:, o:])
    return fock, g, arrays


def _stream(
    arrays: dict[str, np.ndarray],
    cuda: bool,
    *,
    df: bool = True,
    budget: int = 1 << 30,
    diis: int = 6,
    hoist: bool = True,
    matrix: bool = True,
    batch_limit: int = 8,
    canonical_eps: np.ndarray | None = None,
    level_shift: float = 0.0,
    packed_diis: bool = False,
    replay_matrix: bool = True,
    max_iterations: int = 100,
) -> bytes:
    o, v = arrays["t1"].shape
    q = len(arrays["bov"]) if df else 0
    header = np.array(
        [
            o,
            v,
            q,
            budget,
            max_iterations,
            diis,
            int(cuda)
            | (0 if hoist else 4)
            | (0 if matrix else 8)
            | (16 if canonical_eps is not None else 0)
            | (32 if packed_diis else 0)
            | (0 if replay_matrix else 64)
            | (batch_limit << 8),
        ],
        dtype=np.uint64,
    )
    omitted = ("ovvv", "vvvv") if df else ("bov", "bvv")
    data = header.tobytes() + b"".join(
        np.asarray(arrays[name], dtype=np.float64).tobytes()
        for name in FIELDS
        if name not in omitted
    )
    if canonical_eps is not None:
        data += np.asarray(canonical_eps, dtype=np.float64).tobytes()
        data += np.float64(level_shift).tobytes()
    return data


@pytest.mark.parametrize(
    "df,hoist,matrix",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_disabled_diis_evaluates_each_accepted_state_once(
    solver_probe: tuple[Path, bool], df: bool, hoist: bool, matrix: bool
) -> None:
    """Jacobi-only solves must not evaluate a trial and then repeat it next iteration."""
    _, _, arrays = _case(2, 3, 5)
    status, _, _ = _run(
        solver_probe,
        arrays,
        df=df,
        hoist=hoist,
        matrix=matrix,
        diis=0,
        max_iterations=4,
    )
    assert status["iterations_called"] == status["iterations"]


@pytest.mark.parametrize(
    "df,hoist,matrix",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_canonical_spectrum_preserves_solver_trajectory(
    solver_probe: tuple[Path, bool], df: bool, hoist: bool, matrix: bool
) -> None:
    """Compare each early trajectory prefix and final independent residual replay."""
    fock, g, arrays = _case(2, 3, 5)
    eps = np.diag(fock).copy()
    shift = 0.173
    gaps = eps[:2, None] - eps[None, 2:]
    # Independent ordered arithmetic, including a nonzero level shift.
    arrays["d1"] = gaps - shift
    arrays["d2"] = gaps[:, None, :, None] + gaps[None, :, None, :] - 2.0 * shift
    schedule = {"df": df, "hoist": hoist, "matrix": matrix}
    for limit in (1, 2, 3, 4, 100):
        explicit, x1, x2 = _run(solver_probe, arrays, max_iterations=limit, **schedule)
        derived, d1, d2 = _run(
            solver_probe,
            arrays,
            max_iterations=limit,
            canonical_eps=eps,
            level_shift=shift,
            **schedule,
        )
        assert explicit["status"] == derived["status"]
        assert explicit["iterations"] == derived["iterations"]
        assert explicit["denominator_identity"] != derived["denominator_identity"]
        np.testing.assert_allclose(
            explicit["energy"], derived["energy"], atol=2e-12, rtol=0
        )
        np.testing.assert_allclose(x1, d1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(x2, d2, atol=2e-11, rtol=0)
        assert (
            derived["derived_d2_iteration_evaluations"]
            == arrays["d2"].size * derived["iterations_called"]
        )
        assert explicit["derived_d2_iteration_evaluations"] == 0
    energy, r1, r2 = DeterminantOracle(fock, g, 2).evaluate_full(d1, d2)
    np.testing.assert_allclose(derived["energy"], energy, atol=2e-12, rtol=0)
    assert max(np.max(np.abs(r1)), np.max(np.abs(r2))) <= 1e-10


def test_canonical_capacity_admits_previously_rejected_problem(
    solver_probe: tuple[Path, bool],
) -> None:
    """Force the bounded schedule, then test its actual byte boundary."""
    fock, _, arrays = _case(2, 6, 1)
    eps = np.diag(fock).copy()
    schedule = {"hoist": False, "matrix": False}
    explicit, _, _ = _run(solver_probe, arrays, **schedule)
    derived, _, _ = _run(solver_probe, arrays, canonical_eps=eps, **schedule)
    assert derived["capacity"] < explicit["capacity"]
    assert (
        explicit["capacity"] - derived["capacity"] >= arrays["d2"].nbytes - eps.nbytes
    )
    if solver_probe[1]:
        assert explicit["h2d"] - derived["h2d"] == arrays["d2"].nbytes - eps.nbytes
        assert derived["device_bytes"] < explicit["device_bytes"]
    budget = int(derived["capacity"])
    _run(solver_probe, arrays, budget=budget, canonical_eps=eps, **schedule)
    for spectrum, amount in ((eps, budget - 1), (None, budget)):
        rejected = subprocess.run(
            [str(solver_probe[0])],
            input=_stream(
                arrays,
                solver_probe[1],
                budget=amount,
                canonical_eps=spectrum,
                **schedule,
            ),
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert rejected.returncode and b"budget" in rejected.stderr


@pytest.mark.parametrize(
    "df,hoist,matrix", [(False, False, False), (True, False, False), (True, True, True)]
)
def test_packed_history_preserves_trajectories_and_physical_oracle(
    solver_probe: tuple[Path, bool], df: bool, hoist: bool, matrix: bool
) -> None:
    if not solver_probe[1]:
        pytest.skip("packed storage is a CUDA consumer")
    fock, g, arrays = _case(2, 3, 5)
    schedule = {"df": df, "hoist": hoist, "matrix": matrix, "diis": 8}
    for limit in (1, 2, 3, 4, 100):
        full, f1, f2 = _run(solver_probe, arrays, max_iterations=limit, **schedule)
        packed, p1, p2 = _run(
            solver_probe, arrays, max_iterations=limit, packed_diis=True, **schedule
        )
        assert packed["packed_diis"] and not packed["packed_diis_refused"]
        assert full["status"] == packed["status"]
        assert full["iterations"] == packed["iterations"]
        np.testing.assert_allclose(full["energy"], packed["energy"], atol=2e-12, rtol=0)
        np.testing.assert_allclose(f1, p1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(f2, p2, atol=2e-11, rtol=0)
        assert packed["device_bytes"] < full["device_bytes"]
    energy, r1, r2 = DeterminantOracle(fock, g, 2).evaluate_full(p1, p2)
    np.testing.assert_allclose(packed["energy"], energy, atol=2e-12, rtol=0)
    assert max(np.max(np.abs(r1)), np.max(np.abs(r2))) <= 1e-10


def test_packed_history_capacity_and_asymmetry_fallback(
    solver_probe: tuple[Path, bool],
) -> None:
    if not solver_probe[1]:
        pytest.skip("packed storage is a CUDA consumer")
    _, _, arrays = _case(2, 6, 1)
    schedule = {"hoist": False, "matrix": False, "diis": 8}
    full, _, _ = _run(solver_probe, arrays, **schedule)
    packed, _, _ = _run(solver_probe, arrays, packed_diis=True, **schedule)
    assert packed["capacity"] < full["capacity"]
    assert packed["diis_history_capacity_bytes"] < full["diis_history_capacity_bytes"]
    budget = packed["capacity"]
    exact, _, _ = _run(
        solver_probe, arrays, packed_diis=True, budget=budget, **schedule
    )
    assert exact["status"] == 0
    for selected, amount in ((True, budget - 1), (False, budget)):
        p = subprocess.run(
            [str(solver_probe[0])],
            input=_stream(
                arrays, True, packed_diis=selected, budget=amount, **schedule
            ),
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert p.returncode and b"budget" in p.stderr
    for initial in (True, False):
        changed = {k: v.copy() for k, v in arrays.items()}
        if initial:
            # Even a one-ULP supplied asymmetry must preserve the explicit input.
            changed["t2"][0, 1, 0, 1] = np.nextafter(changed["t2"][0, 1, 0, 1], 1.0)
        else:
            # Arbitrary supplied denominators can break symmetry in the first
            # update despite symmetric initial amplitudes: runtime refusal.
            changed["d2"][0, 1, 0, 1] *= 1.1
        expected, e1, e2 = _run(solver_probe, changed, **schedule)
        actual, a1, a2 = _run(solver_probe, changed, packed_diis=True, **schedule)
        assert actual["packed_diis_refused"] and not actual["packed_diis"]
        assert actual["status"] == expected["status"] == 0
        np.testing.assert_allclose(a1, e1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(a2, e2, atol=2e-11, rtol=0)


def _run(
    probe: tuple[Path, bool], arrays: dict[str, np.ndarray], **kwargs: typing.Any
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    executable, cuda = probe
    process = subprocess.run(
        [str(executable)],
        input=_stream(arrays, cuda, **kwargs),
        capture_output=True,
        check=True,
        timeout=120,
    )
    lines = process.stdout.decode().splitlines()
    status = {
        key: float(value)
        if key in ("energy", "r1", "r2", "seconds", "diis_maximum_pair_asymmetry")
        else int(value)
        for key, value in zip(COLUMNS, lines[0].split(), strict=True)
    }
    values = np.fromstring(lines[1], sep=" ")
    o, v = arrays["t1"].shape
    return status, values[: o * v].reshape(o, v), values[o * v :].reshape(o, o, v, v)


@pytest.mark.parametrize("o,v,q,diis", [(1, 1, 1, 0), (2, 3, 4, 6), (3, 2, 3, 6)])
def test_solver_matches_dense_and_independent_determinants(
    solver_probe: tuple[Path, bool], o: int, v: int, q: int, diis: int
) -> None:
    fock, g, arrays = _case(o, v, q)
    actual, t1, t2 = _run(solver_probe, arrays, diis=diis)
    dense, dense_t1, dense_t2 = _run(solver_probe, arrays, df=False, diis=diis)
    assert actual["status"] == dense["status"] == 0
    np.testing.assert_allclose(actual["energy"], dense["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, dense_t1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, dense_t2, atol=2e-11, rtol=0)
    energy, r1, r2 = DeterminantOracle(fock, g, o).evaluate_full(t1, t2)
    np.testing.assert_allclose(actual["energy"], energy, atol=2e-12, rtol=0)
    assert (
        max(np.max(np.abs(r1)), np.max(np.abs(r2)), actual["r1"], actual["r2"]) <= 1e-10
    )
    calls = actual["iterations_called"] + actual["replays_called"]
    assert actual["q_calls"] == q * calls
    if solver_probe[1]:
        tiled = actual["hoisted_evaluations"]
        one_q = q * (calls - tiled)
        assert actual["q_tiles"] == one_q + tiled * int(
            np.ceil(q / actual["batch_size"])
        )
        assert actual["accumulations"] == one_q + actual["q_tiles"]
        assert actual["accumulation_bytes"] > 0
    else:
        assert (
            actual["accumulations"]
            == 2 * q * calls + 4 * q * actual["hoisted_evaluations"]
        )
    assert actual["preparation_calls"] == actual["hoisted_evaluations"]
    assert actual["hoisted_evaluations"] <= actual["iterations_called"]
    assert actual["q_operations"] > actual["q_calls"]
    assert dense["q_calls"] == dense["q_operations"] == dense["accumulations"] == 0
    if solver_probe[1]:
        assert dense["conventional_prepared"] == 1
        assert dense["conventional_calls"] > 0 and dense["conventional_summands"] > 0
        assert dense["conventional_provider_capacity"] == 96 << 20
        assert dense["conventional_binding_host"] > 0
        expected = sum(
            arrays[name].nbytes for name in FIELDS if name not in ("ovvv", "vvvv")
        )
        assert actual["h2d"] == expected
        assert actual["amplitude_d2h"] == t1.nbytes + t2.nbytes


def test_conventional_prepared_binding_retains_exact_budget_fallback(
    solver_probe: tuple[Path, bool],
) -> None:
    """Resource pressure retains the original scientific iteration/replay."""
    if not solver_probe[1]:
        pytest.skip("prepared conventional binding is a native CUDA consumer")
    _, _, arrays = _case()
    bound, t1, t2 = _run(solver_probe, arrays, df=False)
    exact, _, _ = _run(solver_probe, arrays, df=False, budget=int(bound["capacity"]))
    assert exact["conventional_prepared"] == 1
    scalar_capacity = int(
        bound["capacity"]
        - bound["conventional_provider_capacity"]
        - bound["conventional_binding_host"]
    )
    for budget in (int(bound["capacity"]) - 1, scalar_capacity):
        fallback, f1, f2 = _run(solver_probe, arrays, df=False, budget=budget)
        assert fallback["status"] == bound["status"] == 0
        assert fallback["conventional_prepared"] == fallback["conventional_calls"] == 0
        assert fallback["capacity"] == scalar_capacity <= budget
        assert fallback["replays_called"] == bound["replays_called"] == 1
        np.testing.assert_allclose(
            fallback["energy"], bound["energy"], atol=2e-12, rtol=0
        )
        np.testing.assert_allclose(f1, t1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(f2, t2, atol=2e-11, rtol=0)
    refused = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(arrays, True, df=False, budget=scalar_capacity - 1),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode and b"budget" in refused.stderr


def test_exact_memory_admission_and_symmetric_factor_gate(
    solver_probe: tuple[Path, bool],
) -> None:
    _, _, arrays = _case()
    actual, _, _ = _run(solver_probe, arrays)
    fallback, _, _ = _run(solver_probe, arrays, hoist=False)
    conventional_admission = bytearray(_stream(arrays, solver_probe[1]))
    conventional_admission[6 * 8 : 7 * 8] = np.uint64(2).tobytes()
    process = subprocess.run(
        [str(solver_probe[0])],
        input=conventional_admission,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode and b"factorized execution owner" in process.stderr
    _run(solver_probe, arrays, budget=int(actual["capacity"]))
    bounded, _, _ = _run(solver_probe, arrays, budget=int(fallback["capacity"]))
    assert actual["capacity"] > fallback["capacity"]
    assert actual["hoisted_evaluations"] > 0 and bounded["hoisted_evaluations"] == 0
    process = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(
            arrays,
            solver_probe[1],
            budget=int(min(actual["capacity"], fallback["capacity"])) - 1,
        ),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode and b"budget" in process.stderr
    arrays["bvv"][0, 0, 1] += 0.01
    process = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(arrays, solver_probe[1]),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode and b"symmetric" in process.stderr


def test_matrix_schedule_matches_scalar_and_budget_fallback(
    solver_probe: tuple[Path, bool],
) -> None:
    if not solver_probe[1]:
        pytest.skip("matrix provider is a CUDA execution option")
    _, _, arrays = _case(2, 3, 4)
    fast, t1, t2 = _run(solver_probe, arrays)
    scalar, s1, s2 = _run(solver_probe, arrays, matrix=False)
    assert fast["status"] == scalar["status"] == 0
    assert fast["matrix_gemm"] == 1 and scalar["matrix_gemm"] == 0
    assert 0 < fast["gemm_summands"] <= fast["contraction_terms"]
    assert fast["gemm_calls"] > 0 and fast["packing_bytes"] > 0
    assert fast["provider_capacity"] == 96 << 20
    assert (
        scalar["gemm_calls"]
        == scalar["packing_bytes"]
        == scalar["provider_capacity"]
        == 0
    )
    np.testing.assert_allclose(fast["energy"], scalar["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, s1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, s2, atol=2e-11, rtol=0)
    admitted, _, _ = _run(solver_probe, arrays, budget=int(fast["capacity"]))
    assert admitted["matrix_gemm"] == 1
    one_q, _, _ = _run(solver_probe, arrays, batch_limit=1, replay_matrix=False)
    for budget in (int(one_q["capacity"]) - 1, int(scalar["capacity"])):
        bounded, b1, b2 = _run(solver_probe, arrays, budget=budget)
        assert bounded["matrix_gemm"] == 0 and bounded["hoisted_evaluations"] > 0
        assert bounded["capacity"] <= budget
        np.testing.assert_allclose(b1, s1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(b2, s2, atol=2e-11, rtol=0)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_auxiliary_tiles_preserve_tail_and_budget(
    solver_probe: tuple[Path, bool], batch: int
) -> None:
    """Uneven tails and exact/short budgets retain independently replayed results."""
    if not solver_probe[1]:
        pytest.skip("Q tiles are a CUDA matrix execution option")
    _, _, arrays = _case(2, 3, 5)
    one, s1, s2 = _run(solver_probe, arrays, batch_limit=1)
    tiled, t1, t2 = _run(solver_probe, arrays, batch_limit=batch)
    assert tiled["batch_size"] == min(batch, 5)
    assert tiled["status"] == one["status"] == 0
    assert tiled["iterations"] == one["iterations"]
    np.testing.assert_allclose(tiled["energy"], one["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, s1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, s2, atol=2e-11, rtol=0)
    exact, _, _ = _run(
        solver_probe, arrays, batch_limit=batch, budget=int(tiled["capacity"])
    )
    assert exact["batch_size"] == tiled["batch_size"]
    if batch > 1:
        assert tiled["q_tiles"] < one["q_tiles"]
        assert tiled["q_operations"] < one["q_operations"]
        assert tiled["gemm_calls"] < one["gemm_calls"]
        assert tiled["accumulation_bytes"] < one["accumulation_bytes"]
        for budget in (int(tiled["capacity"]) - 1, int(one["capacity"])):
            short, b1, b2 = _run(solver_probe, arrays, batch_limit=batch, budget=budget)
            assert short["batch_size"] < tiled["batch_size"] or (
                tiled["replay_matrix"] and not short["replay_matrix"]
            )
            assert short["capacity"] <= budget
            np.testing.assert_allclose(b1, s1, atol=2e-11, rtol=0)
            np.testing.assert_allclose(b2, s2, atol=2e-11, rtol=0)


def test_expanded_matrix_replay_ablation_and_optional_capacity(
    solver_probe: tuple[Path, bool],
) -> None:
    """Audit storage is optional and cannot displace an admitted primal tile."""
    if not solver_probe[1]:
        pytest.skip("matrix replay is a CUDA execution plan")
    _, _, arrays = _case(2, 3, 5)
    original, reference_t1, reference_t2 = _run(
        solver_probe, arrays, replay_matrix=False
    )
    selected, actual_t1, actual_t2 = _run(solver_probe, arrays)
    assert selected["replay_matrix"] and not original["replay_matrix"]
    assert selected["gemm_calls"] > original["gemm_calls"]
    assert selected["iterations"] == original["iterations"]
    assert selected["replays_called"] == original["replays_called"]
    assert selected["q_calls"] == original["q_calls"]
    assert selected["accumulations"] == original["accumulations"]
    assert (
        original["contraction_terms"] - selected["contraction_terms"]
        == 5 * 2 * 3 * 3 * selected["replays_called"]
    )
    np.testing.assert_allclose(actual_t1, reference_t1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual_t2, reference_t2, atol=2e-11, rtol=0)
    exact, _, _ = _run(solver_probe, arrays, budget=int(selected["capacity"]))
    assert exact["replay_matrix"] and exact["capacity"] <= selected["capacity"]
    for budget in (int(selected["capacity"]) - 1, int(original["capacity"])):
        refused, bounded_t1, bounded_t2 = _run(solver_probe, arrays, budget=budget)
        assert not refused["replay_matrix"] and refused["matrix_gemm"]
        assert refused["batch_size"] == original["batch_size"]
        assert refused["capacity"] <= budget
        np.testing.assert_allclose(bounded_t1, reference_t1, atol=2e-11, rtol=0)
        np.testing.assert_allclose(bounded_t2, reference_t2, atol=2e-11, rtol=0)


@pytest.mark.parametrize("o,v,q", [(2, 3, 4), (4, 1, 1), (2, 6, 1)])
def test_hoisted_solver_matches_forced_bounded_fallback(
    solver_probe: tuple[Path, bool], o: int, v: int, q: int
) -> None:
    """Charge complete solves, including the unchanged expanded replay.

    Low-Q and occupied-rich cases protect scheduling outside the large
    virtual-rich workload, without interpreting work counts as wall time.
    """
    _, _, arrays = _case(o, v, q)
    fast, t1, t2 = _run(solver_probe, arrays)
    old, old_t1, old_t2 = _run(solver_probe, arrays, hoist=False)
    assert fast["status"] == old["status"] == 0
    assert fast["replays_called"] == old["replays_called"] == 1
    assert fast["hoisted_evaluations"] == fast["iterations_called"] > 0
    assert old["hoisted_evaluations"] == old["preparation_calls"] == 0
    assert 0 < fast["contraction_terms"] < old["contraction_terms"]
    np.testing.assert_allclose(fast["energy"], old["energy"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(t1, old_t1, atol=2e-11, rtol=0)
    np.testing.assert_allclose(t2, old_t2, atol=2e-11, rtol=0)


@pytest.mark.parametrize("field", ["t1", "oooo"])
def test_prepare_and_core_overflow_remain_sticky(
    solver_probe: tuple[Path, bool], field: str
) -> None:
    """Finite inputs overflow in prepare/core, with ordinary later Q slices."""
    _, _, arrays = _case()
    arrays[field].fill(1e200 if field == "t1" else 1e308)
    if field == "oooo":
        arrays["t2"].fill(8.0)
    status, _, _ = _run(solver_probe, arrays)
    assert status["status"] == 2


def test_early_auxiliary_overflow_survives_later_slices(
    solver_probe: tuple[Path, bool],
) -> None:
    _, _, arrays = _case()
    arrays["bvv"][0] *= 1e200
    # Later Q slices remain well behaved. A per-slice reset must not erase the
    # first error before convergence checks (including optimized DIIS paths).
    status, _, _ = _run(solver_probe, arrays)
    assert status["status"] == 2


@pytest.mark.parametrize("name", ["h2", "h2o", "ch4"])
def test_molecular_exact_factorization_recovers_pinned_ccsd(
    solver_probe: tuple[Path, bool], name: str
) -> None:
    """Exact positive-pair factorization is a solver oracle, not a DF-fit claim."""
    from test_df_ccsd_factorized import _factor_eri

    from tools.cc_endpoint_fixtures import load

    meta, source = load(name)
    o = int(np.count_nonzero(source["occ"]))
    eps = source["eps"]
    fock = source["C"].T @ source["F"] @ source["C"]
    factors = _factor_eri(source["g"])
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    arrays = dense_feeds(
        fock,
        source["g"],
        fock[:o, o:] / d1,
        source["g"][:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2,
    )
    arrays.update(d1=d1, d2=d2, bov=factors[:, :o, o:], bvv=factors[:, o:, o:])
    status, t1, t2 = _run(solver_probe, arrays)
    assert status["status"] == 0
    np.testing.assert_allclose(
        status["energy"], meta["correlation_energy"], atol=3e-10, rtol=0
    )
    np.testing.assert_allclose(t1, source["t1"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(t2, source["t2"], atol=2e-9, rtol=0)
