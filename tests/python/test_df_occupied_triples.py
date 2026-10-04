"""Independent triples algebra and native DF occupied-tile qualification."""

from __future__ import annotations

import ctypes as ct
import itertools
import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import (
    DF_TRIPLES_W_FP32_QUALIFICATION,
    PERMUTATIONS,
    df_panel_program,
    energy_scalar_program,
    moment_program,
    w_fp32_candidate_program,
)
from generativeqc_compiler.cc.triples import _LABELS, VP, triples_energy
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import describe_precision, execute
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.generate_df_occupied_triples import header

ROOT = Path(__file__).resolve().parents[2]
GPU = os.environ.get("GENERATIVEQC_DF_TRIPLES_CUDA_TEST") == "1"


def case(o: int, v: int, q: int = 4) -> tuple[list[np.ndarray], np.ndarray]:
    """Independent physical Gram integrals with pair-symmetric doubles."""
    rng = np.random.default_rng(1763 + 100 * o + v + q)
    b = rng.normal(scale=0.2, size=(q, o + v, o + v))
    b = (b + b.transpose(0, 2, 1)) / 2
    eri = np.einsum("Qpq,Qrs->pqrs", b, b)
    t2 = rng.normal(scale=0.03, size=(o, o, v, v))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    inputs = [
        b[:, :o, o:],
        b[:, o:, o:],
        eri[:o, o:, :o, :o],
        eri[:o, o:, :o, o:],
        rng.normal(scale=0.01, size=(o, v)),
        rng.normal(scale=0.04, size=(o, v)),
        t2,
        np.linspace(-1.1, -0.6, o),
        np.linspace(0.2, 1.3, v),
    ]
    return [np.ascontiguousarray(x) for x in inputs], eri[:o, o:, o:, o:]


def reference(inputs: list[np.ndarray], ovvv: np.ndarray) -> float:
    _, _, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    return float(
        triples_energy(len(eo), len(ev), ovvv, ovoo, ovov, fov, t1, t2, eo, ev)
    )


@pytest.fixture(scope="module")
def scalar_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    directory = tmp_path_factory.mktemp("df-triples-scalar")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    (directory / "generated.hpp").write_text(header())
    program = energy_scalar_program()
    names = tuple(
        sorted(n.attrs["name"] for n in program.live_nodes if n.op == "input")
    )
    args = ",".join(f"p[{i}]" for i in range(len(names)))
    source = directory / "probe.cpp"
    source.write_text(
        '#include "generated.hpp"\nextern "C" double point(const double* p) {'
        f"double result=0; generativeqc::cc::triples::generated_df::energy_element({args},result);"
        "return result;}\n"
    )
    obj, library = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(library))
    call = dll.point
    call.argtypes, call.restype = [ct.POINTER(ct.c_double)], ct.c_double
    return call, names


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2), (3, 4)])
def test_occupied_domain_preserves_all_permutations_and_multiplicities(
    scalar_probe: typing.Any, o: int, v: int
) -> None:
    inputs, ovvv = case(o, v)
    _, _, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    call, names = scalar_probe
    energy = 0.0
    for i in range(o):
        for j in range(i + 1):
            for k in range(j + 1):
                occupied = i, j, k
                moments = {}
                for label, order in zip(_LABELS, PERMUTATIONS, strict=True):
                    I, J, K = (occupied[x] for x in order)
                    # Independent direct seeds in the pinned source's layout.
                    moments[label] = (
                        np.einsum("afb,cf->abc", ovvv[I], t2[K, J])
                        - np.einsum("am,mbc->abc", ovoo[I, :, J], t2[:, K]),
                        np.einsum("ab,c->abc", ovov[I, :, J], t1[K])
                        + np.einsum("ab,c->abc", t2[I, J], fov[K]),
                    )
                degeneracy = 6 if i == k else 2 if i == j or j == k else 1
                for abc in itertools.product(range(v), repeat=3):
                    feeds = {
                        "denominator": (eo[i] + eo[j] + eo[k] - sum(ev[a] for a in abc))
                        * degeneracy
                    }
                    for occ in _LABELS:
                        w, vv = moments[occ]
                        feeds[f"v_{occ}"] = vv[abc]
                        for vir in _LABELS:
                            feeds[f"w_{occ}_{vir}"] = w[tuple(abc[x] for x in VP[vir])]
                    values = np.array([feeds[name] for name in names])
                    energy += call(values.ctypes.data_as(ct.POINTER(ct.c_double)))
    np.testing.assert_allclose(energy, reference(inputs, ovvv), atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("o,v,q", [(2, 3, 4), (3, 2, 1)])
def test_panel_and_each_moment_match_source_index_loops(o: int, v: int, q: int) -> None:
    inputs, ovvv = case(o, v, q)
    bov, bvv, ovoo, ovov, fov, t1, t2, _, _ = inputs
    for i, j, k in itertools.product(range(o), repeat=3):
        panel = execute(
            df_panel_program(v, q), {"bov_i": bov[:, i], "bvv": bvv}
        ).outputs["panel"]
        np.testing.assert_allclose(
            panel, ovvv[i].transpose(0, 2, 1), atol=2e-15, rtol=2e-15
        )
        actual = execute(
            moment_program(o, v),
            {
                "panel": panel,
                "t2_kj": t2[k, j],
                "ovoo_ij": ovoo[i, :, j],
                "t2_mk": t2[:, k],
                "ovov_ij": ovov[i, :, j],
                "t1_k": t1[k],
                "t2_ij": t2[i, j],
                "fov_k": fov[k],
            },
        ).outputs
        for a, b, c in itertools.product(range(v), repeat=3):
            w = sum(ovvv[i, a, f, b] * t2[k, j, c, f] for f in range(v))
            w -= sum(ovoo[i, a, j, m] * t2[m, k, b, c] for m in range(o))
            vv = ovov[i, a, j, b] * t1[k, c] + t2[i, j, a, b] * fov[k, c]
            np.testing.assert_allclose(
                [actual["w"][a, b, c], actual["v"][a, b, c]],
                [w, vv],
                atol=2e-15,
                rtol=2e-15,
            )


@pytest.mark.parametrize("o,v,q", [(2, 3, 4), (3, 4, 5)])
def test_w_fp32_candidate_keeps_sensitive_triples_algebra_fp64(
    o: int, v: int, q: int
) -> None:
    inputs, _ = case(o, v, q)
    _, _, ovoo, ovov, fov, t1, t2, _, _ = inputs
    panel = execute(
        df_panel_program(v, q), {"bov_i": inputs[0][:, 0], "bvv": inputs[1]}
    ).outputs["panel"]
    feeds = {
        "panel": panel,
        "t2_kj": t2[0, 0],
        "ovoo_ij": ovoo[0, :, 0],
        "t2_mk": t2[:, 0],
        "ovov_ij": ovov[0, :, 0],
        "t1_k": t1[0],
        "t2_ij": t2[0, 0],
        "fov_k": fov[0],
    }
    strict = moment_program(o, v)
    candidate = w_fp32_candidate_program(o, v)
    schedule = describe_precision(candidate)
    einsums = [value for value in schedule.values if value.op == "einsum"]
    lowered = [value for value in einsums if value.compute_dtype == "float32"]
    retained = [value for value in einsums if value.compute_dtype == "float64"]
    assert len(lowered) == 2 and len(retained) == 2
    assert all(
        (value.storage_dtype, value.compute_dtype, value.accumulation_dtype)
        == ("float32", "float32", "float32")
        for value in lowered
    )
    assert all(
        (value.storage_dtype, value.compute_dtype, value.accumulation_dtype)
        == ("float64", "float64", "float64")
        for value in retained
    )
    assert schedule.strict_audit_dtype == "float64"
    assert schedule.request_identity is not None
    assert {qualification for _, qualification in schedule.qualification_scope} == {
        DF_TRIPLES_W_FP32_QUALIFICATION
    }
    fp32_steps = [
        step
        for step in plan_cuda(candidate, cuda_target_info("sm_120")).steps
        if step.node.op == "einsum" and step.node.spec.dtype == "float32"
    ]
    assert len(fp32_steps) == 2
    assert all(step.gemm != "none" for step in fp32_steps)
    actual = execute(candidate, feeds).outputs
    expected = execute(strict, feeds).outputs
    np.testing.assert_array_equal(actual["v"], expected["v"])
    np.testing.assert_allclose(actual["w"], expected["w"], atol=2e-7, rtol=2e-6)
    assert actual["w"].dtype == np.float64


def test_generated_native_w_fp32_contract_uses_candidate_precision_identity() -> None:
    candidate = w_fp32_candidate_program(2, 3)
    schedule = describe_precision(candidate)
    generated = header()
    assert schedule.identity in generated
    assert schedule.request_identity is not None
    assert schedule.request_identity in generated
    assert "build_w_fp32" in generated
    assert "const float* ovoo" in generated
    assert "const float* t2" in generated
    assert "accumulate(scratch" in generated


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if not GPU:
        pytest.skip("requires finite Slurm GPU allocation and built library")
    directory = tmp_path_factory.mktemp("df-triples-native")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj, output = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-c",
            str(ROOT / "tests/native/df_triples_probe.cpp"),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            str(obj),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(output))
    call = dll.df_triples_probe_v2
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.POINTER(ct.c_double)),
        ct.c_double,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_int,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_size_t),
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run(
    call: typing.Any,
    inputs: list[np.ndarray],
    budget: int = 1 << 30,
    panels: int = 3,
    threshold: float = 1e-10,
    mixed: bool = False,
    generated: bool = False,
    cutensor: bool = False,
    reject_cutensor: bool = False,
) -> tuple:
    q, o, v = inputs[0].shape
    arrays = [np.ascontiguousarray(x) for x in inputs]
    pointers = (ct.POINTER(ct.c_double) * 9)(
        *(x.ctypes.data_as(ct.POINTER(ct.c_double)) for x in arrays)
    )
    values = np.full(3, np.nan)
    counts = np.zeros(26, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = call(
        o,
        v,
        q,
        pointers,
        threshold,
        budget,
        panels,
        int(mixed) + 2 * int(generated) + 4 * int(cutensor) + 8 * int(reject_cutensor),
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        len(counts),
        error,
        len(error),
    )
    return status, values, counts, error.value.decode()


def test_native_probe_rejects_short_diagnostic_buffer(native_probe: typing.Any) -> None:
    """The benchmark/probe ABI rejects old output capacities before any work."""
    values = np.full(3, np.nan)
    counts = np.full(1, 17, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = native_probe(
        1,
        1,
        1,
        None,
        1e-10,
        1 << 30,
        1,
        0,
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        len(counts),
        error,
        len(error),
    )
    assert status != 0 and b"diagnostic buffer" in error.value
    assert np.isnan(values).all() and counts[0] == 17


@pytest.mark.parametrize(
    "o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 1), (3, 5, 7), (4, 3, 2)]
)
def test_native_energy_work_budget_fallback_and_repeatability(
    native_probe: typing.Any, o: int, v: int, q: int, tmp_path: Path
) -> None:
    inputs, ovvv = case(o, v, q)
    want = reference(inputs, ovvv)
    status, values, counts, error = run(native_probe, inputs)
    assert status == 0, error
    np.testing.assert_allclose(values[0], want, atol=3e-12, rtol=3e-12)
    tiles = o * (o + 1) * (o + 2) // 6
    assert counts[0] == v * (v + 1) * (v + 2) // 6
    assert counts[1] == tiles and counts[5] == min(3, o)
    assert counts[7] == 12 * tiles
    assert counts[8] == tiles and counts[9] == tiles + 1
    assert counts[10] == tiles * v**3
    assert counts[11] == counts[6] * q * v**3 + 6 * tiles * (v**4 + o * v**3)
    assert counts[12] == sum(x.nbytes for x in inputs) and counts[13] == 12
    assert counts[4] <= 96 << 20 and counts[23] > 0
    assert counts[2] == counts[3] + (96 << 20) + counts[23]
    assert values[1] == pytest.approx(3 * (inputs[-1].min() - inputs[-2].max()))
    status, small_values, small_counts, error = run(native_probe, inputs, panels=1)
    assert status == 0, error
    assert small_counts[5] == 1 and small_counts[6] >= counts[6]
    np.testing.assert_array_equal(small_values[:2], values[:2])
    budget = int(small_counts[2])
    status, _, fallback, error = run(native_probe, inputs, budget=budget)
    assert status == 0, error
    # At very small shapes the extra panels can occupy alignment padding and
    # require no additional bytes. Fallback is needed only when the plans differ.
    assert fallback[5] == (counts[5] if counts[2] == budget else 1)
    status, low_values, low_counts, error = run(native_probe, inputs, budget=budget - 1)
    assert status == 0, error
    assert low_counts[20] == low_counts[22] == 1
    assert low_counts[4] == 0
    assert low_counts[2] == low_counts[3] + low_counts[23]
    assert tuple(low_counts[17:20]) == (64, 64, 64)
    np.testing.assert_allclose(low_values[0], want, atol=3e-12, rtol=3e-12)
    # Without the optional library, only arena and prepared descriptor storage
    # remain. Reject one byte below the smallest complete generated endpoint.
    minimum = int(small_counts[3] + small_counts[23])
    status, unpublished, _, error = run(native_probe, inputs, budget=minimum - 1)
    assert status != 0 and "budget" in error
    assert np.isnan(unpublished).all()
    status, repeated, _, error = run(native_probe, inputs)
    assert status == 0, error
    np.testing.assert_array_equal(repeated[:2], values[:2])
    (tmp_path / "record.json").write_text(
        json.dumps(
            {
                "shape": [o, v, q],
                "values": values.tolist(),
                "counts": counts.tolist(),
                "energy_error": abs(values[0] - want),
            },
            indent=2,
        )
        + "\n"
    )


def test_native_w_fp32_matches_strict_and_reports_actual_precision(
    native_probe: typing.Any,
) -> None:
    inputs, _ = case(2, 3, 4)
    strict_status, strict_values, strict_counts, strict_error = run(
        native_probe, inputs
    )
    mixed_status, mixed_values, mixed_counts, mixed_error = run(
        native_probe, inputs, mixed=True
    )
    assert strict_status == 0, strict_error
    assert mixed_status == 0, mixed_error
    np.testing.assert_allclose(mixed_values[0], strict_values[0], atol=2e-7, rtol=2e-4)
    assert strict_counts[15] == 0
    assert strict_counts[14] == strict_counts[6] + strict_counts[7]
    assert tuple(strict_counts[17:20]) == (64, 64, 64)
    assert mixed_counts[14] == mixed_counts[6]
    assert mixed_counts[15] == mixed_counts[7]
    assert mixed_counts[15] > 0
    assert mixed_counts[16] > 0
    assert tuple(mixed_counts[17:20]) == (32, 32, 32)
    assert mixed_counts[2] > strict_counts[2]


@pytest.mark.parametrize("bad", ["nan", "pair", "gap", "threshold", "overflow"])
def test_native_failure_does_not_publish(native_probe: typing.Any, bad: str) -> None:
    inputs, _ = case(2, 3)
    threshold = 1e-10
    if bad == "nan":
        inputs[0][0, 0, 0] = np.nan
    elif bad == "pair":
        inputs[1][0, 1, 0] += 0.1
    elif bad == "gap":
        inputs[-1][0] = inputs[-2].max() + 1e-12
    elif bad == "threshold":
        threshold = np.nan
    else:
        inputs[0].fill(1e300)
        inputs[1].fill(1e300)
    status, values, _, _ = run(native_probe, inputs, threshold=threshold)
    assert status != 0
    assert np.isnan(values).all()


@pytest.mark.parametrize("o,v", [(2, 1 << 16), (256, 1024)])
def test_dimension_and_complete_work_preflight_precedes_input_access(
    native_probe: typing.Any, o: int, v: int
) -> None:
    """Huge logical shapes must fail before dereferencing even null inputs."""
    pointers = (ct.POINTER(ct.c_double) * 9)()
    values = np.full(3, np.nan)
    counts = np.full(26, 17, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = native_probe(
        o,
        v,
        1,
        pointers,
        1e-10,
        ct.c_size_t(-1).value,
        3,
        0,
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        len(counts),
        error,
        len(error),
    )
    assert status != 0
    assert b"BLAS indexing" in error.value or b"overflow" in error.value
    assert np.isnan(values).all()
    assert (counts == 17).all()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("h2o", -6.731393342463869e-05),
        ("nh3", -1.122922812723691e-04),
        ("ch4", -1.555665872715297e-04),
    ],
)
@pytest.mark.parametrize("cutensor", [False, True])
def test_native_pinned_independent_molecular_energies(
    native_probe: typing.Any, name: str, expected: float, cutensor: bool
) -> None:
    """Exact Gram factors of committed ERIs reproduce pinned PySCF 2.14.0 (T).

    This factorization is test-only: the native production owner receives DF
    factors, and neither imports PySCF nor factors a dense four-index tensor.
    """
    if cutensor and os.environ.get("GENERATIVEQC_CUTENSOR_CUDA_TEST") != "1":
        pytest.skip("requires explicit optional cuTENSOR qualification")
    with np.load(ROOT / "tests/reference_data/cc/endpoints" / f"{name}.npz") as z:
        eps, occ, c, f, g, t1, t2 = (
            z[key] for key in ("eps", "occ", "C", "F", "g", "t1", "t2")
        )
    o, n = int(np.sum(occ > 0)), len(eps)
    eigenvalues, vectors = np.linalg.eigh(g.reshape(n * n, n * n))
    assert eigenvalues.min() > -1e-12
    b = (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T.reshape(n * n, n, n)
    b = (b + b.transpose(0, 2, 1)) / 2
    np.testing.assert_allclose(
        np.einsum("Qpq,Qrs->pqrs", b, b), g, atol=3e-13, rtol=3e-13
    )
    inputs = [
        b[:, :o, o:],
        b[:, o:, o:],
        g[:o, o:, :o, :o],
        g[:o, o:, :o, o:],
        (c.T @ f @ c)[:o, o:],
        t1,
        t2,
        eps[:o],
        eps[o:],
    ]
    status, values, counts, error = run(
        native_probe, inputs, cutensor=cutensor, budget=2 << 30
    )
    assert status == 0, error
    np.testing.assert_allclose(values[0], expected, atol=3e-12, rtol=3e-12)
    mixed_status, mixed_values, mixed_counts, mixed_error = run(
        native_probe, inputs, mixed=True, cutensor=cutensor, budget=2 << 30
    )
    assert mixed_status == 0, mixed_error
    np.testing.assert_allclose(mixed_values[0], expected, atol=2e-7, rtol=2e-4)
    assert mixed_counts[15] > 0
    assert tuple(mixed_counts[17:20]) == (32, 32, 32)
    assert counts[24] == mixed_counts[24] == int(cutensor)


@pytest.mark.parametrize("mixed", [False, True])
def test_cutensor_complete_endpoint_resources_and_transactional_fallback(
    native_probe: typing.Any, mixed: bool, tmp_path: Path
) -> None:
    """Real method execution, not just a microkernel: admission, casts, panels,
    W, denominators, reductions, transfer, stream drain and cleanup all run.
    Test-only scores select the provider without asserting a performance win.
    """
    if os.environ.get("GENERATIVEQC_CUTENSOR_CUDA_TEST") != "1":
        pytest.skip("requires explicit optional cuTENSOR qualification")
    inputs, ovvv = case(3, 4, 5)
    want = reference(inputs, ovvv)
    timings, work = [], []
    for _ in range(3):
        status, values, counts, error = run(
            native_probe, inputs, mixed=mixed, cutensor=True, budget=2 << 30
        )
        assert status == 0, error
        np.testing.assert_allclose(
            values[0],
            want,
            atol=2e-7 if mixed else 3e-12,
            rtol=2e-4 if mixed else 3e-12,
        )
        assert counts[24] == 1 and counts[25] >= 20800
        assert counts[21] == counts[22] == 0
        assert counts[2] == counts[3] + (3 * 320 << 20) + counts[23]
        assert counts[23] >= 3 * 64 << 20
        assert counts[11] == counts[6] * 5 * 4**3 + 60 * (4**4 + 3 * 4**3)
        assert tuple(counts[17:20]) == ((32, 32, 32) if mixed else (64, 64, 64))
        assert values[2] > 0
        timings.append(float(values[2]))
        work.append(counts.tolist())
    (tmp_path / "complete_endpoint.json").write_text(
        json.dumps(
            {
                "precision": "mixed" if mixed else "strict",
                "seconds": timings,
                "work": work,
                "scope": "whole endpoint with per-call preparation and cleanup",
                "selection": "synthetic test-only scores, no speedup claim",
            },
            indent=2,
        )
        + "\n"
    )
    status, expected, generated, error = run(
        native_probe, inputs, mixed=mixed, generated=True, panels=1
    )
    assert status == 0, error
    # The optional provider's simultaneous reservations do not fit. Retain the
    # same precision via generated execution, without preparing any library.
    status, values, bounded, error = run(
        native_probe,
        inputs,
        mixed=mixed,
        cutensor=True,
        panels=1,
        budget=int(generated[2]),
    )
    assert status == 0, error
    assert bounded[20] == bounded[22] == 1 and bounded[24] == 0
    assert bounded[2] <= generated[2]
    np.testing.assert_array_equal(values[:2], expected[:2])
    # Reject the third plan after the panel and first W plan have been prepared.
    # The partial W table and the already published panel table both drain before
    # the generated retry; no method work has been enqueued yet.
    status, values, fallback, error = run(
        native_probe,
        inputs,
        mixed=mixed,
        cutensor=True,
        reject_cutensor=True,
        budget=2 << 30,
        panels=1,
    )
    assert status == 0, error
    assert fallback[20] == fallback[22] == 1 and fallback[24] == 0
    np.testing.assert_array_equal(values[:2], expected[:2])
    np.testing.assert_array_equal(fallback[6:20], generated[6:20])


@pytest.mark.parametrize("mixed", [False, True])
def test_generated_provider_matches_independent_triples_endpoint(
    native_probe: typing.Any, mixed: bool
) -> None:
    inputs, ovvv = case(3, 4, 5)
    status, values, counts, error = run(
        native_probe, inputs, mixed=mixed, generated=True
    )
    if status and "test hook unavailable" in error:
        pytest.skip("requires library built with test hooks")
    assert status == 0, error
    np.testing.assert_allclose(
        values[0],
        reference(inputs, ovvv),
        atol=2e-7 if mixed else 3e-12,
        rtol=2e-4 if mixed else 3e-12,
    )
    assert counts[20] == counts[21] == 1
    assert tuple(counts[17:20]) == ((32, 32, 32) if mixed else (64, 64, 64))


def test_mixed_storage_budget_retains_strict_candidate(
    native_probe: typing.Any,
) -> None:
    inputs, _ = case(3, 9, 5)
    status, expected, strict, error = run(
        native_probe, inputs, panels=1, generated=True
    )
    if status and "test hook unavailable" in error:
        pytest.skip("requires library built with test hooks")
    assert status == 0, error
    status, values, actual, error = run(
        native_probe, inputs, panels=1, mixed=True, budget=int(strict[2])
    )
    assert status == 0, error
    assert actual[20] == actual[22] == 1
    assert actual[2] <= strict[2]
    assert tuple(actual[17:20]) == (64, 64, 64)
    np.testing.assert_array_equal(values[:2], expected[:2])
