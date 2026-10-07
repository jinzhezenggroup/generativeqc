"""Independent full-Fock gates for the occupied-domain DF resolvent."""

from __future__ import annotations

import ctypes as ct
import itertools
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import PERMUTATIONS
from generativeqc_compiler.cc.occupied_triples_fock import (
    moment_program,
    resolvent_scalar_program,
)
from generativeqc_compiler.cc.triples import _LABELS, VP, triples_energy
from generativeqc_compiler.cc.triples_fock_response import (
    build_runtime_triples_resolvent_program,
)
from generativeqc_compiler.tensor import execute
from test_cc_triples_fock_response import NAMES, _canonical_energy, _dense_reference
from test_df_occupied_triples import case


def conventional(inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Independent test-only materialization for the original triples oracle."""
    return [np.einsum("Qia,Qfb->iafb", *inputs[:2]), *inputs[2:7]]


def vectors(inputs: list[np.ndarray], i: int, j: int, k: int) -> dict[str, np.ndarray]:
    """Tiny host traversal; no production fallback uses these contractions."""
    bov, bvv, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    v = len(ev)
    moments = []
    for perm in PERMUTATIONS:
        I, J, K = ((i, j, k)[x] for x in perm)
        panel = np.einsum("Qa,Qbf->abf", bov[:, I], bvv)
        w = np.einsum("abf,cf->abc", panel, t2[K, J])
        w -= np.einsum("am,mbc->abc", ovoo[I, :, J], t2[:, K])
        vv = np.einsum("ab,c->abc", ovov[I, :, J], t1[K])
        vv += np.einsum("ab,c->abc", t2[I, J], fov[K])
        moments.append((w, vv))
    program = resolvent_scalar_program()
    result = {name: np.empty((v, v, v)) for name in ("x", "y")}
    for abc in itertools.product(range(v), repeat=3):
        feed = {"gap": np.array(eo[i] + eo[j] + eo[k] - sum(ev[a] for a in abc))}
        for occ, (w, vv) in zip(_LABELS, moments, strict=True):
            for vir in _LABELS:
                address = tuple(abc[x] for x in VP[vir])
                feed[f"w_{occ}_{vir}"] = np.array(w[address])
                feed[f"v_{occ}_{vir}"] = np.array(vv[address])
        values = execute(program, feed).outputs
        for name, output in result.items():
            output[abc] = values[name]
    return result


def paged(inputs: list[np.ndarray], capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """Two live occupied pages, with zero padding and explicit mirrored scatter."""
    o, v = len(inputs[-2]), len(inputs[-1])
    oo, vv = (moment_program(v, capacity, block) for block in ("oo", "vv"))
    bo, bv = np.zeros((o, o)), np.zeros((v, v))
    for j in range(o):
        for k in range(j + 1):
            weight = 1 if j == k else 2

            def page(start: int, j: int = j, k: int = k) -> dict[str, np.ndarray]:
                out = {name: np.zeros((capacity, v, v, v)) for name in ("x", "y")}
                for index, i in enumerate(range(start, min(start + capacity, o))):
                    for name, value in vectors(inputs, i, j, k).items():
                        out[name][index] = value
                return out

            for start in range(0, o, capacity):
                left = page(start)
                count = min(capacity, o - start)
                for i in range(count):
                    products = execute(
                        vv, {name + "_left": value[i] for name, value in left.items()}
                    ).outputs
                    bv += 0.5 * weight * sum(products.values())
                for other in range(start, o, capacity):
                    right = left if other == start else page(other)
                    other_count = min(capacity, o - other)
                    feed = {name + "_left": value for name, value in left.items()}
                    feed.update(
                        {name + "_right": value for name, value in right.items()}
                    )
                    products = execute(oo, feed).outputs
                    block = -0.5 * weight * sum(products.values())[:count, :other_count]
                    bo[start : start + count, other : other + other_count] += block
                    if start != other:
                        bo[other : other + other_count, start : start + count] += (
                            block.T
                        )
    return bo, bv


@pytest.mark.parametrize("o,v,q", [(1, 2, 2), (2, 3, 4), (3, 2, 3)])
def test_occupied_vectors_match_independent_virtual_page_storage(
    o: int, v: int, q: int
) -> None:
    inputs, _ = case(o, v, q)
    program = build_runtime_triples_resolvent_program(o, v, capacity=1)
    outputs = {
        (i, j, k): vectors(inputs, i, j, k)
        for i, j, k in itertools.product(range(o), repeat=3)
    }
    for abc in itertools.product(range(v), repeat=3):
        feed = dict(
            zip(NAMES, conventional(inputs), strict=True),
            eps_o=inputs[-2],
            eps_v=inputs[-1],
            active=np.ones(1),
        )
        feed.update(
            {
                f"{name}_map": np.array([index], dtype=np.int64)
                for name, index in zip("abc", abc, strict=True)
            }
        )
        actual = execute(program, feed).outputs
        for occupied, values in outputs.items():
            for name in ("x", "y"):
                np.testing.assert_allclose(
                    values[name][abc],
                    actual[name][(0, *occupied)],
                    atol=2e-15,
                    rtol=2e-13,
                )


@pytest.mark.parametrize("capacity", (1, 2, 3))
def test_paged_moments_match_full_inverse_and_recanonicalized_energy(
    capacity: int,
) -> None:
    inputs, _ = case(3, 2, 3)
    inputs[-2][:2] = -1.0
    inputs[-1][:] = 0.3
    actual = paged(inputs, capacity)
    arrays = conventional(inputs)
    reference = _dense_reference(arrays, *inputs[-2:])[:2]
    for observed, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-13, rtol=2e-11)
    rng = np.random.default_rng(1799)
    directions = [rng.normal(size=x.shape) for x in actual]
    directions = [(x + x.T) / 2 for x in directions]
    derivative = sum(
        float(np.sum(x * d)) for x, d in zip(actual, directions, strict=True)
    )
    assert abs(actual[0][0, 1]) > 1e-10
    for step in (1e-4, 3e-5):
        energies = [
            _canonical_energy(
                arrays,
                np.diag(inputs[-2]) + sign * step * directions[0],
                np.diag(inputs[-1]) + sign * step * directions[1],
            )
            for sign in (-1, 1)
        ]
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), derivative, atol=1e-10, rtol=2e-6
        )


@pytest.fixture(scope="module")
def native_fock_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_TRIPLES_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-triples-fock-native")
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
            "-I" + str(root / "src"),
            "-c",
            str(root / "tests/native/df_triples_fock_probe.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
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
        timeout=60,
    )
    dll = ct.CDLL(str(output))
    call = dll.df_triples_fock_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = (
        [ct.c_size_t] * 3
        + [ct.POINTER(dp), ct.c_double]
        + [ct.c_size_t] * 4
        + [ct.POINTER(dp), dp, ct.POINTER(ct.c_size_t), ct.c_void_p, ct.c_size_t]
    )
    call.restype = ct.c_int
    return call


@pytest.fixture(scope="module")
def native_combined_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_TRIPLES_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-triples-combined-native")
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
            "-I" + str(root / "src"),
            "-c",
            str(root / "tests/native/df_triples_fock_probe.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
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
        timeout=60,
    )
    dll = ct.CDLL(str(output))
    call = dll.df_triples_combined_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = (
        [ct.c_size_t] * 3
        + [ct.POINTER(dp), ct.c_double]
        + [ct.c_size_t] * 4
        + [ct.POINTER(dp), dp, ct.POINTER(ct.c_size_t), ct.c_void_p, ct.c_size_t]
    )
    call.restype = ct.c_int
    return call


def run_combined(
    call: typing.Any,
    inputs: list[np.ndarray],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
    rows: int = 0,
    panels: int = 3,
    threshold: float = 1e-10,
) -> tuple:
    q, o, v = inputs[0].shape
    arrays = [np.ascontiguousarray(x) for x in inputs]
    output = [
        np.full((o, o), np.nan),
        np.full((v, v), np.nan),
        *(np.full_like(x, np.nan) for x in arrays[:7]),
    ]
    values = np.full(3, np.nan)
    counts = np.full(14, 19, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    dp = ct.POINTER(ct.c_double)
    status = call(
        o,
        v,
        q,
        (dp * 9)(*(x.ctypes.data_as(dp) for x in arrays)),
        threshold,
        budget,
        caller_bytes,
        rows,
        panels,
        (dp * 2)(*(x.ctypes.data_as(dp) for x in output)),
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, output, values, counts, error.value.decode()


def run_native(
    call: typing.Any,
    inputs: list[np.ndarray],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
    rows: int = 0,
    panels: int = 3,
    threshold: float = 1e-10,
) -> tuple:
    q, o, v = inputs[0].shape
    arrays = [np.ascontiguousarray(x) for x in inputs]
    output = [np.full((o, o), np.nan), np.full((v, v), np.nan)]
    values = np.full(2, np.nan)
    counts = np.full(21, 19, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    dp = ct.POINTER(ct.c_double)
    status = call(
        o,
        v,
        q,
        (dp * 9)(*(x.ctypes.data_as(dp) for x in arrays)),
        threshold,
        budget,
        caller_bytes,
        rows,
        panels,
        (dp * 2)(*(x.ctypes.data_as(dp) for x in output)),
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, output, values, counts, error.value.decode()


def test_combined_response_reuses_one_input_upload(
    native_fock_probe: typing.Any, native_combined_probe: typing.Any
) -> None:
    from test_df_occupied_triples_response import reverse

    inputs, _ = case(2, 3, 4)
    caller = 12345
    status, combined, values, counts, error = run_combined(
        native_combined_probe, inputs, caller_bytes=caller
    )
    assert status == 0, error
    standalone_status, standalone, _, standalone_counts, standalone_error = run_native(
        native_fock_probe, inputs, caller_bytes=caller
    )
    assert standalone_status == 0, standalone_error
    for actual, expected in zip(combined[:2], standalone, strict=True):
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(combined[2:], reverse(inputs)[:7], strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)

    o, v = inputs[5].shape
    ovvv = np.einsum("Qia,Qfb->iafb", inputs[0], inputs[1])
    expected_energy = triples_energy(
        o,
        v,
        ovvv,
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
    )
    np.testing.assert_allclose(values[0], expected_energy, atol=2e-12, rtol=0)
    assert values[1] > 0 and values[2] > 0

    input_bytes = sum(x.nbytes for x in inputs)
    assert counts[1] == counts[2] == input_bytes
    assert counts[3] == 0
    assert input_bytes <= counts[0] < input_bytes + 10 * 256
    assert counts[6] == counts[7] == input_bytes
    assert counts[4] == counts[5] <= 1 << 30
    assert counts[4] >= standalone_counts[0]

    tiles = o * (o + 1) * (o + 2) // 6
    pairs = o * (o + 1) // 2
    fock_cubes = pairs * o
    assert counts[8] == counts[12] == tiles
    assert counts[9] == 12 * tiles
    assert counts[10] == fock_cubes
    assert counts[11] == 12 * fock_cubes
    # Pullback no longer owns another forward W traversal.
    assert counts[13] == 0
    assert counts[11] + counts[9] == 12 * (fock_cubes + tiles)
    assert counts[11] < 12 * (fock_cubes + tiles)

    status, minimal, _, minimal_counts, error = run_combined(
        native_combined_probe, inputs, caller_bytes=caller, rows=1, panels=1
    )
    assert status == 0, error
    for actual, expected in zip(minimal, combined, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    exact_budget = int(max(minimal_counts[4], minimal_counts[5]))

    status, exact, _, exact_counts, error = run_combined(
        native_combined_probe, inputs, budget=exact_budget, caller_bytes=caller
    )
    assert status == 0, error
    for actual, expected in zip(exact, minimal, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    assert max(exact_counts[4], exact_counts[5]) == exact_budget

    status, refused, refused_values, refused_counts, error = run_combined(
        native_combined_probe, inputs, budget=exact_budget - 1, caller_bytes=caller
    )
    assert status != 0 and "budget" in error
    assert all(np.isnan(x).all() for x in (*refused, refused_values))
    np.testing.assert_array_equal(refused_counts, 19)


@pytest.mark.parametrize(
    "o,v,q,rows,panels",
    [
        (1, 3, 2, 0, 3),
        (2, 3, 4, 0, 3),
        (3, 2, 1, 2, 1),
        (3, 5, 7, 1, 3),
        (4, 3, 2, 2, 3),
        (3, 3, 2, 5, 2),
    ],
)
def test_native_full_fock_and_complete_replay_work(
    native_fock_probe: typing.Any, o: int, v: int, q: int, rows: int, panels: int
) -> None:
    inputs, _ = case(o, v, q)
    if o > 1:
        inputs[-2][:2] = inputs[-2][0]
    inputs[-1][:2] = inputs[-1][0]
    caller = 12345
    status, outputs, values, counts, error = run_native(
        native_fock_probe, inputs, rows=rows, panels=panels, caller_bytes=caller
    )
    assert status == 0, error
    for actual, expected in zip(
        outputs, _dense_reference(conventional(inputs), *inputs[-2:])[:2], strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-11)
        np.testing.assert_allclose(actual, actual.T, atol=2e-15, rtol=0)
    assert values[0] > 0 and values[1] > 0
    c = min(rows, o) if rows else o
    pages = (o + c - 1) // c
    pairs = o * (o + 1) // 2
    builds = pairs * pages * (pages + 1) // 2
    cubes = pairs * sum((index + 1) * min(c, o - index * c) for index in range(pages))
    assert tuple(counts[5:12]) == (
        c,
        pages,
        min(o, panels),
        pairs,
        pairs * o,
        cubes,
        builds,
    )
    assert counts[13] == 12 * cubes
    assert counts[14] == 2 * (pairs * o + builds)
    summands = (
        int(counts[12]) * q * v**3
        + 6 * cubes * (v**4 + o * v**3)
        + 2 * pairs * o * v**4
        + 2 * builds * c * c * v**3
    )
    assert counts[15] == summands
    assert counts[16] == cubes * v**3
    assert counts[17] == sum(counts[12:15])
    assert counts[18] == builds
    assert counts[19] == counts[1] == sum(x.nbytes for x in inputs)
    assert counts[20] == sum(x.nbytes for x in outputs) + ct.sizeof(ct.c_int)
    assert counts[0] == caller + counts[1] + counts[2] + sum(x.nbytes for x in outputs)
    assert counts[2] == counts[3] + (96 << 20)
    assert counts[4] <= 96 << 20


@pytest.mark.parametrize("splitting", (0.0, 1e-12, 1e-8, 0.2))
def test_native_recanonicalized_fock_differences(
    native_fock_probe: typing.Any, splitting: float
) -> None:
    inputs, _ = case(3, 3, 4)
    inputs[-2][1] = inputs[-2][0] + splitting
    inputs[-1][1] = inputs[-1][0] + splitting
    status, output, _, _, error = run_native(native_fock_probe, inputs, rows=2)
    assert status == 0, error
    rng = np.random.default_rng(1799)
    directions = [rng.normal(scale=0.1, size=x.shape) for x in output]
    directions = [(x + x.T) / 2 for x in directions]
    derivative = sum(
        float(np.sum(x * d)) for x, d in zip(output, directions, strict=True)
    )
    assert abs(output[0][0, 1]) > 1e-9
    for step in (1e-4, 3e-5):
        energies = [
            _canonical_energy(
                conventional(inputs),
                np.diag(inputs[-2]) + sign * step * directions[0],
                np.diag(inputs[-1]) + sign * step * directions[1],
            )
            for sign in (-1, 1)
        ]
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), derivative, atol=2e-12, rtol=3e-7
        )


def test_native_internal_degenerate_covariance(native_fock_probe: typing.Any) -> None:
    from test_cc_triples_fock_response import _rotate

    inputs, _ = case(3, 3, 4)
    inputs[-2][:2] = -1.0
    inputs[-1][:2] = 0.3
    status, output, _, _, error = run_native(native_fock_probe, inputs)
    assert status == 0, error
    uo, uv = np.eye(3), np.eye(3)
    for u, angle in ((uo, 0.4), (uv, -0.7)):
        u[:2, :2] = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    old = conventional(inputs)
    rotated = _rotate(old, uo, uv)
    new = [
        np.einsum("Qia,ij,ab->Qjb", inputs[0], uo, uv),
        np.einsum("Qab,ai,bj->Qij", inputs[1], uv, uv),
        *rotated[1:],
        *inputs[-2:],
    ]
    status, observed, _, _, error = run_native(native_fock_probe, new, rows=1, panels=1)
    assert status == 0, error
    for actual, bar, u in zip(observed, output, (uo, uv), strict=True):
        np.testing.assert_allclose(actual, u.T @ bar @ u, atol=2e-13, rtol=2e-11)


def test_native_budget_fallback_and_exact_complete_admission(
    native_fock_probe: typing.Any,
) -> None:
    inputs, _ = case(5, 7, 3)
    status, expected, _, counts, error = run_native(
        native_fock_probe, inputs, rows=1, panels=1, caller_bytes=256
    )
    assert status == 0, error
    budget = int(counts[0])
    status, actual, _, full, error = run_native(
        native_fock_probe, inputs, budget=budget, caller_bytes=256
    )
    assert status == 0, error
    assert full[5] == full[7] == 1
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    status, outputs, values, counts, error = run_native(
        native_fock_probe, inputs, budget=budget - 1, caller_bytes=256
    )
    assert status != 0 and "budget" in error
    assert all(np.isnan(x).all() for x in (*outputs, values))
    np.testing.assert_array_equal(counts, 19)


@pytest.mark.parametrize(
    "fault", ("nan", "asymmetric", "gap", "overflow", "intermediate", "threshold")
)
def test_native_failure_does_not_publish_partial_response(
    native_fock_probe: typing.Any, fault: str
) -> None:
    inputs, _ = case(2, 3, 4)
    threshold = 1e-10
    if fault == "nan":
        inputs[0][0, 0, 0] = np.nan
    elif fault == "asymmetric":
        inputs[1][0, 0, 1] += 0.1
    elif fault == "gap":
        inputs[-1][0] = inputs[-2].max()
    elif fault == "overflow":
        inputs[0] *= 1e200
        inputs[1] *= 1e200
    elif fault == "intermediate":
        # The final mathematical V can cancel; an overflowing first W GEMM
        # must remain rejected even if subsequent resolvent R3 cancels it.
        inputs[6] *= 1e308
        inputs[0] *= 1e100
    else:
        threshold = 0
    status, outputs, values, counts, _ = run_native(
        native_fock_probe, inputs, threshold=threshold
    )
    assert status != 0
    assert all(np.isnan(x).all() for x in (*outputs, values))
    np.testing.assert_array_equal(counts, 19)


@pytest.mark.parametrize("o,v,q", ((2, 2048, 1), (256, 1024, 4)))
def test_native_preflight_rejects_before_null_input_access(
    native_fock_probe: typing.Any, o: int, v: int, q: int
) -> None:
    dp = ct.POINTER(ct.c_double)
    nulls = (dp * 9)()
    values = np.full(2, np.nan)
    counts = np.full(21, 19, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = native_fock_probe(
        o,
        v,
        q,
        nulls,
        1e-10,
        np.iinfo(np.uintp).max,
        0,
        0,
        3,
        nulls,
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    assert status != 0 and any(
        word in error.value.decode() for word in ("indexing", "overflow")
    )
    assert np.isnan(values).all()
    np.testing.assert_array_equal(counts, 19)
