"""Independent gates for a generic streamed runtime-indexed reduction schedule."""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples_response import gap_vjp
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    TensorSpec,
    add,
    cast,
    input_tensor,
    multiply,
    reduce_sum,
)
from generativeqc_compiler.tensor.indexed_cuda_reduction import (
    IndexedReductionSchedule,
    emit_indexed_reduction_cuda,
    plan_indexed_reduction,
)
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.program import Program

from tools.generate_rccsd_native import _cuda_program, _dim, _required_function


def emission(schedule: IndexedReductionSchedule | None = None) -> tuple[str, str, str]:
    """Bind only shapes and buffers; the generated graph remains authoritative."""
    return emit_indexed_reduction_cuda(
        plan_indexed_reduction(gap_vjp(3), schedule),
        "probe_parallel",
        dimension=_dim,
        parameters=("o", "v"),
        input_bindings={"bar_gap": "state.bar_gap"},
        state_type="ProbeState",
        output_type="ProbeOutputs",
    )


def test_region_identity_preserves_equations_and_aliases() -> None:
    program = gap_vjp(3)
    original = program.logical_hash
    plan = plan_indexed_reduction(program)
    assert len(plan.roots) == 2
    assert [len(frontier) for frontier in plan.frontiers] == [1, 3]
    assert program.logical_hash == original == plan.program.logical_hash
    assert plan.identity == plan_indexed_reduction(gap_vjp(3)).identity
    assert (
        plan.identity
        != plan_indexed_reduction(
            gap_vjp(3), IndexedReductionSchedule(maximum_partials=128)
        ).identity
    )
    assert plan.to_payload()["producer_storage"] == "streamed"


@pytest.mark.parametrize(
    "options",
    [
        {"threads": 33},
        {"threads": 0},
        {"threads": True},
        {"maximum_partials": 0},
        {"maximum_partials": 65536},
        {"scalar_tile_elements": 0},
    ],
)
def test_invalid_schedule_is_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        IndexedReductionSchedule(**options)


def test_nonlinear_frontier_is_not_silently_reinterpreted() -> None:
    program = gap_vjp(3)
    root = program.outputs["bar_eps_i"]
    with pytest.raises(ValueError, match="compose reductions"):
        plan_indexed_reduction(Program({"square": multiply(root, root)}))


def test_fp32_is_not_silently_admitted() -> None:
    root = gap_vjp(3).outputs["bar_eps_i"]
    with pytest.raises(ValueError, match="FP64"):
        plan_indexed_reduction(Program({"narrow": cast(root, "float32")}))


@pytest.mark.parametrize("placement", ["producer", "output"])
@pytest.mark.parametrize(
    ("coefficient", "literal"),
    [
        (Fraction(1, 3), "0x1.5555555555555p-2"),
        (Fraction(9007199254740993, 9007199254740995), "0x1.ffffffffffffep-1"),
        (Fraction(10**400, 10**400 + 1), "0x1.0000000000000p+0"),
        (Fraction(-1, 10**400), "-0x0.0p+0"),
    ],
)
def test_rational_coefficients_match_tensorir_fp64_rounding(
    placement: str, coefficient: Fraction, literal: str
) -> None:
    space = IndexSpace("coefficient_probe", "virtual", 1)
    source = input_tensor(
        "seed", TensorSpec((Index("a", space), Index("b", space)), role="input")
    )
    scaled = add(source, coefficients=(coefficient,))
    root = (
        reduce_sum(scaled, (0,))
        if placement == "producer"
        else add(reduce_sum(source, (0,)), coefficients=(coefficient,))
    )
    program = Program({"result": root})
    expected = execute(program, {"seed": np.ones((1, 1))}).outputs["result"]
    assert expected[0] == float.fromhex(literal) == float(coefficient)
    emitted, _, _ = emit_indexed_reduction_cuda(
        plan_indexed_reduction(program),
        "coefficient_probe",
        dimension=_dim,
        parameters=("v",),
        input_bindings={"seed": "state.seed"},
        state_type="ProbeState",
        output_type="ProbeOutputs",
    )
    assert f"__dmul_rn({literal}," in emitted


def test_nonrepresentable_coefficient_is_rejected_during_emission() -> None:
    program = gap_vjp(3)
    root = add(program.outputs["bar_eps_v"], coefficients=(10**400,))
    with pytest.raises(ValueError, match="outside finite FP64"):
        emit_indexed_reduction_cuda(
            plan_indexed_reduction(Program({"result": root})),
            "coefficient_probe",
            dimension=_dim,
            parameters=("o", "v"),
            input_bindings={"bar_gap": "state.bar_gap"},
            state_type="ProbeState",
            output_type="ProbeOutputs",
        )


def test_parallel_emission_has_bounded_partials_and_no_cube_slots() -> None:
    source, helper, identity = emission()
    assert source.count("__global__ void probe_parallel_group_") == 2
    assert "__global__ void probe_parallel_drain" in source
    assert "cursor=std::max<std::size_t>" in source
    assert "gridDim.x" in source and "__dadd_rn" in source
    assert "__shfl_down_sync(0xffffffffu" in source
    assert "atomicAdd" not in source
    assert "double* slot" not in source
    assert "probe_parallel_partial_blocks(checked_product({v,v,v}))" in helper
    assert "std::min<std::size_t>(256,tiles)" in helper
    assert identity in helper


def test_df_consumer_keeps_serial_fallback_and_runtime_sized_arena() -> None:
    from tools.generate_df_occupied_triples import (
        response_cuda_source,
        response_scalar_header,
    )

    header = response_scalar_header()
    source = response_cuda_source()
    assert "gap_response_serial_arena_elements" in header
    assert "gap_response_parallel_arena_elements" in header
    assert "enabled && v>=4" in header
    assert "run_gap_response_parallel(s):run_gap_response(s)" in source
    assert "gap_response_node_2" in source


@pytest.fixture(scope="module")
def cuda_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    if os.environ.get("GENERATIVEQC_INDEXED_REDUCTION_CUDA_TEST") != "1":
        pytest.skip("requires an explicit finite Slurm GPU allocation")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires nvcc and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("indexed-reduction-cuda")
    source, helper, _ = emission()
    program = gap_vjp(3)
    serial = _cuda_program(
        program,
        "probe_serial",
        "ProbeOutputs",
        state_type="ProbeState",
        input_overrides={"bar_gap": "s.bar_gap"},
        output_fields=tuple(program.outputs),
        reset_error=False,
    )
    (directory / "generated_indexed_reduction_probe.cuh").write_text(
        "\n".join(
            [
                _required_function(program, "probe_serial_arena_elements"),
                helper,
                serial,
                source,
            ]
        )
    )
    library = directory / "probe.so"
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=native",
            "-shared",
            "-Xcompiler=-fPIC",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-I" + str(directory),
            str(root / "tests/native/indexed_reduction_probe.cu"),
            "-o",
            str(library),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    assert compiled.returncode == 0, compiled.stderr
    call = ct.CDLL(str(library)).indexed_reduction_probe
    pointer = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_size_t,
        pointer,
        ct.c_bool,
        pointer,
        pointer,
        ct.POINTER(ct.c_size_t),
        pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run_probe(call: Any, seed: np.ndarray, parallel: bool) -> tuple:
    seed = np.ascontiguousarray(seed, dtype=np.float64)
    occupied = np.full(3, np.nan)
    virtual = np.full(seed.shape[0], np.nan)
    counts = np.zeros(6, dtype=np.uintp)
    seconds = np.array([np.nan])
    error = ct.create_string_buffer(2048)
    pointer = ct.POINTER(ct.c_double)
    status = call(
        seed.shape[0],
        seed.ctypes.data_as(pointer),
        parallel,
        occupied.ctypes.data_as(pointer),
        virtual.ctypes.data_as(pointer),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        seconds.ctypes.data_as(pointer),
        error,
        len(error),
    )
    return status, occupied, virtual, counts, seconds[0], error.value.decode()


@pytest.mark.parametrize("virtuals", [1, 3, 4, 7, 17, 25, 58, 221])
def test_actual_emission_matches_independent_signed_gap_seeds(
    cuda_probe: Any, virtuals: int
) -> None:
    rng = np.random.default_rng(1763 + virtuals)
    seed = rng.normal(scale=1e-3, size=(virtuals,) * 3)
    scalar = math.fsum(seed.ravel())
    vector = -(seed.sum(axis=(0, 1)) + seed.sum(axis=(0, 2)) + seed.sum(axis=(1, 2)))
    tolerance = 64 * np.finfo(np.float64).eps * np.abs(seed).sum()
    receipts = []
    for parallel in (False, True):
        status, occupied, actual, counts, seconds, error = run_probe(
            cuda_probe, seed, parallel
        )
        assert status == 0, error
        np.testing.assert_allclose(occupied, scalar, atol=tolerance, rtol=2e-13)
        np.testing.assert_allclose(actual, vector, atol=tolerance, rtol=2e-13)
        assert seconds >= 0
        receipts.append(
            {
                "parallel": parallel,
                "device_seconds": seconds,
                "workspace_bytes": int(counts[0]),
                "kernels": int(counts[1]),
                "materialized_elements": int(counts[2]),
                "value_reads": int(counts[3]),
                "value_writes": int(counts[4]),
                "reduction_summands": int(counts[5]),
                "scalar_error": float(np.max(np.abs(occupied - scalar))),
                "vector_error": float(np.max(np.abs(actual - vector))),
            }
        )
        if parallel:
            partials = min(256, math.ceil(virtuals**3 / 4096))
            temporary = partials if partials > 1 else 0
            assert counts[0] == 8 * (temporary + 1 + virtuals)
            assert counts[1] == 2 + (partials > 1)
            assert counts[2] == temporary
            assert counts[3] == 4 * virtuals**3 + temporary
            assert counts[4] == 1 + virtuals + temporary
            assert counts[5] == 4 * virtuals**3 + temporary
        else:
            assert counts[1] == 8
    if destination := os.environ.get("GENERATIVEQC_INDEXED_REDUCTION_RECEIPT_DIR"):
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"virtuals-{virtuals}.json").write_text(
            json.dumps({"virtuals": virtuals, "receipts": receipts}, indent=2) + "\n"
        )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_seed_never_publishes_outputs(
    cuda_probe: Any, parallel: bool, bad: float
) -> None:
    seed = np.zeros((17,) * 3)
    seed[7, 3, 5] = bad
    status, occupied, vector, _, _, error = run_probe(cuda_probe, seed, parallel)
    assert status and "nonfinite" in error
    assert np.isnan(occupied).all() and np.isnan(vector).all()


@pytest.mark.parametrize("parallel", [False, True])
def test_finite_partials_cannot_hide_scalar_overflow(
    cuda_probe: Any, parallel: bool
) -> None:
    seed = np.full((58,) * 3, 1e304)
    status, occupied, vector, _, _, error = run_probe(cuda_probe, seed, parallel)
    assert status and "nonfinite" in error
    assert np.isnan(occupied).all() and np.isnan(vector).all()


def test_fixed_tree_is_repeatable(cuda_probe: Any) -> None:
    seed = np.random.default_rng(1763).normal(scale=1e-3, size=(25,) * 3)
    first = run_probe(cuda_probe, seed, True)
    second = run_probe(cuda_probe, seed, True)
    assert first[0] == second[0] == 0
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
