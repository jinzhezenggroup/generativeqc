"""Source-bound opt-in (T) endpoint A/B; retain negative and missing evidence.

Run inside an explicitly allocated CUDA Job. CUPTI API callbacks observe launch
symbols and allocation high-water marks without changing scientific kernels.
Timing samples run with callbacks disabled. This does not measure SM instruction
counts or the complete preceding SCF/CCSD solve.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import dataclasses
import hashlib
import json
import platform
import statistics
import subprocess
import threading
import time
import typing
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from generativeqc_compiler.cc.triples import triples_energy
from generativeqc_compiler.cc.triples_tiles import (
    TriplesTileEnumerator,
    build_runtime_tile_triples_program,
    runtime_tile_control_batches,
    runtime_tile_static_feeds,
)
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_resources import parse_resources
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from generativeqc_compiler.tensor.cuda_resident import (
    PreparedResident,
    compile_resident,
)
from generativeqc_compiler.tensor.cuda_search import (
    estimate_schedule,
    execution_key,
    require_compiled_resources,
)
from generativeqc_compiler.tensor.interpreter import execute


class CallbackData(ct.Structure):
    """Prefix of CUpti_CallbackData, verified against cupti_callbacks.h."""

    _fields_ = [
        ("site", ct.c_int),
        ("function", ct.c_char_p),
        ("params", ct.c_void_p),
        ("result", ct.c_void_p),
        ("symbol", ct.c_char_p),
    ]


class AllocationParams(ct.Structure):
    _fields_ = [("pointer", ct.POINTER(ct.c_void_p)), ("bytes", ct.c_size_t)]


class CuptiObserver:
    """Observe successful runtime launches and runtime/driver allocations."""

    def __init__(self) -> None:
        import nvidia.cuda_cupti

        root = Path(next(iter(nvidia.cuda_cupti.__path__)))
        self.path = root / "lib/libcupti.so.12"
        self.header_sha = digest(root / "include/cupti_callbacks.h")
        self.library = ct.CDLL(str(self.path))
        self.subscriber = ct.c_void_p()
        self.allocations: dict[int, int] = {}
        self.peak = 0
        self.launches: Counter = Counter()
        self.driver_launches: Counter = Counter()
        self.runtime_launch_depth: Counter = Counter()
        self.failures: list[str] = []
        callback_type = ct.CFUNCTYPE(
            None, ct.c_void_p, ct.c_int, ct.c_uint32, ct.c_void_p
        )
        self.callback = callback_type(self._observe)
        self.library.cuptiSubscribe.argtypes = [
            ct.POINTER(ct.c_void_p),
            callback_type,
            ct.c_void_p,
        ]
        self.library.cuptiEnableDomain.argtypes = [ct.c_uint32, ct.c_void_p, ct.c_int]
        self.library.cuptiUnsubscribe.argtypes = [ct.c_void_p]
        self.check(
            self.library.cuptiSubscribe(ct.byref(self.subscriber), self.callback, None)
        )

    @staticmethod
    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUPTI returned {status}")

    def _observe(self, _userdata: typing.Any, domain: int, _id: int, data: int) -> None:
        try:
            info = ct.cast(data, ct.POINTER(CallbackData)).contents
            name = info.function.decode()
            runtime_launch = domain == 2 and name.startswith("cudaLaunchKernel")
            thread = threading.get_ident()
            if runtime_launch:
                self.runtime_launch_depth[thread] += 1 if info.site == 0 else -1
            if (
                info.site != 1
                or not info.result
                or ct.cast(info.result, ct.POINTER(ct.c_int))[0]
            ):
                return
            if runtime_launch:
                self.launches[(info.symbol or b"unknown").decode()] += 1
            elif (
                domain == 1
                and name.startswith("cuLaunchKernel")
                and not self.runtime_launch_depth[thread]
            ):
                self.driver_launches[(info.symbol or b"unknown").decode()] += 1
            allocate = name in (
                "cudaMalloc",
                "cudaMallocAsync",
                "cudaMallocManaged",
                "cuMemAlloc",
                "cuMemAlloc_v2",
                "cuMemAllocAsync",
                "cuMemAllocManaged",
            )
            free = name in (
                "cudaFree",
                "cudaFreeAsync",
                "cuMemFree",
                "cuMemFree_v2",
                "cuMemFreeAsync",
            )
            if allocate:
                params = ct.cast(info.params, ct.POINTER(AllocationParams)).contents
                pointer = params.pointer[0]
                self.allocations[pointer] = int(params.bytes)
                self.peak = max(self.peak, sum(self.allocations.values()))
            elif free:
                pointer = ct.cast(info.params, ct.POINTER(ct.c_void_p))[0]
                self.allocations.pop(pointer, None)
        except (ValueError, TypeError, AttributeError, UnicodeError) as error:
            self.failures.append(repr(error))

    def enable(self, enabled: bool) -> None:
        for domain in (1, 2):
            self.check(
                self.library.cuptiEnableDomain(int(enabled), self.subscriber, domain)
            )

    def close(self) -> None:
        self.check(self.library.cuptiUnsubscribe(self.subscriber))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(nocc: int, nvir: int) -> dict:
    rng = np.random.default_rng(783 + nocc * 100 + nvir)
    arrays = {
        name: rng.normal(size=shape)
        for name, shape in (
            ("ovvv", (nocc, nvir, nvir, nvir)),
            ("ovoo", (nocc, nvir, nocc, nocc)),
            ("ovov", (nocc, nvir, nocc, nvir)),
            ("fov", (nocc, nvir)),
            ("t1", (nocc, nvir)),
            ("t2", (nocc, nocc, nvir, nvir)),
        )
    }
    arrays["t2"] = (arrays["t2"] + arrays["t2"].transpose(1, 0, 3, 2)) / 2
    arrays["eps_o"] = np.linspace(-1.0, -0.5, nocc)
    arrays["eps_v"] = np.linspace(0.5, 1.5, nvir)
    return arrays


def counted_scalar_execution(
    plan: typing.Any, compiler: typing.Any, cache: Path, feeds: dict
) -> dict:
    """One separate instrumented artifact, excluded from endpoint timings.

    Count entries to the original checked virtual scalar bodies, including the
    bodies inlined around shared caches. The production emitter files and the
    running production artifact are unchanged. Retain both generated hashes.
    This is diagnostic emission substitution, not platform/JIT compatibility.
    """
    from generativeqc_compiler.tensor import cuda_emit, cuda_execute, cuda_resident_emit

    original_value, original_emit = cuda_emit._value, cuda_emit.emit_cuda
    emitted = Counter()

    def counted_value(
        plan: typing.Any, index: int, *args: typing.Any, **kwargs: typing.Any
    ) -> str:
        body = original_value(plan, index, *args, **kwargs)
        if plan.steps[index].virtual:
            emitted[index] += 1
            return f"atomicAdd(&qualification_counts[{index}], 1ULL); " + body
        return body

    def counted_emit(plan: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> str:
        with patch.object(cuda_emit, "_value", counted_value):
            source = original_emit(plan, *args, **kwargs)
        declaration = f"__device__ unsigned long long qualification_counts[{len(plan.steps)}] = {{0}};\n"
        functions = """
extern "C" int qualification_reset() {
    void* pointer = nullptr;
    cudaError_t status = cudaGetSymbolAddress(&pointer, qualification_counts);
    if (status != cudaSuccess) return int(status);
    return int(cudaMemset(pointer, 0, sizeof(qualification_counts)));
}
extern "C" int qualification_read(unsigned long long* output) {
    return int(cudaMemcpyFromSymbol(output, qualification_counts, sizeof(qualification_counts)));
}
"""
        return declaration + source + functions

    with (
        patch.object(cuda_execute, "emit_cuda", counted_emit),
        patch.object(cuda_resident_emit, "emit_cuda", counted_emit),
    ):
        artifact = compile_resident(plan, compiler, cache)
    counters = np.zeros(len(plan.steps), dtype=np.uint64)
    expected = execute(plan.program, feeds).outputs
    with PreparedResident(plan, artifact) as counted:
        reset, read = (
            counted._library.qualification_reset,
            counted._library.qualification_read,
        )
        reset.argtypes, reset.restype = [], ct.c_int
        read.argtypes, read.restype = [ct.c_void_p], ct.c_int
        assert reset() == 0
        counted.upload(feeds)
        leases, _ = counted.run()
        actual = {name: counted.download(lease) for name, lease in leases.items()}
        assert read(counters.ctypes.data) == 0
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, rtol=2e-11, atol=2e-11)
    return {
        "kind": "separate instrumented virtual-scalar body entries; not timing or SM instructions",
        "generated_artifact": artifact.metadata,
        "compiled_files": {
            str(path): digest(path)
            for path in artifact.library.parent.iterdir()
            if path.is_file()
        },
        "scalar_entries_by_step": {
            str(index): int(counters[index])
            for index, step in enumerate(plan.steps)
            if step.virtual
        },
        "total_virtual_scalar_entries": int(counters.sum()),
        "rematerialized_scalar_entries": sum(
            max(0, int(counters[index]) - step.node.spec.size)
            for index, step in enumerate(plan.steps)
            if step.virtual
        ),
        "emitted_scalar_body_occurrences": dict(emitted),
        "oracle_outputs": {name: float(value) for name, value in expected.items()},
    }


def qualify(
    nocc: int,
    nvir: int,
    enabled: bool,
    args: argparse.Namespace,
    mode: str | None = None,
) -> dict:
    arrays = inputs(nocc, nvir)
    oracle = float(triples_energy(nocc, nvir, **arrays))
    compiler = CudaCompilerAdapter(args.nvcc, cuda_target_info(args.architecture))
    start = time.perf_counter()
    program = build_runtime_tile_triples_program(nocc, nvir, capacity=args.capacity)
    mode = mode or ("shared" if enabled else "unfused")
    schedule = (
        TensorSchedule()
        if mode == "default"
        else TensorSchedule(
            stream_reductions=True,
            streamed_gemm_reduction=True,
            shared_streamed_reductions=enabled,
        )
    )
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=schedule,
    )
    plan_seconds = time.perf_counter() - start
    estimates = estimate_schedule(plan)
    start = time.perf_counter()
    artifact = compile_resident(plan, compiler, args.cache)
    compile_seconds = time.perf_counter() - start
    compiled_resources = [
        dataclasses.asdict(row)
        for row in parse_resources(
            (artifact.library.parent / "compiler.log").read_text()
        )
    ]
    require_compiled_resources(plan, compiled_resources)
    controls = [
        page
        for tile in TriplesTileEnumerator(nocc, nvir, vir_chunk_size=3)
        for page in runtime_tile_control_batches(tile, args.capacity)
    ]
    static = runtime_tile_static_feeds(arrays)

    def lifecycle() -> dict:
        # Include per-call input/control preparation, owner creation, all
        # transfers, executions, output publication and owner cleanup. Planning
        # and compilation are recorded separately as preparation of one reusable
        # specialization. No reference work is timed as production execution.
        beginning = time.perf_counter()
        pages = [
            page
            for tile in TriplesTileEnumerator(nocc, nvir, vir_chunk_size=3)
            for page in runtime_tile_control_batches(tile, args.capacity)
        ]
        values = runtime_tile_static_feeds(arrays)
        total = 0.0
        with PreparedResident(plan, artifact) as current:
            current.upload(values)
            for page in pages:
                current.upload(page)
                leases, _ = current.run()
                total += float(current.download(leases["triples_energy"]))
        elapsed = time.perf_counter() - beginning
        np.testing.assert_allclose(total, oracle, rtol=2e-11, atol=2e-11)
        return {
            "full_lifecycle_s": elapsed,
            "energy": total,
            "runtime_batches": len(pages),
        }

    full_lifecycle_samples = [lifecycle() for _ in range(args.repeats)]
    observer = CuptiObserver()
    observer.enable(True)
    start = time.perf_counter()
    owner = PreparedResident(plan, artifact)
    prepare_seconds = time.perf_counter() - start
    observer.enable(False)

    def endpoint() -> dict:
        timing = {"upload_s": 0.0, "run_s": 0.0, "download_s": 0.0}
        total = 0.0
        beginning = time.perf_counter()
        start = time.perf_counter()
        owner.upload(static)
        timing["upload_s"] += time.perf_counter() - start
        for page in controls:
            start = time.perf_counter()
            owner.upload(page)
            timing["upload_s"] += time.perf_counter() - start
            start = time.perf_counter()
            leases, _metrics = owner.run()
            timing["run_s"] += time.perf_counter() - start
            start = time.perf_counter()
            total += float(owner.download(leases["triples_energy"]))
            timing["download_s"] += time.perf_counter() - start
        timing["endpoint_s"] = time.perf_counter() - beginning
        timing["energy"] = total
        np.testing.assert_allclose(total, oracle, rtol=2e-11, atol=2e-11)
        return timing

    try:
        warmup = endpoint()
        samples = [endpoint() for _ in range(args.repeats)]
        observer.enable(True)
        observed = endpoint()
        observer.enable(False)
        # Invalid controls must not publish successful output; recovery reuses
        # this exact resident owner and compiled artifact.
        invalid = {name: value.copy() for name, value in controls[0].items()}
        invalid["a_map"][0] = nvir
        owner.upload(invalid)
        try:
            owner.run()
        except (RuntimeError, ValueError) as error:
            invalid_error = str(error)
        else:
            raise AssertionError("invalid runtime map was accepted")
        recovered = endpoint()
        device = owner.device
        transfers = dict(owner.transfers)
    finally:
        observer.enable(True)
        start = time.perf_counter()
        owner.close()
        cleanup_seconds = time.perf_counter() - start
        observer.enable(False)
        observer.close()
    assert not observer.failures, observer.failures
    assert not observer.allocations, "observed CUDA allocations survive owner cleanup"
    scalar_counts = (
        counted_scalar_execution(plan, compiler, args.cache, {**static, **controls[0]})
        if not args.skip_scalar_counts
        else {"status": "SKIP/requested; separate prior source-bound raw retained"}
    )
    return {
        "shape": [nocc, nvir],
        "shared": enabled,
        "mode": mode,
        "status": "PASS",
        "oracle": oracle,
        "input_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in arrays.items()
        },
        "plan_identity": plan.identity,
        "execution_key": execution_key(plan),
        "equation_hash": program.logical_hash,
        "device": device,
        "artifact": artifact.metadata,
        "compiled_files": {
            str(path): digest(path)
            for path in artifact.library.parent.iterdir()
            if path.is_file()
        },
        "static_estimates": estimates,
        "compiled_resident_resources": compiled_resources,
        "full_lifecycle_samples": full_lifecycle_samples,
        "full_lifecycle_median_s": statistics.median(
            x["full_lifecycle_s"] for x in full_lifecycle_samples
        ),
        "plan_s": plan_seconds,
        "compile_s": compile_seconds,
        "prepare_s": prepare_seconds,
        "cleanup_s": cleanup_seconds,
        "warmup": warmup,
        "samples": samples,
        "observed_sample": observed,
        "warm_endpoint_median_s": statistics.median(
            row["endpoint_s"] for row in samples
        ),
        "runtime_batches": len(controls),
        "transfers": transfers,
        "invalid_map_error": invalid_error,
        "recovered": recovered,
        "cupti": {
            "library_sha256": digest(observer.path),
            "header_sha256": observer.header_sha,
            "successful_runtime_kernel_symbols": dict(observer.launches),
            "successful_driver_only_launch_symbols": dict(observer.driver_launches),
            "observed_allocation_peak_bytes": observer.peak,
            "remaining_observed_allocations": observer.allocations,
        },
        "instrumented_scalar_counts_first_runtime_page": scalar_counts,
        "limits": [
            "Scalar entry counters are a separate instrumented artifact and one runtime page, not SM instructions",
            "Allocation peak covers observed malloc APIs, not all driver-reserved storage",
            "Launch APIs do not enumerate device-side graph/child kernels; error memset is reported separately by static contract",
            "Complete (T) correction endpoint only; preceding SCF/CCSD solve excluded",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--shape", action="append", help="nocc,nvir; default 4,8 and 8,16"
    )
    parser.add_argument("--include-default", action="store_true")
    parser.add_argument("--skip-scalar-counts", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {"platform": platform.platform(), "rows": [], "status": "INCOMPLETE"}
    try:
        record["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        record["tracked_source_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        )
        record["tracked_patch_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"])
        ).hexdigest()
        record["qualification_sources_sha256"] = {
            path: digest(Path(path))
            for path in (
                "tools/qualify_shared_streamed_reductions.py",
                "tests/python/test_tensor_cuda_shared_reduction.py",
                "python/generativeqc_compiler/tensor/cuda_emit.py",
                "python/generativeqc_compiler/tensor/cuda_reduction.py",
                "python/generativeqc_compiler/tensor/cuda_plan.py",
                "python/generativeqc_compiler/tensor/cuda_search.py",
            )
        }
        shapes = (
            [tuple(map(int, shape.split(","))) for shape in args.shape]
            if args.shape
            else ((4, 8), (8, 16))
        )
        for shape in shapes:
            choices = [("unfused", False), ("shared", True)]
            if args.include_default:
                choices.insert(0, ("default", False))
            for mode, enabled in choices:
                record["rows"].append(qualify(*shape, enabled, args, mode))
                args.output.write_text(json.dumps(record, indent=2))
        record["status"] = "NUMERICAL_PASS_ENDPOINT_RECORDED_QUALIFICATION_INCOMPLETE"
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
