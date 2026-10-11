"""Explicit native executors for public Array API TensorIR programs.

This is runtime orchestration, not a second IR or code generator. Native
programs and derivative programs use the same compiler-owned lowering paths.
CUDA's host staging is explicit in the selected backend; it never falls back
to reference execution when a device, compiler, or resource budget is missing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from threading import RLock
from typing import Any


class _Executor:
    def __init__(self, backend: str, options: dict[str, Any]) -> None:
        from generativeqc_compiler.tensor.types import checked_size

        allowed = {
            "cache",
            "max_bytes",
            "max_work",
            "max_nodes",
            "compile_timeout",
            "cuda_arch",
            "device",
        }
        if set(options) - allowed:
            raise TypeError("unknown native compilation options")
        options = {
            "cache": None,
            "max_bytes": 64 * 1024 * 1024,
            "max_work": 100_000_000,
            "max_nodes": 4096,
            "compile_timeout": 300.0,
            "cuda_arch": "sm_120",
            "device": 0,
            **options,
        }
        for name in ("max_bytes", "max_work", "max_nodes", "device"):
            checked_size(options[name], name)
        if not 1 <= options["max_nodes"] <= 16384:
            raise ValueError("max_nodes must lie in [1, 16384]")
        timeout = options["compile_timeout"]
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("compile_timeout must be finite and positive")
        self.backend = backend
        self.options = options
        self.executables: dict[str, Any] = {}
        self.lock = RLock()
        self.closed = False
        from ._array_api_molecular import _MolecularRuntime

        self.molecular = _MolecularRuntime()

    def _build(self, program: Any) -> Any:
        cache = self.options["cache"]
        if cache is None:
            cache = (
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "generativeqc"
                / "array-api"
            )
        cache = Path(cache) / self.backend
        if self.backend == "cpu":
            from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
            from generativeqc_compiler.tensor.cpu import NativeTensorProgram

            return NativeTensorProgram(
                program,
                compiler=lambda: CppCompilerAdapter(
                    Path(os.environ.get("CXX", "c++")),
                    compile_timeout=self.options["compile_timeout"],
                ),
                cache=cache,
                max_bytes=self.options["max_bytes"],
                max_work=self.options["max_work"],
                max_nodes=self.options["max_nodes"],
            )
        from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
        from generativeqc_compiler.common.cuda_target import cuda_target_info
        from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
        from generativeqc_compiler.tensor.cuda_plan import plan_cuda

        from generativeqc.profiles import find_nvcc

        # Plan from static IR before discovering a compiler or allocating a GPU.
        target = cuda_target_info(self.options["cuda_arch"])
        plan = plan_cuda(program, target, max_bytes=self.options["max_bytes"])
        if len(plan.program.live_nodes) > self.options["max_nodes"]:
            raise ValueError("CUDA program exceeds node budget")
        # CUDA's planner owns semantic work counts, including contractions.
        work = plan.estimated_flops + sum(step.node.spec.size for step in plan.steps)
        if work > self.options["max_work"]:
            raise ValueError("CUDA program exceeds work budget")
        nvcc = find_nvcc()
        if nvcc is None:
            raise RuntimeError("backend='cuda' requires an available NVCC toolchain")
        compiler = CudaCompilerAdapter(
            nvcc, target, compile_timeout=self.options["compile_timeout"]
        )
        return PreparedCuda(
            plan, compile_cuda(plan, compiler, cache), device=self.options["device"]
        )

    def execute(self, program: Any, feeds: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                raise RuntimeError("compiled function is closed")
            program = self.molecular.bind_snapshots(program)
            if self.backend == "reference":
                from generativeqc_compiler.tensor.interpreter import execute

                return execute(
                    program, feeds, max_bytes=self.options["max_bytes"]
                ).outputs
            executable = self.executables.get(program.logical_hash)
            if executable is None:
                executable = self._build(program)
                self.executables[program.logical_hash] = executable
            self.molecular.materialize(self, program, feeds)
            result = executable.execute(feeds)
            return result if self.backend == "cpu" else result.outputs

    def close(self) -> None:
        with self.lock:
            self.molecular.close()
            for executable in self.executables.values():
                if self.backend == "cuda":
                    executable.close()
            self.executables.clear()
            self.closed = True
