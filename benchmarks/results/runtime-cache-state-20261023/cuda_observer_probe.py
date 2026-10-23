"""CPU-only NVCC observation probe; compiled sm120 code is never executed."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from benchmarks.runtime_cache_state import (
    Observer,
    cache_environment,
    digest,
    new_cache,
    save,
)


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("requires real owned CPU Slurm allocation")
    root = Path("/data/srxu/projects/gqc-runtime-cache-state-20261023")
    probe = root / "cuda-observer-probe"
    probe.mkdir(exist_ok=False)
    cache = probe / "cache"
    new_cache(cache)
    environment = cache_environment(
        cache,
        {
            "library": str(probe / "unused-no-sdk.so"),
            "python_paths": [str(root / "source/python")],
        },
        dict(os.environ),
    )
    environment["PATH"] = (
        str(root / "runtime-env/bin") + ":/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
    )
    os.environ.clear()
    os.environ.update(environment)
    from generativeqc_compiler.common import compiler_process
    from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    launcher = resolve_compiler_cache()
    nvcc = Path("/group/software/cuda-12.9.1/bin/nvcc")
    adapter = CudaCompilerAdapter(nvcc, cuda_target_info("sm_120"), compile_timeout=90)
    source = probe / "probe.cu"
    source.write_text('extern "C" __global__ void addone(double *p) {p[0] += 1.0;}\n')
    observer = Observer()
    with observer:
        dependency = compiler_process.run_compiler(
            [str(nvcc), "-M", "-std=c++17", "-arch=sm_120", str(source)],
            30,
            label="NVCC dependency identity",
        )
        if dependency.returncode:
            raise RuntimeError(dependency.stderr)
        for name in ("first.o", "cached.o"):
            result = adapter.compile(
                source,
                probe / name,
                options=("--relocatable-device-code=true", "-Xcompiler=-fPIC"),
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
        result = adapter.link_shared_objects((probe / "first.o",), probe / "probe.so")
        if result.returncode:
            raise RuntimeError(result.stderr)
    elf = subprocess.check_output(
        [
            "/group/software/cuda-12.9.1/bin/cuobjdump",
            "--list-elf",
            str(probe / "probe.so"),
        ],
        text=True,
        timeout=30,
    )
    if "sm_120" not in elf:
        raise ValueError("compiled code object is not sm120")
    save(
        probe / "observations.json",
        {
            "scope": "CPU NVCC dependency/compile/link and private compiler cache; no GPU kernel executed",
            "source_sha256": digest(source),
            "elf": elf,
            "launcher": {"path": str(launcher.path), "version": launcher.version},
            "observation": observer.receipt(),
        },
    )
    save(
        probe / "outcome.json",
        {"nvcc_observation": "PASS", "gpu_execution": "NOT_RUN", "endpoint": "NOT_RUN"},
    )


if __name__ == "__main__":
    main()
