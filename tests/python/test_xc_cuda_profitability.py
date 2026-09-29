"""Compiled CUDA resource evidence for shared XC profitability."""

from types import SimpleNamespace

import pytest
from generativeqc_compiler.common.cuda_resources import (
    KernelResources,
    compiled_gpu_profitability,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.xc.cuda import XCArtifact, compiled_xc_profitability


def test_ptxas_rows_normalize_to_compiled_profitability() -> None:
    target = cuda_target_info("sm_120")
    resources = (
        KernelResources(
            function="a",
            registers=64,
            stack_bytes=0,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=1024,
            local_bytes=0,
        ),
        KernelResources(
            function="b",
            registers=128,
            stack_bytes=16,
            spill_store_bytes=24,
            spill_load_bytes=8,
            shared_bytes=2048,
            local_bytes=32,
        ),
    )
    profitability = compiled_gpu_profitability(
        resources,
        target,
        128,
        object_bytes=4096,
        compile_seconds=1.5,
    )
    assert profitability.compiled_registers_per_thread == 128
    assert profitability.spill_store_bytes == 24
    assert profitability.spill_load_bytes == 8
    assert profitability.local_bytes == 32
    assert profitability.shared_bytes == 2048
    assert profitability.compiled_occupancy_upper_bound == pytest.approx(1 / 3)
    assert profitability.object_bytes == 4096
    assert profitability.compile_seconds == 1.5


def test_ptxas_profitability_preserves_unknown_local_memory() -> None:
    profitability = compiled_gpu_profitability(
        (
            KernelResources(
                function="a",
                registers=32,
                stack_bytes=0,
                spill_store_bytes=0,
                spill_load_bytes=0,
                shared_bytes=0,
                local_bytes=None,
            ),
        ),
        cuda_target_info("sm_120"),
        128,
    )
    assert profitability.local_bytes is None


def test_xc_artifact_exposes_scoped_compiled_profitability(tmp_path) -> None:
    library = tmp_path / "runtime.so"
    library.write_bytes(b"compiled-xc")
    metadata = {
        "identity": {"target": {"architecture": "sm_120"}},
        "resources": [
            {
                "function": "runtime_check_scale",
                "registers": 200,
                "stack_bytes": 0,
                "spill_store_bytes": 64,
                "spill_load_bytes": 64,
                "shared_bytes": 4096,
                "local_bytes": 128,
            },
            {
                "function": "_Z10xc_group_0PKdPdmPi",
                "registers": 80,
                "stack_bytes": 0,
                "spill_store_bytes": 16,
                "spill_load_bytes": 8,
                "shared_bytes": 512,
                "local_bytes": 32,
            },
        ],
        "compile_seconds": 2.0,
    }
    artifact = XCArtifact(
        SimpleNamespace(metadata=metadata, library=library),
        {"threads": 128},
    )
    profitability = compiled_xc_profitability(artifact)
    assert profitability.compiled_registers_per_thread == 80
    assert profitability.spill_bytes == 24
    assert profitability.shared_bytes == 512
    assert profitability.object_bytes == library.stat().st_size
    assert profitability.compile_seconds == 2.0
