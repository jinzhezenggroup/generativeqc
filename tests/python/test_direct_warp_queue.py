"""Packet coverage and actual C++ queue proofs; GPU evidence remains separate."""

import importlib.util
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/scf/cuda/direct_bounded_fallback.cu"
HEADER = ROOT / "src/scf/cuda/direct_warp_queue.cuh"


def test_actual_queue_constexpr_proofs() -> None:
    """Evaluate actual queue/selector assertions without emitting build products."""
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or shutil.which(command[0]) is None:
        pytest.skip("C++17 compiler unavailable for syntax/constexpr checking")
    # Syntax-only validation is not a linked build or a GPU qualification.
    cache = shutil.which("ccache")
    if cache and Path(command[0]).name != "ccache":
        subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
        command.insert(0, cache)
    subprocess.run(
        command
        + [
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/cuda/direct_warp_queue_constexpr.cpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("threads", [128, 256])
@pytest.mark.parametrize("indexed", [False, True])
def test_candidate_and_static_fallback_ownership(threads: int, indexed: bool) -> None:
    """Every page/tail remains single-counted, including the 128-thread control."""
    for size in range(1025):
        pages = range(16) if indexed else range(1)
        admitted = []
        for page in pages:
            begin = page * 64 if indexed else 0
            end = min(size, begin + 64) if indexed else size
            for packet in range(begin, end, threads):
                candidates = [
                    packet + lane for lane in range(threads) if packet + lane < end
                ]
                admitted.extend(candidates)
                count = len(candidates)
                drained = [
                    slot
                    for warp in range(threads // 32)
                    for slot in range(warp, count, threads // 32)
                ]
                assert sorted(drained) == list(range(count))
        assert admitted == list(range(size))


def test_scientific_owner_and_packet_lifetime_are_preserved() -> None:
    source = SOURCE.read_text()
    assert source.count("direct_shell_quartet_survives_screening<Unrestricted, Purpose>") == 1
    assert source.count("atomicAdd(global_cursor, 1ULL)") == 1
    assert "candidate_begin += candidate_packet" in source
    assert "Force ? blockDim.x : detail::kBoundedDirectQueueCapacity" in source
    assert "if constexpr (DynamicWarpPull) warp_queue.classes[slot] = shell_class;" in source
    publish = source.index("warp_queue.prepare(queue_count)")
    consume = source.index("task_cursor.next(warp_queue, lane, queue_count)")
    assert "__syncthreads();" in source[publish:consume]
    retire = source.index("/**\n * Diagnostic angular partition", consume)
    assert "__syncthreads();" in source[consume:retire]
    # The queue handles are separate from the mutable per-task tile field.
    assert "if (lane == 0) queue[slot].tile = tile;\n            __syncwarp();" in source
    assert source.count("GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_SCHEDULE") == 1
    launch = source[
        source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(") :
    ]
    assert "if (purpose == DirectScreeningPurpose::Force)" in launch.split("std::getenv")[0]
    range_launches = source[
        source.index("void launch_bounded_direct_range_exchange_force_kernel(") :
    ]
    assert "GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_SCHEDULE" not in range_launches


def test_queue_has_no_raw_domain_or_scientific_dependency() -> None:
    header = HEADER.read_text()
    for forbidden in ("global_cursor", "DeviceBatch", "schwarz", "contract_", "cudaMalloc"):
        assert forbidden not in header
    assert "__shfl_sync(0xffffffffU, slot, 0)" in header
    assert "return atomicAdd(cursor, 1U);" in header
    assert "DisabledClassWarpQueue" in header


def test_queue_dependency_boundary(tmp_path: Path) -> None:
    """The real ownership checker admits only the leaf queue dependency."""
    spec = importlib.util.spec_from_file_location(
        "warp_queue_scf_structure", ROOT / "tools/check_scf_structure.py"
    )
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    cuda = tmp_path / "src/scf/cuda"
    cuda.mkdir(parents=True)
    (cuda / HEADER.name).write_text(HEADER.read_text())
    (cuda / SOURCE.name).write_text('#include "scf/cuda/direct_warp_queue.cuh"\n')
    report = checker.audit_scf_structure(tmp_path)
    assert not report["errors"]
    assert len(report["modules"]) == 2
    assert report["edges"] == [
        {
            "from": "scf/cuda/direct_bounded_fallback.cu",
            "to": "scf/cuda/direct_warp_queue.cuh",
        }
    ]
    # A queue must not acquire scientific recurrence ownership.
    (cuda / "direct_native_cartesian.cuh").write_text("#pragma once\n")
    with (cuda / HEADER.name).open("a") as stream:
        stream.write('#include "scf/cuda/direct_native_cartesian.cuh"\n')
    errors = checker.audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden cuda_direct_queues dependency" in errors[0]


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_WARP_QUEUE_GPU_TESTS") != "1",
    reason="requires explicit real-GPU qualification allocation",
)
def test_real_cuda_queue_ownership(tmp_path: Path) -> None:
    """Actual CUDA atomics, warp masks, mutable tile ownership and packet reuse."""
    cache = shutil.which("ccache")
    nvcc = shutil.which("nvcc")
    assert cache and nvcc, "GPU qualification requires verified ccache and nvcc"
    subprocess.run([cache, "--version"], check=True)
    executable = tmp_path / "warp-queue-probe"
    architecture = os.environ.get("GENERATIVEQC_WARP_QUEUE_TEST_ARCH", "compute_80")
    subprocess.run(
        [
            cache,
            nvcc,
            "-std=c++17",
            f"-arch={architecture}",
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/cuda/direct_warp_queue_probe.cu"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run([str(executable)], check=True, timeout=120)
