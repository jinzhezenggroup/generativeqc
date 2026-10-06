"""Real-GPU qualification of shared high-order primitive-pair recurrence."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)
from generativeqc_compiler.integral.direct_pair_cache_cuda import (
    emit_direct_pair_cache_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_materialized_pair_recurrence_and_jk(tmp_path: Path) -> None:
    """Check all high orders, both spins and exact preparation/consume counts."""
    if os.environ.get("GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU qualification")
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        "generated_direct_pair_cache.cuh": emit_direct_pair_cache_header(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "generated_direct_source_contraction.cuh": emit_direct_source_contraction_header(),
        "generated_direct_fock_accumulation.cuh": emit_direct_fock_accumulation_header(),
    }
    for name, text in headers.items():
        (tmp_path / name).write_text(text)
    executable = tmp_path / "pair_materialized"
    object_file = tmp_path / "pair_materialized.o"
    stream_object = os.environ.get("GENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_OBJECT")
    if stream_object:
        assert Path(stream_object).is_file()
    command = [
        cache,
        compiler,
        "-std=c++20",
        "-O2",
        "-arch=sm_120",
        f"-I{tmp_path}",
        f"-I{ROOT / 'src'}",
        f"-I{ROOT / 'include'}",
        "-c",
        str(ROOT / "tests/native/test_direct_pair_materialized.cu"),
        "-o",
        str(object_file),
    ]
    if stream_object:
        command[2:2] = [
            "-rdc=true",
            "-DGENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_TEST=1",
        ]
    (tmp_path / "compile-command.json").write_text(json.dumps(command, indent=2))
    environment = {**os.environ, "CCACHE_BASEDIR": str(ROOT)}
    for stage in ("before", "after"):
        if stage == "after":
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=360,
                check=False,
            )
            (tmp_path / "compile.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
            # Compile separately so ccache can reuse the object; CUDA linking
            # is intentionally outside the cacheable compilation invocation.
            link = [compiler, "-arch=sm_120", str(object_file), "-o", str(executable)]
            if stream_object:
                link[1:1] = ["-rdc=true", stream_object]
            (tmp_path / "link-command.json").write_text(json.dumps(link, indent=2))
            subprocess.run(link, env=environment, check=True, timeout=120)
        stats = subprocess.check_output([cache, "--show-stats"], env=environment)
        (tmp_path / f"cache-{stage}.txt").write_bytes(stats)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=360, check=False
    )
    (tmp_path / "gpu.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "order5..12, RHF/UHF J/K and exact work PASS" in result.stdout
    assert "whole-shell admission and exact work PASS" in result.stdout
    if stream_object:
        assert (
            "DFT native dddd stream, RHF/UHF J/K and bounded fallbacks PASS"
            in result.stdout
        )
