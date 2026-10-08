"""Allocation-failure recovery for the complete production CPU cache transaction.

The real SystemKey and ensure_systems definitions are compiled unchanged. Only
scientific plan construction is stubbed; numerical endpoint coverage belongs to
test_gfn2_runtime_retention.py and test_gfn2_native_oracle.py.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cpu_cache_transaction(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    source = (ROOT / "src/xtb/native/src/runtime/gfn2_cpu_execution.cpp").read_text()
    folder = tmp_path_factory.mktemp("gfn2-cpu-cache-transaction")
    begin = source.index("struct SystemKey {")
    end = source.index("struct NormalizedExecutionPolicy {", begin)
    (folder / "gfn2_cpu_cache_key.inc").write_text(source[begin:end])
    begin = source.index("  generativeqc_xtb_status_t ensure_systems(")
    end = source.index("  void prepare_staging(", begin)
    (folder / "gfn2_cpu_cache_ensure.inc").write_text(source[begin:end])
    return replace(required_native_cxx, base_dir=ROOT).build_executable(
        [ROOT / "tests/native/test_gfn2_cpu_cache_transaction.cpp"],
        folder / "probe",
        compile_args=[
            "-std=c++17",
            "-O1",
            "-Wall",
            "-Wextra",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            f"-I{folder}",
            f"-I{ROOT / 'src/xtb/native/src'}",
        ],
        link_args=["-fsanitize=undefined"],
    )


@pytest.mark.parametrize(
    "scenario",
    ["key-allocation", "first-key-allocation", "plan-build", "plan-throw", "reuse"],
)
def test_cpu_cache_preparation_is_transactional(
    cpu_cache_transaction: Path, scenario: str
) -> None:
    result = subprocess.run(
        [str(cpu_cache_transaction), scenario],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
