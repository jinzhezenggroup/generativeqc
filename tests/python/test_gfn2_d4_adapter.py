"""Compile the actual GFN2 D4 adapter and exercise its outer storage contract."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory, native_cxx: object) -> Path:
    if sys.platform != "linux":
        pytest.skip("ELF section-GC fixture requires Linux")
    output = tmp_path_factory.mktemp("gfn2-d4-adapter") / "probe"
    runtime = ROOT / "src/xtb/native"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_parameters.py"),
            "--source",
            str(ROOT / "python/generativeqc_compiler/method/method_parameters.json"),
            "--cpp-output",
            str(output.parent / "generated_method_parameters.hpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    native_cxx.build_executable(
        [
            ROOT / "tests/native/test_gfn2_d4_adapter.cpp",
            runtime / "src/model/gfn2/d4.cpp",
            ROOT / "src/dft/dispersion/d4_table_data.cpp",
        ],
        output,
        compile_args=(
            "-std=c++20",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{ROOT / 'src'}",
            f"-I{output.parent}",
            f"-I{runtime}",
            f"-I{runtime / 'src'}",
            f"-I{runtime / 'include'}",
        ),
        link_args=("-Wl,--gc-sections",),
        compile_timeout=90,
    )
    return output


@pytest.mark.parametrize(
    "case",
    [
        "alias_positions",
        "alias_cache",
        "nonfinite_output",
        "late_gradient_failure",
        "late_energy_failure",
        "hotloop_shared_parity",
        "per_system_cache_replay",
        "per_system_failure_atomic",
        "success",
    ],
)
def test_adapter_storage_and_publication(adapter: Path, case: str) -> None:
    result = subprocess.run(
        [str(adapter), case], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
