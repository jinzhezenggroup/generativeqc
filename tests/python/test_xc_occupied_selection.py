"""Exercise the production occupied-XC default and durable opt-out parser."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_occupied_xc_default_and_explicit_rollback(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Admission remains a separate owner contract, not an environment spelling."""
    source = tmp_path / "occupied-selection.cpp"
    source.write_text(
        '#include "dft/cuda_ks_precision.hpp"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "  using generativeqc::dft::cuda_ks_occupied_xc_requested;\n"
        "  assert(!cuda_ks_occupied_xc_requested(nullptr));\n"
        '  assert(!cuda_ks_occupied_xc_requested(""));\n'
        '  assert(cuda_ks_occupied_xc_requested("1"));\n'
        '  for (const char* value : {"0", "false", "true", " 1", "10", "unknown"})\n'
        "    assert(!cuda_ks_occupied_xc_requested(value));\n"
        "}\n"
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "occupied-selection",
        compile_args=[
            "-std=c++20",
            "-O2",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([executable], check=True, timeout=10)
