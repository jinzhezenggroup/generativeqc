"""Host-only overflow and boundary checks for optional reference ERI residency."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_reference_cache_preserves_bounded_fallback(tmp_path: Path) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text(r"""
#include <cstddef>
#include <limits>
#include "scf/cuda/reference_eri_policy.hpp"
using generativeqc::scf::cuda_execution::reference_eri_cache_bytes;
int main() {
  constexpr std::size_t required=123456, values=7*7*7*7, bytes=values*sizeof(double);
  if(reference_eri_cache_bytes(values,1,false,required,required+bytes)!=bytes) return 1;
  if(reference_eri_cache_bytes(values,1,false,required,required+bytes-1)!=0) return 2;
  if(reference_eri_cache_bytes(values,1,false,required,required-1)!=0) return 3;
  if(reference_eri_cache_bytes(values,2,false,required,required+bytes)!=0) return 4;
  if(reference_eri_cache_bytes(values,1,true,required,required+bytes)!=0) return 5;
  constexpr auto maximum=std::numeric_limits<std::size_t>::max();
  if(reference_eri_cache_bytes(maximum,1,false,0,maximum)!=0) return 6;
  if(reference_eri_cache_bytes(values,1,false,maximum,maximum)!=0) return 7;
  constexpr std::size_t ceiling=256ULL<<20;
  if(reference_eri_cache_bytes(ceiling/8,1,false,0,maximum)!=ceiling) return 8;
  if(reference_eri_cache_bytes(ceiling/8+1,1,false,0,maximum)!=0) return 9;
}
""")
    subprocess.run(
        ([cache] if cache else [])
        + [
            compiler,
            "-std=c++20",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
