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
using generativeqc::scf::cuda_execution::reference_quartet_direct;
using generativeqc::scf::cuda_execution::reference_resident_value_allowance;
using generativeqc::scf::cuda_execution::reference_resident_values_auto_refusal;
using generativeqc::scf::cuda_execution::reference_resident_values_selection;
using generativeqc::scf::cuda_execution::ReferenceResidentValuesSelection;
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
  if(reference_quartet_direct(0,2) || reference_quartet_direct(76,1)) return 10;
  if(!reference_quartet_direct(77,1) || !reference_quartet_direct(maximum,0)) return 11;
  if(!reference_quartet_direct(6,2) || !reference_quartet_direct(8,3)) return 12;
  if(reference_quartet_direct(230,4)) return 13;
  if(reference_resident_value_allowance(64,4,100,200,400,500)!=100) return 14;
  if(reference_resident_value_allowance(63,4,100,200,400,500)!=0) return 15;
  if(reference_resident_value_allowance(64,3,100,200,400,500)!=0) return 16;
  if(reference_resident_value_allowance(64,4,100,200,299,500)!=0) return 17;
  if(reference_resident_value_allowance(64,4,100,200,400,50)!=50) return 18;
  if(reference_resident_value_allowance(64,4,maximum,1,maximum,maximum)!=0) return 19;
  if(reference_resident_value_allowance(64,4,0,maximum,maximum,maximum)!=0) return 20;
  if(reference_resident_values_selection(nullptr)!=ReferenceResidentValuesSelection::automatic) return 21;
  if(reference_resident_values_selection("auto")!=ReferenceResidentValuesSelection::automatic) return 22;
  if(reference_resident_values_selection("0")!=ReferenceResidentValuesSelection::disabled) return 23;
  if(reference_resident_values_selection("1")!=ReferenceResidentValuesSelection::forced) return 24;
  for(const char* invalid:{"", "AUTO", "true", "2", " auto"})
    if(reference_resident_values_selection(invalid)!=ReferenceResidentValuesSelection::invalid) return 25;
  if(reference_resident_values_auto_refusal(64,128,8,3,true,true)) return 26;
  if(!reference_resident_values_auto_refusal(64,128,8,3,false,true)) return 27;
  if(!reference_resident_values_auto_refusal(64,128,8,3,true,false)) return 28;
  if(!reference_resident_values_auto_refusal(64,128,8,2,true,true)) return 29;
  if(!reference_resident_values_auto_refusal(64,127,8,3,true,true)) return 30;
  if(!reference_resident_values_auto_refusal(63,128,8,3,true,true)) return 31;
  if(!reference_resident_values_auto_refusal(64,128,7,3,true,true)) return 32;
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
