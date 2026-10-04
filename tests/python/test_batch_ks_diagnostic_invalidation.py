"""Rejected batch attempts must revoke the preceding solve's AO work record."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def definition(source: str, marker: str) -> str:
    """Extract the production function, including its actual invalidation order."""
    begin = source.index(marker)
    opened = source.index("{", begin)
    depth, end = 1, opened + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_rejected_batch_revokes_ks_ao_work(tmp_path: Path) -> None:
    """Execute the real getter and admission prefix without a device or solver."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++20 compiler")
    source = (ROOT / "src/api/c_api_batch.cpp").read_text()
    getter = definition(
        source,
        "generativeqc_status generativeqc_batch_get_ks_ao_selection_diagnostic_v1(",
    )
    execute = definition(source, "generativeqc_status generativeqc_batch_execute(")
    # Only the admission boundary is under test. Stop before input loops or any
    # physical execution and exercise the unchanged getter on the same owner.
    prefix = execute[: execute.index("  const std::uint32_t system_count =")]
    prefix += "  return GENERATIVEQC_STATUS_SUCCESS;\n}\n"
    harness = r"""
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "dft/ao_selection_work.hpp"
namespace generativeqc::runtime::host_trace {
struct Region { explicit Region(const char*) {} };
}
namespace generativeqc::api {
template<class T> bool valid_descriptor(const T* value) {
  return value && value->struct_size == sizeof(T) &&
         value->abi_version == GENERATIVEQC_ABI_VERSION;
}
generativeqc_status map_exception(std::string*) {
  return GENERATIVEQC_STATUS_INTERNAL_ERROR;
}
}
struct Context { std::recursive_mutex mutex; std::string last_detail; };
struct Plan {
  bool valid=true, fail=false;
  void invalidate_result() {
    valid=false;
    if(fail) throw std::runtime_error("invalidation failed");
  }
};
struct Diagnostic { generativeqc::dft::CudaXcAoSelectionWork cuda_ao_selection; };
struct generativeqc_batch {
  Context* context; Plan* plan;
  std::vector<unsigned> last_fock_builds{7};
  std::vector<std::optional<int>> precision_work{7}, initial_guesses{7},
    precision{7}, incremental_direct_jk{7}, scf_diagnostics{7};
  std::vector<std::optional<Diagnostic>> ks_diagnostics{Diagnostic{}};
};
GETTER
PREFIX
int main() {
  for(unsigned mode=0;mode<3;++mode) {
    Context context; Plan plan; plan.fail=mode==1;
    generativeqc_batch batch{&context,&plan};
    batch.ks_diagnostics[0]->cuda_ao_selection.xc_evaluations=9;
    generativeqc_ks_ao_selection_diagnostic_v1 record{};
    record.struct_size=sizeof(record); record.abi_version=GENERATIVEQC_ABI_VERSION;
    if(generativeqc_batch_get_ks_ao_selection_diagnostic_v1(&batch,0,&record)!=
       GENERATIVEQC_STATUS_SUCCESS || record.xc_evaluations!=9) return 1;
    generativeqc_batch_item_result_descriptor output{};
    const auto status=generativeqc_batch_execute(
        &batch,nullptr,0,mode==0?nullptr:&output,1);
    const auto expected=mode==0?GENERATIVEQC_STATUS_INVALID_ARGUMENT:
                        mode==1?GENERATIVEQC_STATUS_INTERNAL_ERROR:
                                GENERATIVEQC_STATUS_SUCCESS;
    if(status!=expected || plan.valid) return 2;
    if(generativeqc_batch_get_ks_ao_selection_diagnostic_v1(&batch,0,&record)!=
       GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
      std::cerr << "stale AO work survived admission mode " << mode; return 3;
    }
    if(batch.last_fock_builds[0] || batch.precision_work[0] || batch.initial_guesses[0] ||
       batch.precision[0] || batch.incremental_direct_jk[0] || batch.scf_diagnostics[0])
      return 4;
  }
}
"""
    cpp, obj, binary = (
        tmp_path / name for name in ("record.cpp", "record.o", "record")
    )
    cpp.write_text(harness.replace("GETTER", getter).replace("PREFIX", prefix))
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
            "-c",
            str(cpp),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    checked = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert checked.returncode == 0, checked.stderr
