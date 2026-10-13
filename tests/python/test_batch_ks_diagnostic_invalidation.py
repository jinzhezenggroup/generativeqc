"""Rejected batch attempts must revoke the preceding solve's AO work record."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_fock_counts_require_complete_cuda_ks_work(tmp_path: Path) -> None:
    """Exercise production publication with counts distinct from iterations.

    A complete final attempt cannot legitimize a partial retry total. A CUDA
    owner also needs the KS diagnostic and both independent census flags.
    """
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++20 compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/api/c_api_batch.cpp").read_text()
    begin = source.index("      const auto& work = item.calculation.precision_work;")
    end = source.index("      const std::uint32_t required_forces", begin)
    publication = source[begin:end]
    cpp, binary = tmp_path / "counts.cpp", tmp_path / "counts"
    cpp.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "scf/target_fock_work.hpp"
struct Work { bool complete{}, operator_inventory_complete{}; };
struct Calculation {
  generativeqc_backend executed_backend;
  Work precision_work;
  std::uint64_t fock_builds{53};
  unsigned iterations{4};
};
struct Item {
  Calculation calculation;
  bool warm_start_fallback{};
  std::optional<generativeqc::scf::TargetFockWork> target_fock_work;
};
struct Batch {
  std::vector<std::optional<int>> ks_diagnostics;
  std::vector<std::uint64_t> last_fock_builds{0};
};
void publish(Batch* batch, const Item& item) {
  const unsigned i = 0;
"""
        + publication
        + r"""
}
struct Case {
  generativeqc_backend backend;
  bool ks, complete, operators, retry;
  std::uint64_t expected;
};
int main() {
  // CPU counters retain their existing contract. Other CUDA methods, both
  // incomplete-census cases, and retries retain the unavailable sentinel.
  const Case cases[] = {
    {GENERATIVEQC_BACKEND_CPU_REFERENCE, false, false, false, false, 53},
    {GENERATIVEQC_BACKEND_CPU_REFERENCE, true, true, true, true, 0},
    {GENERATIVEQC_BACKEND_CUDA, true, true, true, false, 53},
    {GENERATIVEQC_BACKEND_CUDA, true, true, true, true, 0},
    {GENERATIVEQC_BACKEND_CUDA, false, true, true, false, 0},
    {GENERATIVEQC_BACKEND_CUDA, true, false, true, false, 0},
    {GENERATIVEQC_BACKEND_CUDA, true, true, false, false, 0},
    {GENERATIVEQC_BACKEND_CUDA, true, false, false, false, 0},
  };
  for (const auto& test : cases) {
    Batch batch{{test.ks ? std::optional<int>{1} : std::nullopt}};
    Item item{{test.backend, {test.complete, test.operators}}, test.retry, std::nullopt};
    publish(&batch, item);
    assert(batch.last_fock_builds[0] == test.expected);
  }
}
"""
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_rejected_batch_revokes_ks_ao_work(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Exercise both real ABI entries with a fault-injectable prepared owner."""
    from generativeqc import _native

    library = Path(_native.load_library(device="cpu")._name).resolve()
    cpp, binary = (tmp_path / name for name in ("record.cpp", "record"))
    cpp.write_text(
        r"""
#include <cassert>
#include <memory>
#include "api/handles.hpp"

extern "C" generativeqc_status generativeqc_ks_batch_execute_selected_forces_v1(
    generativeqc_batch*, const generativeqc_batch_input_descriptor*, uint32_t,
    generativeqc_batch_item_result_descriptor*, uint32_t);

using namespace generativeqc::methods;
struct Plan final : PreparedBatch {
  bool valid=true, fail=false;
  std::size_t size() const noexcept override { return 1; }
  void invalidate_result() override {
    valid=false;
    if (fail) throw MethodError(GENERATIVEQC_STATUS_INTERNAL_ERROR, "invalidation failed");
  }
  std::vector<BatchItemResult> execute_impl(const Coordinates&, bool) override {
    // A completed dispatch with one failed item publishes no new work record.
    return std::vector<BatchItemResult>(1);
  }
  std::vector<BatchItemResult> execute_selected_forces_impl(
      const Coordinates& coordinates, std::span<const std::uint8_t>) override {
    return execute_impl(coordinates, false);
  }
};

int main() {
  for (auto execute : {generativeqc_batch_execute,
                       generativeqc_ks_batch_execute_selected_forces_v1}) {
    for (unsigned mode=0; mode<5; ++mode) {
      generativeqc_context context;
      generativeqc_batch batch;
      batch.context=&context;
      batch.plan=std::make_unique<Plan>();
      auto& plan=static_cast<Plan&>(*batch.plan);
      plan.fail=mode==1;
      batch.atom_counts={1};
      batch.last_fock_builds={7};
      batch.precision_work.resize(1); batch.precision_work[0].emplace();
      batch.initial_guesses.resize(1); batch.initial_guesses[0].emplace();
      batch.precision.resize(1); batch.precision[0].emplace();
      batch.incremental_direct_jk.resize(1); batch.incremental_direct_jk[0].emplace();
      batch.scf_diagnostics.resize(1); batch.scf_diagnostics[0].emplace();
      batch.ks_diagnostics.resize(1); batch.ks_diagnostics[0].emplace();
      batch.ks_diagnostics[0]->cuda_ao_selection.xc_evaluations=9;
      generativeqc_ks_ao_selection_diagnostic_v1 record{};
      record.struct_size=sizeof(record); record.abi_version=GENERATIVEQC_ABI_VERSION;
      assert(generativeqc_batch_get_ks_ao_selection_diagnostic_v1(&batch,0,&record)==
             GENERATIVEQC_STATUS_SUCCESS && record.xc_evaluations==9);
      generativeqc_batch_item_result_descriptor output{};
      output.struct_size=sizeof(output); output.abi_version=GENERATIVEQC_ABI_VERSION;
      if (mode==3) ++output.abi_version;
      const auto status=execute(&batch,nullptr,0,mode==0?nullptr:&output,mode==2?0:1);
      const auto expected=mode==0 || mode==2 ? GENERATIVEQC_STATUS_INVALID_ARGUMENT :
                          mode==1 ? GENERATIVEQC_STATUS_INTERNAL_ERROR :
                          mode==3 ? GENERATIVEQC_STATUS_ABI_MISMATCH :
                                    GENERATIVEQC_STATUS_SUCCESS;
      assert(status==expected && !plan.valid);
      assert(plan.execution_lifecycle().failed()==(mode!=4));
      assert(generativeqc_batch_get_ks_ao_selection_diagnostic_v1(&batch,0,&record)==
             GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
      assert(!batch.last_fock_builds[0] && !batch.precision_work[0] &&
             !batch.initial_guesses[0] && !batch.precision[0] &&
             !batch.incremental_direct_jk[0] && !batch.scf_diagnostics[0]);
      plan.fail=false;
      output.abi_version=GENERATIVEQC_ABI_VERSION;
      assert(execute(&batch,nullptr,0,&output,1)==GENERATIVEQC_STATUS_SUCCESS);
      assert(!plan.execution_lifecycle().failed());
    }
  }
}
"""
    )
    native_cxx.build_executable(
        [cpp, ROOT / "src/api/c_api_batch.cpp"],
        binary,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    checked = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert checked.returncode == 0, checked.stderr
