"""Compile selector and actual endpoint-dispatch fragments with launch recorders."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _block(source: str, signature: str) -> str:
    start = source.index(signature)
    opened = source.index("{", start)
    depth = 0
    for index in range(opened, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start : index + 1]
    raise AssertionError("unclosed source block")


def _branch(source: str, condition: str) -> str:
    first = _block(source, condition)
    remaining = source[source.index(first) + len(first) :]
    assert remaining.lstrip().startswith("else {")
    return first + " " + _block(remaining, "else {")


def test_hf_automatic_admission_and_explicit_controls(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Exercise the shared host selector, including inactive invalid controls."""
    diis = (ROOT / "src/scf/cuda/scf_diis_kernels.cu").read_text()
    selectors = "\n".join(
        _block(diis, "bool " + name + "(")
        for name in (
            "ordered_incremental_diis_gram_requested",
            "incremental_diis_gram_requested",
            "hf_incremental_diis_gram_requested",
            "hf_ordered_diis_gram_requested",
        )
    )
    source = tmp_path / "hf_admission.cpp"
    source.write_text(
        "#include <cstddef>\n#include <cstdlib>\n#include <cstring>\n"
        "#include <stdexcept>\n#include <initializer_list>\n"
        + selectors
        + r"""
int main() {
  const char* enable = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM";
  const char* reduction = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION";
  for (std::size_t aos : {1, 511, 512, 513, 768, 1024})
    for (std::size_t spins : {1, 2})
      for (std::size_t history : {0, 1, 2, 8, 64})
        for (const char* flag : {static_cast<const char*>(nullptr), "0", "1", "bad"})
          for (const char* reducer : {static_cast<const char*>(nullptr), "cooperative", "ordered", "", "bad"}) {
            if (flag) setenv(enable, flag, 1); else unsetenv(enable);
            if (reducer) setenv(reduction, reducer, 1); else unsetenv(reduction);
            const bool explicit_on = flag && !std::strcmp(flag, "1");
            const bool invalid_flag = flag && std::strcmp(flag, "0") && !explicit_on;
            const bool enabled = history >= 2 &&
                (flag ? explicit_on : (aos >= 512 && spins == 1));
            const bool invalid_reducer = reducer && std::strcmp(reducer, "cooperative") &&
                std::strcmp(reducer, "ordered");
            const bool should_reject = (history >= 2 && invalid_flag) ||
                (enabled && invalid_reducer);
            bool rejected = false, incremental = false, ordered = false;
            try {
              incremental = hf_incremental_diis_gram_requested(aos, history, spins);
              ordered = incremental && hf_ordered_diis_gram_requested();
            } catch (const std::invalid_argument&) { rejected = true; }
            if (rejected != should_reject) throw std::runtime_error("control validation");
            if (rejected) continue;
            const bool expected_ordered = enabled &&
                ((!flag && !reducer) || (reducer && !std::strcmp(reducer, "ordered")));
            if (incremental != enabled || ordered != expected_ordered)
              throw std::runtime_error("HF default or diagnostic admission");
          }
}
"""
    )
    executable = tmp_path / "hf_admission"
    required_native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", "-O2")
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )


def test_selector_matrix_and_reachable_endpoint_dispatch(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    diis = (ROOT / "src/scf/cuda/scf_diis_kernels.cu").read_text()
    hf = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    selectors = "\n".join(
        _block(diis, "bool " + name + "()")
        for name in (
            "ordered_incremental_diis_gram_requested",
            "incremental_diis_gram_requested",
        )
    )
    hf_dispatch = _branch(hf, "if (plan.ordered_diis_gram) {")
    ks_dispatch = _branch(ks, "if (ordered_diis_gram) {")
    ks_second = ks[ks.index("if (ordered_diis_gram) {") + len(ks_dispatch) :]
    ks_iteration_dispatch = _branch(ks_second, "if (ordered_diis_gram) {")
    source = tmp_path / "endpoint_dispatch.cpp"
    prefix = r"""
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
unsigned ordered_calls{}, pending_calls{}, legacy_calls{};
constexpr int cudaSuccess = 0;
int cuda_status(int status) { return status; }
void check(int status) { if (status) throw std::runtime_error("launch failure"); }
template <class... Args> void launch_update_diis_cached_gram(Args...) { ++ordered_calls; }
template <class... Args> int launch_diis_pending_gram(Args...) { ++pending_calls; return 0; }
template <class... Args> void launch_update_diis_kernel(Args...) { ++legacy_calls; }
"""
    bindings = r"""
  struct { bool ordered_diis_gram, incremental_diis_gram; } plan{ordered, incremental};
  const bool ordered_diis_gram = ordered, incremental_diis_gram = incremental;
  struct { void* stream_ = nullptr; } resources;
  void* stream = nullptr;
  void* diis_work = nullptr;
  const int batch_size=2, matrix_reduction_threads=32, nbf=3, n=3, spins=1;
  const bool unrestricted=false;
  const unsigned diis_history=4, history=4;
  double value{};
  double *fock=&value, *residual=&value, *fock_history=&value, *residual_history=&value,
         *diis_linear_system=&value, *diis_coefficients=&value, *eigensystem=&value,
         *gram=&value, *weights=&value, *effective=&value;
  double *diis_raw_gram=incremental?&value:nullptr, *raw_gram=diis_raw_gram;
  unsigned count{}, head{};
  unsigned *diis_count=&count, *diis_head=&head, *history_count=&count, *history_head=&head;
  unsigned char active_value=1;
  unsigned char *active=&active_value, *enabled=&active_value;
"""
    bodies = ""
    for name, body in (
        ("hf_endpoint", hf_dispatch),
        ("ks_chunk", ks_dispatch),
        ("ks_iteration", ks_iteration_dispatch),
    ):
        bodies += (
            f"int {name}(bool incremental, bool ordered) {{\n"
            + bindings
            + body
            + "\nreturn 0;\n}\n"
        )
    source.write_text(
        prefix
        + selectors
        + bodies
        + r"""
int main() {
  const char* enable = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM";
  const char* reduction = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION";
  for (const char* flag : {static_cast<const char*>(nullptr), "0", "1", "invalid"})
    for (const char* reducer : {static_cast<const char*>(nullptr), "cooperative", "ordered", "", "invalid", "1"})
      for (unsigned history : {0U, 1U, 2U, 64U}) {
        if (flag) setenv(enable, flag, 1); else unsetenv(enable);
        if (reducer) setenv(reduction, reducer, 1); else unsetenv(reduction);
        const bool admitted = history >= 2;
        const bool enabled = flag && std::strcmp(flag, "1") == 0;
        const bool valid_flag = !flag || std::strcmp(flag, "0") == 0 || enabled;
        const bool ordered = reducer && std::strcmp(reducer, "ordered") == 0;
        const bool valid_reducer = !reducer || std::strcmp(reducer, "cooperative") == 0 || ordered;
        const bool should_reject = admitted && (!valid_flag || (enabled && !valid_reducer));
        bool threw = false, actual_incremental = false, actual_ordered = false;
        try {
          actual_incremental = admitted && incremental_diis_gram_requested();
          actual_ordered = actual_incremental && ordered_incremental_diis_gram_requested();
        } catch (const std::invalid_argument&) { threw = true; }
        if (threw != should_reject) throw std::runtime_error("selector rejection changed");
        if (threw) continue;
        if (actual_incremental != (admitted && enabled) ||
            actual_ordered != (admitted && enabled && ordered))
          throw std::runtime_error("selector chose wrong route");
        ordered_calls = pending_calls = legacy_calls = 0;
        hf_endpoint(actual_incremental, actual_ordered);
        ks_chunk(actual_incremental, actual_ordered);
        ks_iteration(actual_incremental, actual_ordered);
        if (ordered_calls != (actual_ordered ? 3U : 0U) ||
            pending_calls != (actual_incremental && !actual_ordered ? 3U : 0U) ||
            legacy_calls != (actual_ordered ? 0U : 3U))
          throw std::runtime_error("endpoint dispatch changed or doubled Gram work");
      }
}
"""
    )
    executable = tmp_path / "endpoint_dispatch"
    required_native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", "-O2")
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )


def test_reducer_is_owned_by_plan_and_invalidates_hf_replay(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    bucket = (ROOT / "src/scf/cuda/rhf_bucket.cpp").read_text()
    driver = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    retire = _block(bucket, "if (*plan != nullptr && (*plan)->initialized &&")
    replay_source = driver[
        driver.index("  const bool requested_incremental_diis_gram") :
    ]
    replay = _block(replay_source, "if (!first_setup &&")
    stores = []
    for field in ("incremental_diis_gram", "ordered_diis_gram", "diis_work_enabled"):
        start = driver.index(f"plan.{field} =")
        stores.append(driver[start : driver.index(";", start) + 1])
    unit = tmp_path / "reducer_lifetime.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "scf/cuda/arena.hpp"
#include "scf/diis_work.hpp"
using generativeqc::scf::diis_work_requested;
void diis_work_control(bool enabled) {
 if(enabled)setenv("GENERATIVEQC_DF_PROGRESS_TRACE","unused-host-probe-path",1);
 else unsetenv("GENERATIVEQC_DF_PROGRESS_TRACE");
}
using ProviderInitializationUnavailable=std::runtime_error;
struct Plan {
 struct {int device_id_{};} resources;
 int topology{},options{};
 bool initialized=true,unrestricted{},shell_class_profiling{},inactive_eigensolver_profiling{};
 bool bounded_fock_class_timing{},bounded_streaming_override{},fock_only_diagnostic{};
 unsigned primary_streaming_fock_mask{};
 bool graph_native_eigensolver_override{},reuse_converged_fock{},incremental_diis_gram{},ordered_diis_gram{};
 bool one_electron_value_mapping{},mixed_precision_fock{},bounded_direct_streaming{},diis_work_enabled{};
 bool fp32_block_screening{},incremental_direct_jk{};
 double mixed_precision_fock_threshold{};
};
template<class... T> bool same_topology(T&&...) {return true;}
template<class... T> bool compatible_hf_bucket_options(T&&...) {return true;}
template<class... T> bool same_hf_bucket_options(T&&...) {return true;}
bool bounded_fock_class_timing_requested() {return false;}
bool bounded_direct_streaming_override_requested() {return false;}
bool bounded_direct_fock_only_diagnostic_requested() {return false;}
std::optional<unsigned> bounded_direct_primary_streaming_fock_mask_requested() {return {};}
namespace cuda_policy {bool one_electron_value_mapping_requested() {return false;}}
bool retirement_entered=false,retirement_allowed=true;
bool try_retire_rhf_cuda_bucket_plan(Plan** plan) {
 retirement_entered=true;if(!retirement_allowed)return false;*plan=nullptr;return true;
}
void fill_global_failure(std::vector<int>& outputs,generativeqc_status status) {outputs[0]=status;}
void cached_probe(bool saved_incremental,bool saved_ordered,bool saved_work,bool incremental_diis_gram,
                  bool ordered_diis_gram,bool requested_work,bool initialized,bool allow_retirement) {
 Plan storage;storage.incremental_diis_gram=saved_incremental;storage.ordered_diis_gram=saved_ordered;
 storage.diis_work_enabled=saved_work;diis_work_control(requested_work);
 storage.initialized=initialized;Plan* value=&storage;Plan** plan=&value;
 const int device_id=0,candidate=0,options=0;
 const bool unrestricted=false,shell_class_profiling=false,inactive_eigensolver_profiling=false;
 const bool graph_native_eigensolver_override=false,reuse_converged_fock=false,mixed_precision_fock=false;
 const double mixed_precision_fock_threshold=0.;
 retirement_entered=false;retirement_allowed=allow_retirement;bool refused=false;
 try {
"""
        + retire
        + r"""
 } catch(const ProviderInitializationUnavailable&) {refused=true;}
 const bool mismatch=saved_incremental!=incremental_diis_gram || saved_ordered!=ordered_diis_gram || saved_work!=requested_work;
 assert(retirement_entered==(initialized && mismatch));
 assert(refused==(initialized && mismatch && !allow_retirement));
 assert((value==nullptr)==(initialized && mismatch && allow_retirement));
 if(refused)assert(value==&storage);
}
std::vector<int> replay_probe(bool saved_incremental,bool saved_ordered,bool saved_work,
 bool requested_incremental_diis_gram,bool requested_ordered_diis_gram,bool requested_work,bool first_setup) {
 Plan plan;plan.incremental_diis_gram=saved_incremental;plan.ordered_diis_gram=saved_ordered;
 plan.diis_work_enabled=saved_work;diis_work_control(requested_work);
 const int device_id=0,host=0,options=0;
 const bool unrestricted=false,requested_bounded_direct_streaming=false,requested_fp32_block_screening=false;
 const bool shell_class_profiling=false,inactive_eigensolver_profiling=false,bounded_fock_class_timing=false;
 const bool bounded_direct_fock_only_diagnostic=false,requested_graph_native_eigensolver_override=false;
 const bool requested_reuse_converged_fock=false,requested_incremental_direct_jk=false,requested_mixed_precision_fock=false;
 const unsigned requested_primary_streaming_fock_mask=0;
 const std::optional<double> requested_mixed_precision_fock_threshold;
 std::vector<int> outputs{GENERATIVEQC_STATUS_SUCCESS};
"""
        + replay
        + "\n if(first_setup) {\n"
        + "\n".join(stores)
        + r"""
  assert(plan.incremental_diis_gram==requested_incremental_diis_gram);
  assert(plan.ordered_diis_gram==requested_ordered_diis_gram);
  assert(plan.diis_work_enabled==requested_work);
 }
 return outputs;
}
int main() {
 for(bool saved_incremental:{false,true})for(bool saved_ordered:{false,true})for(bool saved_work:{false,true})
 for(bool incremental:{false,true})for(bool ordered:{false,true})for(bool work:{false,true})for(bool first:{false,true}) {
  for(bool retire:{false,true})cached_probe(saved_incremental,saved_ordered,saved_work,incremental,ordered,work,!first,retire);
  const auto result=replay_probe(saved_incremental,saved_ordered,saved_work,incremental,ordered,work,first);
  const bool reject=!first && (saved_incremental!=incremental || saved_ordered!=ordered || saved_work!=work);
  assert(result[0]==(reject?GENERATIVEQC_STATUS_INVALID_ARGUMENT:GENERATIVEQC_STATUS_SUCCESS));
 }
 diis_work_control(false);
 const auto arena=[](const char* reducer) {
  setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION",reducer,1);
  generativeqc::scf::cuda_execution::ArenaLayout layout;
  assert(generativeqc::scf::cuda_execution::make_layout(1,2,2,2,2,3,3,0,0,0,0,0,0,0,0,0,2,4,8,1,
   false,false,false,false,false,false,false,false,layout,true));
  return layout;
 };
 const auto cooperative=arena("cooperative"),ordered=arena("ordered");
 assert(cooperative.diis_raw_gram && cooperative.bytes==ordered.bytes &&
        cooperative.diis_raw_gram==ordered.diis_raw_gram);
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "reducer_lifetime"
    required_native_cxx.build_executable(
        [unit, ROOT / "src/scf/cuda/arena.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)
