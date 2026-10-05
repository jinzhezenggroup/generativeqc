"""Source-extracted optional-cache retry and failed-attempt reporting contracts.

Only resource-policy control executes; molecular/GPU operations are injected
boundaries. The fake phase owner records destruction before cache release.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""

#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point x) {
  return std::chrono::duration<double>(Clock::now() - x).count();
}
constexpr int GENERATIVEQC_STATUS_OUT_OF_MEMORY = 1;
struct MethodError : std::runtime_error {
  int code;
  MethodError(int c) : std::runtime_error("method failure"), code(c) {}
  int status() const { return code; }
};
int calls = 0, live = 0, drains = 0, clears = 0, late_calls = 0, failure = 0;
const char* phase = "triples";
struct ResourceOwner {
  ResourceOwner() { ++live; }
  ~ResourceOwner() {
    --live;
    ++drains;
  }
};
struct Cache {
  std::size_t retained = 40;
  std::size_t storage_bytes() const { return retained; }
  void clear() {
    assert(live == 0);
    assert(drains > 0);
    retained = 0;
    ++clears;
  }
};
namespace runtime {
struct ExecutionContext {
  bool cuda_requested() { return true; }
};
namespace df_progress {
struct Scope {
  Scope(const char*) {}
  bool enabled() { return false; }
  static void label(const char*, const char*) {}
};
}  // namespace df_progress
}  // namespace runtime
namespace core {
struct Atom {
  int position{}, atomic_number{};
};
struct System {
  std::vector<Atom> atoms;
};
}  // namespace core
namespace hf {
struct RHFFrameResponseOptions {
  Cache* recycling{};
};
}  // namespace hf
struct generativeqc_method_descriptor {
  std::size_t limit = 100;
};
struct RccsdNativeState {
  struct Solved {
    bool converged() const { return true; }
  } solved;
  std::size_t budget{};
  std::unique_ptr<ResourceOwner> resource;
};
struct DFCCSDTResult {
  bool recycling_discarded_primal_attempt = false;
  double total_seconds = 0;
  int semantic_work = 13;
};
RccsdNativeState run_rccsd_native_state(runtime::ExecutionContext&, const core::System&,
                                        const generativeqc_method_descriptor& d, void*, void*,
                                        void*, std::size_t reserve, const core::System*, bool, bool,
                                        void*, std::size_t, bool) {
  ++calls;
  RccsdNativeState s;
  s.resource = std::make_unique<ResourceOwner>();
  s.budget = d.limit - reserve;
  if (s.budget < 50) throw std::length_error("primal capacity");
  return s;
}
Cache* current_cache = nullptr;
void later_phase(std::size_t budget) {
  ++late_calls;
  if (failure == 5) {
    current_cache->retained = 10;
    throw std::bad_alloc();
  }
  if (failure == 1) throw std::runtime_error("physical CUDA failure");
  if (failure == 2) throw MethodError(2);
  if (failure == 3) throw std::bad_alloc();
  if (failure == 4) throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  if (budget < 80) throw std::length_error(phase);
}
"""
_TAIL = r"""

(void)with_triples;
(void)df_auxiliary_reduction;
(void)lambda_matrix_gemm;
(void)lambda_batch_limit;
later_phase(state.budget);
DFCCSDTResult result;
result.total_seconds = elapsed(started);
return result;
}
"""
_MAIN = r"""

int main() {
  runtime::ExecutionContext execution;
  core::System system, auxiliary;
  generativeqc_method_descriptor descriptor;
  hf::RHFFrameResponseOptions opts;
  Cache cache;
  opts.recycling = &cache;
  current_cache = &cache;
  auto call = [&] {
    return run_df_ccsdt_native(execution, system, auxiliary, descriptor, true, true, true, true,
                               true, 8, 8, true, &opts);
  };
  auto reset = [&] {
    assert(live == 0);
    calls = drains = clears = late_calls = 0;
    failure = 0;
    cache.retained = 40;
    descriptor.limit = 100;
  };
  for (const char* p : {"triples", "Lambda", "source"}) {
    reset();
    phase = p;
    auto r = call();
    assert(calls == 2 && late_calls == 2 && drains == 2 && live == 0 && clears == 1 &&
           cache.retained == 0 && r.recycling_discarded_primal_attempt && r.semantic_work == 13 &&
           r.total_seconds >= 0);
  }
  for (int mode : {1, 2}) {
    reset();
    failure = mode;
    bool threw = false;
    try {
      (void)call();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw && calls == 1 && late_calls == 1 && drains == 1 && clears == 0 &&
           cache.retained == 40);
  }
  for (int mode : {3, 4}) {
    reset();
    failure = mode;
    bool threw = false;
    try {
      (void)call();
    } catch (const std::exception&) {
      threw = true;
    }
    assert(threw && calls == 2 && late_calls == 2 && drains == 2 && clears == 1 &&
           cache.retained == 0);
  }
  reset();
  descriptor.limit = 70;
  cache.retained = 10;
  bool failed = false;
  try {
    (void)call();
  } catch (const std::length_error&) {
    failed = true;
  }
  assert(failed && calls == 2 && drains == 2 && clears == 1);
  reset();
  descriptor.limit = 70;
  cache.retained = 0;
  failed = false;
  try {
    (void)call();
  } catch (const std::length_error&) {
    failed = true;
  }
  assert(failed && calls == 1 && drains == 1 && clears == 0);
  reset();
  cache.retained = 60;
  auto r = call();
  assert(calls == 2 && late_calls == 1 && drains == 2 && clears == 1 &&
         r.recycling_discarded_primal_attempt);
  reset();
  cache.retained = 60;
  failure = 5;
  bool republished_failure = false;
  try {
    (void)call();
  } catch (const std::bad_alloc&) {
    republished_failure = true;
  }
  assert(republished_failure && calls == 2 && drains == 2 && clears == 1 && cache.retained == 10);
  puts(
      "PASS: three late phases retry once after drain+clear; physical errors propagate; repeated "
      "resource failure stops; cold failure stays single; early primal refusal stays bounded; "
      "republished-cache second failure cannot cause third attempt; unsuccessful attempt work "
      "flagged unavailable");
}
"""


def test_retained_cache_retry_covers_late_phases_once(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    marker = "static DFCCSDTResult run_df_ccsdt_native_attempt("
    public = "\nDFCCSDTResult run_df_ccsdt_native("
    assert source.count(marker) == source.count(public) == 1
    start = source.index(marker)
    end = source.index("\n  DFCCSDTResult result;", start)
    prefix = source[start:end]
    wrapper = source[
        source.index(public, end) : source.rindex(
            "}  // namespace generativeqc::methods::detail"
        )
    ]
    probe = tmp_path / "retry.cpp"
    probe.write_text(_SHIM + prefix + _TAIL + wrapper + _MAIN)
    binary = tmp_path / "retry"
    compile_owner(compiler, tmp_path, [probe], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def test_discarded_attempt_work_is_unavailable_in_benchmark(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = (ROOT / "benchmarks/df_ccsdt_force_endpoint.cpp").read_text()
    start = source.index("      const auto field =")
    stop = source.index('      field("nbf",', start)
    reporting = source[start:stop]
    fields = {
        name: kind
        for kind, name in re.findall(
            r'(?<![A-Za-z_])(work_field|field)\("([^"\n]+)"', source[stop:]
        )
    }
    suffixes = (
        "_calls",
        "_actions",
        "_terms",
        "_summands",
        "_iterations",
        "_bytes",
        "_capacity",
        "_tiles",
        "_slices",
        "_operations",
        "_evaluations",
        "_work",
        "_values",
        "_visits",
        "_kernels",
        "_restarts",
        "_updates",
        "_seconds",
    )
    for name, kind in fields.items():
        if name.endswith(suffixes) and name not in {
            "native_seconds",
            "previous_endpoint_seconds",
        }:
            assert kind == "work_field", name
    assert fields["native_seconds"] == fields["total_energy"] == "field"
    probe = tmp_path / "report.cpp"
    probe.write_text(
        r"""
#include <cassert>
#include <iomanip>
#include <sstream>
struct Result {bool recycling_discarded_primal_attempt;};
int main() {Result result{true};std::ostringstream output;
"""
        + reporting
        + r"""
work_field("triples_work",17);work_field("lambda_actions",19);work_field("jk_actions",23);
work_field("orbital_seconds",2.0);field("native_seconds",3.0);
assert(output.str()=="  \"triples_work\": null,\n  \"lambda_actions\": null,\n  \"jk_actions\": null,\n  \"orbital_seconds\": null,\n  \"native_seconds\": 3,\n");
output.str("");result.recycling_discarded_primal_attempt=false;work_field("triples_work",17);
assert(output.str()=="  \"triples_work\": 17,\n");}
"""
    )
    binary = tmp_path / "report"
    compile_owner(compiler, tmp_path, [probe], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
