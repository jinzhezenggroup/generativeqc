"""Execute XC density re-preparation with bounded provider/runtime host doubles.

Agent: dot. These owner-admission tests do not qualify GPU numerics or performance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


SIGNATURES = (
    "void CudaXcPlan::prepare_density(",
    "const tensor::PreparedPanelProduct* CudaXcPlan::density_execution_provider(",
    "const CudaXcDensityBinding& CudaXcPlan::density_binding(",
    "bool CudaXcPlan::select_local_ao(",
)

PRELUDE = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/lowering_binding.hpp"
using namespace generativeqc::runtime;
using cudaStream_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0, cudaMemcpyHostToDevice = 0;
constexpr std::size_t provider_allowance = 96ULL << 20, host_reservation = 16U << 10;
int capture{}, query_failure{}, copy_failure{}, device_failure{};
int binding_calls{}, provider_calls{}, launcher_calls{}, binding_failure_at{};
bool qualified{true}, unavailable{}, provider_failure{}, binding_allocation_failure{};
bool discovery_empty{};
std::size_t live_contexts{}, peak_contexts{}, live_cache{}, peak_cache{}, live_host{}, peak_host{};
std::size_t numeric_limit = std::numeric_limits<std::size_t>::max();
void check(int status) { if (status) throw std::runtime_error("runtime failure"); }
int cudaStreamIsCapturing(int, int* value) { *value = capture; return query_failure; }
int cudaMemcpyAsync(void*, const void*, std::size_t, int, int) { return copy_failure; }
int cudaStreamSynchronize(int) { return 0; }
std::size_t size_mul(std::size_t a, std::size_t b, const char*) { return a*b; }
std::size_t size_add(std::size_t a, std::size_t b, const char*) { return a+b; }
struct CudaXcDensityBinding {
  NativeLoweringCandidate candidate{};
  NativeLoweringPrecision precision{};
  std::size_t count{}, replays{};
  static void* operator new(std::size_t n) {
    if (binding_allocation_failure) throw std::bad_alloc();
    return ::operator new(n);
  }
  static void operator delete(void* p) { ::operator delete(p); }
};
namespace tensor {
struct Diagnostic { NativeLoweringCandidate candidate{}; };
// Model the production ordering: opaque context first, ledger cache second.
// Counters represent reservations; no GPU or large host allocation is made.
struct PreparedPanelProduct {
  bool active{};
  std::size_t cache{};
  Diagnostic info{};
  PreparedPanelProduct(bool enable, std::size_t bytes) {
    live_host += host_reservation; peak_host = std::max(peak_host, live_host);
    if (enable) {
      ++live_contexts; peak_contexts = std::max(peak_contexts, live_contexts);
      if (bytes <= numeric_limit - live_cache) {
        active = true; cache = bytes; live_cache += cache;
        peak_cache = std::max(peak_cache, live_cache);
      } else { --live_contexts; }
    }
    info.candidate.provider = active ? "cublas" : "generated.cuda";
  }
  ~PreparedPanelProduct() {
    if (active) { --live_contexts; live_cache -= cache; }
    live_host -= host_reservation;
  }
  bool enabled() const { return active; }
  const Diagnostic& diagnostic() const { return info; }
};
}
struct Layout {
  std::size_t nao{3}, npoint{7}, tile_points{3}, spins{2}, work_jets{4}, jets{4};
  bool local_ao{}, response{}, mixed_physical{true};
  struct { bool mixed_density_precision{true}; } fast_paths;
  std::size_t device_bytes{};
};
struct Capabilities { bool mixed_density_contraction; };
Capabilities cuda_xc_execution_capabilities(const Layout& l) {
  return {l.mixed_physical && !l.local_ao && !l.response};
}
bool cuda_xc_capability_qualified(bool value) { return value; }
struct CudaXcAoTiles {
  std::vector<std::size_t> indices, offsets;
  int derivative_order{-1};
};
struct Bound { std::size_t device_bytes{64}, host_peak_bytes{64}, max_entries{9}, tiles{3}; };
Bound cuda_xc_ao_selection_resources(const Layout&) { return {}; }
Layout cuda_xc_local_ao_layout(Layout l, const CudaXcAoTiles& maps) {
  if (maps.derivative_order != (l.jets == 1 ? 0 : 1))
    throw std::invalid_argument("discovery lost derivative capability");
  l.local_ao = true; return l;
}
std::vector<int> local_density_launchers(const Layout&, const std::vector<std::size_t>& offsets) {
  ++launcher_calls; return std::vector<int>(offsets.size()-1, 1);
}
struct CudaXcAoSelectionWork {
  bool requested{}, selected{};
  double cutoff{}, discovery_seconds{};
  std::size_t reserved_device_bytes{}, host_peak_bytes{};
  std::size_t tiles{}, empty_tiles{}, min_active{3}, max_active{}, active_sum{};
  std::size_t point_ao_visits{}, point_ao_square_sum{}, discovery_ao_jet_values{};
  std::size_t dense_point_ao_square_sum{}, discovery_d2h_bytes{};
};
namespace cuda_xc_detail {
CudaXcDensityBinding prepare_density_binding(std::size_t, std::size_t count,
    std::size_t, std::size_t, PrecisionDirective precision, std::uint64_t replays) {
  ++binding_calls;
  if (binding_failure_at == binding_calls) throw std::bad_alloc();
  if (!replays || (!precision.is_strict_fp64() &&
      precision.qualification != "dft.cuda.auto/density-contraction-v1"))
    throw std::invalid_argument("invalid binding admission");
  CudaXcDensityBinding result;
  result.precision.arithmetic = precision;
  result.candidate.provider = "generated.cuda";
  result.count = count; result.replays = replays;
  return result;
}
std::unique_ptr<tensor::PreparedPanelProduct> prepare_density_provider(
    const Layout& l, int, std::size_t budget) {
  ++provider_calls;
  if (provider_failure) throw std::bad_alloc();
  const auto bytes = l.spins*l.nao*l.nao*sizeof(double);
  return std::make_unique<tensor::PreparedPanelProduct>(
      qualified && !unavailable && !l.response && budget >= provider_allowance+bytes, bytes);
}
void select_ao(const Layout& l, int, void*, const double*, std::size_t, double,
               void*, void*, int*, unsigned* flags) {
  for (std::size_t i=0; i<l.nao; ++i) flags[i] = !discovery_empty && i != 1;
}
}
struct CudaXcPlan {
  Layout layout_;
  bool evaluation_started_{};
  double* point_batch_arena_{};
  int stream_{7};
  std::array<CudaXcDensityBinding, 2> strict_density_{}, admitted_density_{};
  std::unique_ptr<CudaXcDensityBinding> provider_density_binding_;
  std::unique_ptr<tensor::PreparedPanelProduct> density_provider_;
  std::vector<int> local_density_launchers_;
  std::vector<std::size_t> ao_offsets_{0, 2, 4, 6};
  std::array<double, 21> points_storage{};
  const double* points_{points_storage.data()};
  std::array<std::size_t, 128> arena_storage{};
  void* arena_{arena_storage.data()};
  std::size_t arena_bytes_{1024};
  void *basis_{}, *ao_{}, *work_{};
  int* error_{};
  std::size_t* ao_ids_{};
  struct { std::size_t setup_h2d_bytes{}, synchronizations{}; } transfers_;
  CudaXcAoSelectionWork ao_selection_work_;
  void check_device() const { check(device_failure); }
  void prepare_density(PrecisionDirective, std::uint64_t = 1, std::size_t = 0);
  const tensor::PreparedPanelProduct* density_execution_provider(PrecisionPhase) const;
  const CudaXcDensityBinding& density_binding(PrecisionPhase) const;
  bool select_local_ao(double, std::size_t);
};
"""

CASES = (
    "overlap",
    "ledger",
    "replacement",
    "failures",
    "mixed",
    "validation",
    "empty",
    "response",
    "discovery",
    "batched",
)

TESTS = r"""
const auto strict = strict_fp64_precision();
const auto mixed = fp32_compute_fp64_accumulation("dft.cuda.auto/density-contraction-v1");
constexpr std::size_t budget = provider_allowance + 2*3*3*sizeof(double);
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> bool rejected(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
struct Snapshot {
  const void* provider;
  const void* binding;
  std::array<std::size_t, 8> tables;
  explicit Snapshot(const CudaXcPlan& p) : provider(p.density_provider_.get()),
      binding(p.provider_density_binding_.get()) {
    for (std::size_t i=0; i<2; ++i) {
      tables[4*i] = p.strict_density_[i].count; tables[4*i+1] = p.strict_density_[i].replays;
      tables[4*i+2] = p.admitted_density_[i].replays;
      tables[4*i+3] = p.admitted_density_[i].precision.arithmetic.is_strict_fp64();
    }
  }
  void unchanged(const CudaXcPlan& p) const {
    Snapshot current(p);
    require(provider == current.provider && binding == current.binding && tables == current.tables,
            "rejected replacement changed published owner or tables");
  }
};
void overlap(bool tight_ledger) {
  CudaXcPlan p; p.prepare_density(strict, 11, budget);
  const Snapshot before(p);
  const auto old_bindings = binding_calls, old_providers = provider_calls;
  if (tight_ledger) numeric_limit = live_cache;
  const auto did_reject = rejected([&]{ p.prepare_density(strict, 12, budget); });
  std::cout << "peak_contexts=" << peak_contexts
            << " peak_context_reservation=" << peak_contexts*provider_allowance
            << " peak_cache=" << peak_cache << " peak_host=" << peak_host << '\n';
  require(did_reject, "nonzero re-preparation accepted a second live optional owner");
  before.unchanged(p);
  require(binding_calls == old_bindings && provider_calls == old_providers,
          "guard ran after temporary binding/provider preparation");
  require(peak_contexts == 1 && peak_cache == live_cache && peak_host == host_reservation,
          "re-preparation exceeded a single live reservation");
  require(rejected([&]{ p.prepare_density(strict, 12, 1); }), "small nonzero reprepare accepted");
  before.unchanged(p);
}
void replacement() {
  CudaXcPlan p;
  p.prepare_density(strict, 1); p.prepare_density(mixed, 2);
  require(!p.density_provider_ && provider_calls == 0, "generated replacement acquired provider");
  qualified = false; p.prepare_density(strict, 3, budget);
  require(p.density_provider_ && !p.density_provider_->enabled(), "unqualified owner enabled");
  unavailable = true; qualified = true; p.prepare_density(strict, 4, budget);
  require(!p.density_provider_->enabled(), "unavailable owner enabled");
  unavailable = false; p.prepare_density(strict, 5, budget);
  require(p.density_provider_->enabled(), "disabled-to-enabled replacement blocked");
  p.prepare_density(mixed, 6, 0);
  require(!p.density_provider_ && !p.provider_density_binding_ && !live_contexts && !live_cache,
          "zero-budget replacement did not release enabled owner");
  require(!p.admitted_density_[0].precision.arithmetic.is_strict_fp64(), "release lost new table");
  p.prepare_density(strict, 7, budget);
  require(p.density_provider_->enabled() && peak_contexts == 1, "release/reprepare overlapped");
}
void failures() {
  CudaXcPlan p; p.prepare_density(strict, 7);
  for (int fail=1; fail<=4; ++fail) {
    const Snapshot before(p); binding_failure_at = binding_calls+fail;
    require(rejected([&]{ p.prepare_density(mixed, 9, budget); }), "binding failure ignored");
    before.unchanged(p); binding_failure_at = 0;
  }
  qualified = false; p.prepare_density(strict, 7, budget); qualified = true;
  const Snapshot disabled(p);
  provider_failure = true;
  require(rejected([&]{ p.prepare_density(strict, 9, budget); }), "provider failure ignored");
  disabled.unchanged(p); provider_failure = false;
  binding_allocation_failure = true;
  require(rejected([&]{ p.prepare_density(strict, 9, budget); }), "binding allocation failure ignored");
  disabled.unchanged(p); binding_allocation_failure = false;
  require(!live_contexts && !live_cache, "failed candidate leaked resources");
  p.prepare_density(strict, 7, budget);
  const Snapshot enabled(p);
  require(rejected([&]{ p.prepare_density(strict, 0, 0); }), "invalid replay admission accepted");
  enabled.unchanged(p);
  const auto invalid = fp32_compute_fp64_accumulation("unqualified");
  require(rejected([&]{ p.prepare_density(invalid, 8, 0); }), "invalid arithmetic accepted");
  enabled.unchanged(p);
}
void mixed_tables() {
  CudaXcPlan p; p.prepare_density(mixed, 10, budget);
  require(p.density_binding(PrecisionPhase::StrictAudit).candidate.provider == "cublas" &&
          p.density_binding(PrecisionPhase::Admitted).candidate.provider == "generated.cuda",
          "provider replaced mixed arithmetic");
  require(p.strict_density_[0].count == 3 && p.strict_density_[1].count == 1 &&
          p.admitted_density_[0].count == 3 && p.admitted_density_[1].count == 1,
          "full/tail table shapes changed");
  Snapshot before(p);
  require(rejected([&]{ p.prepare_density(strict, 11, budget); }), "mixed owner reprepare accepted");
  before.unchanged(p);
  p.prepare_density(mixed, 12, 0);
  require(p.density_binding(PrecisionPhase::StrictAudit).precision.arithmetic.is_strict_fp64() &&
          !p.density_binding(PrecisionPhase::Admitted).precision.arithmetic.is_strict_fp64(),
          "zero-budget release lost strict/mixed distinction");
}
void validation() {
  CudaXcPlan p; p.prepare_density(strict, 3, budget); const Snapshot before(p);
  const auto old_calls = binding_calls, old_providers = provider_calls;
  for (int route=0; route<6; ++route) {
    p.evaluation_started_ = route == 0; capture = route == 1; query_failure = route == 2;
    device_failure = route == 3; p.layout_.mixed_physical = route != 4;
    p.layout_.fast_paths.mixed_density_precision = route != 5;
    require(rejected([&]{ p.prepare_density(mixed, 4, 0); }), "validation refusal changed");
    before.unchanged(p);
  }
  require(binding_calls == old_calls && provider_calls == old_providers,
          "invalid preparation started binding/provider work");
}
void empty() {
  CudaXcPlan p; p.layout_.local_ao = true; p.ao_offsets_ = {0, 0, 0, 0};
  p.prepare_density(strict, 1, budget); p.prepare_density(strict, 2, budget);
  require(!p.density_provider_ && !provider_calls && launcher_calls == 1,
          "empty domain acquired provider or rebuilt launchers");
  require(p.density_binding(PrecisionPhase::StrictAudit).candidate.provider == "generated.cuda",
          "empty domain lost generated binding");
}
void response() {
  CudaXcPlan p; p.layout_.response = true;
  p.prepare_density(strict, 1, budget); p.prepare_density(strict, 2, budget);
  require(!p.density_provider_->enabled() && !live_contexts && !live_cache,
          "response acquired optional resources");
}
void discovery() {
  for (std::size_t jets : {1U, 4U})
  for (bool empty_map : {false, true}) {
    discovery_empty = false;
    CudaXcPlan p; p.layout_.jets = jets;
    p.prepare_density(strict, 1, budget); const Snapshot dense(p);
    require(!p.select_local_ao(1e-16, 0), "discovery ignored insufficient resources");
    dense.unchanged(p);
    copy_failure = 1;
    require(rejected([&]{ p.select_local_ao(1e-16, 64); }), "discovery copy failure ignored");
    dense.unchanged(p); copy_failure = 0; discovery_empty = empty_map;
    require(p.select_local_ao(1e-16, 64), "discovery did not complete");
    require(!p.density_provider_ && !p.provider_density_binding_ && !live_contexts && !live_cache,
            "discovery did not reset dense owner");
    p.prepare_density(strict, 2, budget);
    require(bool(p.density_provider_) == !empty_map, "discovery reset blocked correct new domain");
    if (!empty_map) require(p.density_provider_->enabled(), "indexed preparation not enabled");
  }
}
void batched() {
  CudaXcPlan p; p.prepare_density(strict, 3);
  double retained_panel{}; p.point_batch_arena_ = &retained_panel;
  const Snapshot before(p);
  const auto old_bindings = binding_calls, old_providers = provider_calls;
  const auto old_launchers = launcher_calls;
  require(rejected([&]{ p.prepare_density(mixed, 4, budget); }),
          "batched owner accepted mixed density arithmetic");
  require(rejected([&]{ p.select_local_ao(1e-16, 64); }),
          "batched owner accepted map discovery");
  before.unchanged(p);
  require(binding_calls == old_bindings && provider_calls == old_providers &&
          launcher_calls == old_launchers && !p.layout_.local_ao,
          "batched guard ran after binding or map preparation");
  p.prepare_density(strict, 4);
  require(p.admitted_density_[0].precision.arithmetic.is_strict_fp64() &&
          p.point_batch_arena_ == &retained_panel, "strict rebind lost batched owner");
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected case name"); const std::string name = argv[1];
    if (name == "overlap") overlap(false);
    else if (name == "ledger") overlap(true);
    else if (name == "replacement") replacement();
    else if (name == "failures") failures();
    else if (name == "mixed") mixed_tables();
    else if (name == "validation") validation();
    else if (name == "empty") empty();
    else if (name == "response") response();
    else if (name == "discovery") discovery();
    else if (name == "batched") batched();
    else throw std::runtime_error("unknown case");
    require(!live_contexts && !live_cache && !live_host, "owner teardown leaked resources");
    std::cout << name << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
"""


def probe_source(owner: str) -> str:
    return PRELUDE + "\n".join(_definition(owner, sig) for sig in SIGNATURES) + TESTS


@pytest.fixture(scope="module")
def reprepare_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    directory = tmp_path_factory.mktemp("xc-density-repreparation")
    source = directory / "probe.cpp"
    source.write_text(probe_source((ROOT / "src/dft/cuda_xc.cpp").read_text()))
    binary = directory / "probe"
    compile_owner(compiler, directory, [source], binary)
    return binary


@pytest.mark.parametrize("case", CASES)
def test_density_repreparation(reprepare_probe: Path, case: str) -> None:
    result = subprocess.run(
        [str(reprepare_probe), case], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
