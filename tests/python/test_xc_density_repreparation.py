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
    "bool CudaXcPlan::density_provider_retirement_pending(",
    "void CudaXcPlan::retire_density_provider(",
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
#include "runtime/df_progress_trace.hpp"
#include "dft/ao_selection_work.hpp"
#include "generativeqc/generativeqc.hpp"
#include "tensor/cuda_error.hpp"
namespace runtime = generativeqc::runtime;
using generativeqc::dft::CudaXcAoSelectionWork;
enum class CudaXcAoDiscovery { HostTiles, DeviceChunks };
using namespace generativeqc::runtime;
using cudaStream_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0, cudaMemcpyHostToDevice = 0;
constexpr std::size_t provider_allowance = 96ULL << 20, host_reservation = 16U << 10;
int capture{}, query_failure{}, copy_failure{}, device_failure{};
int binding_calls{}, provider_calls{}, launcher_calls{}, binding_failure_at{};
bool qualified{true}, unavailable{}, provider_failure{}, binding_allocation_failure{};
bool discovery_empty{};
int provider_error_kind{}, release_error_kind{}, synchronization_error{}, synchronization_calls{};
bool stream_drained{};
void (*provider_retirement_observer)(){};
std::size_t live_contexts{}, peak_contexts{}, live_cache{}, peak_cache{}, live_host{}, peak_host{};
std::size_t numeric_limit = std::numeric_limits<std::size_t>::max();
void check(int status) {
  if (status) throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, "runtime failure");
}
int cudaStreamIsCapturing(int, int* value) { *value = capture; return query_failure; }
int cudaMemcpyAsync(void*, const void*, std::size_t, int, int) { return copy_failure; }
int cudaStreamSynchronize(int) {
  ++synchronization_calls;
  if (!synchronization_error) stream_drained = true;
  return synchronization_error;
}
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
struct Diagnostic {
  NativeLoweringCandidate candidate{};
  std::size_t matrix_bytes{}, provider_allowance{}, host_bytes{};
};
// Model the production ordering: opaque context first, ledger cache second.
// Counters represent reservations; no GPU or large host allocation is made.
struct PreparedPanelProduct {
  bool active{}, owns_context{};
  std::size_t cache{};
  Diagnostic info{};
  PreparedPanelProduct(bool enable, std::size_t bytes) {
    live_host += host_reservation; peak_host = std::max(peak_host, live_host);
    if (enable) {
      ++live_contexts; peak_contexts = std::max(peak_contexts, live_contexts);
      if (bytes <= numeric_limit - live_cache) {
        active = owns_context = true; cache = bytes; live_cache += cache;
        peak_cache = std::max(peak_cache, live_cache);
      } else { --live_contexts; }
    }
    info.candidate.provider = active ? "cublas" : "generated.cuda";
    info.matrix_bytes = cache;
    info.provider_allowance = active ? provider_allowance : 0;
    info.host_bytes = host_reservation;
  }
  ~PreparedPanelProduct() {
    if ((owns_context || cache) && provider_retirement_observer) provider_retirement_observer();
    if (owns_context) --live_contexts;
    live_cache -= cache;
    live_host -= host_reservation;
  }
  void release() {
    active = false;
    if (provider_retirement_observer) provider_retirement_observer();
    if (release_error_kind == 1)
      throw generativeqc_tensor::DeviceAllocationError("provider release allocation failed");
    if (release_error_kind == 2)
      throw generativeqc_tensor::DeviceRuntimeError("provider release failed");
    live_cache -= cache; cache = 0;
    if (release_error_kind == 3)
      throw generativeqc_tensor::DeviceRuntimeError("provider destruction failed");
    if (owns_context) --live_contexts;
    owns_context = false;
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
Layout cuda_xc_local_ao_layout_shape(Layout l, std::size_t, int derivative_order) {
  if (derivative_order != (l.jets == 1 ? 0 : 1))
    throw std::invalid_argument("discovery lost derivative capability");
  l.local_ao = true; return l;
}
std::vector<int> local_density_launchers(const Layout&, const std::vector<std::size_t>& offsets) {
  ++launcher_calls; return std::vector<int>(offsets.size()-1, 1);
}
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
    const Layout& l, int, std::size_t budget, std::size_t automatic_budget, std::uint64_t) {
  ++provider_calls;
  if (provider_failure) throw std::bad_alloc();
  if (provider_error_kind == 1)
    throw generativeqc_tensor::DeviceAllocationError("provider allocation failed");
  if (provider_error_kind == 2)
    throw generativeqc_tensor::DeviceRuntimeError("provider launch failed");
  if (provider_error_kind == 3)
    throw std::runtime_error("unclassified provider failure");
  const auto bytes = l.spins*l.nao*l.nao*sizeof(double);
  const bool indexed_gemm=automatic_budget && l.local_ao && !l.response;
  if(indexed_gemm) budget=automatic_budget;
  return std::make_unique<tensor::PreparedPanelProduct>(
      (qualified || indexed_gemm) && !unavailable && !l.response &&
          budget >= provider_allowance+bytes, bytes);
}
void select_ao(const Layout& l, int, void*, const double*, std::size_t, double,
               void*, void*, int*, unsigned* flags) {
  for (std::size_t i=0; i<l.nao; ++i) flags[i] = !discovery_empty && i != 1;
}
// Model the external compaction/transfer port; owner admission and publication
// still execute the production body below, for both discovery schedules.
void select_ao_chunk(const Layout& l, int, void*, const double*, std::size_t,
    std::size_t tiles, double, void*, void*, double*, int*, std::size_t* indices,
    std::size_t capacity, std::size_t offset, std::size_t* host_offsets) {
  check(copy_failure);
  for (std::size_t tile=0; tile<tiles; ++tile) {
    for (std::size_t ao=0; ao<l.nao; ++ao) {
      if (discovery_empty || ao == 1) continue;
      if (offset >= capacity) throw std::runtime_error("discovery capacity exceeded");
      indices[offset++] = ao;
    }
    host_offsets[tile] = offset;
  }
}
}
struct CudaXcPlan {
  Layout layout_;
  bool evaluation_started_{};
  bool mixed_matrix_contractions_{}, mixed_pbe_ao_{};
  double* point_batch_arena_{};
  double* point_totals_{};
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
  void prepare_density(PrecisionDirective, std::uint64_t = 1, std::size_t = 0, std::size_t = 0);
  void retire_density_provider();
  bool density_provider_retirement_pending() const noexcept;
  const tensor::Diagnostic* density_provider_diagnostic() const noexcept {
    return density_provider_ ? &density_provider_->diagnostic() : nullptr;
  }
  const tensor::PreparedPanelProduct* density_execution_provider(PrecisionPhase) const;
  const CudaXcDensityBinding& density_binding(PrecisionPhase) const;
  bool select_local_ao(double, std::size_t, CudaXcAoDiscovery = CudaXcAoDiscovery::HostTiles);
};
"""

CASES = (
    "overlap",
    "ledger",
    "replacement",
    "failures",
    "provider_errors",
    "mixed",
    "validation",
    "empty",
    "response",
    "discovery",
    "batched",
    "indexed_qualification",
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
void provider_errors() {
  CudaXcPlan p;
  qualified = false; p.prepare_density(strict, 7, budget); qualified = true;
  const Snapshot before(p);
  for (int kind : {1, 2, 3}) {
    provider_error_kind = kind;
    bool caught = false;
    try { p.prepare_density(strict, 9, budget); }
    catch (const std::bad_alloc&) { caught = kind == 1; }
    catch (const generativeqc::Error& error) {
      caught = kind == 2 && error.status() == GENERATIVEQC_STATUS_CUDA_ERROR &&
               std::string(error.what()) == "provider launch failed";
    }
    catch (const std::runtime_error& error) {
      caught = kind == 3 && std::string(error.what()) == "unclassified provider failure";
    }
    require(caught, "provider error lost its native status or diagnostic");
    before.unchanged(p);
    require(!live_contexts && !live_cache && live_host == host_reservation,
            "provider error changed the retained reservation");
  }
  provider_error_kind = 0;
  p.prepare_density(strict, 9, budget);
  require(p.density_provider_->enabled(), "owner could not recover after failed preparation");
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
  for (auto schedule : {CudaXcAoDiscovery::HostTiles, CudaXcAoDiscovery::DeviceChunks})
  for (std::size_t jets : {1U, 4U})
  for (bool empty_map : {false, true}) {
    discovery_empty = false;
    CudaXcPlan p; p.layout_.jets = jets;
    p.prepare_density(strict, 1, budget); const Snapshot dense(p);
    require(rejected([&]{ p.select_local_ao(1e-16, 64,
                static_cast<CudaXcAoDiscovery>(255)); }), "unknown discovery schedule accepted");
    dense.unchanged(p);
    require(!p.select_local_ao(1e-16, 0, schedule), "discovery ignored insufficient resources");
    dense.unchanged(p);
    copy_failure = 1;
    require(rejected([&]{ p.select_local_ao(1e-16, 64, schedule); }), "discovery copy failure ignored");
    dense.unchanged(p); copy_failure = 0; discovery_empty = empty_map;
    require(p.select_local_ao(1e-16, 64, schedule), "discovery did not complete");
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
void indexed_qualification() {
  qualified = false;
  CudaXcPlan owner;
  owner.prepare_density(strict, 3, 0, budget);
  require(!owner.density_provider_->enabled(), "dense domain accepted indexed qualification");
  owner.layout_.local_ao = true;
  owner.layout_.response = true;
  owner.prepare_density(strict, 3, 0, budget);
  require(!owner.density_provider_->enabled(), "response accepted physical indexed qualification");
  owner.layout_.response = false;
  const Snapshot before_mixed(owner);
  require(rejected([&] { owner.prepare_density(mixed, 3, 0, budget); }),
          "mixed arithmetic accepted strict indexed qualification");
  before_mixed.unchanged(owner);
  owner.prepare_density(strict, 3, 0, 1);
  require(!owner.density_provider_->enabled(), "small budget bypassed indexed admission");
  owner.prepare_density(strict, 3, 0, budget);
  require(owner.density_provider_->enabled(), "explicit indexed qualification was lost");
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected case name"); const std::string name = argv[1];
    if (name == "overlap") overlap(false);
    else if (name == "ledger") overlap(true);
    else if (name == "replacement") replacement();
    else if (name == "failures") failures();
    else if (name == "provider_errors") provider_errors();
    else if (name == "mixed") mixed_tables();
    else if (name == "validation") validation();
    else if (name == "empty") empty();
    else if (name == "response") response();
    else if (name == "discovery") discovery();
    else if (name == "batched") batched();
    else if (name == "indexed_qualification") indexed_qualification();
    else throw std::runtime_error("unknown case");
    require(!live_contexts && !live_cache && !live_host, "owner teardown leaked resources");
    std::cout << name << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
"""


def write_cuda_error_stub(directory: Path) -> None:
    """Use production typed exceptions with only the CUDA error-query port doubled."""
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaError_t = int;\n"
        "constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;\n"
        'inline const char* cudaGetErrorString(int) { return "CUDA error"; }\n'
    )


def probe_source(owner: str) -> str:
    return PRELUDE + "\n".join(_definition(owner, sig) for sig in SIGNATURES) + TESTS


@pytest.fixture(scope="module")
def reprepare_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    directory = tmp_path_factory.mktemp("xc-density-repreparation")
    write_cuda_error_stub(directory)
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


FORCE_PRELUDE = r"""
// Device capacity, completion and graph lifetime are boundary ports. Both the
// force-memory transaction and XC retirement/dispatch below are production code.
std::size_t free_capacity{}, other_retained{};
int memory_queries{}, memory_failure_at{}, graph_calls{}, graph_failure_at{};
bool lifetime_violation{};
int cudaMemGetInfo(std::size_t* available, std::size_t* total) {
  if (++memory_queries == memory_failure_at) return 1;
  *total = 2ULL << 30;
  *available = free_capacity - live_cache - live_contexts*provider_allowance - other_retained;
  return 0;
}
struct CudaKsFinalStateToken {
  unsigned version{1};
  std::uint64_t generation{17};
  bool operator==(const CudaKsFinalStateToken&) const = default;
};
struct Region {
  bool live{true};
  int invalidations{};
  void invalidate() {
    if (++graph_calls == graph_failure_at) throw std::bad_alloc();
    live = false;
    ++invalidations;
  }
};
struct ForceOwner {
  int stream{7};
  std::unique_ptr<CudaXcPlan> xc = std::make_unique<CudaXcPlan>();
  std::size_t fast_anchor_arena{}, nonlocal_anchor_arena{}, phase_source{}, phase_device_bytes{};
  struct Resources {
    std::size_t state_device_bytes{4096}, xc_device_bytes{8192}, provider_device_bytes{},
                retained_host_numeric_bytes{1024};
    bool operator==(const Resources&) const = default;
  } resource;
  struct { std::size_t synchronizations{}; } movement;
  Region device_chunk_region, solver_region_executor;
  CudaKsFinalStateToken token() const { return {}; }
  void current_device() const { check(device_failure); }
  std::size_t retire_anchor(std::size_t& bytes) {
    const auto released = bytes;
    other_retained -= bytes; resource.state_device_bytes -= bytes; bytes = 0;
    return released;
  }
  std::size_t retire_nonlocal_anchor_storage() { return retire_anchor(nonlocal_anchor_arena); }
  std::size_t retire_fast_anchor_storage() { return retire_anchor(fast_anchor_arena); }
  void retire_phase_source(bool) {
    device_chunk_region.invalidate(); solver_region_executor.invalidate();
    other_retained -= phase_device_bytes;
    resource.provider_device_bytes -= phase_device_bytes;
    phase_device_bytes = phase_source = 0;
  }
};
ForceOwner* retiring_owner{};
void observe_retirement() {
  lifetime_violation |= !stream_drained || retiring_owner->device_chunk_region.live ||
                        retiring_owner->solver_region_executor.live;
}
struct CudaKsPlan {
  std::unique_ptr<ForceOwner> impl_ = std::make_unique<ForceOwner>();
  generativeqc_status prepare_force_memory(const CudaKsFinalStateToken&, std::size_t,
                                           std::array<std::uint64_t, 4>&, std::string&);
};
"""

FORCE_TESTS = r"""
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
constexpr std::size_t budget = provider_allowance + 2*3*3*sizeof(double);
void prepare(CudaKsPlan& p, std::size_t anchors = 0, std::size_t phase = 0) {
  auto& owner = *p.impl_;
  owner.xc->prepare_density(strict_fp64_precision(), 11, budget);
  owner.xc->evaluation_started_ = true;
  const auto& diag = *owner.xc->density_provider_diagnostic();
  owner.resource.xc_device_bytes += diag.matrix_bytes;
  owner.resource.provider_device_bytes += diag.provider_allowance + phase;
  owner.resource.retained_host_numeric_bytes += diag.host_bytes;
  owner.resource.state_device_bytes += anchors;
  owner.fast_anchor_arena = anchors;
  owner.phase_source = owner.phase_device_bytes = phase;
  other_retained = anchors + phase;
  free_capacity = live_cache + live_contexts*provider_allowance + other_retained + 128;
  retiring_owner = &owner;
  provider_retirement_observer = observe_retirement;
}
void pressure(bool release_prior) {
  CudaKsPlan p; prepare(p, release_prior ? 64 : 0, release_prior ? 64 : 0);
  auto& owner = *p.impl_;
  const auto before = owner.resource;
  const auto token = owner.token();
  std::array<std::uint64_t, 4> receipt;
  std::string detail;
  const std::size_t requested = release_prior ? 257 : 129;
  require(p.prepare_force_memory(token, requested, receipt, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "force-memory pressure was not handled");
  require(!lifetime_violation && !owner.device_chunk_region.live &&
          !owner.solver_region_executor.live, "provider outlived its completion/graph fence");
  require(!owner.xc->density_provider_diagnostic() && !live_contexts && !live_cache && !live_host,
          "force pressure retained optional density resources");
  require(receipt == std::array<std::uint64_t,4>{requested, 128, free_capacity,
                                               budget + (release_prior ? 128 : 0)},
          "force-memory receipt lost retired capacity");
  require(owner.resource.xc_device_bytes == before.xc_device_bytes - 2*3*3*sizeof(double) &&
          owner.resource.provider_device_bytes == 0 &&
          owner.resource.retained_host_numeric_bytes == 1024 &&
          owner.resource.state_device_bytes == 4096, "retirement lost resource accounting");
  require(owner.token() == token && owner.xc->evaluation_started_,
          "retirement changed the final-state generation");
  for (auto phase : {PrecisionPhase::StrictAudit, PrecisionPhase::Admitted}) {
    const auto& binding = owner.xc->density_binding(phase);
    require(!owner.xc->density_execution_provider(phase) &&
            binding.candidate.provider == "generated.cuda" &&
            binding.precision.arithmetic.is_strict_fp64() && binding.replays == 11,
            "future warm execution lost its prepared generated fallback");
  }
  const auto syncs = synchronization_calls, invalidations = graph_calls;
  require(p.prepare_force_memory(token, free_capacity+1, receipt, detail) ==
          GENERATIVEQC_STATUS_SUCCESS && receipt[3] == 0 &&
          synchronization_calls == syncs && graph_calls == invalidations,
          "repeated handoff retired or synchronized again");
  provider_retirement_observer = nullptr;
}
void retained(bool reclaim_anchor, bool disabled) {
  qualified = !disabled;
  CudaKsPlan p; prepare(p, reclaim_anchor ? 64 : 0);
  auto& owner = *p.impl_;
  const auto* provider = owner.xc->density_provider_diagnostic();
  const auto before = owner.resource;
  std::array<std::uint64_t, 4> receipt;
  std::string detail;
  const std::size_t requested = reclaim_anchor || disabled ? 129 : 128;
  require(p.prepare_force_memory(owner.token(), requested, receipt, detail) ==
          GENERATIVEQC_STATUS_SUCCESS, "unpressured handoff failed");
  require(owner.xc->density_provider_diagnostic() == provider &&
          owner.device_chunk_region.live && owner.solver_region_executor.live && !graph_calls,
          "handoff unnecessarily retired or invalidated density provider");
  require(owner.resource.xc_device_bytes == before.xc_device_bytes &&
          owner.resource.provider_device_bytes == before.provider_device_bytes &&
          owner.resource.retained_host_numeric_bytes == before.retained_host_numeric_bytes &&
          receipt[3] == (reclaim_anchor ? 64 : 0) &&
          synchronization_calls == (reclaim_anchor ? 1 : 0),
          "unchanged provider acquired retirement charges");
  provider_retirement_observer = nullptr;
}
void refusals() {
  CudaKsPlan p; prepare(p);
  auto& owner = *p.impl_;
  const auto* provider = owner.xc->density_provider_diagnostic();
  const auto before = owner.resource;
  for (int route = 0; route < 8; ++route) {
    auto token = owner.token();
    if (route == 0) token.version = 2;
    if (route == 1) ++token.generation;
    capture = route == 2;
    query_failure = route == 3;
    device_failure = route == 4;
    memory_failure_at = route == 5 ? memory_queries+1 : 0;
    synchronization_error = route == 6;
    graph_failure_at = route == 7 ? graph_calls+2 : 0;
    std::array<std::uint64_t, 4> receipt{1, 2, 3, 4};
    std::string detail;
    const auto expected = route < 3 ? GENERATIVEQC_STATUS_INVALID_ARGUMENT :
                          route == 7 ? GENERATIVEQC_STATUS_OUT_OF_MEMORY :
                                       GENERATIVEQC_STATUS_CUDA_ERROR;
    require(p.prepare_force_memory(token, 129, receipt, detail) == expected &&
            receipt == std::array<std::uint64_t,4>{} && !detail.empty(),
            "failed handoff lost public status or published a success receipt");
    require(owner.xc->density_provider_diagnostic() == provider && owner.resource == before &&
            owner.token() == CudaKsFinalStateToken{} && !lifetime_violation && live_contexts == 1,
            "failed handoff destroyed the live provider or final-state generation");
  }
  synchronization_error = graph_failure_at = memory_failure_at = 0;
  std::array<std::uint64_t, 4> receipt;
  std::string detail;
  require(p.prepare_force_memory(owner.token(), 129, receipt, detail) ==
          GENERATIVEQC_STATUS_SUCCESS && !lifetime_violation && !live_contexts,
          "handoff could not recover after a failed fence");
  provider_retirement_observer = nullptr;
}
void release_failure(int kind) {
  CudaKsPlan p; prepare(p);
  auto& owner = *p.impl_;
  const auto* provider = owner.xc->density_provider_diagnostic();
  const auto* binding = owner.xc->provider_density_binding_.get();
  const auto before = owner.resource;
  const auto token = owner.token();
  std::array<std::uint64_t, 4> receipt{1, 2, 3, 4};
  std::string detail;
  release_error_kind = kind;
  const auto expected = kind == 1 ? GENERATIVEQC_STATUS_OUT_OF_MEMORY :
                                    GENERATIVEQC_STATUS_CUDA_ERROR;
  require(p.prepare_force_memory(token, 129, receipt, detail) == expected &&
          receipt == std::array<std::uint64_t, 4>{} && !detail.empty(),
          "failed live release admitted success or lost its public status");
  require(owner.xc->density_provider_diagnostic() == provider &&
          owner.xc->provider_density_binding_.get() == binding &&
          owner.resource == before && owner.token() == token && live_contexts == 1 &&
          owner.xc->density_provider_retirement_pending() && !lifetime_violation,
          "failed live release lost its owner, charges or final-state generation");
  require(!owner.xc->density_execution_provider(PrecisionPhase::StrictAudit) &&
          owner.xc->density_binding(PrecisionPhase::StrictAudit).candidate.provider ==
              "generated.cuda" &&
          owner.xc->density_binding(PrecisionPhase::Admitted).candidate.provider ==
              "generated.cuda",
          "partial release retained an executable provider binding");
  // Partial matrix release can already satisfy the original request. The same
  // owner must still finish checked teardown before publishing a success receipt.
  release_error_kind = 0;
  const std::size_t retry_request = kind == 3 ? 129 : 0;
  require(p.prepare_force_memory(token, retry_request, receipt, detail) ==
              GENERATIVEQC_STATUS_SUCCESS && detail.empty() && !lifetime_violation &&
          !owner.xc->density_provider_diagnostic() &&
          !owner.xc->density_provider_retirement_pending() && !live_contexts && !live_cache &&
          !live_host && receipt[3] == budget && owner.resource.provider_device_bytes == 0 &&
          owner.resource.xc_device_bytes == 8192 &&
          owner.resource.retained_host_numeric_bytes == 1024 && owner.token() == token,
          "failed live release could not finish with exact reserved charges");
  provider_retirement_observer = nullptr;
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("one scenario required");
    const std::string name = argv[1];
    if (name == "pressure") pressure(false);
    else if (name == "prior_pressure") pressure(true);
    else if (name == "no_pressure") retained(false, false);
    else if (name == "anchor_sufficient") retained(true, false);
    else if (name == "disabled") retained(false, true);
    else if (name == "refusals") refusals();
    else if (name == "release_allocation") release_failure(1);
    else if (name == "release_runtime") release_failure(2);
    else if (name == "release_partial") release_failure(3);
    else throw std::runtime_error("unknown scenario");
    require(!live_contexts && !live_cache && !live_host, "force owner leaked resources");
    std::cout << name << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
"""


@pytest.fixture(scope="module")
def force_memory_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    directory = tmp_path_factory.mktemp("xc-force-memory")
    write_cuda_error_stub(directory)
    owner = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    handoff = _definition(
        (ROOT / "src/dft/cuda_ks.cpp").read_text(),
        "generativeqc_status CudaKsPlan::prepare_force_memory(",
    )
    source = directory / "force.cpp"
    source.write_text(
        PRELUDE
        + "\n".join(_definition(owner, sig) for sig in SIGNATURES)
        + FORCE_PRELUDE
        + handoff
        + FORCE_TESTS
    )
    binary = directory / "force"
    compile_owner(compiler, directory, [source], binary)
    return binary


@pytest.mark.parametrize(
    "case",
    (
        "pressure",
        "prior_pressure",
        "no_pressure",
        "anchor_sufficient",
        "disabled",
        "refusals",
        "release_allocation",
        "release_runtime",
        "release_partial",
    ),
)
def test_force_memory_density_lifetime(force_memory_probe: Path, case: str) -> None:
    result = subprocess.run(
        [str(force_memory_probe), case],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
