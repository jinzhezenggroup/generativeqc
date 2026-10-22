"""CPU shared-library source contracts, not installed CUDA/scientific qualification.

The external client includes the scalar contract only, never native observer TLS.
Synthetic prepared plans test protocol delivery, not real CUDA resource admission.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="ELF/dlopen CC source-observer contract"
)

BRIDGE = r"""

#include "runtime/residency_observer.hpp"
extern "C" int generativeqc_residency_observer_bind_v1(generativeqc::runtime::ResidencyObserver cb,
                                                       void* ctx) {
  return generativeqc::runtime::bind_residency_observer(cb, ctx);
}
extern "C" int generativeqc_residency_observer_unbind_v1(
    generativeqc::runtime::ResidencyObserver cb, void* ctx, std::uint64_t* e) {
  return generativeqc::runtime::unbind_residency_observer(cb, ctx, e);
}
"""
EMITTER = r"""

#include <stdexcept>

#include "cc/phase_observation.hpp"
#include "cc/replay_tiles.hpp"
namespace phase = generativeqc::cc::phase_observation;
extern "C" int generativeqc_cc_phase_enable_v1(std::uint64_t version, phase::Observer cb, void* ctx,
                                               std::uint64_t* permit) {
  return phase::enable(version, cb, ctx, permit);
}
extern "C" int generativeqc_cc_phase_finish_v1(std::uint64_t permit, std::uint64_t* errors) {
  return phase::finish(permit, errors);
}
extern "C" void fixture_controlled_source(int mode, void (*between)()) {
  phase::Scope root(phase::Site::endpoint_inclusive);
  {
    phase::Scope tile(phase::Site::q_tile_inclusive, 4, 8);
    if (between) between();
    if (mode == 1) throw std::runtime_error("scientific exception preserved");
  }
}
extern "C" std::uint64_t fixture_generation() { return phase::source_generation.load(); }
extern "C" std::uint64_t fixture_binding_epoch() {
  return generativeqc::runtime::residency_binding_epoch.value;
}
extern "C" bool fixture_binding_exhausted() {
  return generativeqc::runtime::residency_binding_epoch.exhausted;
}
extern "C" void fixture_set_epoch(std::uint64_t value) {
  generativeqc::runtime::residency_binding_epoch = {value, false};
}
extern "C" void fixture_exhaust_source_ids() {
  phase::source_generation.store(std::numeric_limits<std::uint64_t>::max());
}
extern "C" void fixture_plan(int mode) {
  phase::Scope root(phase::Site::endpoint_inclusive);
  phase::Scope solve(phase::Site::ccsd_solve_inclusive);
  // CPU protocol double only: the synthetic prepared plan is read after the
  // fallback has happened. This does not qualify real Owner capacity selection.
  unsigned replay_batch = 8;
  {
    phase::Scope setup(phase::Site::setup_inclusive);
    // Optional prepare/allocation refusal leaves the real stored sentinel zero.
    replay_batch = (mode == 1 || mode == 2) ? 0 : 4;
  }
  phase::admitted_plan(phase::PlanOwner::ccsd, 488, 8,
                        generativeqc::cc::detail::replay_tile_width(488,replay_batch,true),
                       phase::hoisted | phase::matrix | phase::replay_matrix, 8000000, 4000000,
                       3000000);
  {
    phase::Scope iteration(phase::Site::q_iteration_inclusive, 0, 488);
    phase::Scope tile(phase::Site::q_tile_inclusive, 0, 8);
    {
      phase::Scope host(phase::Site::q_host_preparation, 0, 8);
    }
    phase::Scope production(phase::Site::q_production_submission, 0, 8);
    const auto producer_id = production.identity();
    production.close();
    {
      phase::Scope accumulation(phase::Site::q_accumulation_submission, 0, 8, 0, producer_id);
    }
  }
}
extern "C" void fixture_details(bool unknown) {
  phase::Scope root(phase::Site::endpoint_inclusive);
  if (unknown) { phase::Scope invalid(static_cast<phase::Site>(999)); return; }
  { phase::Scope source(phase::Site::source_inclusive, 230, 488);
    phase::Scope production(phase::Site::source_gpu_submission, 0, 488); }
  { phase::Scope triples(phase::Site::triples_inclusive, 9, 221);
    phase::admitted_triples(9,221,488,3,4096,0,64,64,64,8000000,4000000);
    phase::Scope tile(phase::Site::triples_tile_inclusive,5,171);
    phase::Scope panel(phase::Site::triples_panel_submission,2,0);
    const auto producer=panel.identity();panel.close();
    phase::Scope moment(phase::Site::triples_moment_submission,0,1,0,producer); }
}
extern "C" std::uint64_t fixture_late_panel_cache(void (*enable_inside)()) {
  phase::PanelSources sources;
  enable_inside();
  return sources.dependency(0); // Existing logical cache hit, no observed producer.
}
extern "C" std::uint64_t fixture_panel_cache(bool remember) {
  static phase::PanelSources sources;
  if (remember) sources.remember(0, 123);
  return sources.dependency(0);
}
extern "C" std::uint64_t fixture_panel_bounds() {
  phase::PanelSources sources;
  return sources.dependency(3);
}
extern "C" void fixture_replay_rule() {
  using generativeqc::cc::detail::replay_tile_width;
  static_assert(replay_tile_width(0,0,true)==0);
  static_assert(replay_tile_width(13,0,true)==1);
  static_assert(replay_tile_width(13,4,true)==4);
  static_assert(replay_tile_width(3,8,true)==3);
  for (auto stored : {0u,4u}) {
    std::size_t offset=0,tiles=0;
    const std::size_t expected_batched[4]{4,4,4,1};
    while(offset<13) {
      const auto width=replay_tile_width(13-offset,stored,true);
      if(width!=(stored?expected_batched[tiles]:1))throw std::runtime_error("actual replay rule mismatch");
      offset+=width;++tiles;
    }
    if(tiles!=(stored?4u:13u))throw std::runtime_error("actual replay traversal mismatch");
  }
  if(replay_tile_width(13,4,false)!=1)throw std::runtime_error("non-physical scalar route changed");
}
extern "C" void fixture_no_q_projection() {
  phase::Scope root(phase::Site::endpoint_inclusive);
  phase::Scope solve(phase::Site::ccsd_solve_inclusive);
  phase::admitted_plan(phase::PlanOwner::ccsd,0,1,
      generativeqc::cc::detail::replay_tile_width(0,0,true),
      phase::conventional,8000000,4000000,3000000);
}
"""
CLIENT = r"""

#include <dlfcn.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "cc/phase_observation_contract.hpp"
using Callback = generativeqc::cc::phase_observation::Observer;
using Bind = int (*)(Callback, void*);
using Unbind = int (*)(Callback, void*, std::uint64_t*);
namespace {
using Enable = int (*)(std::uint64_t, Callback, void*, std::uint64_t*);
using Finish = int (*)(std::uint64_t, std::uint64_t*);
Bind bind;
Unbind unbind;
Enable enable;
Finish finish;
std::uint64_t (*binding_epoch)();
bool mutate_in_callback = false;
unsigned plan_seen = 0;
unsigned expected_replay = 0;
std::uint64_t plan_id = 0, producer_id = 0;
std::uint64_t end_statuses[16]{};
unsigned end_count = 0;
unsigned events = 0;
unsigned triples_admissions = 0, detailed_sites = 0;
std::uint64_t triples_plan = 0;
std::uint64_t last_admitted_auxiliary=999,last_admitted_replay=999;
std::uint64_t token = 0;
int context = 1, newcontext = 2;
bool busy_rejected = false;
void require(bool x, const char* s) {
  if (!x) throw std::runtime_error(s);
}
void callback(const std::uint64_t* v, std::size_t n, void*) {
  ++events;
  if(n==14 && v[0]==0x434350410001ULL){last_admitted_auxiliary=v[5];last_admitted_replay=v[7];}
  if (n == 17 && v[0] == 0x434354410001ULL) {
    require(v[4]==9 && v[5]==221 && v[6]==488 && v[7]==3 && v[8]==4096 &&
            v[10]==64 && v[11]==64 && v[12]==64 && v[13]==8000000 &&
            v[14]==4000000 && v[15]==token && v[16]==0, "typed reserved triples plan");
    ++triples_admissions;triples_plan=v[2];
  }
  if (n == 12 && v[0] == 0x434350480001ULL && v[4] == 28) {
    require(v[5]==5 && v[6]==171 && v[7]==triples_plan && v[10]==token,
            "canonical triples tuple and admitted plan");
    ++detailed_sites;
  }
  if (n == 12 && v[0] == 0x434350480001ULL && v[1] == 2) {
    require(end_count < 16, "bounded end-status fixture");
    end_statuses[end_count++] = v[11];
  }
  if (expected_replay) {
    if (v[0] == 0x434350410001ULL) {
      require(n == 14 && v[5] == 488 && v[6] == 8 && v[7] == expected_replay && v[8] == 7 &&
                  v[9] == 8000000 && v[10] == 4000000 && v[11] == 3000000 && v[12] == token,
              "actual supplied plan facts");
      ++plan_seen;
      plan_id = v[2];
    } else if (v[4] >= 14 && v[4] <= 15)
      require(n == 12 && v[7] == plan_id && v[10] == token, "phase pinned to admitted plan");
    else if (v[4] == 6 && v[1] == 1)
      producer_id = v[2];
    else if (v[4] == 7)
      require(v[8] == producer_id && producer_id != 0,
              "actual source dependency retained after close");
  }
  if (mutate_in_callback) {
    auto before = binding_epoch();
    std::uint64_t e = 0;
    require(bind(callback, &context) != 0, "reentrant bind rejected");
    require(unbind(callback, &context, &e) != 0, "reentrant unbind rejected");
    require(binding_epoch() == before, "failed reentrant mutation not advanced");
  }
}
void throwing(const std::uint64_t*, std::size_t, void*) {
  throw std::runtime_error("observer failed");
}
void busy() {
  std::uint64_t e = 0;
  busy_rejected = finish(token, &e) != 0;
}
void aba() {
  std::uint64_t e = 0;
  require(unbind(callback, &context, &e) == 0, "ABA detach");
  require(bind(callback, &context) == 0, "ABA rebind");
}
void replacement() {
  std::uint64_t e = 0;
  require(unbind(callback, &context, &e) == 0, "replace detach");
  require(bind(callback, &newcontext) == 0, "replace rebind");
}
void enable_detailed_inside() {
  require(enable(generativeqc::cc::phase_observation::kDetailedContract,
                 callback, &context, &token) == 0, "late explicit permit");
}
template <class F>
F symbol(void* l, const char* n) {
  auto p = dlsym(l, n);
  require(p, n);
  return reinterpret_cast<F>(p);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    void* l = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    require(l, dlerror());
    bind = symbol<Bind>(l, "generativeqc_residency_observer_bind_v1");
    unbind = symbol<Unbind>(l, "generativeqc_residency_observer_unbind_v1");
    enable = symbol<Enable>(l, "generativeqc_cc_phase_enable_v1");
    finish = symbol<Finish>(l, "generativeqc_cc_phase_finish_v1");
    auto emit = symbol<void (*)(int, void (*)())>(l, "fixture_controlled_source");
    auto generation = symbol<std::uint64_t (*)()>(l, "fixture_generation");
    binding_epoch = symbol<std::uint64_t (*)()>(l, "fixture_binding_epoch");
    std::string mode = argv[2];
    std::uint64_t errors = 0;
    constexpr auto contract = 0x434350480001ULL;
    require(enable(contract, callback, &context, &token) != 0,
            "cannot enable without actual binding");
    if (mode == "null-context") {
      require(bind(callback, nullptr) == 0, "legacy null context");
      require(enable(contract, callback, nullptr, &token) == 0, "explicit null-context owner");
      emit(0, nullptr);
      require(events == 4, "null-context delivery");
      require(finish(token, &errors) == 0 && errors == 0, "null-context finish");
      require(unbind(callback, nullptr, &errors) == 0, "null-context detach");
      require(dlclose(l) == 0, "null-context module lifetime");
      return 0;
    }
    require(bind(callback, &context) == 0, "bind");
    auto binding_before = binding_epoch();
    require(binding_before > 0, "successful bind advances epoch");
    require(bind(callback, &context) != 0, "failed replacement");
    require(unbind(callback, &newcontext, &errors) != 0, "failed foreign context");
    require(binding_epoch() == binding_before, "failed operations do not advance");
    auto before = generation();
    emit(0, nullptr);
    require(events == 0 && generation() == before,
            "legacy binding no new layout or generation work");
    require(enable(contract + 1, callback, &context, &token) != 0, "unknown diagnostic contract");
    require(enable(contract, callback, &newcontext, &token) != 0, "foreign live owner");
    require(enable(contract, callback, &context, &token) == 0 && token, "explicit contract enable");
    if (mode == "admission" || mode == "fallback-admission" || mode=="allocation-fallback-admission") {
      expected_replay = mode == "admission" ? 4 : 1;
      symbol<void (*)(int)>(l, "fixture_plan")(mode == "admission" ? 0 : mode=="fallback-admission"?1:2);
      symbol<void (*)()>(l,"fixture_replay_rule")();
      require(plan_seen == 1 && events == 17, "admission after setup and phase inventory");
      require(finish(token, &errors) == 0 && errors == 0, "admission complete control lifetime");
    } else if (mode == "normal") {
      mutate_in_callback = true;
      emit(0, busy);
      require(events == 4 && busy_rejected, "busy finish rejected");
      require(end_count == 2 && end_statuses[0] == 0 && end_statuses[1] == 0, "normal ends have zero status");
      require(finish(token, &errors) == 0 && errors == 0, "complete control lifecycle");
    } else if (mode == "aba" || mode == "replace") {
      emit(0, mode == "aba" ? aba : replacement);
      require(events == 2, "old permission not reused after binding epoch change");
      require(finish(token, &errors) == 0 && errors != 0, "ABA/replacement sticky incomplete");
      if (mode == "replace") {
        require(unbind(callback, &newcontext, &errors) == 0, "new owner cleanup");
        require(bind(callback, &context) == 0, "restore owner");
      }
    } else if (mode == "thread") {
      auto n = generation();
      std::thread t([&] {
        std::uint64_t other = 0;
        require(enable(contract, callback, &context, &other) != 0, "foreign TLS enable");
        emit(0, nullptr);
      });
      t.join();
      require(events == 0 && generation() == n, "foreign thread no inherited permission");
      emit(0, nullptr);
      require(finish(token, &errors) == 0 && errors == 0 && events == 4, "owner remains live");
    } else if (mode == "exception") {
      bool caught = false;
      try {
        emit(1, nullptr);
      } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "scientific exception preserved";
      }
      require(caught && events == 4, "native exception and end delivery");
      require(end_count == 2 && end_statuses[0] != 0 && end_statuses[1] != 0, "both exception ends retain status");
      require(finish(token, &errors) == 0 && errors == 0, "unwind finished");
    } else if (mode == "exhaustion") {
      symbol<void (*)(std::uint64_t)>(l, "fixture_set_epoch")(~std::uint64_t{0} - 1);
      require(unbind(callback, &context, &errors) == 0 && binding_epoch() == ~std::uint64_t{0},
              "legacy detach at max");
      require(bind(callback, &context) == 0 && binding_epoch() == ~std::uint64_t{0},
              "legacy bind remains allowed, no wrap");
      require(symbol<bool (*)()>(l, "fixture_binding_exhausted")(), "sticky exhausted");
      emit(0, nullptr);
      require(events == 0, "exhausted permission not used");
      require(finish(token, &errors) == 0 && errors != 0, "exhausted trace incomplete");
      require(enable(contract, callback, &context, &token) != 0,
              "new CC admission refused at saturation");
    } else if (mode == "legacy-details" || mode == "detailed" || mode == "unknown-site") {
      if (mode != "legacy-details") {
        require(finish(token,&errors)==0 && errors==0,"finish legacy view");
        require(enable(generativeqc::cc::phase_observation::kDetailedContract,
                       callback,&context,&token)==0,"explicit detailed v2 contract");
      }
      symbol<void (*)(bool)>(l,"fixture_details")(mode=="unknown-site");
      if(mode=="legacy-details")require(events==2 && triples_admissions==0 && detailed_sites==0,
                                         "v1 never receives new layouts/sites");
      else if(mode=="detailed")require(triples_admissions==1 && detailed_sites==2,"v2 detailed delivery");
      require(finish(token,&errors)==0 && (errors!=0)==(mode=="unknown-site"),
              "unknown site fails closed without scientific exception");
    } else if (mode == "late-panel-cache") {
      require(finish(token,&errors)==0 && errors==0,"finish preexisting view");
      require(symbol<std::uint64_t (*)(void (*)())>(l,"fixture_late_panel_cache")(
                  enable_detailed_inside)==0,"late cache hit is explicitly unobserved, not uninitialized");
      auto cache=symbol<std::uint64_t (*)(bool)>(l,"fixture_panel_cache");
      require(cache(true)==123,"same permit diagnostic producer");
      require(finish(token,&errors)==0 && errors==0,"finish first permit");
      enable_detailed_inside();
      require(cache(false)==0,"new permit cannot inherit cached source identity");
      require(finish(token,&errors)==0 && errors==0,"finish new permit");
    } else if (mode == "panel-bounds") {
      require(symbol<std::uint64_t (*)()>(l,"fixture_panel_bounds")()==0,
              "invalid diagnostic slot returns unknown");
      require(finish(token,&errors)==0 && errors!=0,"bounds failure remains incomplete");
    } else if(mode=="no-q-admission") {
      symbol<void (*)()>(l,"fixture_no_q_projection")();
      require(last_admitted_auxiliary==0 && last_admitted_replay==0,"no-Q stays absent, never scalar width1");
      require(finish(token,&errors)==0 && errors==0,"no-Q observation closes");
    } else if (mode == "source-id-exhaustion") {
      symbol<void (*)()>(l, "fixture_exhaust_source_ids")();
      emit(0, nullptr);
      require(events == 0, "no zero or wrapped source IDs emitted");
      require(finish(token, &errors) == 0 && errors != 0, "source ID exhaustion incomplete");
    } else if (mode == "callback-throws") {
      require(finish(token, &errors) == 0, "finish prior control");
      require(unbind(callback, &context, &errors) == 0, "detach prior callback");
      require(bind(throwing, &context) == 0, "throwing callback bind");
      require(enable(contract, throwing, &context, &token) == 0, "enable throwing callback");
      emit(0, nullptr);
      require(finish(token, &errors) == 0 && errors != 0, "finish throwing capture");
      require(unbind(throwing, &context, &errors) == 0 && errors == 4,
              "actual C ABI reports callback errors");
      require(bind(callback, &context) == 0, "restore");
    } else
      return 2;
    require(unbind(callback, &context, &errors) == 0, "detach before unload");
    before = generation();
    emit(0, nullptr);
    require(generation() == before, "disabled after finish/detach");
    require(dlclose(l) == 0, "unload");
    std::cout << "PASS " << mode << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def phase_probe(
    required_native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("cc-phase-dso")
    inputs = {}
    for name, value in (("bridge", BRIDGE), ("emitter", EMITTER), ("client", CLIENT)):
        path = folder / (name + ".cpp")
        path.write_text(value)
        inputs[name] = path
    library = folder / "libphase.so"
    executable = folder / "client"
    args = ("-std=c++20", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'src'}")
    required_native_cxx.build_shared(
        [inputs["bridge"], inputs["emitter"]],
        library,
        compile_args=args,
        link_args=("-pthread",),
    )
    required_native_cxx.build_executable(
        [inputs["client"]],
        executable,
        compile_args=args,
        link_args=("-pthread", "-ldl"),
    )
    return executable, library


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "aba",
        "replace",
        "thread",
        "exception",
        "callback-throws",
        "exhaustion",
        "admission",
        "fallback-admission",
        "allocation-fallback-admission",
        "null-context",
        "source-id-exhaustion",
        "legacy-details",
        "detailed",
        "unknown-site",
        "late-panel-cache",
        "panel-bounds",
        "no-q-admission",
    ],
)
def test_cc_source_control_and_admission(
    phase_probe: tuple[Path, Path], mode: str
) -> None:
    executable, library = phase_probe
    subprocess.run(
        [str(executable), str(library), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
