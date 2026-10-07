"""Execute native AUTO/DF/SolverRegion integration guards without a CUDA device."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _span(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def _run(tmp_path: Path, source: str, native_cxx: object) -> None:
    unit, executable = tmp_path / "integration.cpp", tmp_path / "integration"
    unit.write_text(source)
    native_cxx.build_executable(
        [unit],
        executable,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        compile_timeout=45,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_pbe0_chunks_and_semilocal_replay_exclude_each_auto_component(
    tmp_path: Path,
    native_cxx: object,
) -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    chunk = _span(
        source, "    const bool pure_semilocal_chunk", "    if (device_chunk_mode)"
    )
    replay = _span(source, "    const bool replay_point_program", "    auto graph =")
    _run(
        tmp_path,
        r"""
#include <cassert>
#include <vector>
#include "dft/semilocal_family.hpp"
#include "dft/xc_capabilities.hpp"
#include "scf/types.hpp"
#include "runtime/execution_precision.hpp"
using namespace generativeqc;
using namespace generativeqc::dft;
namespace generativeqc::scf::cuda_execution {
constexpr std::size_t kSmallEigensolverLimit = 32;
}
constexpr unsigned kCudaKsChunkCapacity = 2;
bool is_semilocal_family(std::uint32_t code, SemilocalFamily family) {
  return code == semilocal_family_code(family);
}
"""
        + r"""
runtime::ExecutionPrecisionSchedule test_precision_schedule(bool coulomb, bool density) {
  runtime::ExecutionPrecisionSchedule schedule;
  schedule.add_region("test.coulomb",
                      coulomb ? runtime::fp32_compute_fp64_accumulation("test/coulomb")
                              : runtime::strict_fp64_precision());
  schedule.add_region("test.density",
                      density ? runtime::fp32_compute_fp64_accumulation("test/density")
                              : runtime::strict_fp64_precision());
  return schedule;
}
struct Owner {
  scf::ScfOptions options;
  runtime::ExecutionPrecisionSchedule precision_schedule;
  struct { CudaXcFastPathCapabilities fast_paths; } xc_layout;
  bool has_exchange{}, has_range_correction{}, fitted_coulomb{}, device_chunk_mode{};
  bool device_nonlocal{}, incremental_direct_jk{};
  void* nonlocal_correlation{};
  std::optional<scf::ResolvedFockBuild> range_correction;
  unsigned spins{1}, functional{semilocal_family_code(SemilocalFamily::Pbe)}, width{2};
  double exchange_coefficient{-0.125};
  std::size_t n{8};
  struct Provider {
    struct System { std::vector<int> ecp_terms; } value;
    const System& system() const { return value; }
  } provider;
  Owner() {
    options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::DeviceFused;
    xc_layout.fast_paths.graph_replay = CudaXcCapability::Qualified;
  }
  unsigned configured_chunk_width() const { return width; }
  bool configured_replay_enabled() const { return true; }
  bool chunk() {
"""
        + chunk
        + "    return device_chunk_mode;\n  }\n  bool replay() {\n"
        + replay
        + r"""
    return replay;
  }
};
int main() {
  for (bool coulomb : {false, true}) for (bool density : {false, true})
    for (bool hybrid : {false, true}) {
      Owner p;
      p.precision_schedule = test_precision_schedule(coulomb, density);
      p.has_exchange = hybrid;
      p.options.semilocal_exchange_scale = hybrid ? 0.75 : 1.0;
      assert(p.chunk() == (!coulomb && !density));
      assert(p.replay() == (!coulomb && !density && !hybrid));
      p.incremental_direct_jk = true;
      assert(!p.chunk());
      p.incremental_direct_jk = false;
      p.options.semilocal_exchange_scale = 0.73;
      assert(!p.chunk() && !p.replay());
    }
  for (auto capability : {CudaXcCapability::Unavailable,
                          CudaXcCapability::QualificationRequired,
                          CudaXcCapability::Qualified}) {
    Owner p;
    p.xc_layout.fast_paths.graph_replay = capability;
    assert(p.chunk());
    assert(p.replay() == (capability == CudaXcCapability::Qualified));
  }
  for (int excluded = 0; excluded != 7; ++excluded) {
    Owner p;
    if (excluded == 0) p.fitted_coulomb = true;
    if (excluded == 1) {
      // A correction-only model must not enter through the semilocal arm.
      p.has_range_correction = true;
      p.range_correction.emplace();
      p.range_correction->backend = scf::FockBackend::Cuda;
      p.range_correction->spec.derivative_order = 0;
      p.range_correction->spec.coulomb.present = false;
      p.range_correction->spec.exchange.op = scf::FockOperator::LongRange;
      p.range_correction->spec.exchange.omega = 0.3;
    }
    if (excluded == 2) p.nonlocal_correlation = &p;
    if (excluded == 3) p.spins = 2;
    if (excluded == 4) p.provider.value.ecp_terms.push_back(1);
    if (excluded == 5) p.width = 1;
    if (excluded == 6)
      p.options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::HostUnfused;
    assert(!p.chunk());
    if (excluded < 3) assert(!p.replay());
  }
  for (bool coulomb : {false, true}) for (bool density : {false, true}) {
    Owner rsh;
    rsh.has_exchange = rsh.has_range_correction = true;
    rsh.range_correction.emplace();
    rsh.range_correction->backend = scf::FockBackend::Cuda;
    rsh.range_correction->spec.derivative_order = 0;
    rsh.range_correction->spec.coulomb.present = false;
    rsh.range_correction->spec.exchange.op = scf::FockOperator::LongRange;
    rsh.range_correction->spec.exchange.omega = 0.3;
    rsh.precision_schedule = test_precision_schedule(coulomb, density);
    assert(rsh.chunk() == (!coulomb && !density));
    assert(!rsh.replay());
    rsh.incremental_direct_jk = true;
    assert(!rsh.chunk());
    rsh.incremental_direct_jk = false;
    rsh.nonlocal_correlation = &rsh;
    assert(!rsh.chunk());
    rsh.device_nonlocal = true;
    assert(rsh.chunk() == (!coulomb && !density));
    assert(!rsh.replay());
    rsh.range_correction.reset();
    assert(!rsh.chunk());
  }
  Owner large;
  large.n = 33;
  assert(!large.replay());
}
""",
        native_cxx,
    )


def test_native_global_hybrid_admission_combines_direct_auto_and_strict_df(
    tmp_path: Path,
    native_cxx: object,
) -> None:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    admission = _span(
        source, "  const bool cuda_global_hybrid", "  bool cuda_split_hybrid"
    )
    _run(
        tmp_path,
        r"""
#include <cassert>
#include "generativeqc/generativeqc.h"
#include "dft/semilocal_family.hpp"
#include "scf/types.hpp"
using namespace generativeqc;
bool admitted(bool fitted, bool automatic, bool unrestricted, bool b3lyp,
              unsigned excluded = 0) {
  const auto backend = excluded == 1 ? GENERATIVEQC_BACKEND_CPU_REFERENCE
                                     : GENERATIVEQC_BACKEND_CUDA;
  struct Plan {
    bool range_exchange{}, nonlocal_correlation{};
    dft::SemilocalFamily semilocal_family;
  } execution_plan{excluded == 2, excluded == 3,
                   b3lyp ? dft::SemilocalFamily::B3lyp : dft::SemilocalFamily::Pbe};
  const auto& semilocal_metadata = dft::semilocal_family_metadata(execution_plan.semilocal_family);
  scf::ScfOptions options;
  options.precision_mode = automatic ? GENERATIVEQC_PRECISION_AUTO : GENERATIVEQC_PRECISION_FP64;
  options.density_fitting_mode = fitted ? GENERATIVEQC_DENSITY_FITTING_AUTO
                                       : GENERATIVEQC_DENSITY_FITTING_NONE;
  options.semilocal_exchange_scale = b3lyp ? 1.0 : 0.75;
  options.semilocal_correlation_scale = 1.0;
  scf::FockBuildSpec fock;
  fock.spin = unrestricted ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  fock.exchange.present = excluded != 4;
  fock.exchange.coefficient = (b3lyp ? -0.1 : -0.125) * (unrestricted ? 2 : 1);
  if (excluded == 5) fock.exchange.coefficient = -0.3;
  if (excluded == 6) options.semilocal_exchange_scale = 0.7;
  if (excluded == 7) options.semilocal_correlation_scale = 0.9;
"""
        + admission
        + r"""
  return cuda_curated_global_hybrid;
}
int main() {
  for (bool fitted : {false, true}) for (bool automatic : {false, true})
    for (bool unrestricted : {false, true}) for (bool b3lyp : {false, true}) {
      assert(admitted(fitted, automatic, unrestricted, b3lyp) == !(fitted && automatic));
      for (unsigned excluded = 1; excluded != 8; ++excluded)
        assert(!admitted(fitted, automatic, unrestricted, b3lyp, excluded));
    }
}
""",
        native_cxx,
    )
