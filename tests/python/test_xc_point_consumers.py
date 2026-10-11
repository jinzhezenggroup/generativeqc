"""Execute the emitted bounded point selector without loading CUDA or the runtime."""

import os
import subprocess
from pathlib import Path

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.ao_cuda import (
    emit_grid_source,
    emit_native_xc_contraction_kernels,
    emit_native_xc_point_dispatch,
)
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_device_point_value,
    emit_point_traversal,
)

from tools.generate_xc_split_hybrid_registry import emit_registry

ROOT = Path(__file__).resolve().parents[2]


def test_shared_traversal_forwards_static_family(
    tmp_path: Path, native_cxx: object
) -> None:
    """Execute the emitted adapter through physical, response and ragged batches.

    Host CUDA builtins and point algebra are stubs; the actual generated adapter
    and traversal carry the family key and publish compact coefficients. This
    checks C++ composition without claiming device or numerical qualification.
    """
    emitted = emit_native_xc_contraction_kernels()
    start = emitted.index("template <I static_family>\nstruct CuratedPointEvaluator")
    adapter = emitted[start : emitted.index(emit_point_traversal(), start)]
    source = tmp_path / "family-traversal.cpp"
    source.write_text(
        "#include <algorithm>\n#include <cstddef>\n#include <cmath>\n"
        "#define __device__\n#define __forceinline__ inline\n"
        "using I = int; using std::min;\n"
        "struct { I x; } blockIdx{0}, blockDim{1}, threadIdx{0}, gridDim{1};\n"
        "void atomicCAS(int* error, int old, int value) { if (*error == old) *error = value; }\n"
        "double finite(double value, int* error, int code) {\n"
        "  if (!std::isfinite(value)) *error = code; return value;\n}\n"
        + emit_device_point_value()
        + r"""
int selected_family = -99, physical_calls = 0, response_calls = 0;
template <I family> DevicePointValue evaluate_semilocal_point(
    I, const double*, const double (*)[3], const double*, double, double) {
  selected_family = family; ++physical_calls;
  DevicePointValue value;
  value.energy = 7; value.rho[0] = 9; value.rho[1] = 11;
  return value;
}
namespace cuda_xc_detail {
template <I family> DevicePointValue response_point(
    const double*, const double*, I, I, I, I, I) {
  selected_family = family; ++response_calls;
  DevicePointValue value;
  value.energy = 13; value.rho[0] = 15; value.rho[1] = 17;
  return value;
}
}
"""
        + adapter
        + emit_point_traversal()
        + r"""
template <I family, bool response, bool batched> int check(I spins) {
  constexpr I points = 5;
  double features[40], coefficients[40]{}, totals[15]{}, delta[40]{};
  double weights[points]{1, 1, 1, 1, 1};
  std::fill_n(features, 40, 2.0);
  int error = 0;
  physical_calls = response_calls = 0;
  evaluate_point_tiles<4, response, batched, CuratedPointEvaluator<family>>(
      features, weights, points, spins, coefficients, totals, &error, 1, 1, 1,
      delta, batched ? 3 : 0);
  if (error || selected_family != family) return 1;
  if (physical_calls != (response ? 0 : points) ||
      response_calls != (response ? points : 0)) return 2;
  for (I begin = 0; begin < points; begin += batched ? 3 : points) {
    const I count = batched ? min(3, points - begin) : points;
    for (I p = 0; p < count; ++p) {
      if (totals[3 * begin + p] != (response ? 13 : 7)) return 3;
      for (I spin = 0; spin < spins; ++spin) {
        const double expected = spins == 1 ? (response ? 16 : 10)
                                           : (response ? 15 : 9) + 2 * spin;
        if (coefficients[begin * spins * 4 + spin * 4 * count + p] != expected)
          return 4;
      }
    }
  }
  return 0;
}
int main() {
  for (I spins : {1, 2}) {
    if (check<-1, false, false>(spins) || check<1, false, false>(spins) ||
        check<-1, true, false>(spins) || check<1, true, false>(spins) ||
        check<-1, false, true>(spins) || check<1, false, true>(spins)) return 1;
  }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "family-traversal"
    native_cxx.build_executable([source], binary, compile_args=("-std=c++17", "-O2"))
    subprocess.run([str(binary)], check=True)


def test_admitted_point_consumers(tmp_path: Path, native_cxx: object) -> None:
    """Admitted legacy and split keys resolve; unsupported keys must fail.

    Stub launchers record template arguments, so this tests the emitted host
    dispatch itself without duplicating its conditional implementation in Python.
    Generated MGGA codes share a launcher keyed by ingredient shape; the
    functional code remains separate layout data rather than a template key.
    Spin, AO precision and point counts are deliberately absent from the AOT key:
    they remain validated data/layout arguments rather than extra code variants.
    """
    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(
        emit_registry(), encoding="utf-8"
    )
    emitted = emit_native_xc_contraction_kernels()
    batch_start = emitted.index(
        "CudaXcPointBatchLauncher resolve_point_batch_launcher("
    )
    batch_end = emitted.index("bool compact_point_batch_admitted(", batch_start)
    batch_dispatch = emitted[batch_start:batch_end]
    source = tmp_path / "dispatch.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cstdlib>\n#include <stdexcept>\n#include <limits>\n"
        '#include "generated_split_hybrid_registry.cuh"\n'
        '#include "dft/semilocal_family.hpp"\n'
        "using generativeqc::dft::SemilocalFamily;\n"
        "using generativeqc::dft::semilocal_family_code;\n"
        "namespace generated = generativeqc::dft::generated;\n"
        "using CudaXcPointLauncher = void (*)();\n"
        "using CudaXcPointBatchLauncher = void (*)();\n"
        "namespace generativeqc::dft::generated {\n"
        "struct AutomaticPointStub { CudaXcPointLauncher launch{}; };\n"
        "const AutomaticPointStub* automatic_cuda_point_entry(std::uint32_t) { return nullptr; }\n"
        "}\n"
        "struct CudaXcPointCapabilities {\n"
        "  bool local_ao_selection{}, mixed_density_contraction{};\n"
        "};\n"
        "unsigned selected_functional; bool selected_response, selected_specialized;\n"
        "template <unsigned F, bool R, bool S = false> void launch_points() {\n"
        "  selected_functional = F; selected_response = R; selected_specialized = S;\n}\n"
        "template <unsigned Mask> void launch_split_hybrid_points() {\n"
        "  selected_functional = Mask; selected_response = false; selected_specialized = false;\n}\n"
        "template <unsigned Terms, unsigned Threads, bool S = false> void launch_point_batches() {\n"
        "  selected_specialized = S;\n}\n"
        + emit_native_xc_point_dispatch()
        + batch_dispatch
        + r"""
int main() {
  CudaXcPointLauncher entries[7]{};
  unsigned count = 0;
  for (unsigned f = 0; f < 6; ++f) {
    for (unsigned r = 0; r < 2; ++r) {
      const bool admitted = r ? f < 2 : f < 5;
      try {
        auto launch = resolve_point_launcher(f, r);
        if (!admitted || !launch) return 1;
        const auto capability = resolve_point_capabilities(f, r);
        if (capability.local_ao_selection != !bool(r)) return 10;
        if (capability.mixed_density_contraction != (!r && f < 3)) return 11;
        launch();
        if (selected_functional != f || selected_response != bool(r)) return 2;
        const char* setting = std::getenv("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION");
        if (selected_specialized != (f == 1 && setting && setting[0] == '1')) return 13;
        for (unsigned i = 0; i < count; ++i)
          if (entries[i] == launch) return 3;
        entries[count++] = launch;
      } catch (const std::invalid_argument&) {
        if (admitted) return 4;
      }
    }
  }
  if (count != 7) return 5;
  auto pbe_generic = &launch_points<semilocal_family_code(SemilocalFamily::Pbe), false>;
  auto pbe_specialized = &launch_points<semilocal_family_code(SemilocalFamily::Pbe), false, true>;
  if (resolve_point_batch_launcher(1, pbe_generic) != &launch_point_batches<4, 32>) return 14;
  if (resolve_point_batch_launcher(1, pbe_specialized) !=
      &launch_point_batches<4, 32, true>) return 15;
  try {
    resolve_point_batch_launcher(1, &launch_points<semilocal_family_code(SemilocalFamily::Pbe), true>);
    return 16;
  } catch (const std::invalid_argument&) {}
  const std::uint32_t generated_codes[]{generated::kM062XFunctionalCode,
                                        generated::kMN15FunctionalCode};
  for (const auto code : generated_codes) {
    auto launch = resolve_point_launcher(code, false);
    if (launch != &launch_split_hybrid_points<5>) return 7;
    const auto capability = resolve_point_capabilities(code, false);
    if (capability.local_ao_selection || capability.mixed_density_contraction) return 12;
    launch();
    if (selected_functional != 5 || selected_response) return 8;
    try {
      resolve_point_launcher(code, true);
      return 9;
    } catch (const std::invalid_argument&) {}
  }
  try {
    resolve_point_launcher(std::numeric_limits<std::uint32_t>::max(), false);
    return 6;
  } catch (const std::invalid_argument&) {}
}
"""
    )
    binary = tmp_path / "dispatch"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=("-std=c++17", "-O2", f"-I{tmp_path}", f"-I{ROOT / 'src'}"),
    )
    environment = os.environ.copy()
    environment.pop("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", None)
    subprocess.run([str(binary)], check=True, env=environment)
    for value in ("0", "1"):
        subprocess.run(
            [str(binary)],
            check=True,
            env={**environment, "GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION": value},
        )
    invalid = subprocess.run(
        [str(binary)],
        check=False,
        env={**environment, "GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION": "yes"},
    )
    assert invalid.returncode == 4


def test_pbe_point_source_identity_contains_both_prepared_entries(
    monkeypatch: object,
) -> None:
    """The runtime selector binds one entry from a single versioned source artifact."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "0")
    generic_source, generic_key, _ = emit_grid_source(native_ks=True)
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "1")
    specialized_source, specialized_key, _ = emit_grid_source(native_ks=True)
    assert (generic_source, generic_key) == (specialized_source, specialized_key)
    assert generic_key == canonical_hash(
        {"schema": "generativeqc.grid-policy.v1", "source": generic_source}
    )
    assert (
        "&launch_points<semilocal_family_code(SemilocalFamily::Pbe), false, true>"
        in generic_source
    )
    assert (
        "&launch_points<semilocal_family_code(SemilocalFamily::Pbe), false>"
        in generic_source
    )
    assert "point::evaluate(true, rho, gradient, exchange_scale," in generic_source
    assert emit_grid_source()[1] != generic_key
