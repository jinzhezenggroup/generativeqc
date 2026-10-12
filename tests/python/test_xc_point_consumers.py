"""Execute the emitted bounded point selector without loading CUDA or the runtime."""

import os
import subprocess
from pathlib import Path

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.ao_cuda import (
    emit_grid_source,
    emit_native_xc_point_batch_dispatch,
    emit_native_xc_point_dispatch,
)

from tools.generate_xc_split_hybrid_registry import emit_registry

ROOT = Path(__file__).resolve().parents[2]


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
    batch_dispatch = emit_native_xc_point_batch_dispatch()
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


def test_pbe_policy_is_runtime_selection_with_one_cache_identity(
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
    assert emit_grid_source()[1] != generic_key
