"""Execute CPU/CUDA admission metadata without launching or mocking science."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_C_TPSS"))
def test_generated_descriptor_authenticates_individual_cpu_contracts(
    name: str, tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Reject stale/missing spin identities and domains, retaining the pair hash."""
    provider = emit_automatic_cuda_provider(name)
    cpu = emit_automatic_point(name)
    restricted = (
        emit_automatic_point(name, spin="unpolarized").expression_identity
        if provider.feature_terms == 1
        else ""
    )
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "cuda_runtime_api.h").write_text("using cudaStream_t = void*;\n")
    source = tmp_path / "identity.cpp"
    source.write_text(
        "#include <cassert>\n#include <string_view>\n"
        '#include "dft/cuda_xc_point_program.hpp"\n'
        f"namespace {provider.namespace} {{\n"
        "void launch(cudaStream_t, const double*, const double*, std::size_t, "
        "std::size_t, double*, double*, int*, std::uint32_t, double, double, "
        "const double*) {}\n"
        "void launch_batch(cudaStream_t, const double*, const double*, std::size_t, "
        "std::size_t, double*, double*, int*, std::uint32_t, double, double, "
        "std::size_t) {}\n"
        "void launch_stationary(cudaStream_t, const double*, std::size_t, "
        "std::size_t, DevicePointValue*, int*, const int*) {}\n}\n"
        + emit_automatic_cuda_descriptor(provider)
        + f"""
int main() {{
  auto p = {provider.namespace}::program;
  const std::string_view polarized = "{cpu.expression_identity}";
  const std::string_view restricted = "{restricted}";
  const auto matches = [&] {{
    return p.matches_cpu_science(polarized, restricted,
                                {cpu.ingredient_mask}, {cpu.domain_version});
  }};
  assert(matches());
  assert(p.expression_identity != polarized);
  assert(!p.matches_cpu_science(p.expression_identity, restricted,
                                {cpu.ingredient_mask}, {cpu.domain_version}));
  assert(!p.matches_cpu_science("stale", restricted,
                                {cpu.ingredient_mask}, {cpu.domain_version}));
  assert(!p.matches_cpu_science(polarized, "stale",
                                {cpu.ingredient_mask}, {cpu.domain_version}));
  assert(!p.matches_cpu_science(polarized, restricted, 0, {cpu.domain_version}));
  assert(!p.matches_cpu_science(polarized, restricted, {cpu.ingredient_mask}, 0));
  p.polarized_expression_identity = nullptr;
  assert(!matches());
}}
"""
    )
    binary = native_cxx.build_executable(
        [source],
        tmp_path / "identity",
        compile_args=("-std=c++20", "-I", str(tmp_path), "-I", str(root / "src")),
    )
    subprocess.run([binary], check=True, capture_output=True, timeout=10)
