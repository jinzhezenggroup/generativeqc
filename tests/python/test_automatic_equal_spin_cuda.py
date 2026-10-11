"""Qualify registry-wide scalar sharing without granting automatic CUDA admission."""

from __future__ import annotations

import ctypes
import re
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_contraction_kernels
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_device_point_value,
    emit_equal_spin_point_selector,
    emit_point_traversal,
)
from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_functional_code,
    automatic_registry_entries,
)
from test_libxc_automatic_point_native import NAMES, _inputs, _oracle

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", [entry.name for entry in automatic_registry_entries()])
def test_every_automatic_registration_has_a_producer_bound_sibling(name: str) -> None:
    """The structural census, not a positive name/code whitelist, owns coverage."""
    provider = emit_automatic_cuda_provider(name)
    ordinary = emit_automatic_point(name, backend="cuda")
    restricted = emit_automatic_point(name, backend="cuda", equal_spin=True)
    assert provider.point == ordinary
    assert restricted.expression_identity == ordinary.expression_identity
    assert restricted.features == ordinary.features
    assert restricted.outputs == ordinary.outputs
    assert restricted.source_identity != ordinary.source_identity
    assert ordinary.source in provider.source
    assert restricted.source in provider.source
    assert provider.source.count(emit_point_traversal()) == 1
    assert provider.source.count(emit_equal_spin_point_selector()) == 1
    for batched in ("false", "true"):
        assert (
            f"automatic_evaluate_points<{provider.feature_terms}, {batched}, "
            f"{provider.provider_name}_equal_spin>" in provider.source
        )
    assert provider.source.count("if (use_equal_spin_points(spins))") == 2


def test_cpu_cannot_request_an_unproven_equal_spin_slice() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        emit_automatic_point("GGA_X_APBE", equal_spin=True)
    source = emit_native_xc_contraction_kernels()
    assert source.count(emit_equal_spin_point_selector()) == 1


@pytest.mark.parametrize("name", NAMES)
def test_automatic_equal_spin_preserves_polarized_partials_and_libxc(
    name: str, tmp_path: Path, native_cxx: object
) -> None:
    """Execute exact CUDA scalar bodies as strict host C++, including work floors."""
    ordinary = emit_automatic_point(name, backend="cuda")
    restricted = emit_automatic_point(name, backend="cuda", equal_spin=True)
    source = tmp_path / "points.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstddef>\nusing std::isfinite;\n"
        + emit_device_point_value()
        + ordinary.source.replace("__device__ ", "")
        + "\n"
        + restricted.source.replace("__device__ ", "")
        + f"""
extern "C" void evaluate(bool restricted, const double* inputs,
                         double* outputs, std::size_t count) {{
  for (std::size_t point = 0; point < count; ++point) {{
    const double* input = inputs + 10 * point;
    double* output = outputs + 12 * point;
    const double rho[2]{{input[0], input[1]}};
    const double gradient[2][3]{{
        {{input[2], input[3], input[4]}}, {{input[5], input[6], input[7]}}}};
    const double tau[2]{{input[8], input[9]}};
    const auto value = restricted ? {restricted.function_name}(rho, gradient, tau)
                                  : {ordinary.function_name}(rho, gradient, tau);
    output[0] = value.energy;
    for (unsigned spin = 0; spin < 2; ++spin) {{
      output[1 + spin] = value.rho[spin];
      for (unsigned axis = 0; axis < 3; ++axis)
        output[3 + 3 * spin + axis] = value.gradient[spin][axis];
      output[9 + spin] = value.kinetic[spin];
    }}
    output[11] = value.valid;
  }}
}}
""",
        encoding="utf-8",
    )
    binary = native_cxx.build_shared(
        [source],
        tmp_path / "points.so",
        compile_args=("-std=c++17", "-O3", "-ffp-contract=off"),
    )
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (ctypes.c_bool, pointer, pointer, ctypes.c_size_t)
    library.evaluate.restype = None
    inputs = np.ascontiguousarray(_inputs(name, 0), dtype=np.float64)
    results = []
    for selected in (False, True):
        output = np.empty((len(inputs), 12))
        library.evaluate(
            selected,
            inputs.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            len(inputs),
        )
        assert np.all(output[:, 11] == 1)
        results.append(output)
    np.testing.assert_array_equal(
        results[0].view(np.uint64), results[1].view(np.uint64)
    )
    np.testing.assert_allclose(
        results[1][:, :11], _oracle(name, inputs, 1), rtol=2e-10, atol=2e-10
    )
    for selected in (False, True):
        invalid = inputs.copy()
        invalid[:, :2] = -1
        output = np.empty((len(invalid), 12))
        library.evaluate(
            selected,
            invalid.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            len(invalid),
        )
        assert np.all(output[:, 11] == 0)


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_C_TPSS"))
def test_actual_automatic_launchers_select_only_proven_rks(
    name: str, tmp_path: Path, native_cxx: object
) -> None:
    """Run generated host launchers with a CUDA stub; keep UKS/scaling/response gates."""
    provider = emit_automatic_cuda_provider(name)
    launchers = provider.source.split("void validate_launch", 1)[1].split(
        "}  // namespace", 1
    )[0]
    launchers = re.sub(
        r"<<<.*?>>>", "", "void validate_launch" + launchers, flags=re.DOTALL
    )
    source = tmp_path / "launchers.cpp"
    source.write_text(
        "#include <algorithm>\n#include <cassert>\n#include <cstdlib>\n"
        "#include <cstring>\n#include <cstdint>\n#include <limits>\n"
        "#include <stdexcept>\n#include <type_traits>\n"
        "using I = long;\nusing cudaStream_t = void*;\n"
        f"struct {provider.provider_name} {{}};\nstruct {provider.provider_name}_equal_spin {{}};\n"
        "bool observed_restricted = false, observed_batch = false;\nunsigned calls = 0;\n"
        f"""
template <I width, bool batched, class Provider, class... Args>
void automatic_evaluate_points(Args...) {{
  static_assert(width == {provider.feature_terms});
  observed_restricted = std::is_same_v<Provider, {provider.provider_name}_equal_spin>;
  observed_batch = batched;
  ++calls;
}}
"""
        + emit_equal_spin_point_selector()
        + launchers
        + f"""
int main() {{
  constexpr std::uint32_t code = 0x{automatic_functional_code(name):x}U;
  for (const char* selection : {{"0", "1"}}) {{
    setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", selection, 1);
    for (unsigned spins : {{1, 2}}) {{
      launch(nullptr,nullptr,nullptr,37,spins,nullptr,nullptr,nullptr,code,1,1,nullptr);
      assert(!observed_batch && observed_restricted == (selection[0] == '1' && spins == 1));
      launch_batch(nullptr,nullptr,nullptr,37,spins,nullptr,nullptr,nullptr,code,1,1,16);
      assert(observed_batch && observed_restricted == (selection[0] == '1' && spins == 1));
    }}
  }}
  unsetenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN");
  launch(nullptr,nullptr,nullptr,37,1,nullptr,nullptr,nullptr,code,1,1,nullptr);
  assert(!observed_restricted);
  const auto previous_calls = calls;
  launch(nullptr,nullptr,nullptr,0,1,nullptr,nullptr,nullptr,code,1,1,nullptr);
  assert(calls == previous_calls);
  setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "yes", 1);
  try {{ launch(nullptr,nullptr,nullptr,37,1,nullptr,nullptr,nullptr,code,1,1,nullptr); assert(false); }}
  catch (const std::invalid_argument&) {{}}
  for (unsigned spins : {{0, 3}}) {{
    try {{ validate_launch(37,spins,code,1,1); assert(false); }}
    catch (const std::invalid_argument&) {{}}
  }}
  for (double scale : {{0.0, 0.5}}) {{
    try {{ validate_launch(37,1,code,scale,1); assert(false); }}
    catch (const std::invalid_argument&) {{}}
    try {{ validate_launch(37,1,code,1,scale); assert(false); }}
    catch (const std::invalid_argument&) {{}}
  }}
  try {{ validate_launch(37,1,code + 1,1,1); assert(false); }}
  catch (const std::invalid_argument&) {{}}
  try {{ launch_batch(nullptr,nullptr,nullptr,37,1,nullptr,nullptr,nullptr,code,1,1,0); assert(false); }}
  catch (const std::invalid_argument&) {{}}
  const double delta = 0;
  try {{ launch(nullptr,nullptr,nullptr,37,1,nullptr,nullptr,nullptr,code,1,1,&delta); assert(false); }}
  catch (const std::invalid_argument&) {{}}
}}
""",
        encoding="utf-8",
    )
    binary = native_cxx.build_executable(
        [source], tmp_path / "launchers", compile_args=("-std=c++17",)
    )
    subprocess.run([str(binary)], check=True, timeout=30)
