"""Automatic force transport preserves the independently qualified point ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.method.stationary_cuda import (
    _qualified_aot_profile,
    stationary_aot_profile_for_plan,
)
from generativeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_functional_code,
    automatic_functional_name,
    automatic_scf_domain,
)
from generativeqc_compiler.xc.geometry_cuda import _emit_automatic_point
from generativeqc_compiler.xc.libxc_work import automatic_work_policy
from generativeqc_compiler.xc.spec import UnsupportedXC
from test_libxc_automatic_point_native import NAMES

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_stationary_point_transport_matches_cpu_worker(
    name: str, spin: str, tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the adapted CUDA body on host, including density-floor edges.

    This is ABI/domain coverage, not a GPU scientific oracle. The canonical
    worker has separate Libxc gates, and public GPU forces have reconverged FDs.
    """
    method = resolve_bulk_ks(name, spin=spin, backend="cuda").method
    spec = method.primitives[0].functional
    code = automatic_functional_code(name)
    domain = automatic_scf_domain(name)
    plan = StationaryGradientPlan(method, StationaryMeanField(domain))
    profile = _qualified_aot_profile(
        f"libxc_{code}_{'rks' if spin == 'unpolarized' else 'uks'}"
    )
    assert profile.plan.identity == plan.identity
    assert stationary_aot_profile_for_plan(code, spin, plan) == profile
    assert (
        stationary_aot_profile_for_plan(
            code, "polarized" if spin == "unpolarized" else "unpolarized", plan
        )
        is None
    )
    point_spin = spin if spec.ingredients == ("rho",) else "polarized"
    cpu = emit_automatic_point(name, backend="cpu", spin=point_spin)
    source = tmp_path / "point.cpp"
    source.write_text(
        '#include <cmath>\n#include <stdexcept>\n#include "dft/xc.hpp"\n'
        "#define __device__\nusing std::isfinite;\n"
        "namespace gpu {\n" + _emit_automatic_point(code, spec) + "\n}\n"
        "namespace cpu {\nusing generativeqc::dft::SemilocalPointValue;\n"
        + cpu.source
        + "\n}\n"
        f"""
extern "C" void evaluate(const double* in, double* out, int device) {{
  double rho[2]{{in[0], in[1]}}, gradient[2][3]{{{{in[2],in[3],in[4]}},{{in[5],in[6],in[7]}}}}, tau[2]{{in[8],in[9]}};
  auto copy = [&](const auto& value) {{
    out[0] = value.energy;
    for (unsigned s = 0; s < 2; ++s) {{
      out[1+s] = value.rho[s]; out[9+s] = value.kinetic[s];
      for (unsigned k=0;k<3;++k) out[3+3*s+k] = value.gradient[s][k];
    }}
  }};
  out[11] = 1;
  if (device) {{
    auto value = gpu::stationary_evaluate_point(rho, gradient, tau);
    out[11] = value.valid; copy(value);
  }} else try {{ copy(cpu::{cpu.function_name}(rho, gradient, tau)); }}
    catch (const std::domain_error&) {{ out[11] = 0; }}
}}
"""
    )
    binary = tmp_path / "point.so"
    native_cxx.build_shared(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = [pointer, pointer, ctypes.c_int]
    library.evaluate.restype = None
    floor = automatic_work_policy(name).density_threshold
    for channels in (
        (0.3, 0.2),
        (0.0, 0.0),
        (0.75 * floor, 0.75 * floor),
        (1e-3 * floor, 0.3),
    ):
        rho = (
            (sum(channels) / 2, sum(channels) / 2)
            if spin == "unpolarized"
            else channels
        )
        inputs = np.array(
            [*rho, 0.01, -0.02, 0.03, -0.02, 0.01, 0.0, 0.4, 0.3], dtype=np.float64
        )
        outputs = []
        for device in (0, 1):
            out = np.zeros(12)
            library.evaluate(
                inputs.ctypes.data_as(pointer), out.ctypes.data_as(pointer), device
            )
            outputs.append(out)
        assert outputs[0][11] == outputs[1][11]
        if outputs[0][11]:
            np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-13, atol=1e-25)


@pytest.mark.parametrize("code", (True, 196609.0, 0x30000, 0x40000, 0x3FFFF))
def test_unregistered_or_truncated_code_rejects(code: object) -> None:
    with pytest.raises(UnsupportedXC, match="not registered"):
        automatic_functional_name(code)


@pytest.mark.parametrize(
    "name", ("libxc_0196639_rks", "libxc_196639_bad", "libxc_196608_rks")
)
def test_invalid_profile_rejects(name: str) -> None:
    with pytest.raises((ValueError, UnsupportedXC)):
        _qualified_aot_profile(name)
