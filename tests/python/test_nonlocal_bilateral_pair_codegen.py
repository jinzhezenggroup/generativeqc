"""Independent high-precision checks of both generated VV10 feature legs."""

import ctypes as ct
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tools.generate_nonlocal_pair_native import native_header

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def bilateral_pair(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: "NativeCxx"
) -> Any:
    """Compile the actual emitted scalar roots without any CUDA dependency."""
    root = tmp_path_factory.mktemp("vv10-bilateral-host")
    source = root / "pair.cpp"
    source.write_text(
        "#include <cmath>\nnamespace generativeqc::dft::nlc { "
        "enum class Vv10Variant { vv10, rvv10 }; }\n"
        + native_header()
        + r"""
extern "C" void pair(const double* x, double* out) {
  using namespace generativeqc::dft::nlc::generated;
  const auto p=pair_values_vv10_bilateral<true>(x[0],x[1],x[2],x[3],x[4]);
  out[0]=p.phi; out[1]=p.dphi_domega; out[2]=p.dphi_dkappa;
  out[3]=p.dphi_domega_j; out[4]=p.dphi_dkappa_j; out[5]=p.dphi_dr2;
  const auto f=pair_values_vv10_bilateral<false>(x[0],x[1],x[2],x[3],x[4]);
  out[6]=f.phi; out[7]=f.dphi_domega; out[8]=f.dphi_dkappa;
  out[9]=f.dphi_domega_j; out[10]=f.dphi_dkappa_j;
}
"""
    )
    library = native_cxx.build_shared(
        [source],
        root / "pair.so",
        compile_args=["-std=c++17", "-O2", "-ffp-contract=off"],
    )
    dll = ct.CDLL(str(library))
    dll.pair.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    dll.pair.restype = None
    return dll.pair


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 1.0, 2.0, 0.5, 3.0),
        (1.25, 0.7, 2.1, 4.5, 0.3),
        (2.0**32, 2.0**32, 2.0**-32, 2.0**32, 2.0**-32),
        (2.0**-32, 2.0**-32, 2.0**32, 2.0**-32, 2.0**32),
        (0.0, 2.0**-32, 2.0**-32, 2.0**-32, 2.0**-32),
    ],
)
def test_both_feature_legs_match_independent_decimal(
    bilateral_pair: Any, values: tuple[float, float, float, float, float]
) -> None:
    """The reference differentiates the rational energy directly at 90 digits."""
    with localcontext() as context:
        context.prec = 90
        r2, wi, wj, ki, kj = map(Decimal.from_float, values)
        gi, gj = wi * r2 + ki, wj * r2 + kj
        phi = -Decimal("1.5") / (gi * gj * (gi + gj))
        di = -phi * (1 / gi + 1 / (gi + gj))
        dj = -phi * (1 / gj + 1 / (gi + gj))
        expected = [phi, r2 * di, di, r2 * dj, dj, wi * di + wj * dj]
        expected += expected[:5]
    actual = (ct.c_double * 11)()
    bilateral_pair((ct.c_double * 5)(*values), actual)
    assert list(actual) == pytest.approx(list(map(float, expected)), rel=3e-14, abs=0)


def test_partner_demand_refuses_unsupported_science() -> None:
    from generativeqc_compiler.method.nonlocal_pair import build_nonlocal_pair_program

    for options in ({}, {"features": False}, {"rational_derivatives": False}):
        with pytest.raises(ValueError, match="partner features"):
            build_nonlocal_pair_program("vv10", partner_features=True, **options)
    with pytest.raises(ValueError):
        build_nonlocal_pair_program("rvv10", partner_features=True)


def test_parallel_output_leases_preserve_live_inputs(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Execute the real native alias gate with adjacent and overlapping spans."""
    from test_nonlocal_signed_weight_kernels import _definition

    root = Path(__file__).resolve().parents[2]
    source = (root / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    probe = tmp_path / "leases.cpp"
    probe.write_text(
        "#include <array>\n#include <utility>\n#include <cstddef>\n#include <cstdint>\n"
        + _definition(source, "bool parallel_pair_outputs_disjoint(")
        + r"""
int main() {
  double storage[128]{};
  const auto admit = [&](double* vrho, double* vsigma, double* point, double* weight) {
    return parallel_pair_outputs_disjoint(4, storage, storage+12,
                                          storage+16, 32*sizeof(double),
                                          vrho, vsigma, point, weight);
  };
  // Touching endpoints are disjoint, including the workspace's upper edge.
  if (!admit(storage+48, storage+52, storage+56, storage+68)) return 1;
  if (!admit(storage+48, storage+52, nullptr, nullptr)) return 2;
  if (admit(storage+47, storage+52, storage+56, storage+68)) return 3;
  if (admit(storage+48, storage+51, storage+56, storage+68)) return 4;
  if (admit(storage+48, storage+52, storage+56, storage+67)) return 5;
  if (admit(storage+9, storage+52, storage+56, storage+68)) return 6;
  if (admit(storage+12, storage+52, storage+56, storage+68)) return 7;
  if (admit(storage+48, storage+52, storage+40, storage+68)) return 8;
}
"""
    )
    executable = native_cxx.build_executable(
        [probe], tmp_path / "leases", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([executable], check=True, capture_output=True, text=True)
