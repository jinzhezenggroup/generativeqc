"""Execute actual typed geometry algebra, with independent Becke derivatives."""

import ctypes
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from generativeqc_compiler.xc.geometry_cuda import emit_geometry_cuda
from test_grid_response import CENTERS, DC, POINTS, decimal_partition

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def geometry_probe(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """No CUDA scheduling emulation: one lane runs the emitted scientific body."""
    semilocal = resolve_method("PBE0", spin="unpolarized").primitives[0].functional
    source = _STATIONARY_SCIENTIFIC_KERNELS
    begin = source.rfind(
        "template <", 0, source.index("__device__ bool geometry_point_setup(")
    )
    end = source.index("/** Evaluate one shared", begin)
    kernel = source.rfind(
        "template <", 0, source.index("__global__ void geometry_kernel(")
    )
    folder = tmp_path_factory.mktemp("geometry-fp32")
    path = folder / "probe.cpp"
    path.write_text(
        r"""
#include <algorithm>
#include <cstdint>
#include <vector>
#define __device__
#define __global__
#define __constant__
using std::size_t;
struct { size_t x; } blockIdx{0}, threadIdx{0}, blockDim{1};
int atomicExch(int* pointer, int value) { const auto previous=*pointer; *pointer=value; return previous; }
double finite(double value, int* error, double replacement) {
  if (std::isfinite(value)) return value;
  *error=1; return replacement;
}
namespace generativeqc::dft {
struct GridTaskView {
  size_t npoint{}, nactive{}, nao{};
  const double *features{}, *ao{}, *points{};
  const size_t* ao_ids{};
  const int* error{};
};
}
""".replace(
            "#include <algorithm>",
            "#include <algorithm>\n#include <cmath>\nusing std::isfinite;",
        )
        + emit_geometry_cuda(functional=1, semilocal=semilocal)
        + source[begin:end]
        + source[
            kernel : source.index(
                "}  // namespace generativeqc_stationary_cuda", kernel
            )
        ]
        + r"""
extern "C" int geometry(bool mixed, bool cached, size_t atoms, size_t points, size_t aos,
    const double* centers, const double* xyz, const int64_t* owners, const int64_t* ao_atoms,
    const double* features, const double* ao, const double* work,
    const double* weights, const double* raw, double* output) {
  std::vector<double> partial(9*atoms), scratch(9*atoms);
  std::vector<generativeqc_grid_adjoint::CenterPair> pairs(atoms*(atoms-1)/2);
  if (!generativeqc_grid_adjoint::prepare_center_geometry(centers, atoms, 1e-12,
      pairs.data(), local_norm, local_ratio_geometry)) return 2;
  generativeqc::dft::GridTaskView view{points,aos,aos,features,ao,xyz,nullptr,nullptr};
  int error=0;
  const auto execute=[&](auto arithmetic) {
    geometry_kernel<decltype(arithmetic)>(view,work,ao_atoms,owners,0,0,centers,atoms,
        weights,raw,nullptr,0,0,1,partial.data(),scratch.data(),cached?pairs.data():nullptr,&error,false);
  };
  if (mixed) execute(float{}); else execute(double{});
  if (!error) std::copy(partial.begin(),partial.end(),output);
  return error;
}
"""
    )
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [path],
                folder / "probe.so",
                compile_args=(
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    f"-I{ROOT / 'src'}",
                ),
            )
        )
    )
    floating = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    function = library.geometry
    function.argtypes = (
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        floating,
        floating,
        integer,
        integer,
        floating,
        floating,
        floating,
        floating,
        floating,
        floating,
    )
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("aos", (0, 17, 192))
def test_typed_geometry_against_strict_and_decimal(
    geometry_probe: Any, cached: bool, aos: int
) -> None:
    generator = np.random.default_rng(42)
    atoms, points = len(CENTERS), len(POINTS)
    owners = np.arange(points, dtype=np.int64)
    ao_atoms = np.arange(aos, dtype=np.int64) % atoms
    features = np.zeros((2, 5, points))
    features[:, 0] = 0.3
    features[:, 1:4] = generator.normal(size=(2, 3, points)) * 0.02
    ao = generator.normal(size=(10, points, aos)) * 0.1
    work = generator.normal(size=(8, points, aos)) * 0.2
    weights = np.ones(points) * 0.2
    raw = np.array([0.3, 0.2, 0.7])
    results = []
    for mixed in (False, True):
        output = np.zeros((3, atoms, 3))
        assert (
            geometry_probe(
                mixed,
                cached,
                atoms,
                points,
                aos,
                CENTERS,
                POINTS,
                owners,
                ao_atoms,
                features,
                ao,
                work,
                weights,
                raw,
                output,
            )
            == 0
        )
        results.append(output)
    np.testing.assert_allclose(results[1], results[0], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(results[1].sum(axis=(0, 1)), 0, atol=1e-12)
    if aos:
        assert np.any(results[0] != results[1])
        return
    libxc = pytest.importorskip("pyscf.dft.libxc")
    energy, *_ = libxc.eval_xc("0.75*GGA_X_PBE+GGA_C_PBE", features[:, :4], spin=1)
    seeds = raw * energy * features[:, 0].sum(axis=0)
    with localcontext() as context:
        context.prec = 60
        step = Decimal("1e-16")

        def move(values: Any, directions: Any, sign: int) -> list:
            return [
                [
                    Decimal(str(value)) + sign * step * Decimal(str(direction))
                    for value, direction in zip(row, motion, strict=True)
                ]
                for row, motion in zip(values, directions, strict=True)
            ]

        plus, minus = [
            decimal_partition(
                move(POINTS, DC[owners], sign), move(CENTERS, DC, sign), 3
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (left[owner] - right[owner]) / (2 * step)
            for seed, left, right, owner in zip(seeds, plus, minus, owners, strict=True)
        )
    assert abs(np.sum(results[1][2] * DC) - float(expected)) < 3e-6


def test_float_failure_does_not_publish_partial_geometry(geometry_probe: Any) -> None:
    features = np.ones((2, 5, 3))
    ao = np.full((10, 3, 1), 1e30)
    work = np.full((8, 3, 1), 1e30)
    output = np.full((3, 3, 3), 123.0)
    assert (
        geometry_probe(
            True,
            True,
            3,
            3,
            1,
            CENTERS,
            POINTS,
            np.arange(3, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            features,
            ao,
            work,
            np.ones(3),
            np.ones(3),
            output,
        )
        == 1
    )
    np.testing.assert_array_equal(output, 123.0)
