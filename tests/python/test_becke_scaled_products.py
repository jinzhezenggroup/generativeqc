"""Independent host qualification of the opt-in binary product representation."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from decimal import Decimal, localcontext

import numpy as np
import pytest
import test_becke_cooperative as retained
import test_becke_phased as phased
from generativeqc_compiler.xc.becke_scaled_products import emit_binary_scaled_becke
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from generativeqc_compiler.xc.grid_phased import emit_phased_becke

PRODUCT_HARNESS = r"""
extern "C" int product_probe(size_t atoms, size_t factors_per_atom,
    const double* factors, size_t owner, double seed, double* weights,
    double* derivatives) {
  using namespace generativeqc_grid_scaled;
  if (!atoms || atoms > maximum_scaled_atoms || factors_per_atom >= maximum_scaled_atoms ||
      owner >= atoms || !std::isfinite(seed)) return -2;
  std::vector<BlockedScaledProduct> products(atoms);
  int scale = std::numeric_limits<int>::min();
  for (size_t atom = 0; atom < atoms; ++atom) {
    for (size_t factor = 0; factor < factors_per_atom; ++factor)
      if (!products[atom].multiply(factors[atom * factors_per_atom + factor])) return -2;
    products[atom].finalize();
    if (!products[atom].zeros) scale = std::max(scale, products[atom].exponent);
  }
  if (scale == std::numeric_limits<int>::min()) return -2;
  std::vector<double> values(atoms), result_weights(atoms);
  std::vector<double> result_derivatives(atoms * factors_per_atom);
  double total = 0;
  for (size_t atom = 0; atom < atoms; ++atom) {
    values[atom] = products[atom].scaled_value(scale);
    total += values[atom];
  }
  const auto objective = normalized_product_objective(atoms, owner, values, local_ratio);
  for (size_t atom = 0; atom < atoms; ++atom) {
    result_weights[atom] = values[atom] / total;
    const double bar = normalized_product_bar(atom, owner, seed, objective);
    for (size_t factor = 0; factor < factors_per_atom; ++factor) {
      const size_t index = atom * factors_per_atom + factor;
      result_derivatives[index] = bar * products[atom].normal_factor_excluded_value(factors[index], scale);
      if (!std::isfinite(result_derivatives[index])) return -2;
    }
  }
  std::copy(result_weights.begin(), result_weights.end(), weights);
  std::copy(result_derivatives.begin(), result_derivatives.end(), derivatives);
  return 0;
}

extern "C" int encoding_probe(size_t count, const double* factors,
    double* mantissas, int64_t* state, double* observable) {
  using namespace generativeqc_grid_scaled;
  ScaledProduct baseline;
  BlockedScaledProduct blocked;
  for (size_t index = 0; index < count; ++index)
    if (!baseline.multiply(factors[index]) || !blocked.multiply(factors[index])) return -2;
  blocked.finalize();
  blocked.finalize();
  mantissas[0] = baseline.mantissa;
  mantissas[1] = blocked.mantissa;
  state[0] = baseline.exponent;
  state[1] = blocked.exponent;
  state[2] = baseline.zeros;
  state[3] = blocked.zeros;
  observable[0] = baseline.scaled_value(0);
  observable[1] = blocked.scaled_value(0);
  for (size_t index = 0; index < count; ++index) {
    observable[2 + 2 * index] = baseline.excluded_value(factors[index], 0);
    observable[3 + 2 * index] = blocked.normal_factor_excluded_value(factors[index], 0);
  }
  return 0;
}

struct PullbackGeometryProbe {
  size_t* calls;
  std::array<double, 4> separation(size_t, size_t, bool&) const {
    ++*calls;
    return {1, 1, 0, 0};
  }
  std::array<double, 3> coordinate(double, size_t, size_t, double) const {
    ++*calls;
    return {0, 1, 1};
  }
};

extern "C" int pullback_probe(size_t first_zeros, size_t second_zeros,
    double factor, double slope, double first_bar, double second_bar,
    double first_partial, double second_partial,
    double* output, size_t* calls) {
  using namespace generativeqc_grid_scaled;
  double pairs[4]{first_partial, second_partial, factor, slope};
  double fields[22]{};
  size_t zeros[2]{second_zeros, first_zeros};
  double maximum[1]{};
  Workspace work{2, 1, pairs, fields, zeros, maximum};
  for (size_t atom = 0; atom < 2; ++atom) {
    work.field(4, 0)[atom] = 0.5;
    work.field(7, 0)[atom] = pack_relative_exponent(1);
    work.field(6, 0)[atom] = atom ? first_bar : second_bar;
  }
  *calls = 0;
  PairPullbackCoefficients result;
  const bool valid = scaled_pair_pullback_coefficients(work, 0, 1, 0,
      PullbackGeometryProbe{calls}, result);
  std::copy(result.values.begin(), result.values.end(), output);
  std::copy(result.separation.begin(), result.separation.end(), output + 2);
  return valid ? 0 : -2;
}

extern "C" int finite_excluded_bound_probe(double mantissa, int exponent,
    size_t zeros, double factor, double* output) {
  using namespace generativeqc_grid_scaled;
  const ScaledProduct product{mantissa, exponent, zeros};
  *output = product.normal_factor_excluded_value(factor, 0);
  return product.finite_normal_excluded_bound(factor);
}

extern "C" int normalized_division_probe(size_t count, const double* mantissas,
    int exponent, const double* factors, double* raw, double* normalized) {
  using namespace generativeqc_grid_scaled;
  for (size_t index = 0; index < count; ++index) {
    raw[index] = std::ldexp(mantissas[index] / factors[index], exponent);
    normalized[index] = ScaledProduct{mantissas[index], exponent, 0}
        .normal_factor_excluded_value(factors[index], 0);
  }
  return 0;
}

extern "C" int zero_bar_encoding_probe(double mantissa, int exponent,
    size_t first_zeros, double factor, double first_bar,
    double first_partial, double second_partial, double* output) {
  using namespace generativeqc_grid_scaled;
  double pairs[4]{first_partial, second_partial, factor, 0.2};
  double fields[22]{};
  size_t zeros[2]{2, first_zeros};
  double maximum[1]{};
  Workspace work{2, 1, pairs, fields, zeros, maximum};
  work.field(4, 0)[1] = mantissa;
  work.field(7, 0)[1] = pack_relative_exponent(exponent);
  work.field(6, 0)[1] = first_bar;
  size_t calls = 0;
  PairPullbackCoefficients result;
  const bool valid = scaled_pair_pullback_coefficients(work, 0, 1, 0,
      PullbackGeometryProbe{&calls}, result);
  std::copy(result.values.begin(), result.values.end(), output);
  output[2] = static_cast<double>(calls);
  return valid ? 0 : -2;
}

extern "C" int cached_coordinate_probe(double* output, size_t* phase_calls) {
  using namespace generativeqc_grid_scaled;
  double pairs[4]{};
  double fields[22]{};
  size_t zeros[2]{};
  double maximum[1]{};
  Workspace work{2, 1, pairs, fields, zeros, maximum};
  size_t calls = 0;
  const PullbackGeometryProbe geometry{&calls};
  const auto logarithm = [](double value) {
    return std::array<double, 2>{std::log(value), 1 / value};
  };
  const auto pair = [](double coordinate) {
    return std::array<double, 2>{0.5 * (1 + coordinate), 0.5};
  };
  bool valid = scaled_pair_primal_phase(work, 0, 1, 0, geometry, logarithm, pair);
  output[0] = work.pair(0, 0, 0);
  output[1] = work.pair(1, 0, 0);
  phase_calls[0] = calls;
  for (size_t atom = 0; atom < 2; ++atom) {
    ScaledProduct product;
    valid = product.multiply(atom ? pairs[2] : 1 - pairs[2]) && valid;
    work.field(4, 0)[atom] = product.mantissa;
    work.field(7, 0)[atom] = pack_relative_exponent(product.exponent);
    work.field(6, 0)[atom] = atom ? 0.7 : 0.4;
  }
  PairPullbackCoefficients result;
  valid = scaled_pair_pullback_coefficients(work, 0, 1, 0, geometry, result) && valid;
  output[2] = result.values[0];
  output[3] = result.values[1];
  phase_calls[1] = calls - phase_calls[0];
  return valid ? 0 : -2;
}

extern "C" int primal_validated_probe(double tail, bool validated, double* output) {
  using namespace generativeqc_grid_scaled;
  double pairs[24]{};
  double fields[44]{};
  size_t zeros[4]{};
  double maximum[1]{};
  Workspace work{4, 1, pairs, fields, zeros, maximum};
  size_t calls = 0;
  size_t cursor = 0;
  const std::array<double, 6> factors{0.5, 0.5, 0.5, 0, 0, tail};
  const auto logarithm = [](double value) {
    return std::array<double, 2>{std::log(value), 1 / value};
  };
  const auto pair = [&](double) {
    return std::array<double, 2>{factors[cursor++], 0.5};
  };
  for (size_t first = 1; first < 4; ++first)
    for (size_t second = 0; second < first; ++second)
      if (!scaled_pair_primal_phase(work, 0, first, second,
          PullbackGeometryProbe{&calls}, logarithm, pair)) return -2;
  const bool valid = validated ? scaled_atom_products_phase<true>(work, 0, 3)
      : scaled_atom_products_phase<false>(work, 0, 3);
  const auto product = atom_product(work, 0, 3);
  output[0] = static_cast<double>(product.zeros);
  output[1] = product.mantissa;
  output[2] = product.scaled_value(0);
  for (size_t neighbor = 0; neighbor < 3; ++neighbor)
    output[3 + neighbor] = product.excluded_value(factors[3 + neighbor], 0);
  return valid ? 0 : -2;
}

extern "C" int unvalidated_dead_row_probe(double tail) {
  using namespace generativeqc_grid_scaled;
  double pairs[24]{};
  double fields[44]{};
  size_t zeros[4]{};
  double maximum[1]{};
  Workspace work{4, 1, pairs, fields, zeros, maximum};
  work.pair(2, center_pair_index(3, 2), 0) = tail;
  return scaled_atom_products_phase(work, 0, 3) ? 0 : -2;
}

extern "C" int excluded_encoding_probe(size_t count, const double* factors,
    int scale_offset, double* baseline, double* optimized) {
  using namespace generativeqc_grid_scaled;
  if (count >= maximum_scaled_atoms) return -2;
  BlockedScaledProduct product;
  for (size_t index = 0; index < count; ++index)
    if (!product.multiply(factors[index])) return -2;
  product.finalize();
  const int scale = product.exponent + scale_offset;
  for (size_t index = 0; index < count; ++index) {
    baseline[index] = product.excluded_value(factors[index], scale);
    optimized[index] = product.normal_factor_excluded_value(factors[index], scale);
  }
  return 0;
}
extern "C" int budget_probe(size_t count, const double* factors, uint64_t* state) {
  using namespace generativeqc_grid_scaled;
  BlockedScaledProduct product;
  for (size_t index = 0; index < count; ++index) {
    if (!product.multiply(factors[index])) return -2;
    if (!std::isnormal(product.mantissa) || product.mantissa >= 1 ||
        product.pending_cost > 1021) return -3;
  }
  state[0] = product.pending_cost;
  state[1] = product.zeros;
  product.finalize();
  return product.pending_cost ? -4 : 0;
}

extern "C" int relative_exponent_probe(int exponent, double* carrier) {
  using namespace generativeqc_grid_scaled;
  *carrier = pack_relative_exponent(exponent);
  return unpack_relative_exponent(*carrier);
}
"""


def scaled_harness(*, primal_validated: bool = False) -> str:
    """Reuse the incumbent geometry, owner validation and guarded output harness."""
    source = phased.HARNESS.replace(
        "using namespace generativeqc_grid_phased;",
        "using namespace generativeqc_grid_phased;\n"
        "  using namespace generativeqc_grid_scaled;",
    )
    source = source.replace("pair_primal_phase(", "scaled_pair_primal_phase(")
    source = source.replace(
        "atom_logs_phase(work, point, atom);",
        "if (!scaled_atom_products_phase"
        + ("<true>" if primal_validated else "<false>")
        + "(work, point, atom)) return false;",
    )
    source = source.replace("point_normalize_phase(", "scaled_point_normalize_phase(")
    return source.replace("pair_reverse_phase(", "scaled_pair_reverse_phase(")


@pytest.fixture(
    scope="module",
    params=[
        (iterations, primal_validated)
        for iterations in (1, 3, 5)
        for primal_validated in (False, True)
    ],
)
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    compiler = shutil.which("c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and verified compiler cache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    iterations, primal_validated = request.param
    directory = tmp_path_factory.mktemp("becke-scaled-products")
    source, library = directory / "probe.cpp", directory / "probe.so"
    source.write_text(
        emit_grid_adjoint()
        + emit_grid_partials(iterations)
        + emit_phased_becke()
        + emit_binary_scaled_becke()
        + scaled_harness(primal_validated=primal_validated)
        + PRODUCT_HARNESS
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    result.run.argtypes = [
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        pointer,
        ct.c_bool,
        ct.c_size_t,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    result.run.restype = ct.c_int
    result.product_probe.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.c_double,
        pointer,
        pointer,
    ]
    result.product_probe.restype = ct.c_int
    result.encoding_probe.argtypes = [
        ct.c_size_t,
        pointer,
        pointer,
        ct.POINTER(ct.c_int64),
        pointer,
    ]
    result.encoding_probe.restype = ct.c_int
    result.pullback_probe.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    result.pullback_probe.restype = ct.c_int
    result.finite_excluded_bound_probe.argtypes = [
        ct.c_double,
        ct.c_int,
        ct.c_size_t,
        ct.c_double,
        pointer,
    ]
    result.finite_excluded_bound_probe.restype = ct.c_int
    result.normalized_division_probe.argtypes = [
        ct.c_size_t,
        pointer,
        ct.c_int,
        pointer,
        pointer,
        pointer,
    ]
    result.normalized_division_probe.restype = ct.c_int
    result.zero_bar_encoding_probe.argtypes = [
        ct.c_double,
        ct.c_int,
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        pointer,
    ]
    result.zero_bar_encoding_probe.restype = ct.c_int
    result.cached_coordinate_probe.argtypes = [pointer, ct.POINTER(ct.c_size_t)]
    result.cached_coordinate_probe.restype = ct.c_int
    result.primal_validated_probe.argtypes = [ct.c_double, ct.c_bool, pointer]
    result.primal_validated_probe.restype = ct.c_int
    result.unvalidated_dead_row_probe.argtypes = [ct.c_double]
    result.unvalidated_dead_row_probe.restype = ct.c_int
    result.excluded_encoding_probe.argtypes = [
        ct.c_size_t,
        pointer,
        ct.c_int,
        pointer,
        pointer,
    ]
    result.excluded_encoding_probe.restype = ct.c_int
    result.budget_probe.argtypes = [ct.c_size_t, pointer, ct.POINTER(ct.c_uint64)]
    result.budget_probe.restype = ct.c_int
    result.relative_exponent_probe.argtypes = [ct.c_int, pointer]
    result.relative_exponent_probe.restype = ct.c_int
    result.iterations = iterations
    return result


@pytest.mark.parametrize(
    "scale_offset",
    [-2048, -1074, -1024, -1023, -53, -1, 0, 1, 53, 1022, 1023, 1024, 1074, 2048],
)
@pytest.mark.parametrize(
    "case",
    ["ordinary", "normal_floor", "subnormal", "one_zero", "two_zeros", "underflow"],
)
def test_normal_factor_exclusion_is_bit_identical(
    helper: ct.CDLL, case: str, scale_offset: int
) -> None:
    """Exact exponent edits retain oracle bits, including underflow/overflow."""
    minimum = np.finfo(np.float64).tiny
    rng = np.random.default_rng(11310)
    factors = rng.uniform(0.1, 0.99, size=127)
    if case == "normal_floor":
        factors = np.array(
            [minimum, np.nextafter(minimum, np.inf), 2 * minimum, 0.5, 1.0]
        )
    elif case == "subnormal":
        factors = np.array(
            [np.nextafter(0.0, 1.0), np.nextafter(minimum, 0.0), 0.5, 1.0]
        )
    elif case == "one_zero":
        factors = np.array([0.0, minimum, 0.7, 1.0])
    elif case == "two_zeros":
        factors = np.array([0.0, minimum, 0.0, 1.0])
    elif case == "underflow":
        factors[:] = 2.0**-54
    baseline = np.full_like(factors, np.nan)
    optimized = np.full_like(factors, np.nan)
    assert (
        helper.excluded_encoding_probe(
            len(factors),
            factors.ctypes.data_as(ct.POINTER(ct.c_double)),
            scale_offset,
            baseline.ctypes.data_as(ct.POINTER(ct.c_double)),
            optimized.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        == 0
    )
    np.testing.assert_array_equal(baseline.view(np.uint64), optimized.view(np.uint64))


@pytest.mark.parametrize(
    "exponent", [-2147483648, -137472, -2048, -1074, -1023, -53, -1, 0, 1, 1024]
)
def test_unit_significand_division_preserves_raw_correctly_rounded_bits(
    helper: ct.CDLL, exponent: int
) -> None:
    """Exact binary rescaling keeps raw quotient bits through underflow and overflow."""
    rng = np.random.default_rng(192831)
    mantissas = rng.uniform(0.5, 1.0, size=1026)
    mantissas[:3] = [0.5, 0.75, np.nextafter(1.0, 0.0)]
    significands = rng.uniform(1.0, 2.0, size=mantissas.size)
    significands[-3:] = [1.0, np.nextafter(1.0, 2.0), np.nextafter(2.0, 1.0)]
    powers = -(np.arange(mantissas.size) % 1023)
    factors = np.ldexp(significands, powers)
    factors = np.minimum(factors, 1.0)
    factors[:3] = [np.finfo(float).tiny, np.nextafter(np.finfo(float).tiny, 1), 1.0]
    baseline = np.full_like(factors, np.nan)
    normalized = np.full_like(factors, np.nan)
    pointer = ct.POINTER(ct.c_double)
    assert (
        helper.normalized_division_probe(
            factors.size,
            mantissas.ctypes.data_as(pointer),
            exponent,
            factors.ctypes.data_as(pointer),
            baseline.ctypes.data_as(pointer),
            normalized.ctypes.data_as(pointer),
        )
        == 0
    )
    np.testing.assert_array_equal(baseline.view(np.uint64), normalized.view(np.uint64))


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 31, 32, 127])
@pytest.mark.parametrize(
    "case",
    [
        "dense",
        "ones_zeros",
        "threshold",
        "tiny_first",
        "tiny_last",
        "mixed",
        "budget_boundary",
        "budget_crossing",
        "budget_floor",
    ],
)
def test_ordered_block_active_encoding_and_dead_observables_are_bit_identical(
    helper: ct.CDLL, count: int, case: str
) -> None:
    """Active encodings stay exact; dead rows retain every observable derivative."""
    rng = np.random.default_rng(11309 + count)
    factors = rng.uniform(0.1, 0.99, size=count)
    if case == "ones_zeros":
        factors[::3] = 1
        factors[1::7] = 0
    elif case == "threshold":
        factors[:] = 2.0**-54
        factors[::2] = np.nextafter(2.0**-54, 0.0)
    elif case == "tiny_first" and count:
        factors[0] = np.nextafter(0.0, 1.0)
    elif case == "tiny_last" and count:
        factors[-1] = np.nextafter(0.0, 1.0)
    elif case == "mixed":
        factors = np.ldexp(factors, -rng.integers(0, 1074, size=count))
        factors[::7] = 1
    elif case == "budget_boundary":
        factors[:] = 1
        if count:
            factors[0] = 2.0**-1000
        if count > 1:
            factors[1] = 2.0**-21
    elif case == "budget_crossing":
        factors[:] = 1
        if count:
            factors[0] = 2.0**-1000
        if count > 1:
            factors[1] = np.nextafter(2.0**-21, 0.0)
    elif case == "budget_floor":
        factors[:] = 1
        factors[::2] = np.nextafter(2.0**-1021, 0.0)
        factors[1::2] = np.nextafter(2.0**-1021, np.inf)
    mantissas = np.full(2, np.nan)
    state = np.full(4, -999, dtype=np.int64)
    observable = np.full(2 * (count + 1), np.nan)
    assert (
        helper.encoding_probe(
            count,
            factors.ctypes.data_as(ct.POINTER(ct.c_double)),
            mantissas.ctypes.data_as(ct.POINTER(ct.c_double)),
            state.ctypes.data_as(ct.POINTER(ct.c_int64)),
            observable.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        == 0
    )
    assert state[2] == state[3] == np.count_nonzero(factors == 0)
    if state[2] < 2:
        assert mantissas.view(np.uint64)[0] == mantissas.view(np.uint64)[1]
        assert state[0] == state[1]
    else:
        np.testing.assert_array_equal(observable, 0)
        np.testing.assert_array_equal(
            observable[::2].view(np.uint64), observable[1::2].view(np.uint64)
        )


@pytest.mark.parametrize(
    "factors,expected_cost,expected_zeros",
    [
        ([0.5] * 127, 127, 0),
        ([2.0**-1000, 2.0**-21], 1021, 0),
        ([2.0**-1000, np.nextafter(2.0**-21, 0.0)], 22, 0),
        ([2.0**-1021], 1021, 0),
        ([np.nextafter(2.0**-1021, 0.0)], 0, 0),
        ([np.nextafter(0.0, 1.0)], 0, 0),
        ([0.5, 0, 0, 0.1], 1, 2),
    ],
)
def test_exponent_budget_preserves_normal_intermediates(
    helper: ct.CDLL, factors: list[float], expected_cost: int, expected_zeros: int
) -> None:
    """Long ordinary rows need one final decomposition, not fixed-size blocks."""
    array = np.asarray(factors, dtype=np.float64)
    state = np.full(2, 9999, dtype=np.uint64)
    assert (
        helper.budget_probe(
            len(array),
            array.ctypes.data_as(ct.POINTER(ct.c_double)),
            state.ctypes.data_as(ct.POINTER(ct.c_uint64)),
        )
        == 0
    )
    np.testing.assert_array_equal(state, [expected_cost, expected_zeros])


@pytest.mark.parametrize(
    "exponent", [-2147483648, -137472, -1024, -1023, -1, 0, 1, 1023, 137472, 2147483647]
)
def test_relative_exponent_carrier_roundtrip_is_finite_and_exact(
    helper: ct.CDLL, exponent: int
) -> None:
    carrier = np.full(1, np.nan)
    assert (
        helper.relative_exponent_probe(
            exponent, carrier.ctypes.data_as(ct.POINTER(ct.c_double))
        )
        == exponent
    )
    assert np.isfinite(carrier[0]) and 0.5 <= carrier[0] < 1
    assert int(carrier.view(np.uint64)[0]) & 0xFFFFFFFF == exponent & 0xFFFFFFFF


def pair_probe(
    helper: ct.CDLL,
    counts: tuple[int, int],
    factor: float = 0.3,
    slope: float = 0.2,
    first_bar: float = 0.7,
    second_bar: float = 0.4,
    partials: tuple[float, float] = (1.0, 1.0),
) -> tuple[int, np.ndarray, int]:
    """Observe geometric work and publication separately from row encoding."""
    output = np.full(6, np.nan)
    calls = ct.c_size_t(99)
    status = helper.pullback_probe(
        *counts,
        factor,
        slope,
        first_bar,
        second_bar,
        *partials,
        output.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(calls),
    )
    return status, output, calls.value


@pytest.mark.parametrize(
    "counts", [(2, 2), (3, 2), (2, 1), (1, 2), (1, 1), (1, 0), (0, 1), (0, 0)]
)
@pytest.mark.parametrize("factor", [0.0, 0.3, 1.0])
def test_pair_pullback_skips_only_zero_excluded_derivatives(
    helper: ct.CDLL, counts: tuple[int, int], factor: float
) -> None:
    """Single-zero rows can prune nonzero slots, never their excluded-zero slot."""
    status, output, calls = pair_probe(helper, counts, factor)
    if (counts[0] == 0 and factor == 0) or (counts[1] == 0 and factor == 1):
        assert status == -2 and calls == 1
        return
    assert status == 0
    first_inactive = counts[0] > 1 or (counts[0] == 1 and factor != 0)
    second_inactive = counts[1] > 1 or (counts[1] == 1 and factor != 1)
    if first_inactive and second_inactive:
        assert calls == 0
        np.testing.assert_array_equal(output, 0)
    else:
        assert calls == 1


@pytest.mark.parametrize(
    "counts,factor,expected", [((1, 2), 0.0, 0.14), ((2, 1), 1.0, -0.08)]
)
def test_single_zero_pair_pullback_keeps_nonzero_excluded_derivative(
    helper: ct.CDLL, counts: tuple[int, int], factor: float, expected: float
) -> None:
    status, output, calls = pair_probe(helper, counts, factor)
    assert status == 0 and calls == 1
    np.testing.assert_allclose(output[:2], expected, rtol=1e-15, atol=0)


def test_primal_retains_coordinate_partials_for_reverse(helper: ct.CDLL) -> None:
    """Reverse reuses the exact primal partials without calling coordinate again."""
    output = np.full(4, np.nan)
    calls = np.zeros(2, dtype=np.uintp)
    status = helper.cached_coordinate_probe(
        output.ctypes.data_as(ct.POINTER(ct.c_double)),
        calls.ctypes.data_as(ct.POINTER(ct.c_size_t)),
    )
    assert status == 0
    np.testing.assert_array_equal(calls, [2, 1])
    np.testing.assert_array_equal(output[:2], [1.0, 1.0])
    expected = 0.7 * 0.5 - 0.4 * 0.5
    np.testing.assert_array_equal(output[2:], [expected, expected])


@pytest.mark.parametrize("partials", [(np.nan, 1.0), (1.0, np.inf)])
def test_live_reverse_rejects_invalid_cached_partials(
    helper: ct.CDLL, partials: tuple[float, float]
) -> None:
    status, _, calls = pair_probe(helper, (0, 0), partials=partials)
    assert status == -2 and calls == 1


@pytest.mark.parametrize("mantissa", [0.5, 0.75, np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("exponent", [-2147483648, -1074, -1, 0])
@pytest.mark.parametrize("factor", [np.finfo(float).tiny, 0.3, 1.0])
def test_zero_bar_bound_only_admits_finite_excluded_products(
    helper: ct.CDLL, mantissa: float, exponent: int, factor: float
) -> None:
    """The independent exact-rational upper bound stays below binary64 overflow."""
    from fractions import Fraction

    output = ct.c_double(np.nan)
    bounded = helper.finite_excluded_bound_probe(
        mantissa, exponent, 0, factor, ct.byref(output)
    )
    assert bounded == 1 and np.isfinite(output.value)
    quotient = Fraction.from_float(mantissa) / Fraction.from_float(factor)
    assert quotient < Fraction(2) ** 1022
    assert exponent <= 0


@pytest.mark.parametrize(
    "mantissa,exponent,zeros,factor",
    [
        (np.nan, 0, 0, 0.3),
        (np.inf, 0, 0, 0.3),
        (0.25, 0, 0, 0.3),
        (1.0, 0, 0, 0.3),
        (0.5, 1, 0, 0.3),
        (0.5, 0, 1, 0.0),
        (0.5, 0, 2, 0.3),
        (0.5, 0, 0, 0.0),
        (0.5, 0, 0, np.nextafter(0.0, 1.0)),
        (0.5, 0, 0, np.nextafter(np.finfo(float).tiny, 0.0)),
        (0.5, 0, 0, np.nan),
        (0.5, 0, 0, np.inf),
    ],
)
def test_zero_bar_bound_does_not_admit_unproven_rows(
    helper: ct.CDLL, mantissa: float, exponent: int, zeros: int, factor: float
) -> None:
    output = ct.c_double(np.nan)
    assert (
        helper.finite_excluded_bound_probe(
            mantissa, exponent, zeros, factor, ct.byref(output)
        )
        == 0
    )


@pytest.mark.parametrize("bar", [0.0, -0.0])
@pytest.mark.parametrize("partials", [(1.0, 1.0), (np.nan, 1.0), (1.0, np.inf)])
def test_zero_bar_reverse_preserves_cached_partial_validation(
    helper: ct.CDLL, bar: float, partials: tuple[float, float]
) -> None:
    """Removing finite scalar terms must not suppress invalid geometry partials."""
    output = np.full(3, np.nan)
    status = helper.zero_bar_encoding_probe(
        0.5,
        0,
        0,
        np.finfo(float).tiny,
        bar,
        *partials,
        output.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    assert output[2] == 1
    if np.isfinite(partials).all():
        assert status == 0
        np.testing.assert_array_equal(output[:2], 0)
    else:
        assert status == -2


@pytest.mark.parametrize(
    "mantissa,exponent,zeros,factor,bar",
    [
        (np.nan, 0, 0, 0.3, 0.0),
        (np.inf, 0, 0, 0.3, 0.0),
        (0.5, 2048, 0, 0.3, 0.0),
        (0.5, 2048, 1, 0.0, 0.0),
        (0.5, 0, 0, 0.0, 0.0),
        (0.5, 0, 0, 0.3, np.nan),
        (0.5, 0, 0, 0.3, np.inf),
    ],
)
def test_zero_bar_reverse_keeps_nonfinite_product_rejections(
    helper: ct.CDLL,
    mantissa: float,
    exponent: int,
    zeros: int,
    factor: float,
    bar: float,
) -> None:
    output = np.full(3, np.nan)
    status = helper.zero_bar_encoding_probe(
        mantissa,
        exponent,
        zeros,
        factor,
        bar,
        1.0,
        1.0,
        output.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    assert status == -2 and output[2] == 1


@pytest.mark.parametrize("counts,slope", [((2, 2), 0.2), ((0, 0), 0.0)])
def test_inactive_reverse_does_not_borrow_cached_partials(
    helper: ct.CDLL, counts: tuple[int, int], slope: float
) -> None:
    status, output, calls = pair_probe(
        helper, counts, slope=slope, partials=(np.nan, np.inf)
    )
    assert status == 0 and calls == 0
    np.testing.assert_array_equal(output, 0)


@pytest.mark.parametrize("tail", [0.0, 0.25])
def test_validated_primal_can_terminate_two_zero_rows(
    helper: ct.CDLL, tail: float
) -> None:
    """Only dead native encodings saturate; products and first pullbacks match."""
    for validated in (False, True):
        output = np.full(6, np.nan)
        status = helper.primal_validated_probe(
            tail, validated, output.ctypes.data_as(ct.POINTER(ct.c_double))
        )
        assert status == 0
        assert output[0] == (2 if validated or tail else 3)
        assert output[1] == 0.5
        np.testing.assert_array_equal(output[2:], 0)


def test_invalid_tail_fails_primal_before_dead_row_elision(helper: ct.CDLL) -> None:
    """A late NaN cannot be hidden by an earlier pair of exact zero factors."""
    for validated in (False, True):
        output = np.full(6, 987654.0)
        status = helper.primal_validated_probe(
            np.nan, validated, output.ctypes.data_as(ct.POINTER(ct.c_double))
        )
        assert status == -2
        np.testing.assert_array_equal(output, 987654.0)


@pytest.mark.parametrize("tail", [np.nan, np.inf, -0.1, 1.1])
def test_unvalidated_dead_row_still_rejects_invalid_tail(
    helper: ct.CDLL, tail: float
) -> None:
    assert helper.unvalidated_dead_row_probe(tail) == -2


@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (2, 2)])
def test_flat_slope_never_borrows_unused_normalization_bars(
    helper: ct.CDLL, counts: tuple[int, int]
) -> None:
    status, output, calls = pair_probe(helper, counts, slope=0, first_bar=np.inf)
    assert status == 0 and calls == 0
    np.testing.assert_array_equal(output, 0)


@pytest.mark.parametrize("counts", [(2, 2), (1, 1), (1, 2)])
@pytest.mark.parametrize("invalid_bar", [np.nan, np.inf])
def test_inactive_pullback_does_not_hide_invalid_normalization_bars(
    helper: ct.CDLL, counts: tuple[int, int], invalid_bar: float
) -> None:
    status, _, calls = pair_probe(helper, counts, first_bar=invalid_bar)
    assert status == -2 and calls == 0


@pytest.mark.parametrize(
    "factor,slope",
    [(np.nan, 0), (np.inf, 0.2), (-0.1, 0.2), (1.1, 0.2), (0.3, np.nan), (0.3, np.inf)],
)
def test_dead_pair_pullback_does_not_hide_invalid_panels(
    helper: ct.CDLL, factor: float, slope: float
) -> None:
    status, _, calls = pair_probe(helper, (2, 2), factor, slope)
    assert status == -2 and calls == 0


@pytest.mark.parametrize("atoms", [1, 2, 5, 32, 33, 48, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_scaled_phases_keep_geometry_and_pair_work(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    rng = np.random.default_rng(11300 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(13, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    moved = centers + rng.normal(size=centers.shape) * 0.1
    for geometry in (centers, moved, centers):
        expected = retained.run(helper, points, geometry, owners, seeds, cached, 0)
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 1)
        assert actual[0] == expected[0] == 0
        np.testing.assert_allclose(actual[1], expected[1], rtol=8e-13, atol=2e-13)
        assert actual[2][0] == len(points) * atoms * (atoms - 1) // 2
        assert expected[2][0] == 2 * actual[2][0]
        assert actual[2][2] == 0


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "positive_zero",
        "negative_zero",
        "coincident",
        "collision",
        "nonfinite",
        "tiny",
        "huge",
        "empty",
    ],
)
def test_scaled_phase_edge_semantics(helper: ct.CDLL, case: str) -> None:
    retained.test_cooperative_edge_semantics(helper, case)


def test_scaled_independent_decimal_fd_and_invariances(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(helper)
    retained.test_single_atom_point_reuse_then_collision_is_collective(helper)


def product_probe(
    helper: ct.CDLL, factors: np.ndarray, owner: int = 0, seed: float = 0.7
) -> tuple[int, np.ndarray, np.ndarray]:
    """Publish no partial output when a representation precondition fails."""
    pointer = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_double))
    weights = np.full(len(factors), 12345.0)
    derivatives = np.full_like(factors, 12345.0)
    status = helper.product_probe(
        *factors.shape,
        pointer(factors),
        owner,
        seed,
        pointer(weights),
        pointer(derivatives),
    )
    return status, weights, derivatives


def decimal_product_oracle(
    factors: np.ndarray, owner: int, seed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate direct Decimal products, without log or binary scaling."""
    with localcontext() as context:
        context.prec = 160
        rows = [[Decimal.from_float(value) for value in row] for row in factors]
        products = [np.prod(row, initial=Decimal(1)) for row in rows]
        total = sum(products)
        weights = np.array([float(product / total) for product in products])
        derivatives = np.empty_like(factors)
        for atom, row in enumerate(rows):
            bar = Decimal.from_float(seed) * (
                (Decimal(1) / total if atom == owner else 0)
                - products[owner] / total**2
            )
            for factor in range(len(row)):
                excluded = np.prod(row[:factor] + row[factor + 1 :], initial=Decimal(1))
                derivatives[atom, factor] = float(bar * excluded)
        return weights, derivatives


@pytest.mark.parametrize(
    "case", ["ordinary", "underflow", "subnormal", "single_zero", "two_zeros"]
)
def test_scaled_products_against_independent_decimal(
    helper: ct.CDLL, case: str
) -> None:
    rng = np.random.default_rng(11308)
    factors = rng.uniform(0.1, 0.9, size=(6, 5))
    if case == "underflow":
        factors = np.full((8, 127), 2.0**-54)
        factors[:, -1] *= np.arange(1, 9)
    elif case == "subnormal":
        factors = np.array([[np.nextafter(0.0, 1.0), 0.5], [1.0, 1.0], [0.0, 0.8]])
    elif case == "single_zero":
        factors[0, 0] = 0
    elif case == "two_zeros":
        factors[0, :2] = 0
    status, weights, derivatives = product_probe(helper, factors)
    assert status == 0
    expected_weights, expected_derivatives = decimal_product_oracle(factors, 0, 0.7)
    np.testing.assert_allclose(weights, expected_weights, rtol=2e-14, atol=1e-300)
    np.testing.assert_allclose(
        derivatives, expected_derivatives, rtol=3e-14, atol=1e-300
    )
    if case == "subnormal":
        assert weights[0] == 0 and derivatives[0, 0] > 0
    if case == "two_zeros":
        np.testing.assert_array_equal(derivatives[0], 0)


@pytest.mark.parametrize(
    "case",
    ["all_zero", "negative", "above_one", "nan", "inf", "owner", "seed", "atoms"],
)
def test_scaled_products_reject_without_publication(helper: ct.CDLL, case: str) -> None:
    factors = np.full((3, 2), 0.5)
    owner, seed = 0, 0.7
    if case == "all_zero":
        factors[:, 0] = 0
    elif case in {"negative", "above_one", "nan", "inf"}:
        factors[0, 0] = {
            "negative": -0.5,
            "above_one": 1.1,
            "nan": np.nan,
            "inf": np.inf,
        }[case]
    elif case == "owner":
        owner = len(factors)
    elif case == "seed":
        seed = np.nan
    elif case == "atoms":
        factors = np.full((129, 2), 0.5)
    status, weights, derivatives = product_probe(helper, factors, owner, seed)
    assert status == -2
    np.testing.assert_array_equal(weights, 12345.0)
    np.testing.assert_array_equal(derivatives, 12345.0)


@pytest.mark.parametrize("invalid", [-0.5, 1.1, np.nan, np.inf])
def test_double_zero_row_still_validates_every_later_factor(
    helper: ct.CDLL, invalid: float
) -> None:
    """An independent nonzero row prevents an all-zero normalization rejection."""
    factors = np.array([[0.0, 0.0, invalid], [0.5, 0.5, 0.5]])
    status, weights, derivatives = product_probe(helper, factors)
    assert status == -2
    np.testing.assert_array_equal(weights, 12345.0)
    np.testing.assert_array_equal(derivatives, 12345.0)


def test_scaled_emission_is_deterministic() -> None:
    source = emit_binary_scaled_becke()
    assert source == emit_binary_scaled_becke()
