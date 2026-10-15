"""Execute generated FP64 root sharing against independent high-precision Wick sums."""

import ctypes
import math
import shutil
import subprocess
from decimal import Decimal, localcontext

import pytest
from generativeqc_compiler.integral.coulomb_recurrence_cuda import (
    emit_high_order_coulomb_support,
)


@pytest.fixture(scope="module")
def spatial_reuse(tmp_path_factory: pytest.TempPathFactory):
    """Compile the emitted scientific arithmetic without a runtime or GPU import."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    directory = tmp_path_factory.mktemp("spatial-coulomb-reuse")
    source = directory / "spatial.cpp"
    source.write_text(
        "#include <cstdlib>\n#define __device__\n"
        "template <typename Scalar> struct Vec3 { Scalar x, y, z; };\n"
        "template <typename Scalar> Scalar vec_axis(const Vec3<Scalar>& value, int axis) {\n"
        "  return axis == 0 ? value.x : axis == 1 ? value.y : value.z;\n}\n"
        "double fourth_order_coulomb(unsigned, double, const Vec3<double>&, const double*) {\n"
        "  std::abort();\n}\n"
        + emit_high_order_coulomb_support()
        + '\nextern "C" void evaluate(unsigned maximum_x, unsigned maximum_y, '
        "double rho, const double* difference, const double* boys, double* values) {\n"
        "  const auto workspace = make_high_order_coulomb_workspace<7>(rho,\n"
        "      Vec3<double>{difference[0], difference[1], difference[2]});\n"
        "  double preweighted_boys[8];\n"
        "  for (unsigned order = 0; order < 8; ++order)\n"
        "    preweighted_boys[order] = workspace.negative_two_rho_powers[order] * boys[order];\n"
        "  const unsigned maximum_z = 6 - maximum_x - maximum_y;\n"
        "  for (unsigned axis_x = 0; axis_x < maximum_x + 2; ++axis_x)\n"
        "    for (unsigned axis_y = 0; axis_y < maximum_y + 2; ++axis_y)\n"
        "      for (unsigned axis_z = 0; axis_z < maximum_z + 2; ++axis_z) {\n"
        "        if ((axis_x > maximum_x) + (axis_y > maximum_y) + (axis_z > maximum_z) > 1) continue;\n"
        "        const unsigned state = axis_x + (axis_y << 3U) + (axis_z << 6U);\n"
        "        values[(axis_x * (maximum_y + 2) + axis_y) * (maximum_z + 2) + axis_z] =\n"
        "            high_order_coulomb<7, true>(state, rho, workspace, preweighted_boys);\n"
        "      }\n"
        "}\n"
    )
    library_path = directory / "spatial.so"
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(library_path),
    ]
    cache = shutil.which("sccache") or shutil.which("ccache")
    if cache:
        subprocess.run([cache, "--version"], check=True, capture_output=True)
        command.insert(0, cache)
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    library = ctypes.CDLL(str(library_path))
    library.evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        *([ctypes.POINTER(ctypes.c_double)] * 3),
    ]
    return library.evaluate


def wick_reference(degrees, rho, difference, boys):
    """Evaluate Gaussian spatial derivatives with independent integer pair counts."""
    with localcontext() as context:
        context.prec = 70
        decimal_rho = Decimal(rho)
        decimal_difference = tuple(Decimal(coordinate) for coordinate in difference)
        result = Decimal(0)
        for pairs_x in range(degrees[0] // 2 + 1):
            for pairs_y in range(degrees[1] // 2 + 1):
                for pairs_z in range(degrees[2] // 2 + 1):
                    pairs = (pairs_x, pairs_y, pairs_z)
                    order = sum(degrees) - sum(pairs)
                    term = Decimal(boys[order])
                    if order:
                        term *= (-2 * decimal_rho) ** order
                    for degree, pair_count, coordinate in zip(
                        degrees, pairs, decimal_difference
                    ):
                        multiplicity = math.factorial(degree) // (
                            2**pair_count
                            * math.factorial(pair_count)
                            * math.factorial(degree - 2 * pair_count)
                        )
                        term *= multiplicity
                        if degree != 2 * pair_count:
                            term *= coordinate ** (degree - 2 * pair_count)
                    result += term
        return float(result)


@pytest.mark.parametrize("rho", (0.0, 0.013, 0.37, 1.4, 3.0))
@pytest.mark.parametrize(
    "difference", ((0.3, -0.8, 0.6), (0.0, 0.0, 0.0), (-1.2, 0.05, 1.7))
)
def test_every_component_domain_matches_independent_wick(
    spatial_reuse, rho: float, difference: tuple[float, float, float]
) -> None:
    """Check all 28 boxes, raised faces, zero coordinates and untouched corners."""
    boys = tuple(0.29 / (order + 1) + (-0.07) ** order for order in range(8))
    difference_array = (ctypes.c_double * 3)(*difference)
    boys_array = (ctypes.c_double * 8)(*boys)
    for maximum_x in range(7):
        for maximum_y in range(7 - maximum_x):
            maximum_z = 6 - maximum_x - maximum_y
            bounds = (maximum_x, maximum_y, maximum_z)
            values = (ctypes.c_double * 64)(*([math.nan] * 64))
            spatial_reuse(
                maximum_x, maximum_y, rho, difference_array, boys_array, values
            )
            for axis_x in range(maximum_x + 2):
                for axis_y in range(maximum_y + 2):
                    for axis_z in range(maximum_z + 2):
                        degrees = (axis_x, axis_y, axis_z)
                        offset = (axis_x * (maximum_y + 2) + axis_y) * (
                            maximum_z + 2
                        ) + axis_z
                        if (
                            sum(
                                degree > bound for degree, bound in zip(degrees, bounds)
                            )
                            > 1
                        ):
                            assert math.isnan(values[offset])
                        else:
                            expected = wick_reference(degrees, rho, difference, boys)
                            assert values[offset] == pytest.approx(
                                expected, rel=3e-12, abs=3e-12
                            )
