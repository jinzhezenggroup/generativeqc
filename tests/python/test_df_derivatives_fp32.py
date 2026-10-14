"""Independent gates for the private typed DF polynomial candidate, not an endpoint."""

from __future__ import annotations

import ctypes
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives_cuda import (
    df_derivative_inventory,
    emit_df_derivatives_cpu,
    emit_df_derivatives_cuda,
    emit_df_polynomial_dot_cuda,
)

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture

if TYPE_CHECKING:
    from conftest import NativeCxx

DOUBLE = ctypes.POINTER(ctypes.c_double)
UNSIGNED = ctypes.POINTER(ctypes.c_uint)


@pytest.fixture(scope="module")
def typed_library(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    """Compile both precisions together; type assertions protect arithmetic, not spelling."""
    directory = tmp_path_factory.mktemp("df-typed-polynomials")
    source = directory / "probe.cpp"
    units = []
    declarations = ["#include <type_traits>\n"]
    for auxiliary_g in (False, True):
        suffix = "auxiliary_g_" if auxiliary_g else ""
        for scalar_type in ("double", "float"):
            precision = "fp32_" if scalar_type == "float" else ""
            namespace = (
                f"generativeqc::scf::generated_df_{precision}{suffix}derivatives"
            )
            declarations.append(
                emit_df_derivatives_cpu(
                    auxiliary_g=auxiliary_g, scalar_type=scalar_type
                )
            )
            declarations.append(
                f"static_assert(std::is_same_v<decltype({namespace}::dot(0, "
                f"(const {scalar_type}*)nullptr, 0, (const {scalar_type}*)nullptr, "
                f"0, (const {scalar_type}*)nullptr, (const {scalar_type}*)nullptr)), "
                f"{scalar_type}>);\n"
            )
            name = f"derivative_{precision}{suffix}probe"
            declarations.append(
                f'extern "C" void {name}(unsigned rank, const double* exponents, '
                "const double* centers, const unsigned* powers, double* output) {\n"
                f"  using namespace {namespace};\n"
                "  Vec3 first{centers[0],centers[1],centers[2]};\n"
                "  Vec3 second{centers[3],centers[4],centers[5]};\n"
                "  Vec3 third{centers[6],centers[7],centers[8]};\n"
                "  Angular first_power{powers[0],powers[1],powers[2]};\n"
                "  Angular second_power{powers[3],powers[4],powers[5]};\n"
                "  Angular third_power{powers[6],powers[7],powers[8]};\n"
                "  const auto result=rank==2\n"
                "    ? metric(exponents[0],first,first_power,exponents[2],third,third_power)\n"
                "    : three_center(exponents[0],first,first_power,exponents[1],second,second_power,exponents[2],third,third_power);\n"
                "  const Vec3 channels[]{result.first,result.second,result.third};\n"
                "  for(unsigned center=0;center<3;++center) {\n"
                "    output[3*center]=channels[center].x;\n"
                "    output[3*center+1]=channels[center].y;\n"
                "    output[3*center+2]=channels[center].z;\n"
                "  }\n}\n"
            )
            units.append(name)
    source.write_text("\n".join(declarations))
    binary = directory / "probe.so"
    native_cxx.build_shared(
        [source], binary, compile_args=("-std=c++17", "-O2"), compile_timeout=240
    )
    library = ctypes.CDLL(str(binary))
    for name in units:
        function = getattr(library, name)
        function.argtypes = (ctypes.c_uint, DOUBLE, DOUBLE, UNSIGNED, DOUBLE)
        function.restype = None
    return library


def evaluate(library: Any, fixture: Any, *, mixed: bool) -> np.ndarray:
    """Contract normalized primitive records without borrowing an oracle density."""
    suffix = (
        "auxiliary_g_"
        if max(shell["angular_momentum"] for shell in fixture.inputs["shells"]) == 4
        else ""
    )
    function = getattr(library, f"derivative_{'fp32_' if mixed else ''}{suffix}probe")
    output = np.empty((len(fixture.records), 3, 3))
    for index, record in enumerate(fixture.records):
        exponents = np.ascontiguousarray(record["exponents"])
        centers = np.ascontiguousarray(record["centers"])
        powers = np.ascontiguousarray(record["angular"])
        function(
            len(fixture.inputs["shells"]),
            exponents.ctypes.data_as(DOUBLE),
            centers.ctypes.data_as(DOUBLE),
            powers.ctypes.data_as(UNSIGNED),
            output[index].ctypes.data_as(DOUBLE),
        )
    output *= fixture.records["weight"][:, None, None]
    if len(fixture.inputs["shells"]) == 2:
        output = output[:, [0, 2]]
    return fixture.contract(output)


@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
def test_typed_moments_match_independent_contracted_blocks(
    typed_library: Any, variant: str
) -> None:
    """All s/p/d/f classes plus auxiliary-g metric/three-center witnesses."""
    pytest.importorskip("pyscf")
    signatures = [
        angular for rank in (2, 3) for angular in product(range(4), repeat=rank)
    ] + [(0, 4), (4, 4), (0, 0, 4), (2, 2, 4)]
    for angular in signatures:
        fixture = make_df_derivative_fixture(angular, variant=variant)
        strict = evaluate(typed_library, fixture, mixed=False)
        candidate = evaluate(typed_library, fixture, mixed=True)
        np.testing.assert_allclose(strict, fixture.reference, atol=8e-11, rtol=8e-11)
        for actual, expected in (
            (candidate, fixture.reference),
            (fixture.spherical(candidate), fixture.spherical_reference),
        ):
            np.testing.assert_allclose(
                actual, expected, atol=1e-5, rtol=3e-6, err_msg=fixture.name
            )
        np.testing.assert_allclose(candidate.sum(axis=0), 0, atol=1e-5)


@pytest.mark.parametrize("scale", (1e-2, 1e3))
@pytest.mark.parametrize("angular", ((2, 3), (1, 2, 3), (2, 2, 4)))
def test_diffuse_tight_and_signed_contractions(
    typed_library: Any, scale: float, angular: tuple[int, ...]
) -> None:
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(
        angular, exponent_scale=scale, separation_scale=scale**-0.5
    )
    candidate = evaluate(typed_library, fixture, mixed=True)
    np.testing.assert_allclose(candidate, fixture.reference, atol=1e-5, rtol=3e-6)


def test_diffuse_metric_is_not_qualified_by_typing_alone(typed_library: Any) -> None:
    """Retain the failed outer-domain gate, rather than widening its tolerance.

    This is negative evidence for the raw float graph. A future native owner
    needs a conditioning guard and observable work/retry evidence before it can
    admit this domain. Neither this primitive test nor the generator enables
    public HF-DF Fast.
    """
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(
        (2, 3), exponent_scale=1e-4, separation_scale=100
    )
    strict = evaluate(typed_library, fixture, mixed=False)
    candidate = evaluate(typed_library, fixture, mixed=True)
    np.testing.assert_allclose(strict, fixture.reference, atol=8e-10, rtol=8e-11)
    assert not np.allclose(candidate, fixture.reference, atol=1e-5, rtol=3e-6)


def test_float_convolutions_are_not_rounded_double_outputs(typed_library: Any) -> None:
    """A type-correct interface alone cannot prove that the DAG actually narrowed."""
    exponents = np.array([0.57123456, 0.92345678, 1.35791234])
    centers = np.array([[0.13, -0.31, 0.24], [-0.43, 0.27, 0.51], [0.68, -0.14, -0.22]])
    powers = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint32)
    results = []
    for name in ("derivative_probe", "derivative_fp32_probe"):
        result = np.empty((3, 3))
        getattr(typed_library, name)(
            3,
            exponents.ctypes.data_as(DOUBLE),
            centers.ctypes.data_as(DOUBLE),
            powers.ctypes.data_as(UNSIGNED),
            result.ctypes.data_as(DOUBLE),
        )
        results.append(result)
    assert np.isfinite(results).all()
    assert not np.array_equal(results[1], results[0].astype(np.float32).astype(float))
    np.testing.assert_allclose(results[1], results[0], atol=1e-6, rtol=3e-6)


@pytest.mark.parametrize(
    "emitter",
    (emit_df_derivatives_cpu, emit_df_derivatives_cuda, emit_df_polynomial_dot_cuda),
)
def test_invalid_arithmetic_is_rejected(emitter: Any) -> None:
    with pytest.raises(ValueError, match="scalar type"):
        emitter(scalar_type="half")


def test_candidate_inventory_preserves_equations_not_precision() -> None:
    """Float source identity must not masquerade as the existing strict artifact."""
    strict = df_derivative_inventory()
    candidate = df_derivative_inventory(scalar_type="float")
    assert candidate["programs"] == strict["programs"]
    assert strict["precision"] == "fp64"
    assert candidate["precision"] == "mixed-fp32-polynomials"
    assert candidate["arithmetic"]["geometry"] == "fp64"
    assert candidate["arithmetic"]["moment_convolution"] == "fp32"
    assert candidate["arithmetic"]["contracted_accumulation"] == "consumer-owned"
