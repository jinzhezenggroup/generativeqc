"""Independent arithmetic gates for typed shell schedules, not public admission."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from generativeqc_compiler.integral.df_rys import emit_df_rys_cuda
from generativeqc_compiler.integral.df_rys_shell import (
    RYS_SHELL_CLASSES,
    emit_df_rys_policy_cpp,
    emit_df_rys_shell_cuda,
)
from generativeqc_compiler.integral.df_shell_derivatives import (
    SHELL_CLASSES,
    emit_df_shell_derivatives_cuda,
)
from test_semilocal_fp32_codegen import _compile_probe

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture


@pytest.fixture(scope="module", params=("host", "cuda"))
def shell_backend(request: pytest.FixtureRequest) -> str:
    """Do not touch a real device outside the explicit scheduled test contract."""
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
            pytest.skip("scheduled CUDA shell qualification is opt-in")
        assert os.environ.get("SLURM_JOB_ID"), "CUDA shell tests require finite srun"
    return request.param


@pytest.fixture(scope="module", params=("polynomial", "rys"))
def shell_probe(
    request: pytest.FixtureRequest,
    tmp_path_factory: Any,
    native_cxx: Any,
    shell_backend: str,
) -> tuple[str, Any]:
    """Compile strict/float owners together, rejecting implicit double promotion.

    Header filenames deliberately contain strict type/namespace spellings. The
    class-local combined-header contract must not rewrite these caller paths.
    Gaussian preparation and final translation channels deliberately stay double.
    """
    algorithm = request.param
    classes = RYS_SHELL_CLASSES if algorithm == "rys" else SHELL_CLASSES
    if shell_backend == "cuda":
        classes = _cuda_classes(algorithm)
    directory = tmp_path_factory.mktemp("df-shell-fp32")
    if shell_backend == "host":
        (directory / "cuda_runtime.h").write_text("")
    (directory / "generated_df_derivatives.cuh").write_text(emit_df_derivatives_cuda())
    (directory / "generated_df_rys.cuh").write_text(emit_df_rys_cuda())
    policy_header = "double_generated_df_shell_policy.hpp"
    (directory / policy_header).write_text(emit_df_rys_policy_cpp())
    generated = [
        "#include <cmath>",
        "#include <type_traits>",
        "#include <cuda_runtime.h>"
        if shell_backend == "cuda"
        else "#define __device__\n#define __forceinline__ inline\n"
        "#define __noinline__ __attribute__((noinline))",
    ]
    for scalar_type in ("double", "float"):
        shell_header = (
            "generated_df_shell_derivatives.cuh"
            if scalar_type == "double"
            else "generated_df_fp32_shell_derivatives.cuh"
        )
        (directory / shell_header).write_text(
            emit_df_shell_derivatives_cuda(classes=classes, scalar_type=scalar_type)
        )
        generated.append(f'#include "{shell_header}"')
        if algorithm == "rys":
            header = f"double_generated_df_shell_{scalar_type}_rys.cuh"
            (directory / header).write_text(
                emit_df_rys_shell_cuda(
                    classes=classes,
                    policy_header=policy_header,
                    scalar_type=scalar_type,
                )
            )
            generated.append(f'#include "{header}"')
    generated.append("using namespace generativeqc::scf;")
    for mode, scalar_type in enumerate(("double", "float")):
        namespace = "generated_df_fp32_shell" if mode else "generated_df_shell"
        owner = "RysShell" if algorithm == "rys" else "Shell"
        prepare = "generated_df_derivatives::prepare_geometry"
        if algorithm == "rys":
            prepare += "_rys_fp32_consumer" if mode else "_rys"
            prepare += "<Math::nroots>"
        generated.append(
            f"template<unsigned AngularA,unsigned AngularB,unsigned AngularC>\n"
            f"__device__ void evaluate_{mode}(const double* input,double* output) {{\n"
            f"  using Math={namespace}::{owner}<AngularA,AngularB,AngularC>;\n"
            f"  static_assert(std::is_same_v<decltype(&Math::accumulate), void(*)(unsigned,"
            f"{scalar_type},{scalar_type},const generated_df_derivatives::Geometry&,"
            f"const {scalar_type}*,{scalar_type},{scalar_type}*)>);\n"
            "  generated_df_derivatives::Geometry geometry;\n"
            "  const generated_df_derivatives::Vec3 first{input[3],input[4],input[5]};\n"
            "  const generated_df_derivatives::Vec3 second{input[6],input[7],input[8]};\n"
            "  const generated_df_derivatives::Vec3 third{input[9],input[10],input[11]};\n"
            f"  {prepare}(input[0],first,input[1],second,input[2],third,"
            "AngularA+AngularB+AngularC,geometry);\n"
            f"  {scalar_type} cache[3*Math::axis_size],independent[6]{{}};\n"
            "  for(unsigned lane=0;lane<32;++lane) Math::prepare(geometry,cache,lane,32);\n"
            f"  Math::accumulate(unsigned(input[12]),{scalar_type}(input[0]),"
            f"{scalar_type}(input[1]),geometry,cache,{scalar_type}(input[13]),independent);\n"
            "  const auto value=Math::finish(independent);\n"
            "  for(unsigned center=0;center<3;++center)\n"
            "    for(unsigned axis=0;axis<3;++axis) output[3*center+axis]=value.gradient[center][axis];\n"
            "}\n"
        )
    cases = "\n".join(
        f"case {16 * first + 4 * second + third}: "
        f"if(mode) evaluate_1<{first},{second},{third}>(input,output); "
        f"else evaluate_0<{first},{second},{third}>(input,output); return 0;"
        for first, second, third in classes
    )
    generated.append(
        "__device__ int point_value(const double* input,double* output,int mode) {\n"
        f"switch(unsigned(input[14])) {{\n{cases}\n}} return 1;\n}}"
    )
    return algorithm, _compile_probe(
        generated, 15, 9, directory, native_cxx, shell_backend, strict_promotion=True
    )


def _evaluate_fixture(probe: Any, fixture: Any, mode: int) -> np.ndarray:
    """Apply signed primitive weights inside the typed kernel, before widening."""
    values = []
    for record in fixture.records:
        degrees = np.sum(record["angular"], axis=1)
        item = 0
        for degree, powers in zip(degrees, record["angular"], strict=True):
            row = int(powers[1] + powers[2])
            item = (
                item * ((degree + 1) * (degree + 2) // 2)
                + row * (row + 1) // 2
                + powers[2]
            )
        inputs = np.concatenate(
            (
                record["exponents"],
                record["centers"].ravel(),
                [item, record["weight"], degrees @ np.array([16, 4, 1])],
            )
        )
        values.append(probe(inputs, mode).reshape(3, 3))
    return fixture.contract(np.asarray(values))


def _cuda_classes(algorithm: str) -> tuple[tuple[int, int, int], ...]:
    """Keep the CUDA translation unit bounded to the executed schedule witnesses."""
    classes = ((0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 3), (2, 2, 2))
    return classes + ((3, 3, 3),) if algorithm == "polynomial" else classes


@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
def test_typed_shells_match_independent_contracted_blocks(
    shell_probe: tuple[str, Any], shell_backend: str, variant: str
) -> None:
    """All host classes and representative CUDA schedules pass the same gates."""
    pytest.importorskip("pyscf")
    algorithm, probe = shell_probe
    classes = RYS_SHELL_CLASSES if algorithm == "rys" else SHELL_CLASSES
    if shell_backend == "cuda":
        classes = _cuda_classes(algorithm)
    for angular in classes:
        fixture = make_df_derivative_fixture(angular, variant=variant)
        for mode in (0, 1):
            actual = _evaluate_fixture(probe, fixture, mode)
            absolute, relative = (1e-5, 3e-6) if mode else (8e-11, 8e-11)
            for result, expected in (
                (actual, fixture.reference),
                (fixture.spherical(actual), fixture.spherical_reference),
            ):
                np.testing.assert_allclose(
                    result, expected, atol=absolute, rtol=relative, err_msg=fixture.name
                )
            np.testing.assert_allclose(actual.sum(axis=0), 0, atol=absolute)


@pytest.mark.parametrize("scale", (1e-2, 1e3))
def test_diffuse_and_tight_shells(shell_probe: tuple[str, Any], scale: float) -> None:
    """Use physical exponent/distance scaling without making it admission policy."""
    pytest.importorskip("pyscf")
    _, probe = shell_probe
    fixture = make_df_derivative_fixture(
        (2, 1, 3), exponent_scale=scale, separation_scale=scale**-0.5
    )
    actual = _evaluate_fixture(probe, fixture, 1)
    np.testing.assert_allclose(actual, fixture.reference, atol=1e-5, rtol=3e-6)


@pytest.mark.parametrize(
    "emitter", (emit_df_shell_derivatives_cuda, emit_df_rys_shell_cuda)
)
def test_invalid_shell_arithmetic_is_rejected(emitter: Any) -> None:
    with pytest.raises(ValueError, match="scalar type"):
        emitter(scalar_type="half")
