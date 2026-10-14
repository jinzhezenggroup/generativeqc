"""Real-device independent checks of the private typed DF polynomial algebra."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from test_semilocal_fp32_codegen import _compile_probe

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1",
    reason="requires explicit scheduled CUDA polynomial qualification",
)


@pytest.fixture(scope="module")
def polynomial_probe(tmp_path_factory: Any, native_cxx: Any) -> Any:
    """Use the cached NVCC transport, never NVRTC or an unscheduled GPU."""
    assert os.environ.get("SLURM_JOB_ID"), "CUDA polynomial tests require finite srun"
    generated = ["#include <cmath>", "#include <cuda_runtime.h>"]
    bodies = []
    for mode, (auxiliary_g, scalar_type) in enumerate(
        ((False, "double"), (False, "float"), (True, "double"), (True, "float"))
    ):
        namespace = "generativeqc::scf::generated_df_"
        namespace += "fp32_" if scalar_type == "float" else ""
        namespace += "auxiliary_g_" if auxiliary_g else ""
        namespace += "derivatives"
        generated.append(
            emit_df_derivatives_cuda(auxiliary_g=auxiliary_g, scalar_type=scalar_type)
        )
        bodies.append(
            f"if(mode=={mode % 2} && unsigned(input[22])=={int(auxiliary_g)}) {{ using namespace {namespace};\n"
            "  const Vec3 first{input[3],input[4],input[5]};\n"
            "  const Vec3 second{input[6],input[7],input[8]};\n"
            "  const Vec3 third{input[9],input[10],input[11]};\n"
            "  const Angular first_power{unsigned(input[12]),unsigned(input[13]),unsigned(input[14])};\n"
            "  const Angular second_power{unsigned(input[15]),unsigned(input[16]),unsigned(input[17])};\n"
            "  const Angular third_power{unsigned(input[18]),unsigned(input[19]),unsigned(input[20])};\n"
            "  const auto response=input[21]==2\n"
            "    ? metric(input[0],first,first_power,input[2],third,third_power)\n"
            "    : three_center(input[0],first,first_power,input[1],second,second_power,input[2],third,third_power);\n"
            "  const Vec3 channels[]{response.first,response.second,response.third};\n"
            "  for(unsigned center=0;center<3;++center) {\n"
            "    output[3*center]=channels[center].x;\n"
            "    output[3*center+1]=channels[center].y;\n"
            "    output[3*center+2]=channels[center].z;\n"
            "  }\n  return 0;\n}\n"
        )
    generated.append(
        "__device__ int point_value(const double* input,double* output,int mode) {\n"
        + "\n".join(bodies)
        + "return 1;\n}\n"
    )
    return _compile_probe(
        generated,
        23,
        9,
        tmp_path_factory.mktemp("df-polynomial-cuda"),
        native_cxx,
        "cuda",
    )


@pytest.mark.parametrize(
    "angular", ((0, 0), (2, 3), (4, 4), (0, 0, 0), (1, 1, 1), (2, 2, 3), (2, 2, 4))
)
@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
def test_cuda_polynomials_match_independent_blocks(
    polynomial_probe: Any, angular: tuple[int, ...], variant: str
) -> None:
    """Cover metric/auxiliary response, signed contraction and both representations."""
    fixture = make_df_derivative_fixture(angular, variant=variant)
    for mixed in (False, True):
        output = []
        for record in fixture.records:
            inputs = np.concatenate(
                (
                    record["exponents"],
                    record["centers"].ravel(),
                    record["angular"].ravel(),
                    [len(angular), int(max(angular) == 4)],
                )
            )
            output.append(polynomial_probe(inputs, int(mixed)).reshape(3, 3))
        values = np.asarray(output) * fixture.records["weight"][:, None, None]
        if len(angular) == 2:
            values = values[:, [0, 2]]
        actual = fixture.contract(values)
        absolute, relative = (1e-5, 3e-6) if mixed else (8e-11, 8e-11)
        np.testing.assert_allclose(
            actual, fixture.reference, atol=absolute, rtol=relative
        )
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=absolute,
            rtol=relative,
        )
