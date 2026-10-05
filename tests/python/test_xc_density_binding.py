"""Independent science and executable preparation gates for density binding."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.dft.xc_bilinear import density_summand
from generativeqc_compiler.dft.xc_contraction_cuda import XcMatrixSchedule, _emit_tiled
from generativeqc_compiler.dft.xc_density_lowering import (
    density_portfolio,
    emit_density_binding,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_density_semantics_preserve_both_matrix_triangles() -> None:
    graph, root = density_summand()
    rng = np.random.default_rng(1889)
    # Deliberately nonsymmetric inputs detect a plain GEMM substitution. This
    # tests the operation itself; the endpoint still validates symmetric input.
    density = rng.normal(size=(2, 7, 7))
    ao = rng.normal(size=(4, 9, 7))
    (summands,) = evaluate_array_graph(
        graph,
        (root,),
        {
            "left": density[:, None, None, :, :],
            "right": density.swapaxes(-1, -2)[:, None, None, :, :],
            "ao": ao[None, :, :, None, :],
        },
    )
    expected = np.einsum("smn,jpn->sjpm", (density + density.swapaxes(-1, -2)) / 2, ao)
    np.testing.assert_allclose(summands.sum(axis=-1), expected, rtol=2e-14, atol=2e-14)


def test_generated_density_binding_executes_shared_selection(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host compiler")
    source = r"""
#include "runtime/lowering_binding.hpp"
using I = std::int64_t;
using namespace generativeqc::runtime;
using CudaXcDensityLauncher=void(*)(int,const double*,const double*,I,I,I,I,double*,int*,const std::size_t*,I);
struct CudaXcDensityBinding {
 CudaXcDensityLauncher launch;
 NativeLoweringCandidate candidate;
 NativeLoweringPrecision precision;
 bool retained_incumbent;
};
template<bool Mixed,bool Tiled>
void launch_density_product(int,const double*,const double*,I,I,I,I,double*,int*,const std::size_t*,I) {}
"""
    emitted = _emit_tiled(XcMatrixSchedule(16))
    source += emitted[
        emitted.index("inline bool tiled_xc_admitted(") : emitted.index(
            "template <bool Mixed, bool Tiled>"
        )
    ]
    source += emit_density_binding(16, emitted)
    source += r"""
int main() {
 const auto strict=strict_fp64_precision();
 const auto mixed=fp32_compute_fp64_accumulation("dft.cuda.auto/density-contraction-v1");
 for (I n: {I(0),I(1),I(15),I(16),I(17),I(128)}) {
   for (I count: {I(1),I(13),I(16),I(17),I(256)}) {
     for (bool low: {false,true}) {
       const auto binding=prepare_density_binding(n,count,2,4,low?mixed:strict,50);
       if(!binding.retained_incumbent || binding.candidate.cost.kernel_ns) return 1;
       const bool tile=n>=16 && count>=16;
       auto expected=low ? (tile?launch_density_product<true,true>:launch_density_product<true,false>)
                         : (tile?launch_density_product<false,true>:launch_density_product<false,false>);
       if(binding.launch!=expected || binding.precision.arithmetic.is_strict_fp64()==low) return 2;
       if(binding.candidate.provider!="generated.cuda" || !binding.candidate.capture_safe) return 3;
       if(low && (binding.precision.refinement.empty() || binding.precision.audit.empty() || binding.precision.casts.empty())) return 4;
     }
   }
 }
 auto reject=[](auto action){try{action();}catch(const std::exception&){return true;}return false;};
 for (auto bad: {PrecisionDirective{PrecisionDtype::Fp32,PrecisionDtype::Fp32,PrecisionDtype::Fp32,"unqualified"},
                  fp32_compute_fp64_accumulation("unqualified")})
   if(!reject([&]{prepare_density_binding(16,16,1,1,bad,1);})) return 5;
 if(!reject([&]{prepare_density_binding(-1,16,1,1,strict,1);})) return 6;
 if(!reject([&]{prepare_density_binding(16,0,1,1,strict,1);})) return 7;
 if(!reject([&]{prepare_density_binding(INT64_MAX,16,2,4,strict,1);})) return 8;
 if(!reject([&]{prepare_density_binding(16,16,1,1,strict,0);})) return 9;
}
"""
    unit, binary = tmp_path / "binding.cpp", tmp_path / "binding"
    unit.write_text(source)
    compile_owner(compiler, tmp_path, [unit], binary)
    result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gemm_competes_for_the_same_science_without_admitting_mixed_rounding() -> None:
    request, candidates, _, _ = density_portfolio(16, "qualification-source")
    assert len(candidates) == 5
    assert dict(request.semantics)["reduction"] == "sum-nu"
    generated, library = candidates[1], candidates[-1]
    assert generated.request is library.request is request
    assert generated.execution is not None and library.execution is not None
    assert generated.execution.precision == library.execution.precision
    assert generated.execution.topology.reduction == "increasing-nu"
    assert library.execution.topology.reduction == "provider-reproducible"
    assert library.execution.precision.directive.compute_dtype == "float64"
    assert library.execution.cache_bytes > 0
    assert candidates[2].execution is not None
    assert candidates[2].execution.precision != library.execution.precision
