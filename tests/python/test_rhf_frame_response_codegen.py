"""Runtime shape lowering agrees with independently exercised RHF matrix IR."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.rhf_orbital_response import build_rhf_frame_response
from generativeqc_compiler.tensor import execute

from tools.generate_rhf_frame_response import (
    STAGES,
    cpu_header,
    cuda_source,
    inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_shapes_match_all_matrix_maps(tmp_path: Path) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ and ccache")
    (tmp_path / "maps.hpp").write_text(cpu_header())
    cases = []
    for o, v in ((1, 1), (1, 3), (2, 2), (3, 2)):
        rng = np.random.default_rng(1807 + o * 10 + v)
        maps = build_rhf_frame_response(o, v)
        for stage in STAGES:
            program = getattr(maps, stage)
            feeds = {
                node.attrs["name"]: np.asarray(rng.normal(size=node.spec.shape))
                for node in program.live_nodes
                if node.op == "input"
            }
            expected = execute(program, feeds).outputs
            block = ["{", f"const std::size_t o={o},v={v};", "Inputs input;"]
            for name in inputs(program):
                values = feeds[name].ravel()
                block += [
                    f"const double {name}[]={{"
                    + ",".join(float(x).hex() for x in values)
                    + "};",
                    f"input.{name}={name};",
                ]
            block += [
                f"std::vector<double> arena({stage}_arena_elements(o,v));",
                f"const auto result=run_{stage}_cpu(o,v,input,arena.data(),arena.size());",
            ]
            for name, array in expected.items():
                block += [
                    f"const double want_{name}[]={{"
                    + ",".join(float(x).hex() for x in array.ravel())
                    + "};",
                    f"for(std::size_t i=0;i<{array.size};++i) if(std::abs(result.{name}[i]-want_{name}[i])>3e-11) return 1;",
                ]
            cases.append("\n".join([*block, "}"]))
    source = tmp_path / "main.cpp"
    source.write_text(
        '#include "maps.hpp"\n#include <vector>\n'
        "using namespace generativeqc::scf::generated::rhf_frame;\n"
        "int main(){\n" + "\n".join(cases) + "\n}"
    )
    executable = tmp_path / "check"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_prepared_lowering_retains_scalar_kernels_and_sticky_audit() -> None:
    source = cuda_source()
    assert "s.contractions[0].execute(" in source
    for name in STAGES:
        assert f"run_{name}_prepared(s) : run_{name}_scalar(s)" in source
        assert f"bind_{name}_prepared(s,context,1,calls,summands)" in source
        assert f"__global__ void {name}_prepared_node_" not in source
    # Multiple maps compose under one sticky arithmetic audit. Only their owner
    # may clear it; a later map must not erase an earlier failed intermediate.
    assert "cudaMemsetAsync" not in source
    assert source == cuda_source()


def test_reference_audit_counts_both_primal_maps(tmp_path: Path) -> None:
    """Exercise the native audit body with the real host-lowered matrix maps.

    Only CUDA transport and the exact provider are replaced; this zero-ERI
    reference has F=h. The audit must charge both full primal executions,
    including the second execution used to validate C.T S C.
    """
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ and ccache")
    (tmp_path / "maps.hpp").write_text(cpu_header())
    native = (ROOT / "src/hf/rhf_frame_response.cu").read_text()
    audit_body = (
        "  void reference_audit("
        + native.split("  void reference_audit(", 1)[1].split("  void weights(", 1)[0]
    )
    source = tmp_path / "audit.cpp"
    source.write_text(
        r"""
#include "maps.hpp"
#include <iostream>
#include <span>
#include <utility>
#include <vector>
namespace generativeqc::scf::generated::rhf_frame {
struct CudaState : Inputs {
  std::size_t o{}, v{}, calls{};
  std::vector<double> arena;
};
PrimalOutputs run_primal_cuda(CudaState& s) {
  ++s.calls;
  return run_primal_cpu(s.o,s.v,s,s.arena.data(),s.arena.size());
}
}
namespace maps=generativeqc::scf::generated::rhf_frame;
using generativeqc::posthf::checked_add;
constexpr double kReferenceTolerance=1e-8;
void require(bool ok,const char* message) {
  if(!ok) throw std::runtime_error(message);
}
struct PhysicalReference {
  std::vector<double> density,fock,hcore,orbital_energies;
};
struct Owner {
  std::size_t n,o,v,nn;
  maps::CudaState state;
  struct { std::size_t contraction_terms{7}; double reference_residual{}; } stats;
  std::vector<double> c,h,smat,fmat,dmat,dfmat;
  double *f,*reference_overlap,*density,*df;
  PhysicalReference ref;
  Owner(std::size_t occupied,std::size_t virtuals)
      : n(occupied+virtuals),o(occupied),v(virtuals),nn(n*n),
        c(nn),h(nn),smat(nn),fmat(nn),dmat(nn),dfmat(nn),
        f(fmat.data()),reference_overlap(smat.data()),density(dmat.data()),df(dfmat.data()) {
    ref.orbital_energies.resize(n);
    for(std::size_t i=0;i<n;++i) {
      c[i*n+i]=smat[i*n+i]=1;
      h[i*n+i]=fmat[i*n+i]=ref.orbital_energies[i]=i<o?-1.0:0.5;
      dmat[i*n+i]=i<o?2.0:0.0;
    }
    ref.density=dmat;ref.fock=fmat;ref.hcore=h;
    state.o=o;state.v=v;state.coefficients=c.data();state.hcore=h.data();state.fock_ao=f;
    state.arena.resize(maps::primal_arena_elements(o,v));
  }
  void begin() {}
  void finish() {}
  std::vector<double> download(const double* source,std::size_t count) {
    return {source,source+count};
  }
  void potential(const double*,double* output) { std::fill_n(output,nn,0.0); }
"""
        + audit_body
        + r"""
};
int main() {
  try {
    for(auto [o,v]:{std::pair{1U,1U},std::pair{2U,3U},std::pair{9U,11U}}) {
      Owner owner(o,v);
      owner.reference_audit(owner.ref);
      require(owner.state.calls==2,"reference audit must execute both primal maps");
      require(owner.state.fock_ao==owner.f,"reference audit did not restore Fock input");
      const auto expected=7+2*maps::primal_contraction_terms(o,v);
      if(owner.stats.contraction_terms!=expected) {
        std::cerr<<"reference audit contraction_terms: "<<owner.stats.contraction_terms
                 <<" expected "<<expected<<'\n';
        return 1;
      }
    }
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
"""
    )
    executable = tmp_path / "audit"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
