"""Execute emitted CUDA AO arithmetic and launch selection without a GPU.

The oracle uses long-double Leibniz derivatives, independent of the emitted DAG.
CUDA indexing/launches and libm are host stand-ins: this is correctness/work-count
coverage, not device scheduling, compiled resources, or endpoint timing evidence.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.dft.ao_cuda import (
    emit_grid_policy,
    emit_grid_scientific_kernels,
    emit_grid_source,
)

ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from conftest import NativeCxx


def _ao_source() -> tuple[str, str, str]:
    policy = emit_grid_policy()
    end = policy.index(";", policy.index("__constant__ int derivatives")) + 1
    source = emit_grid_scientific_kernels(ao_radial_reuse=True)
    kernels = source[
        source.index("__global__ void ao_kernel(") : source.index(
            "__global__ void feature_kernel("
        )
    ]
    schedule_begin = source.index("void scheduled_ao(")
    schedule = source[schedule_begin : source.index("}  // namespace", schedule_begin)]
    return policy[:end] + "\n}\n", kernels, schedule


def test_ao_probe_schedule_excludes_following_kernel_namespaces() -> None:
    """Appending independent grid kernels must not corrupt the AO-only probe."""
    _, _, schedule = _ao_source()
    assert "}  // namespace" not in schedule
    assert "ao_region_" not in schedule


@pytest.fixture(scope="module")
def ao_probe(tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx) -> Path:
    policy, kernels, schedule = _ao_source()
    # Replace only transport syntax. Execute the actual scheduler, including its
    # launch extents, with a small block cap to exercise grid-stride iterations.
    schedule = re.sub(r"(\w+)<<<(.*?)>>>\(", r"launch(\1, \2, ", schedule)
    harness = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#define __device__
#define __noinline__
#define __constant__
#define __global__
using I = std::int64_t;
using cudaStream_t = int;
struct Index { I x; };
Index blockIdx{},threadIdx{},blockDim{},gridDim{};
I exp_calls=0,axis_calls=0,finite_calls=0,launches=0,last_work=0;
I blocks(I work,I threads) { last_work=work; return std::min(I(2),(work+threads-1)/threads); }
template<class Kernel,class... Args>
void launch(Kernel kernel,I grid,I threads,int shared,int stream,Args... args) {
  assert(shared==0 && stream==7);
  ++launches; gridDim.x=grid; blockDim.x=threads;
  for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<threads;++threadIdx.x) kernel(args...);
}
double finite(double value,int* error,int node) {
  ++finite_calls;
  if(!std::isfinite(value) && !*error) *error=node+1;
  return value;
}
float counted_exp(float x) { ++exp_calls; return std::exp(x); }
double counted_exp(double x) { ++exp_calls; return std::exp(x); }
"""
        + policy
        + r"""
template<class T> T counted_axis_jet(int l,int d,T a,T x) {
  ++axis_calls; return generativeqc_grid_policy::axis_jet(l,d,a,x);
}
using generativeqc_grid_policy::derivatives;
"""
        + kernels.replace("exp(", "counted_exp(").replace(
            "axis_jet(", "counted_axis_jet("
        )
        + schedule
        + r"""
long double oracle_axis(unsigned l,unsigned d,long double a,long double x) {
  const long double h[]={1,-2*a*x,4*a*a*x*x-2*a,-8*a*a*a*x*x*x+12*a*a*x};
  const unsigned choose[4][4]={{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}};
  long double sum=0;
  for(unsigned k=0;k<=std::min(l,d);++k) {
    long double falling=1;
    for(unsigned i=0;i<k;++i) falling*=l-i;
    sum+=choose[d][k]*falling*std::pow(x,l-k)*h[d-k];
  }
  return sum;
}
std::vector<double> fixture(bool expanded) {
  std::vector<double> packed={0,0,0, 0.25,-0.5,0.125,
      1.7,0.4, 0.51,-0.17, 0.13,0.8, 20.0,-0.02};
  for(unsigned atom=0;atom<2;++atom) {
    for(unsigned l=0;l<=3;++l)
      for(int x=l;x>=0;--x)
        for(int y=l-x;y>=0;--y) {
          std::array<double,16> r{};
          r[0]=atom;r[1]=atom;r[2]=3;r[3]=1;
          r[4]=x;r[5]=y;r[6]=l-x-y;r[7]=1;
          packed.insert(packed.end(),r.begin(),r.end());
        }
    if(expanded) {
      // Signed real-spherical d_z2 and f_z3 expansions, all three record slots.
      for(unsigned z=0;z<2;++z) {
        std::array<double,16> r{};
        r[0]=atom;r[1]=atom;r[2]=3;r[3]=3;
        r[4]=2;r[6]=z;r[7]=-0.5;
        r[9]=2;r[10]=z;r[11]=-0.5;
        r[14]=2+z;r[15]=1;
        packed.insert(packed.end(),r.begin(),r.end());
      }
    }
  }
  return packed;
}
int main(int argc,char** argv) {
  assert(argc==6);
  const I jets=std::atoi(argv[1]);
  const bool fp32=std::atoi(argv[2]),expanded=std::atoi(argv[3]);
  const int selected_mode=std::atoi(argv[4]),mode=std::atoi(argv[5]);
  auto basis=fixture(expanded);
  const I full_nao=(basis.size()-14)/16;
  std::vector<double> points={0,0,0, .125,-.25,.375, 1.125,.75,-.875,
                             -2.125,.5,1.75, .25,-.5,.125, .5,.5,.5, -.25,-.125,.25};
  std::vector<std::size_t> ids;
  if(selected_mode) ids={std::size_t(full_nao-1),3,0,8,19};
  else for(I i=0;i<full_nao;++i) ids.push_back(i);
  if(mode==1) points.assign(9,1e30);  // Underflow before axis overflow in FP32.
  if(mode==2) points.clear();
  if(mode==3) ids.clear();
  if(mode==4) points[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode==5) basis[14+7]=std::numeric_limits<double>::infinity();
  const auto* selected=selected_mode ? ids.data() : nullptr;
  const I nao=ids.size(),npoint=points.size()/3,count=jets*nao*npoint;
  std::vector<double> output(count+2,12345.0),baseline(count+2,12345.0);
  int error=0,baseline_error=0;
  scheduled_ao(7,basis.data(),2,4,nao,points.data(),npoint,jets,output.data()+1,&error,selected,fp32);
  assert(output.front()==12345.0 && output.back()==12345.0);
  const I scheduled_exp=exp_calls,scheduled_axis=axis_calls;
  const bool reuse=jets==4 || jets==10;
  assert(launches==I(count!=0));
  assert(!count || last_work==(reuse ? npoint*nao : count));
  assert(finite_calls==count);
  assert(scheduled_exp==3*nao*npoint*(reuse ? 1 : jets));
  assert((error!=0)==(mode==4 || mode==5));
  exp_calls=axis_calls=finite_calls=0;
  if(fp32) launch(ao_kernel_fp32,blocks(count,128),128,0,7,basis.data(),I(2),I(4),nao,
                  points.data(),npoint,jets,baseline.data()+1,&baseline_error,selected);
  else launch(ao_kernel,blocks(count,128),128,0,7,basis.data(),I(2),I(4),nao,
              points.data(),npoint,jets,baseline.data()+1,&baseline_error,selected);
  assert(exp_calls==scheduled_exp*(reuse ? jets : 1));
  const I axes_per_term=reuse ? (jets==4 ? 6 : 9) : 3*jets;
  assert(axis_calls*axes_per_term==scheduled_axis*3*jets);
  assert(finite_calls==count && baseline_error==error);
  for(I i=1;i<=count;++i)
    assert(output[i]==baseline[i] || (std::isnan(output[i]) && std::isnan(baseline[i])));
  if(mode==1) {
    assert(scheduled_axis==0);
    assert(std::all_of(output.begin()+1,output.end()-1,[](double x){return x==0;}));
  }
  if(mode) return 0;
  assert(scheduled_axis>0);
  I jet=0;
  for(int degree=0;degree<=3;++degree)
    for(int dx=degree;dx>=0;--dx)
      for(int dy=degree-dx;dy>=0;--dy) {
        if(jet>=jets) continue;
        const unsigned d[]={unsigned(dx),unsigned(dy),unsigned(degree-dx-dy)};
        for(I point=0;point<npoint;++point)
          for(I ao=0;ao<nao;++ao) {
            const double* r=basis.data()+14+16*ids[ao];
            long double xyz[3],r2=0;
            for(unsigned k=0;k<3;++k) {
              xyz[k]=static_cast<long double>(points[3*point+k])-basis[3*I(r[0])+k];
              r2+=xyz[k]*xyz[k];
            }
            long double expected=0;
            for(I p=I(r[1]);p<I(r[1]+r[2]);++p) {
              const long double alpha=basis[6+2*p];
              for(int term=0;term<int(r[3]);++term) {
                long double value=basis[7+2*p]*std::exp(-alpha*r2)*r[7+4*term];
                for(unsigned k=0;k<3;++k)
                  value*=oracle_axis(unsigned(r[4+4*term+k]),d[k],alpha,xyz[k]);
                expected+=value;
              }
            }
            const double actual=output[1+(jet*npoint+point)*nao+ao];
            const long double tolerance=fp32 ? 3e-6L : 8e-13L;
            assert(std::abs(actual-expected)<=tolerance*(1+std::abs(expected)));
          }
        ++jet;
      }
  // Exact dyadic local coordinates remain unchanged under a large translation.
  for(unsigned k=0;k<6;++k) basis[k]+=67108864.0;
  for(auto& value:points) value+=67108864.0;
  scheduled_ao(7,basis.data(),2,4,nao,points.data(),npoint,jets,baseline.data()+1,
               &baseline_error,selected,fp32);
  assert(std::memcmp(output.data(),baseline.data(),output.size()*sizeof(double))==0);
}
"""
    )
    directory = tmp_path_factory.mktemp("cuda-ao-radial")
    source, binary = directory / "probe.cpp", directory / "probe"
    source.write_text(harness)
    native_cxx.build_executable(
        (source,),
        binary,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off"),
        compile_timeout=60,
    )
    return binary


@pytest.mark.parametrize("jets", [1, 4, 10, 20])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_emitted_ao_matches_scalar_and_independent_oracle(
    ao_probe: Path, jets: int, fp32: bool, expanded: bool, selected: bool
) -> None:
    subprocess.run(
        [
            str(ao_probe),
            str(jets),
            str(int(fp32)),
            str(int(expanded)),
            str(int(selected)),
            "0",
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("jets", [1, 4, 10, 20])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("selected", [False, True])
def test_emitted_ao_underflow_empty_and_finite_errors(
    ao_probe: Path, jets: int, fp32: bool, mode: int, selected: bool
) -> None:
    subprocess.run(
        [str(ao_probe), str(jets), str(int(fp32)), "1", str(int(selected)), str(mode)],
        check=True,
        timeout=10,
    )


def test_production_ao_launches_use_one_compiler_schedule() -> None:
    source = emit_grid_scientific_kernels(ao_radial_reuse=True)
    for jets in (4, 10):
        for suffix in ("", "_fp32"):
            assert (
                f"ao_radial_kernel_{jets}{suffix}<<<blocks(npoint * nao, 128)" in source
            )
    assert "value[" not in _ao_source()[1]
    assert "ao_radial_kernel_20" not in source
    assert "__device__ __noinline__ double axis_jet" in emit_grid_policy()
    for native in ("src/dft/cuda_grid.cu", "src/dft/cuda_xc_kernels.cuh"):
        text = (ROOT / native).read_text()
        assert "scheduled_ao(" in text
        assert "ao_kernel<<<" not in text and "ao_kernel_fp32<<<" not in text
    for native_ks in (False, True):
        generated = emit_grid_source(native_ks=native_ks, ao_radial_reuse=True)[0]
        assert generated.count("void scheduled_ao(") == 1
        assert generated.index("void scheduled_ao(") < generated.index(
            '#include "cuda_grid.cu"'
        )


def test_emitted_ao_cuda_compiles_with_specialized_resources(tmp_path: Path) -> None:
    """Optional compile-only check; never initializes a CUDA device."""
    import json
    import os
    from dataclasses import asdict
    from hashlib import sha256

    from generativeqc_compiler.common.compiler_process import run_compiler
    from generativeqc_compiler.common.cuda_resources import parse_resources

    compiler = os.environ.get("GENERATIVEQC_NVCC") or shutil.which("nvcc")
    if compiler is None:
        pytest.skip("requires nvcc for compile-only CUDA resource evidence")
    policy, kernels, schedule = _ao_source()
    source, output = tmp_path / "ao.cu", tmp_path / "ao.o"
    source.write_text(
        "#include <cuda_runtime.h>\n#include <cmath>\n#include <cstddef>\n"
        "#include <cstdint>\nusing I = std::int64_t;\n"
        "unsigned blocks(I n,unsigned threads) { return (n+threads-1)/threads; }\n"
        "__device__ double finite(double x,int* error,int node) {\n"
        "  if (!isfinite(x)) atomicCAS(error,0,node+1); return x;\n}\n"
        + policy
        + "using namespace generativeqc_grid_policy;\n"
        + kernels
        + schedule
    )
    architecture = os.environ.get("GENERATIVEQC_GRID_CUDA_ARCH", "sm_80")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "-Xptxas=-v",
        "-arch=" + architecture,
        "-c",
        str(source),
        "-o",
        str(output),
    ]
    run = run_compiler(command, 180, label="CUDA AO radial")
    (tmp_path / "ptxas.txt").write_text(run.stdout + run.stderr)
    rows = parse_resources(run.stderr)
    # Retain actual PTXAS observations even when compilation/resource admission
    # fails. CI uploads this directory on success and failure, including source.
    (tmp_path / "resources.json").write_text(
        json.dumps(
            {
                "schema": "generativeqc.cuda-ao-radial-resources.v1",
                "architecture": architecture,
                "source_sha256": sha256(source.read_bytes()).hexdigest(),
                "command": command,
                "returncode": run.returncode,
                "timed_out": run.timed_out,
                "compile_seconds": run.duration_seconds,
                "object_bytes": output.stat().st_size if output.exists() else None,
                "resources": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n"
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert output.stat().st_size > 0
    # Match the length-encoded exact names in Itanium CUDA symbols. Checking
    # broad ao_kernel prefixes could accept an inactive specialization instead.
    for name in (
        "ao_kernel",
        "ao_kernel_fp32",
        "ao_radial_kernel_4",
        "ao_radial_kernel_10",
        "ao_radial_kernel_4_fp32",
        "ao_radial_kernel_10_fp32",
    ):
        selected = [row for row in rows if f"{len(name)}{name}P" in row.function]
        assert len(selected) == 1, (name, rows)
        assert selected[0].registers > 0


def test_radial_reuse_is_explicit_and_source_identity_bound(tmp_path: Path) -> None:
    import sys

    for native_ks in (False, True):
        default = emit_grid_source(native_ks=native_ks)
        scalar = emit_grid_source(native_ks=native_ks, ao_radial_reuse=False)
        reuse = emit_grid_source(native_ks=native_ks, ao_radial_reuse=True)
        assert default == scalar
        assert "ao_radial_kernel_" not in default[0]
        assert "ao_kernel<<<blocks(jets * npoint * nao, 128)" in default[0]
        assert "ao_radial_kernel_4<<<blocks(npoint * nao, 128)" in reuse[0]
        assert default[1] != reuse[1]
    output = tmp_path / "opted-in.cu"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_grid_kernels.py"),
            "--output",
            str(output),
            "--ao-radial-reuse",
        ],
        check=True,
        timeout=30,
    )
    assert (
        output.read_text() == emit_grid_source(native_ks=True, ao_radial_reuse=True)[0]
    )
    for invalid in (None, 1, "true"):
        with pytest.raises(TypeError, match="selector must be boolean"):
            emit_grid_source(ao_radial_reuse=invalid)


def test_native_build_defaults_to_reuse_with_explicit_scalar_opt_out() -> None:
    """Native build policy changes without changing the compiler/JIT API default."""
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert (
        "option(GENERATIVEQC_CUDA_AO_RADIAL_REUSE\n"
        '       "Reuse compiler-owned AO radial and axis work for 4/10 jets" ON)'
    ) in cmake
    registration = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    assert "if(GENERATIVEQC_CUDA_AO_RADIAL_REUSE)" in registration
    assert (
        "list(APPEND _generativeqc_grid_ao_schedule_args --ao-radial-reuse)"
        in registration
    )
    assert "ao_radial_kernel_" not in emit_grid_scientific_kernels()
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    resource_step = workflow.split(
        "      - name: Compile native grid/XC region resources\n", 1
    )[1].split("      - name: ", 1)[0]
    # The validator must compare and qualify the schedule CMake now emits.
    assert "--ao-radial-reuse" in resource_step


@pytest.mark.parametrize("jets,axis_count", [(4, 6), (10, 9)])
@pytest.mark.parametrize("fp32", [False, True])
def test_reuse_emits_each_axis_derivative_once(
    jets: int, axis_count: int, fp32: bool
) -> None:
    """Share only identical DAG results; the existing probe checks exact outputs."""
    source = emit_grid_scientific_kernels(ao_radial_reuse=True)
    suffix = "_fp32" if fp32 else ""
    start = source.index(f"__global__ void ao_radial_kernel_{jets}{suffix}(")
    end = source.index("\n}\n", start)
    kernel = source[start:end]
    assert kernel.count("axis_jet(") == axis_count
    assert kernel.index("if (radial ==") < kernel.index(
        "const " + ("float" if fp32 else "double") + " axis_x0"
    )
    assert kernel.count(" *\n            axis_") == 3 * jets


def test_jit_opt_in_uses_distinct_generated_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.dft import cuda

    def compile_standin(
        compiler: object, cache: Path, path: Path, **kwargs: object
    ) -> Path:
        return path

    monkeypatch.setattr(cuda, "compile_runtime", compile_standin)
    default = cuda.compile_cuda(None, tmp_path)
    explicit_scalar = cuda.compile_cuda(None, tmp_path, ao_radial_reuse=False)
    reuse = cuda.compile_cuda(None, tmp_path, ao_radial_reuse=True)
    assert default == explicit_scalar
    assert default != reuse
    assert "ao_radial_kernel_" not in default.read_text()
    assert "ao_radial_kernel_4" in reuse.read_text()


@pytest.mark.parametrize("emitted_reuse", [False, True])
def test_resource_tool_rejects_mismatched_selector_before_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emitted_reuse: bool
) -> None:
    import argparse

    from tools import validate_native_xc_resources

    (tmp_path / "generated_grid_policy.cu").write_text(
        emit_grid_source(native_ks=True, ao_radial_reuse=emitted_reuse)[0]
    )

    def forbidden_compiler(*args: object, **kwargs: object) -> None:
        pytest.fail("mismatched AO selector reached the CUDA compiler")

    monkeypatch.setattr(
        validate_native_xc_resources, "CudaCompilerAdapter", forbidden_compiler
    )
    with pytest.raises(ValueError, match="source differs from compiler emission"):
        validate_native_xc_resources.run(
            argparse.Namespace(
                generated_dir=tmp_path,
                cmake_source=None,
                ao_radial_reuse=not emitted_reuse,
            )
        )


def test_cuda_ci_compiles_opted_in_ao_and_retains_ptxas() -> None:
    import textwrap

    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    cuda = workflow.split("\n  cuda-compile:\n", 1)[1].split(
        "\n  cuda-resources:\n", 1
    )[0]
    marker = "      - name: Compile opted-in AO radial kernels with release resources\n"
    step = cuda.split(marker, 1)[1].split("      - name: ", 1)[0]
    assert "python3-pytest" in cuda
    assert "if:" not in step and "continue-on-error:" not in step
    assert 'test -x "$CUDACXX"' in step
    assert 'exec ccache "$CUDACXX" "$@"' in step
    assert "GENERATIVEQC_GRID_CUDA_ARCH: sm_120" in step
    assert 'CUDA_VISIBLE_DEVICES: ""' in step
    assert (
        "test_cuda_ao_radial_reuse.py::test_emitted_ao_cuda_compiles_with_specialized_resources"
        in step
    )
    assert "--basetemp build/ao-radial-compile" in step
    assert cuda.index(marker) < cuda.index(
        "      - name: Save CUDA ccache immediately after build"
    )
    artifact = cuda.split(
        "      - name: Preserve opted-in AO radial resource reports\n", 1
    )[1].split("      - name: ", 1)[0]
    # Reports survive failed PR/merge-queue builds. Master push runs maintain
    # the trusted compiler cache and deliberately omit diagnostic artifacts.
    assert "if: ${{ always() && github.event_name != 'push' }}" in artifact
    assert "build/ao-radial-compile" in artifact
    assert "name: ao-radial-release-compile" in artifact
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    subprocess.run(["bash", "-n"], input=script, text=True, check=True, timeout=10)


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "ao_radial_kernel_4",
        "ao_radial_kernel_10",
        "ao_radial_kernel_4_fp32",
        "ao_radial_kernel_10_fp32",
    ],
)
def test_compile_gate_requires_exact_radial_resource_rows_and_retains_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    """Synthetic compiler diagnostics validate the gate, not CUDA resources."""
    import json

    from generativeqc_compiler.common import compiler_process

    names = [
        "ao_kernel",
        "ao_kernel_fp32",
        "ao_radial_kernel_4",
        "ao_radial_kernel_10",
        "ao_radial_kernel_4_fp32",
        "ao_radial_kernel_10_fp32",
    ]
    if missing is not None:
        names[names.index(missing)] = missing + "0"  # Inactive near-match.
    diagnostics = "".join(
        f"ptxas info : Function properties for _Z{len(name)}{name}PKd\n"
        "  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info : Used 32 registers, 0 bytes lmem\n"
        for name in names
    )

    def fake_compile(
        command: list[str], timeout: float, *, label: str
    ) -> compiler_process.CompileResult:
        assert "-c" in command and "--fmad=false" in command and "-Xptxas=-v" in command
        Path(command[-1]).write_bytes(b"synthetic object")
        return compiler_process.CompileResult(0, False, 0.25, "", diagnostics)

    monkeypatch.setenv("GENERATIVEQC_NVCC", "/synthetic/nvcc")
    monkeypatch.setattr(compiler_process, "run_compiler", fake_compile)
    if missing is None:
        test_emitted_ao_cuda_compiles_with_specialized_resources(tmp_path)
    else:
        with pytest.raises(AssertionError):
            test_emitted_ao_cuda_compiles_with_specialized_resources(tmp_path)
    assert (tmp_path / "ptxas.txt").read_text() == diagnostics
    report = json.loads((tmp_path / "resources.json").read_text())
    assert len(report["resources"]) == 6
    assert report["returncode"] == 0 and not report["timed_out"]
    assert {
        "registers",
        "stack_bytes",
        "spill_store_bytes",
        "spill_load_bytes",
        "local_bytes",
    } <= report["resources"][0].keys()


@pytest.mark.parametrize("timed_out", [False, True])
def test_compile_gate_retains_failed_compiler_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    import json

    from generativeqc_compiler.common import compiler_process

    monkeypatch.setenv("GENERATIVEQC_NVCC", "/synthetic/nvcc")
    monkeypatch.setattr(
        compiler_process,
        "run_compiler",
        lambda *args, **kwargs: compiler_process.CompileResult(
            124 if timed_out else 1, timed_out, 0.5, "", "synthetic compilation failure"
        ),
    )
    with pytest.raises(AssertionError, match="synthetic compilation failure"):
        test_emitted_ao_cuda_compiles_with_specialized_resources(tmp_path)
    assert (tmp_path / "ptxas.txt").read_text() == "synthetic compilation failure"
    report = json.loads((tmp_path / "resources.json").read_text())
    assert report["timed_out"] == timed_out
    assert report["object_bytes"] is None
