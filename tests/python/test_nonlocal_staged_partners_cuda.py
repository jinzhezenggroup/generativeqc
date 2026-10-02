"""Real-device ordering/barrier gates for the production VV10 partner schedule.

Run with GENERATIVEQC_TEST_WB97MV_CUDA=1 inside a finite Slurm GPU allocation.
The direct schedule is a scheduling oracle; independent mathematical/complete
molecular gates live in the pair-codegen and WB97M-V endpoint tests.
"""

import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_nonlocal_signed_weight_kernels import _definition

from tools.generate_nonlocal_pair_native import native_header

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def staged_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("explicit real-device qualification required")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache, "nvcc and ccache are required"
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    dispatch = _definition(
        source, "template <Vv10Variant Variant, bool Features, bool Geometry>"
    )
    kernel = _definition(
        source,
        "template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows,",
    )
    directory = tmp_path_factory.mktemp("vv10-staged-device")
    cpp, library = directory / "probe.cu", directory / "probe.so"
    cpp.write_text(
        "#include <cuda_runtime.h>\n#include <cmath>\n#include <cstdint>\n"
        "#include <cstddef>\n#include <vector>\n"
        "namespace generativeqc::dft::nlc { enum class Vv10Variant { vv10, rvv10 }; }\n"
        + native_header()
        + "\nusing namespace generativeqc::dft::nlc;\n"
        "using PairKernelValues = generated::PairValues;\n"
        + dispatch
        + kernel
        + r"""
template<bool Geometry, bool Mask, bool Stage>
void launch(size_t n, const double* x, const uint64_t* indices,
            const uint64_t* count, double* out, int* error) {
  pair_kernel_ordered<Vv10Variant::vv10,true,Geometry,Mask,Stage>
    <<<(n+127)/128,128>>>(0,n,0.7,x,x+3*n,x+4*n,x+5*n,x+6*n,x+7*n,
      x+8*n,x+9*n,indices,count,0.03125,out,out+n,out+2*n,out+3*n,out+6*n,error);
}
// Keep output canaries on device so padded row stores cannot pass unnoticed.
extern "C" int run(size_t n, int geometry, int mask, int stage,
                    const double* input, double* output) {
  std::vector<uint64_t> indices;
  for(size_t j=0;j<n;++j) if(input[9*n+j]!=0.0) indices.push_back(j);
  const uint64_t count=indices.size();
  indices.push_back(count);
  struct Storage {
    double *input{},*output{}; uint64_t* indices{}; int* failed{};
    ~Storage() { cudaFree(input);cudaFree(output);cudaFree(indices);cudaFree(failed); }
  } d;
#define CHECK(call) do { auto error=(call); if(error!=cudaSuccess) return int(error); } while(0)
  CHECK(cudaMalloc(&d.input,10*n*sizeof(double)));
  CHECK(cudaMalloc(&d.output,(7*n+2)*sizeof(double)));
  CHECK(cudaMalloc(&d.indices,indices.size()*sizeof(uint64_t)));
  CHECK(cudaMalloc(&d.failed,sizeof(int)));
  CHECK(cudaMemcpy(d.input,input,10*n*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.output,output,(7*n+2)*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.indices,indices.data(),indices.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d.failed,0,sizeof(int)));
#define RUN(G,M,S) launch<G,M,S>(n,d.input,d.indices,d.indices+count,d.output+1,d.failed)
  if(geometry) {
    if(mask) { if(stage) RUN(true,true,true); else RUN(true,true,false); }
    else { if(stage) RUN(true,false,true); else RUN(true,false,false); }
  } else {
    if(mask) { if(stage) RUN(false,true,true); else RUN(false,true,false); }
    else { if(stage) RUN(false,false,true); else RUN(false,false,false); }
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(output,d.output,(7*n+2)*sizeof(double),cudaMemcpyDeviceToHost));
  int failed=0;
  CHECK(cudaMemcpy(&failed,d.failed,sizeof(int),cudaMemcpyDeviceToHost));
  return failed ? -1 : 0;
}
"""
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-O2",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    probe = ct.CDLL(str(library))
    probe.run.argtypes = [
        ct.c_size_t,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    probe.run.restype = ct.c_int
    return probe


@pytest.mark.parametrize("n", [127, 128, 129, 257, 383, 513])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("domain", ["signed", "sparse", "screened_block", "empty"])
def test_staged_rows_keep_order_at_partial_tiles(
    staged_probe: ct.CDLL, n: int, geometry: bool, domain: str
) -> None:
    rng = np.random.default_rng(8173 + n)
    x = rng.uniform(0.1, 2.0, size=10 * n)
    x[: 3 * n] = rng.normal(size=3 * n)
    weighted = x[9 * n :]
    weighted[::7] *= -1
    # Positive-zero rows retain their weight derivative; negative-zero rows are
    # screened only when the explicit molecular row-mask contract is selected.
    weighted[::11] = 0.0
    if domain == "sparse":
        weighted[::3] = -0.0
    elif domain == "screened_block":
        weighted[:128] = -0.0
    elif domain == "empty":
        weighted[:] = -0.0
    mask = domain != "signed"
    outputs = []
    for stage in (False, True):
        out = np.full(7 * n + 2, 9182.5)
        error = staged_probe.run(
            n,
            geometry,
            mask,
            stage,
            x.ctypes.data_as(ct.POINTER(ct.c_double)),
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        assert error == 0
        assert out[0] == out[-1] == 9182.5
        outputs.append(out)
    np.testing.assert_array_equal(outputs[1], outputs[0])


@pytest.mark.parametrize("geometry", [False, True])
def test_staged_exception_publication_matches_direct(
    staged_probe: ct.CDLL, geometry: bool
) -> None:
    n = 129
    x = np.ones(10 * n)
    x[4 * n + 128] = np.inf
    statuses = []
    for stage in (False, True):
        out = np.full(7 * n + 2, 9182.5)
        statuses.append(
            staged_probe.run(
                n,
                geometry,
                False,
                stage,
                x.ctypes.data_as(ct.POINTER(ct.c_double)),
                out.ctypes.data_as(ct.POINTER(ct.c_double)),
            )
        )
        assert out[0] == out[-1] == 9182.5
    assert statuses == [-1, -1]
