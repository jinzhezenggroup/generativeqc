"""Real-device ordering and predicate gates for molecular VV10 admission.

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


@pytest.fixture(
    scope="module",
    params=(0, 4, 8, "packet", "packet-hoisted"),
    ids=(
        "ordered",
        "parallel",
        "parallel-wide",
        "parallel-packet",
        "parallel-packet-hoisted",
    ),
)
def admission_probe(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> ct.CDLL:
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("explicit real-device qualification required")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(0)
    architecture = f"sm_{device['major']}{device['minor']}"
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
    admission = _definition(
        source, "__global__ void admit_molecular_pair_domain_kernel("
    )
    directory = tmp_path_factory.mktemp("vv10-admission-device")
    cpp, library = directory / "probe.cu", directory / "probe.so"
    use_packet = request.param in ("packet", "packet-hoisted")
    pair_rows = request.param if request.param in (4, 8) else 4
    cpp.write_text(
        "#include <cuda_runtime.h>\n#include <cmath>\n#include <cstdint>\n"
        "#include <cstddef>\n#include <vector>\n"
        "namespace generativeqc::dft::nlc { enum class Vv10Variant { vv10, rvv10 }; }\n"
        + native_header()
        + "\nusing namespace generativeqc::dft::nlc;\n"
        "using PairKernelValues = generated::PairValues;\n"
        + dispatch
        + admission
        + kernel
        + f"constexpr bool use_parallel = {str(bool(request.param)).lower()};\n"
        + f"constexpr bool use_packet = {str(use_packet).lower()};\n"
        + f"constexpr unsigned pair_rows = {pair_rows};\n"
        + f"constexpr bool retain_scale_proofs = {str(request.param == 'packet-hoisted').lower()};\n"
        + r"""
template<bool Geometry, bool Mask, bool Admit, Vv10Variant Variant = Vv10Variant::vv10>
void launch(size_t n, const double* x, const uint64_t* indices,
            const uint64_t* count, double* out, int* error, int* rejected,
            bool mixed, uint64_t* work) {
  const int* gate = nullptr;
  if constexpr (Mask && Admit && Variant == Vv10Variant::vv10) {
    cudaMemsetAsync(rejected,0,sizeof(int));
    admit_molecular_pair_domain_kernel<<<(n+127)/128,128>>>(
      n,x,x+3*n,x+4*n,x+5*n,x+6*n,x+7*n,x+8*n,x+9*n,rejected);
    gate=rejected;
  }
  pair_kernel_ordered<Variant,true,Geometry,Mask,false>
    <<<(n+127)/128,128>>>(0,n,0.7,x,x+3*n,x+4*n,x+5*n,x+6*n,x+7*n,
      x+8*n,x+9*n,indices,count,0.03125,out,out+n,out+2*n,out+3*n,out+6*n,error,gate,mixed,work);
  if constexpr (Mask && Admit && Variant == Vv10Variant::vv10) {
    if constexpr(use_parallel) {
      const generated::ParallelPairInputs input{
        n,x,x+3*n,x+4*n,x+5*n,x+6*n,x+7*n,x+8*n,x+9*n,indices,count,0.7,0.03125,rejected,mixed,work};
      const generated::ParallelPairOutputs output{out,out+n,out+2*n,out+3*n,out+6*n,error};
      generated::parallel_pair_initialize<Geometry><<<(n+127)/128,128>>>(input,output);
      if constexpr(use_packet)
        generated::parallel_pair_packet_kernel<Geometry, retain_scale_proofs><<<8,128>>>(input,output);
      else
        generated::parallel_pair_kernel<Geometry, pair_rows, pair_rows == 8><<<8,128>>>(input,output);
      generated::parallel_pair_zero_rows<<<8,128>>>(input,output);
      generated::parallel_pair_finalize<Geometry><<<(n+127)/128,128>>>(input,output);
    } else {
      pair_kernel_ordered<Vv10Variant::vv10,true,Geometry,Mask,true>
        <<<(n+127)/128,128>>>(0,n,0.7,x,x+3*n,x+4*n,x+5*n,x+6*n,x+7*n,
          x+8*n,x+9*n,indices,count,0.03125,out,out+n,out+2*n,out+3*n,out+6*n,error,gate,mixed,work);
    }
  }
}
// Keep output canaries on device so padded row stores cannot pass unnoticed.
uint64_t last_work[2]{};
template<Vv10Variant Variant>
int evaluate(size_t n, int geometry, int mask, int stage,
             const double* input, double* output, bool mixed) {
  std::vector<uint64_t> indices;
  for(size_t j=0;j<n;++j) if(input[9*n+j]!=0.0) indices.push_back(j);
  const uint64_t count=indices.size();
  indices.push_back(count);
  struct Storage {
    double *input{},*output{}; uint64_t *indices{},*work{}; int* failed{}; int* rejected{};
    ~Storage() { cudaFree(input);cudaFree(output);cudaFree(indices);cudaFree(failed);cudaFree(rejected);cudaFree(work); }
  } d;
#define CHECK(call) do { auto error=(call); if(error!=cudaSuccess) return int(error); } while(0)
  CHECK(cudaMalloc(&d.input,10*n*sizeof(double)));
  CHECK(cudaMalloc(&d.output,(7*n+2)*sizeof(double)));
  CHECK(cudaMalloc(&d.indices,indices.size()*sizeof(uint64_t)));
  CHECK(cudaMalloc(&d.failed,sizeof(int)));
  CHECK(cudaMalloc(&d.rejected,sizeof(int)));
  CHECK(cudaMalloc(&d.work,2*sizeof(uint64_t)));
  CHECK(cudaMemset(d.work,0,2*sizeof(uint64_t)));
  CHECK(cudaMemcpy(d.input,input,10*n*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.output,output,(7*n+2)*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.indices,indices.data(),indices.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d.failed,0,sizeof(int)));
#define RUN(G,M,S) launch<G,M,S,Variant>(n,d.input,d.indices,d.indices+count,d.output+1,d.failed,d.rejected,mixed,d.work)
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
  CHECK(cudaMemcpy(last_work,d.work,2*sizeof(uint64_t),cudaMemcpyDeviceToHost));
  return failed ? -1 : 0;
}
extern "C" uint64_t pair_work(int index) { return last_work[index]; }
extern "C" int run(size_t n,int geometry,int mask,int stage,const double* input,double* output) {
  return evaluate<Vv10Variant::vv10>(n,geometry,mask,stage,input,output,false);
}
extern "C" int run_mixed(size_t n,int geometry,int mask,int stage,const double* input,double* output) {
  return evaluate<Vv10Variant::vv10>(n,geometry,mask,stage,input,output,true);
}
extern "C" int run_rvv10(size_t n,int geometry,int mask,int stage,const double* input,double* output) {
  return evaluate<Vv10Variant::rvv10>(n,geometry,mask,stage,input,output,false);
}
extern "C" int run_rvv10_mixed(size_t n,int geometry,int mask,int stage,const double* input,double* output) {
  return evaluate<Vv10Variant::rvv10>(n,geometry,mask,stage,input,output,true);
}
"""
    )
    # Cache CUDA compilation separately; linking a shared object is not cacheable.
    obj = directory / "probe.o"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-O2",
            f"-arch={architecture}",
            "-c",
            "-Xcompiler",
            "-fPIC",
            str(cpp),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        [compiler, "--shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
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
    for name in ("run_mixed", "run_rvv10", "run_rvv10_mixed"):
        getattr(probe, name).argtypes = probe.run.argtypes
        getattr(probe, name).restype = ct.c_int
    probe.pair_work.argtypes = [ct.c_int]
    probe.pair_work.restype = ct.c_uint64
    return probe


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("extreme", [False, True, "partial"])
def test_mixed_pairs_count_executed_arithmetic_and_match_strict(
    admission_probe: ct.CDLL, variant: str, geometry: bool, extreme: bool | str
) -> None:
    """Exercise the production pair consumers, reductions and strict fallback."""
    n = 129
    rng = np.random.default_rng(69281)
    values = rng.uniform(0.1, 2.0, 10 * n)
    values[: 3 * n] = rng.normal(size=3 * n)
    values[9 * n :: 7] = -0.0
    if extreme is True:
        values[4 * n : 5 * n] = 2.0**10
    elif extreme == "partial":
        # Mixed and strict legs must keep their own admission as packets rotate,
        # including padded partners and masked rows. Coincident points and zero
        # feature chains isolate admission from cancellation in derived forces.
        # rVV10 still needs a nonzero kappa derivative for its preconditioner.
        values[: 3 * n] = 0.0
        values[6 * n : 9 * n] = 0.0
        if variant == "rvv10":
            values[8 * n : 9 * n] = 1.0
        values[4 * n : 5 * n : 11] = 2.0**10
        values[5 * n : 6 * n : 13] = 2.0**10
    strict = admission_probe.run if variant == "vv10" else admission_probe.run_rvv10
    mixed = (
        admission_probe.run_mixed
        if variant == "vv10"
        else admission_probe.run_rvv10_mixed
    )
    results = []
    for function in (strict, mixed):
        output = np.full(7 * n + 2, 9182.5)
        assert (
            function(
                n,
                geometry,
                True,
                True,
                values.ctypes.data_as(ct.POINTER(ct.c_double)),
                output.ctypes.data_as(ct.POINTER(ct.c_double)),
            )
            == 0
        )
        assert output[0] == output[-1] == 9182.5
        results.append(output)
    assert (admission_probe.pair_work(0) > 0) == (extreme is not True)
    if extreme is True:
        assert admission_probe.pair_work(1) > 0
        np.testing.assert_array_equal(results[1], results[0])
    else:
        if extreme == "partial":
            assert admission_probe.pair_work(1) > 0
        np.testing.assert_allclose(results[1], results[0], rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 127, 128, 129, 257, 383, 513])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize(
    "domain",
    [
        "signed",
        "sparse",
        "screened_block",
        "empty",
        "translated",
        "oversized_weight",
        "nonzero",
        "masked_nonzero",
        "positive_zero",
    ],
)
def test_admitted_rows_keep_order_at_partial_tiles(
    admission_probe: ct.CDLL, n: int, geometry: bool, domain: str
) -> None:
    rng = np.random.default_rng(8173 + n)
    x = rng.uniform(0.1, 2.0, size=10 * n)
    x[: 3 * n] = rng.normal(size=3 * n)
    weighted = x[9 * n :]
    weighted[::7] *= -1
    # Positive-zero rows retain their weight derivative; negative-zero rows are
    # screened only when the explicit molecular row-mask contract is selected.
    if domain not in ("nonzero", "masked_nonzero"):
        weighted[::11] = 0.0
    if domain == "positive_zero":
        weighted[:] = 0.0
    if domain == "masked_nonzero":
        weighted[::5] = -0.0
    if domain == "sparse":
        weighted[::3] = -0.0
    elif domain == "screened_block":
        weighted[:128] = -0.0
    elif domain == "empty":
        weighted[:] = -0.0
    elif domain == "translated":
        x[: 3 * n] += 2.0**15
    elif domain == "oversized_weight":
        weighted[-1] = 2.0**70
    mask = domain != "signed"
    outputs = []
    for stage in (False, True):
        out = np.full(7 * n + 2, 9182.5)
        error = admission_probe.run(
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
    if domain in ("signed", "translated", "oversized_weight"):
        np.testing.assert_array_equal(outputs[1], outputs[0])
    else:
        np.testing.assert_allclose(outputs[1], outputs[0], rtol=2e-13, atol=1e-11)


@pytest.mark.parametrize("geometry", [False, True])
def test_admitted_exception_publication_matches_direct(
    admission_probe: ct.CDLL, geometry: bool
) -> None:
    n = 129
    x = np.ones(10 * n)
    x[4 * n + 128] = np.inf
    statuses = []
    for stage in (False, True):
        out = np.full(7 * n + 2, 9182.5)
        statuses.append(
            admission_probe.run(
                n,
                geometry,
                True,
                stage,
                x.ctypes.data_as(ct.POINTER(ct.c_double)),
                out.ctypes.data_as(ct.POINTER(ct.c_double)),
            )
        )
        assert out[0] == out[-1] == 9182.5
    assert statuses == [-1, -1]


@pytest.mark.parametrize("n", [33, 129, 257])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_admitted_outputs_match_extended_precision_pair_oracle(
    admission_probe: ct.CDLL, n: int, geometry: bool, mixed: bool
) -> None:
    """Differentiate the original kernel independently of generated pair roots.

    Signed weights, observable positive-zero rows and masked negative-zero rows
    share a partial packet/tile. Long-double reductions avoid using another
    native summation schedule as the scientific acceptance oracle. Mixed sums
    use the independently qualified bilateral pair error times their absolute
    contribution sums, so signed cancellation cannot turn a relative pair
    bound into an unjustified relative bound on a near-zero force component.
    """
    rng = np.random.default_rng(9157 + n)
    inputs = rng.uniform(0.1, 2.0, size=10 * n)
    inputs[: 3 * n] = rng.normal(size=3 * n)
    inputs[9 * n :: 7] *= -1
    inputs[9 * n :: 11] = 0.0
    inputs[9 * n + 5 :: 17] = -0.0
    coordinates = inputs[: 3 * n].astype(np.longdouble).reshape(n, 3)
    rho, omega, kappa, domega_drho, domega_dsigma, dkappa_drho, factors = (
        inputs[3 * n :].astype(np.longdouble).reshape(7, n)
    )
    partners = factors != 0
    screened = (factors == 0) & np.signbit(factors)
    coefficient, beta = np.longdouble("0.7"), np.longdouble("0.03125")
    expected = np.zeros(7 * n)
    absolute_contributions = np.zeros(7 * n)
    for row in np.flatnonzero(~screened):
        delta = coordinates[partners] - coordinates[row]
        radius = np.sum(delta * delta, axis=1)
        first = omega[row] * radius + kappa[row]
        second = omega[partners] * radius + kappa[partners]
        combined = first + second
        phi = -np.longdouble("1.5") / (first * second * combined)
        partial = -phi * (1 / first + 1 / combined)
        sum_phi = np.sum(factors[partners] * phi)
        sum_omega = np.sum(factors[partners] * partial * radius)
        sum_kappa = np.sum(factors[partners] * partial)
        absolute_phi = np.sum(np.abs(factors[partners] * phi))
        absolute_omega = np.sum(np.abs(factors[partners] * partial * radius))
        absolute_kappa = np.sum(np.abs(factors[partners] * partial))
        expected[row] = coefficient * factors[row] * (beta + sum_phi / 2)
        expected[n + row] = coefficient * (
            beta
            + sum_phi
            + rho[row] * (sum_omega * domega_drho[row] + sum_kappa * dkappa_drho[row])
        )
        expected[2 * n + row] = coefficient * rho[row] * sum_omega * domega_dsigma[row]
        absolute_contributions[row] = abs(coefficient * factors[row]) * absolute_phi / 2
        absolute_contributions[n + row] = abs(coefficient) * (
            absolute_phi
            + abs(rho[row])
            * (
                absolute_omega * abs(domega_drho[row])
                + absolute_kappa * abs(dkappa_drho[row])
            )
        )
        absolute_contributions[2 * n + row] = (
            abs(coefficient * rho[row] * domega_dsigma[row]) * absolute_omega
        )
        if geometry:
            radial = -phi * (
                omega[row] / first
                + omega[partners] / second
                + (omega[row] + omega[partners]) / combined
            )
            expected[3 * n + 3 * row : 3 * n + 3 * row + 3] = (
                -2
                * coefficient
                * factors[row]
                * np.sum((factors[partners] * radial)[:, None] * delta, axis=0)
            )
            expected[6 * n + row] = coefficient * rho[row] * (beta + sum_phi)
            absolute_contributions[3 * n + 3 * row : 3 * n + 3 * row + 3] = abs(
                2 * coefficient * factors[row]
            ) * np.sum(np.abs((factors[partners] * radial)[:, None] * delta), axis=0)
            absolute_contributions[6 * n + row] = (
                abs(coefficient * rho[row]) * absolute_phi
            )
    output = np.full(7 * n + 2, 9182.5)
    evaluate = admission_probe.run_mixed if mixed else admission_probe.run
    status = evaluate(
        n,
        geometry,
        True,
        True,
        inputs.ctypes.data_as(ct.POINTER(ct.c_double)),
        output.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    assert status == 0
    assert output[0] == output[-1] == 9182.5
    consumed = 7 * n if geometry else 3 * n
    if mixed:
        error = np.abs(output[1 : 1 + consumed] - expected[:consumed])
        assert np.all(error <= 3e-6 * absolute_contributions[:consumed] + 1e-11)
    else:
        np.testing.assert_allclose(
            output[1 : 1 + consumed], expected[:consumed], rtol=2e-13, atol=1e-11
        )
