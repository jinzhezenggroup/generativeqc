"""Execute production J-only CUDA response with queued host CUDA/cuBLAS doubles.

These tests qualify arithmetic, dispatch, bounded source work and callback
lifetimes. They do not qualify real GPU execution, CUDA synchronization or the
physical integral producer; the native GPU/oracle tests retain those duties.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cpp_source_support import (
    cpp_function_definition,
    cpp_function_definition_signature,
    cpp_record_definition,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _host_launches(source: str) -> str:
    return re.sub(
        r"([\w:]+)\s*<<<([\s\S]*?)>>>\s*\(([\s\S]*?)\);",
        r"simulate(\2, [=] { \1(\3); });",
        source,
    )


_SHIM = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "scf/cuda/df_response_weights.cuh"
#define __global__
#define __device__
using std::isfinite;
struct dim3 { unsigned x; dim3(unsigned x=1): x(x) {} };
dim3 blockIdx, blockDim, threadIdx;
constexpr cudaStream_t test_stream = 19;
std::vector<std::function<void()>> pending;
std::string failure;
int memsets=0, products=0, launches=0, launch_error=0;
std::map<std::string, std::size_t> counters;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void drain() {
  for (auto& operation : pending) operation();
  pending.clear();
}
cudaError_t cudaMemsetAsync(void* pointer, int value, std::size_t bytes, cudaStream_t stream) {
  require(stream == test_stream, "wrong memset stream");
  if (++memsets == 2 && failure == "memset") return 71;
  pending.push_back([=] { std::memset(pointer, value, bytes); });
  return cudaSuccess;
}
cudaError_t cudaGetLastError() { const auto e=launch_error; launch_error=0; return e; }
template<class F>
void simulate(dim3 grid, dim3 block, std::size_t, cudaStream_t stream, F kernel) {
  require(stream == test_stream, "wrong kernel stream");
  if (++launches == 1 && failure == "launch") { launch_error=73; return; }
  pending.push_back([=] {
    blockDim=block;
    for (unsigned b=0;b<grid.x;++b) {
      blockIdx=dim3(b);
      for (unsigned t=0;t<block.x;++t) { threadIdx=dim3(t); kernel(); }
    }
  });
}
cublasStatus_t cublasDgemm(cublasHandle_t handle, int ta, int tb, int m, int n, int k,
                          const double* alpha, const double* a, int lda,
                          const double* b, int ldb, const double* beta, double* c, int ldc) {
  require(handle == test_stream, "wrong BLAS stream");
  if (++products == 1 && failure == "blas") return 79;
  const auto av=*alpha, bv=*beta;
  pending.push_back([=] {
    for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
      double sum=0;
      for (int p=0;p<k;++p)
        sum += a[ta == CUBLAS_OP_N ? i+p*lda : p+i*lda] *
               b[tb == CUBLAS_OP_N ? p+j*ldb : j+p*ldb];
      c[i+j*ldc]=av*sum+(bv == 0 ? 0 : bv*c[i+j*ldc]);
    }
  });
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDgemv(cublasHandle_t handle, int trans, int m, int n,
                          const double* alpha, const double* a, int lda,
                          const double* x, int incx, const double* beta, double* y, int incy) {
  require(handle == test_stream, "wrong BLAS stream");
  if (++products == 1 && failure == "blas") return 79;
  const auto av=*alpha, bv=*beta;
  pending.push_back([=] {
    const auto rows=trans == CUBLAS_OP_N ? m : n;
    const auto cols=trans == CUBLAS_OP_N ? n : m;
    for (int i=0;i<rows;++i) {
      double sum=0;
      for (int j=0;j<cols;++j)
        sum += a[trans == CUBLAS_OP_N ? i+j*lda : j+i*lda]*x[j*incx];
      y[i*incy]=av*sum+(bv == 0 ? 0 : bv*y[i*incy]);
    }
  });
  return CUBLAS_STATUS_SUCCESS;
}
namespace generativeqc::runtime::cuda_trace {
void trace_counter(const char* name, std::size_t count) { counters[name]+=count; }
struct TraceRegion { TraceRegion(const char*, cudaStream_t) {} void finish() {} };
}
"""


_DRIVER = r"""
int main(int argc, char** argv) {
  using namespace generativeqc::scf;
  if (argc != 4) return 90;
  const std::string mode=argv[1]; failure=argv[3];
  const auto tile=std::stoul(argv[2]);
  std::size_t n,a,nt; int full; double threshold;
  std::cin >> n >> a >> nt >> full >> threshold;
  const auto matrix=n*n, aa=a*a;
  const auto read=[](std::size_t count) {
    std::vector<double> values(count);
    for (auto& v : values) std::cin >> v;
    return values;
  };
  auto densities=read(nt*matrix), raw=read(matrix*a), fitted=read(matrix*a);
  auto root=read(aa), vectors=read(aa), eigenvalues=read(a), coefficients=read(nt);
  const auto saved_densities=densities, saved_raw=raw, saved_fitted=fitted;
  const auto saved_root=root, saved_vectors=vectors, saved_eigenvalues=eigenvalues;
  std::vector<DensityFittingDensityResponse> terms;
  for (std::size_t t=0;t<nt;++t)
    terms.push_back({std::span<const double>(densities.data()+t*matrix,matrix), coefficients[t],
                     mode == "mixed" && t == 0 ? 0.27 : 0.0});
  CudaDfMetricView metric{root.data(),vectors.data(),eigenvalues.data(),threshold,bool(full),17};
  const auto capacity=cuda_df_response_workspace_elements(n,a,nt,tile);
  constexpr std::size_t guard=23;
  constexpr double sentinel=-987654.25;
  std::vector<double> storage(capacity+2*guard,sentinel);
  auto* workspace=storage.data()+guard;
  std::vector<double> weights(matrix*a,std::numeric_limits<double>::quiet_NaN());
  std::vector<double> metric_weights(aa,std::numeric_limits<double>::quiet_NaN());
  std::vector<std::size_t> visits(a), writes(matrix*a);
  std::size_t fitted_calls=0, raw_calls=0, consumers=0, metric_consumers=0;
  std::function<void(std::size_t,std::size_t,double*)> reader;
  if (mode != "raw" && mode != "raw_scalar")
    reader=[&](std::size_t begin,std::size_t count,double* destination) {
      require(begin+count <= a && count <= tile,"reader exceeded panel bounds");
      if (++fitted_calls == 2 && failure == "reader") throw std::runtime_error("reader failure");
      for (auto p=begin;p<begin+count;++p) ++visits[p];
      pending.push_back([&,begin,count,destination] {
        // Both supported fitted readers own these scratch intervals. Poison
        // them on every read to expose response storage or lifetime aliases.
        std::fill(workspace+2*aa,workspace+4*aa,12345.0+begin);
        std::fill(workspace+4*aa,workspace+4*aa+matrix,23456.0+begin);
        for (std::size_t p=0;p<count;++p) for (std::size_t ij=0;ij<matrix;++ij)
          destination[p*matrix+ij]=fitted[ij*a+begin+p];
      });
    };
  int status=0;
  try {
    status=contract_cuda_df_response_weights(
      n,a,terms,densities.data(),metric,tile,workspace,test_stream,test_stream,
      mode == "fitted_serial",mode != "fitted_scalar" && mode != "raw_scalar",
      [&](std::size_t p,double* destination) {
        require(p<a,"raw reader exceeded domain");
        ++raw_calls; ++visits[p];
        pending.push_back([&,p,destination] {
          for (std::size_t ij=0;ij<matrix;++ij) destination[ij]=raw[ij*a+p];
        });
      },
      [&](unsigned kind,generativeqc::runtime::StridedRange range,std::size_t count,
          const double* values) {
        if (++consumers == 2 && failure == "consumer") throw std::runtime_error("consumer failure");
        if (kind == 1) {
          ++metric_consumers; require(count == aa,"wrong metric extent");
          pending.push_back([&,values] { std::copy(values,values+aa,metric_weights.begin()); });
        } else {
          require(kind == 0 && count <= tile*matrix,"invalid dense panel");
          pending.push_back([&,range,count,values] {
            for (std::size_t k=0;k<count;++k) {
              const auto index=range.index(k); require(index<weights.size(),"invalid output index");
              weights[index]=values[k]; ++writes[index];
            }
          });
        }
      },nullptr,{},false,{},256,reader,false,nullptr,nullptr);
  } catch (const CudaDfResponseBlasFailure& error) { status=error.status; }
    catch (const std::runtime_error& error) {
      if (std::string(error.what()) == "reader failure") status=81;
      else if (std::string(error.what()) == "consumer failure") status=83;
      else throw;
    }
  drain();
  for (std::size_t i=0;i<guard;++i)
    require(storage[i] == sentinel && storage[guard+capacity+i] == sentinel,
            "response exceeded charged workspace");
  require(densities == saved_densities && raw == saved_raw && fitted == saved_fitted &&
          root == saved_root && vectors == saved_vectors && eigenvalues == saved_eigenvalues,
          "immutable source or density mutated");
  if (status == 0) {
    require(metric_consumers == 1,"metric result missing or repeated");
    for (auto count : writes) require(count == 1,"response output missing or repeated");
  }
  std::cout << std::setprecision(17) << "{\"status\":" << status
            << ",\"fitted_calls\":" << fitted_calls << ",\"raw_calls\":" << raw_calls
            << ",\"consumers\":" << consumers << ",\"metric_consumers\":" << metric_consumers;
  const auto print=[&](const char* name,const auto& values) {
    std::cout << ",\"" << name << "\":[";
    bool first=true;
    for (auto value : values) {
      if (!first) std::cout << ','; first=false;
      if (std::isfinite(double(value))) std::cout << value; else std::cout << "null";
    }
    std::cout << ']';
  };
  print("weights",weights); print("metric",metric_weights); print("visits",visits);
  std::cout << ",\"counters\":{";
  bool first=true;
  for (const auto& [key,value] : counters) {
    if (!first) std::cout << ','; first=false;
    std::cout << '"' << key << "\":" << value;
  }
  std::cout << "}}\n";
}
"""


@pytest.fixture(scope="module")
def streamed_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    from generativeqc_compiler.method.df_hf_response_cuda import (
        emit_df_hf_response_cuda,
    )
    from generativeqc_compiler.method.matrix_function_cuda import (
        emit_symmetric_matrix_function_vjp_cuda,
    )

    directory = tmp_path_factory.mktemp("streamed-coulomb-host")
    (directory / "cuda_runtime.h").write_text(
        "#pragma once\nusing cudaStream_t=int; using cudaError_t=int;\n"
        "constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;\n"
    )
    (directory / "cublas_v2.h").write_text(
        "#pragma once\nusing cublasHandle_t=int; using cublasStatus_t=int;\n"
        "constexpr int CUBLAS_STATUS_SUCCESS=0,CUBLAS_OP_N=0,CUBLAS_OP_T=1;\n"
    )
    source = (ROOT / "src/scf/cuda/df_response_weights.cu").read_text()
    generated = emit_df_hf_response_cuda()
    pieces = [_SHIM]
    # The production response now forwards BLAS through borrowed TensorIR
    # views. Execute those real forwarding adapters against the queued host
    # providers above, so strides and failure propagation stay covered.
    tensor = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    pieces.append("namespace generativeqc_tensor {\n")
    for record, submit in (
        ("ColumnGemmView", "submit_column_gemm"),
        ("ColumnGemvView", "submit_column_gemv"),
    ):
        pieces.append(cpp_record_definition(tensor, record) + ";")
        pieces.append(cpp_function_definition(tensor, submit))
    pieces.append("}\n")
    pieces.append(_host_launches(emit_symmetric_matrix_function_vjp_cuda()))
    pieces.append("namespace generativeqc::scf {\nnamespace generated {\n")
    pieces.append(cpp_function_definition(generated, "df_rhf_charge_contract"))
    pieces.append("}\n")
    for name in (
        "coulomb_weights_kernel",
        "coulomb_metric_kernel",
        "exchange_weights_kernel",
        "exchange_metric_kernel",
        "fitted_exchange_weights_kernel",
    ):
        pieces.append(cpp_function_definition(generated, name))
    pieces.append("constexpr unsigned threads=128;\n")
    for name in (
        "blocks",
        "inverse_kernel",
        "charge_kernel",
        "potential_kernel",
        "right_density_kernel",
        "left_density_kernel",
        "symmetrize_kernel",
        "cuda_df_response_workspace_elements",
        "contract_streamed_coulomb_response",
        "contract_full_rank_panels",
    ):
        pieces.append(_host_launches(cpp_function_definition(source, name)))
    # Dispatch dependencies outside this test's scope fail loudly if selected.
    # Their real signatures preserve the production call contract.
    for name in (
        "contract_retained_coulomb_response",
        "contract_occupied_response",
        "fit_full_tensor_in_place",
        "contract_full_rank_response",
        "contract_resident_response",
    ):
        pieces.append(
            cpp_function_definition_signature(source, name)
            + '{ throw std::runtime_error("unqualified response route"); }'
        )
    pieces.append(
        _host_launches(
            cpp_function_definition(source, "contract_cuda_df_response_weights")
        )
    )
    pieces.extend(("}\n", _DRIVER))
    unit = directory / "probe.cpp"
    unit.write_text("\n".join(pieces))
    return native_cxx.build_executable(
        [unit],
        directory / "probe",
        compile_args=(
            "-std=c++20",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I" + str(directory),
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
        ),
        link_args=("-fsanitize=undefined",),
    )


def _fixture(truncated: bool = False) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(2425)
    n, a = 3, 5
    eigenvalues = np.array([0.02 if truncated else 0.7, 1.1, 1.8, 2.9, 4.3])
    vectors = np.linalg.qr(rng.normal(size=(a, a)))[0]
    inverse_values = np.where(eigenvalues > 0.05 * eigenvalues[-1], 1 / eigenvalues, 0)
    inverse = (vectors * inverse_values) @ vectors.T
    raw = rng.normal(size=(n * n, a))
    return {
        "n": n,
        "a": a,
        "full": int(not truncated),
        "threshold": 0.05,
        "densities": rng.normal(size=(3, n * n)),
        "raw": raw,
        "fitted": raw @ inverse,
        "root": (vectors * np.sqrt(inverse_values)) @ vectors.T,
        "vectors": vectors,
        "eigenvalues": eigenvalues,
        "coefficients": np.array([1.0, -0.31, 0.0]),
    }


def _run(probe: Path, data: dict, mode: str, tile: int, failure: str = "none") -> dict:
    values = [
        data["n"],
        data["a"],
        len(data["coefficients"]),
        data["full"],
        data["threshold"],
    ]
    for name in (
        "densities",
        "raw",
        "fitted",
        "root",
        "vectors",
        "eigenvalues",
        "coefficients",
    ):
        values.extend(data[name].ravel(order="F" if name == "vectors" else "C"))
    result = subprocess.run(
        [str(probe), mode, str(tile), failure],
        input=" ".join(format(value, ".17g") for value in values),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _energy(data: dict, raw: np.ndarray, metric: np.ndarray) -> float:
    values, vectors = np.linalg.eigh(metric)
    inverse = (
        vectors * np.where(values > data["threshold"] * values[-1], 1 / values, 0)
    ) @ vectors.T
    charges = data["densities"] @ raw
    return 0.5 * float(
        np.sum(data["coefficients"][:, None] * charges * (charges @ inverse))
    )


@pytest.mark.parametrize("tile", [1, 2, 3, 5])
def test_streamed_signed_nonsymmetric_adjoint_and_bounded_reads(
    streamed_probe: Path, tile: int
) -> None:
    data = _fixture()
    result = _run(streamed_probe, data, "stream", tile)
    potential = data["densities"] @ data["fitted"]
    expected_raw = np.einsum(
        "t,ti,tp->ip", data["coefficients"], data["densities"], potential
    )
    expected_metric = -0.5 * np.einsum(
        "t,tp,tq->pq", data["coefficients"], potential, potential
    )
    assert result["status"] == 0
    np.testing.assert_allclose(
        np.reshape(result["weights"], expected_raw.shape),
        expected_raw,
        atol=2e-13,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        np.reshape(result["metric"], expected_metric.shape),
        expected_metric,
        atol=2e-13,
        rtol=2e-13,
    )
    assert result["visits"] == [1] * data["a"]
    assert result["raw_calls"] == 0
    assert result["fitted_calls"] == (data["a"] + tile - 1) // tile
    counters = result["counters"]
    assert (
        counters["response_charge_dot_elements"]
        == len(data["coefficients"]) * data["n"] ** 2 * data["a"]
    )
    assert counters["response_auxiliary_blocks"] == result["fitted_calls"]
    assert counters["response_streamed_coulomb_fitted_panels"] == result["fitted_calls"]
    assert counters.get("response_metric_blas_dots", 0) == 0


@pytest.mark.parametrize("mode", ["stream", "raw", "raw_scalar"])
@pytest.mark.parametrize("truncated", [False, True])
def test_actual_response_matches_energy_finite_differences(
    streamed_probe: Path, mode: str, truncated: bool
) -> None:
    # A fitted reader cannot represent discarded raw directions: actual
    # dispatch must ignore it for truncated metrics and retain the Frechet VJP.
    data = _fixture(truncated)
    result = _run(streamed_probe, data, mode, 2)
    assert result["status"] == 0
    assert result["visits"] == [1] * data["a"]
    if truncated or mode != "stream":
        assert result["raw_calls"] == data["a"]
        assert result["fitted_calls"] == 0
    raw = data["raw"]
    metric = (data["vectors"] * data["eigenvalues"]) @ data["vectors"].T
    bar_raw = np.reshape(result["weights"], raw.shape)
    bar_metric = np.reshape(result["metric"], metric.shape)
    eps = 1e-5
    for index in np.ndindex(raw.shape):
        delta = np.zeros_like(raw)
        delta[index] = eps
        numeric = (
            _energy(data, raw + delta, metric) - _energy(data, raw - delta, metric)
        ) / (2 * eps)
        assert numeric == pytest.approx(bar_raw[index], abs=3e-8)
    for p in range(data["a"]):
        for q in range(p + 1):
            delta = np.zeros_like(metric)
            delta[p, q] = delta[q, p] = eps
            numeric = (
                _energy(data, raw, metric + delta) - _energy(data, raw, metric - delta)
            ) / (2 * eps)
            assert numeric == pytest.approx(
                bar_metric[p, q] * (1 if p == q else 2), abs=3e-8
            )


@pytest.mark.parametrize("mode", ["fitted_scalar", "fitted_serial"])
def test_diagnostic_dispatch_retains_numerical_parity(
    streamed_probe: Path, mode: str
) -> None:
    data = _fixture()
    expected = _run(streamed_probe, data, "stream", 2)
    actual = _run(streamed_probe, data, mode, 2)
    assert actual["status"] == 0
    np.testing.assert_allclose(
        actual["weights"], expected["weights"], atol=2e-13, rtol=2e-13
    )
    np.testing.assert_allclose(
        actual["metric"], expected["metric"], atol=2e-13, rtol=2e-13
    )
    assert actual["counters"].get("response_streamed_coulomb_fitted_panels", 0) == 0


def test_exchange_terms_keep_complete_response(streamed_probe: Path) -> None:
    data = _fixture()
    result = _run(streamed_probe, data, "mixed", 2)
    n, a = data["n"], data["a"]
    potentials = data["densities"] @ data["fitted"]
    expected_raw = np.einsum(
        "t,ti,tp->ip", data["coefficients"], data["densities"], potentials
    )
    expected_metric = -0.5 * np.einsum(
        "t,tp,tq->pq", data["coefficients"], potentials, potentials
    )
    density = data["densities"][0].reshape(n, n)
    response = np.stack(
        [density.T @ data["fitted"][:, p].reshape(n, n) @ density for p in range(a)]
    ).reshape(a, n * n)
    expected_raw -= 2 * 0.27 * response.T
    expected_metric += 0.27 * response @ data["fitted"]
    expected_metric = 0.5 * (expected_metric + expected_metric.T)
    assert result["status"] == 0
    np.testing.assert_allclose(
        np.reshape(result["weights"], expected_raw.shape),
        expected_raw,
        atol=2e-13,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        np.reshape(result["metric"], expected_metric.shape),
        expected_metric,
        atol=2e-13,
        rtol=2e-13,
    )
    assert result["counters"].get("response_streamed_coulomb_fitted_panels", 0) == 0


@pytest.mark.parametrize(
    "failure,status",
    [("memset", 71), ("launch", 73), ("blas", 79), ("reader", 81), ("consumer", 83)],
)
def test_streamed_failure_propagates_without_completion(
    streamed_probe: Path, failure: str, status: int
) -> None:
    result = _run(streamed_probe, _fixture(), "stream", 2, failure)
    assert result["status"] == status
    assert result["metric_consumers"] == 0
