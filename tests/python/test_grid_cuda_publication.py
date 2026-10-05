"""Host-execute grid publication control flow with instrumented CUDA stand-ins."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
using I = std::int64_t;
int fault = 0, downloads = 0, fences = 0, point_uploads = 0, pointer_checks = 0;
int gemm_calls = 0, gemm_products = 0, panel_copies = 0;
std::size_t panel_bytes = 0;
using cudaStream_t = std::uintptr_t;
using cudaEvent_t = unsigned;
constexpr unsigned cudaEventDisableTiming = 0;
struct { unsigned x = 0; } blockIdx, threadIdx;
struct { unsigned x = 1; } blockDim, gridDim;
const double* resident_points = nullptr;
const double* resident_alpha = nullptr;
const double* resident_beta = nullptr;
const double* observed_points = nullptr;
void* point_buffer = nullptr;
void require_device_pointer(const void* pointer, int device) {
  ++pointer_checks;
  if ((!pointer || (pointer != resident_points && pointer != resident_alpha && pointer != resident_beta)) || device != 3 || fault == 8)
    throw std::invalid_argument("invalid device point lease");
}
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3;
int cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, int kind, cudaStream_t) {
  if (fault == 11) return 1;
  if (kind == cudaMemcpyHostToDevice && dst == point_buffer) ++point_uploads;
  if (kind == cudaMemcpyDeviceToHost) {
    ++downloads;
    if (fault == 2) throw std::runtime_error("injected output submission failure");
  }
  if (kind == cudaMemcpyDeviceToDevice) { ++panel_copies; panel_bytes += bytes; }
  if (bytes) std::memcpy(dst, src, bytes);
  return 0;
}
int cudaMemsetAsync(void* dst, int value, std::size_t bytes, cudaStream_t) {
  if (fault == 12) return 1;
  if (bytes) std::memset(dst, value, bytes);
  return 0;
}
void cuda_check(int value) { if (value) throw std::runtime_error("CUDA stand-in failure"); }
int cudaGetLastError() { return fault == 6 ? 1 : 0; }
int cudaStreamGetFlags(cudaStream_t, unsigned*) { return fault == 15; }
int cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) { *event = 1; return fault == 16; }
int cudaEventRecord(cudaEvent_t, cudaStream_t) { return fault == 17; }
int cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return fault == 18; }
int cudaEventDestroy(cudaEvent_t) { return fault == 19; }
template<class F> int guarded(char*, std::size_t, F operation) {
  try { operation(); return 0; } catch (...) { return 1; }
}
struct Metrics { double input_ms{},kernel_ms{},packing_ms{},library_ms{},output_ms{}; };
struct Context {
  std::mutex mutex;
  Metrics metrics;
  int error_value{}, *error = &error_value, device = 3;
  cudaStream_t stream = 1;
  void check_device() { if (fault == 13) throw std::runtime_error("wrong device"); }
  template<class F> void section(bool profile, double& metric, F operation) {
    operation();
    if (profile) {
      ++fences;
      if (&metric == &metrics.output_ms && fault == 3)
        throw std::runtime_error("injected output synchronization failure");
    }
  }
};
namespace generativeqc::dft::generated {
struct Descriptor { std::size_t jets, points, active, width; };
Descriptor grid_panel_descriptor(std::size_t j,std::size_t p,std::size_t a,std::size_t w) {
  return {j,p,a,w};
}
}
struct Projection {
  void execute(generativeqc::dft::generated::Descriptor r, cudaStream_t,
               const double* left, const double* right, double* output, int*) {
    ++gemm_calls;
    gemm_products += r.jets;
    for (std::size_t row = 0; row < r.jets*r.points; ++row)
      for (std::size_t column = 0; column < r.width; ++column) {
        double value = 0;
        for (std::size_t item = 0; item < r.active; ++item)
          value += left[row*r.active+item]*right[item*r.width+column];
        output[row*r.width+column] = value;
      }
  }
};
struct GridPlan {
  Context context;
  Projection projection_value;
  Projection* projection = &projection_value;
  bool view_ready=true,features_ready=true,density_jets_ready=true;
  bool local=true,density_ready=true,use_orbitals=false,orbital_ready=false,last_identity_map=false;
  bool identical_spin_density=false,orbital_enabled=true;
  std::size_t generation=7,nao=2,active_capacity=2,capacity=2,jets=4;
  std::size_t last_points{},last_active{},natom=1,nprimitive=2;
  std::size_t orbital_count[2]{},orbital_tile=1;
  std::size_t orbital_capacity[2]{2,2};
  unsigned feature_mask=15;
  double basis[64]{},points[6]{},features[26]{},ao[64]{},density[8]{},local_density[8]{};
  double work[128]{},psi[64]{},factor_panel[64]{}, *factors[2]{};
  double potential[8]{};
  const double* current_points{};
  std::size_t ao_ids[2]{};
};
void scheduled_ao(int, const double*, std::size_t, std::size_t, std::size_t, const double* points,
               std::size_t,std::size_t,double*,int* error,const std::size_t*) {
  observed_points = points;
  if (fault == 1) *error = 7;
}
template<class... A> void gather_factor(A&&...) {}
template<class... A> void orbital_feature_kernel(A&&...) {}
template<class... A> void finish_orbital_sigma(A&&...) {}
void gather_density(const double* density, const std::size_t* ids, I nao, I active,
                    double* output) {
  for (I spin = 0; spin < 2; ++spin)
    for (I row = 0; row < active; ++row)
      for (I column = 0; column < active; ++column)
        output[(spin*active+row)*active+column] = density[(spin*nao+ids[row])*nao+ids[column]];
}
template<class... A> void scheduled_grid_features(A&&...) {}
"""

SPIN_MAIN = r"""
// Compare the admitted route with two actual host GEMMs, including untouched
// jet slots and canaries. The stand-ins instrument work, not numerical policy.
int check_spin_products(unsigned mask, int map, std::size_t npoint, int deferred) {
  GridPlan candidate, control;
  candidate.feature_mask = control.feature_mask = mask;
  candidate.local = control.local = map != 4;
  const std::size_t ids[2]{0,1};
  const std::size_t* selected = map == 1 || map == 4 ? nullptr : ids + (map == 2);
  const std::size_t active = map == 3 ? 0 : (map == 2 ? 1 : 2);
  double points[6]{};
  const double total[4]{0.3,-0.7,-0.7,1.9};
  resident_alpha = total;
  if (grid_cuda_density_device_v1(&candidate,total,nullptr,4,1,
                                reinterpret_cast<void*>(1),nullptr,0)) return 20;
  if (!candidate.identical_spin_density) return 21;
  std::copy(std::begin(candidate.density),std::end(candidate.density),control.density);
  for (std::size_t item = 0; item < 64; ++item)
    candidate.ao[item] = control.ao[item] = (int(item % 7) - 3)*0.17;
  std::fill(std::begin(candidate.work),std::end(candidate.work),-12345.25);
  std::copy(std::begin(candidate.work),std::end(candidate.work),control.work);
  gemm_calls = gemm_products = panel_copies = 0;
  panel_bytes = 0;
  if (grid_cuda_run_selected_impl(&candidate,points,npoint,1,selected,active,
                                 nullptr,nullptr,deferred,0,nullptr,0)) return 22;
  const int first = mask & 7 ? 0 : 1;
  const int count = mask & 8 ? 4-first : 1;
  const bool work = npoint && active;
  if (gemm_calls != int(work) || gemm_products != (work ? count : 0)) return 23;
  if (panel_copies != int(work) || panel_bytes != count*npoint*active*sizeof(double)) return 24;
  gemm_calls = gemm_products = panel_copies = 0;
  if (grid_cuda_run_selected_impl(&control,points,npoint,1,selected,active,
                                 nullptr,nullptr,deferred,0,nullptr,0)) return 25;
  if (gemm_calls != 2*int(work) || gemm_products != (work ? 2*count : 0) || panel_copies) return 26;
  if (std::memcmp(candidate.work,control.work,sizeof(candidate.work))) return 27;
  if (!candidate.view_ready || !candidate.density_jets_ready) return 28;
  return 0;
}

// The witness belongs to a completed, ordered upload, not a method label or
// pointer equality. Every setter, even a rejected replacement, revokes it.
int check_source_transition(int kind, int failure, bool foreign_stream) {
  GridPlan plan;
  double alpha[4]{0.3,-0.7,-0.7,1.9}, beta[4]{0.2,0.1,0.1,0.4};
  resident_alpha = alpha;
  resident_beta = beta;
  void* stream = reinterpret_cast<void*>(foreign_stream ? 2 : 1);
  if (grid_cuda_density_device_v1(&plan,alpha,nullptr,4,1,stream,nullptr,0)) return 30;
  if (!plan.identical_spin_density || !plan.density_ready) return 31;
  alpha[0] = 7.0;
  if (plan.density[0] != 0.15 || plan.density[4] != 0.15) return 32;
  const double host_density[8]{0.1,0.2,0.2,0.3,0.9,0.8,0.8,0.7};
  const double centers[3]{0.01,0.02,0.03};
  const std::size_t counts[2]{};
  const auto elements = failure == 1 ? 99u : 8u;
  fault = failure >= 6 ? failure : 0;
  int status = 0;
  if (kind == 0)
    status = grid_cuda_density_v1(&plan,failure == 2 ? nullptr : host_density,elements,nullptr,0);
  else if (kind == 1)
    status = grid_cuda_centers_v1(&plan,failure == 2 ? nullptr : centers,
                                failure == 1 ? 99 : 3,nullptr,0);
  else if (kind == 2 || kind == 3)
    status = grid_cuda_source_v1(&plan,failure == 2 ? nullptr : host_density,elements,
                                nullptr,nullptr,counts,kind == 3,nullptr,0);
  else
    status = grid_cuda_density_device_v1(&plan,failure == 2 ? nullptr : alpha,beta,
                                        failure == 1 ? 99 : 4,kind == 5 ? 1 : 2,
                                        stream,nullptr,0);
  if (plan.identical_spin_density != (kind == 5 && !status)) return 33;
  if ((status != 0) != (failure != 0)) return 34;
  fault = 0;
  if (grid_cuda_density_device_v1(&plan,alpha,nullptr,4,1,stream,nullptr,0)) return 36;
  if (!plan.identical_spin_density) return 37;
  // Even aliased unrestricted inputs are not evidence of a restricted upload.
  if (grid_cuda_density_device_v1(&plan,alpha,alpha,4,2,stream,nullptr,0)) return 38;
  if (plan.identical_spin_density) return 39;
  return 0;
}

int check_panel_copy_failure() {
  GridPlan plan;
  plan.identical_spin_density = true;
  double points[6]{};
  resident_points = points;
  fault = 11;
  if (!grid_cuda_run_selected_impl(&plan,points,2,1,nullptr,2,nullptr,nullptr,1,1,nullptr,0)) return 40;
  if (plan.view_ready || plan.density_jets_ready || gemm_calls != 1) return 41;
  fault = 0;
  if (grid_cuda_run_selected_impl(&plan,points,2,1,nullptr,2,nullptr,nullptr,1,1,nullptr,0)) return 42;
  if (!plan.view_ready || !plan.density_jets_ready) return 43;
  return 0;
}
"""

MAIN = r"""
int main(int argc,char** argv) {
  if (argc == 6 && std::strcmp(argv[1],"spin") == 0)
    return check_spin_products(std::atoi(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5]));
  if (argc == 5 && std::strcmp(argv[1],"source") == 0)
    return check_source_transition(std::atoi(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]));
  if (argc == 2 && std::strcmp(argv[1],"copy-failure") == 0) return check_panel_copy_failure();
  if (argc != 3) return 1;
  const int mode = std::atoi(argv[1]);
  const int device_points = std::atoi(argv[2]);
  GridPlan p;
  if (mode == 9) p.use_orbitals = p.orbital_ready = true;
  double points[6]{};
  resident_points = points;
  point_buffer = p.points;
  if (mode == 8 && !device_points) points[0] = std::numeric_limits<double>::quiet_NaN();
  const std::size_t ids[2]{0,1};
  double output[26]{};
  const bool deferred = mode == 4 || mode == 5;
  fault = mode == 4 ? 1 : mode;
  const auto npoint = mode == 7 || mode == 9 || mode == 10 ? 0u : 2u;
  const int status = grid_cuda_run_selected_impl(&p,points,npoint,mode == 10 ? 0 : 1,ids,2,
      mode == 5 ? output : nullptr,nullptr,deferred,device_points,nullptr,0);
  const bool expected_success = mode == 0 || mode == 4 || mode == 7 || mode == 9 || mode == 10;
  if ((status == 0) != expected_success) return 2;
  if (p.view_ready != expected_success) return 3;
  if (!expected_success && p.density_jets_ready) return 4;
  if (mode == 4 && (downloads || fences || !*p.context.error || !p.density_jets_ready)) return 5;
  if (mode == 0 && (downloads != 1 || fences != 1 || !p.density_jets_ready)) return 6;
  if ((mode == 7 || mode == 9 || mode == 10) && (p.density_jets_ready != (mode == 7) || downloads || fences != 1)) return 7;
  if (mode == 7 || mode == 9 || mode == 10) {
    const double* jets = nullptr;
    const auto status = grid_cuda_density_jets_v1(&p,p.generation,1,&jets,nullptr,0);
    if ((status == 0) != (mode == 7)) return 15;
    if (mode == 7 && jets != p.work) return 16;
    if (grid_cuda_density_jets_v1(&p,p.generation-1,1,&jets,nullptr,0) == 0) return 17;
  }
  const double* expected_points = device_points ? points : p.points;
  if (expected_success && p.current_points != expected_points) return 10;
  if ((mode == 0 || mode == 4) && observed_points != expected_points) return 11;
  if (point_uploads != int(!device_points && mode != 5 && mode != 7 && mode != 8 && mode != 9 && mode != 10)) return 12;
  if (pointer_checks != int(device_points && mode != 5 && mode != 7 && mode != 9 && mode != 10)) return 13;
  // A later generation switches backends, clears faults, and publishes the
  // new point source instead of retaining a previous borrowed device pointer.
  fault = 0;
  p.use_orbitals = p.orbital_ready = false;
  points[0] = 0;
  const int recovery_device_points = !device_points;
  if (grid_cuda_run_selected_impl(&p,points,2,1,ids,2,nullptr,nullptr,0,
                                  recovery_device_points,nullptr,0)) return 8;
  if (!p.view_ready || !p.density_jets_ready || p.generation != 9) return 9;
  expected_points = recovery_device_points ? points : p.points;
  if (p.current_points != expected_points || observed_points != expected_points) return 14;
}
"""


@pytest.fixture(scope="module")
def publication_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()

    def host_body(signature: str) -> str:
        begin = source.index(signature)
        opening = source.index("{", begin)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return re.sub(r"<<<.*?>>>", "", source[begin:end], flags=re.DOTALL)

    body = host_body("static int grid_cuda_run_selected_impl(")
    setters = "\n".join(
        host_body(signature)
        for signature in (
            "void split_restricted_density(",
            "int grid_cuda_centers_v1(",
            "int grid_cuda_density_v1(",
            "int grid_cuda_density_device_v1(",
            "int grid_cuda_source_v1(",
        )
    )
    getter_begin = source.index("int grid_cuda_density_jets_v1(")
    getter_end = source.index("\nint grid_cuda_xc_v2(", getter_begin)
    getter = source[getter_begin:getter_end]
    directory = tmp_path_factory.mktemp("grid-publication")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(f"{PREFIX}\n{setters}\n{body}\n{getter}\n{SPIN_MAIN}\n{MAIN}")
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("mode", range(11))
@pytest.mark.parametrize("device_points", [False, True])
def test_grid_publication_requires_the_selected_error_gate(
    publication_probe: Path, mode: int, device_points: bool
) -> None:
    process = subprocess.run(
        [str(publication_probe), str(mode), str(int(device_points))],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0, (
        mode,
        device_points,
        process.returncode,
        process.stderr,
    )


@pytest.mark.parametrize("mask", range(1, 16))
@pytest.mark.parametrize("npoint", [0, 1, 2])
@pytest.mark.parametrize(
    ("map_kind", "deferred"),
    [
        (map_kind, deferred)
        for map_kind in range(5)
        for deferred in (False, True)
        if not (map_kind == 4 and deferred)
    ],
)
def test_identical_spin_panels_halve_gemm_work_without_changing_bytes(
    publication_probe: Path, mask: int, map_kind: int, npoint: int, deferred: bool
) -> None:
    result = subprocess.run(
        [
            str(publication_probe),
            "spin",
            str(mask),
            str(map_kind),
            str(npoint),
            str(int(deferred)),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize(
    ("kind", "failure", "foreign_stream"),
    [
        (kind, failure, foreign_stream)
        for kind, failures in enumerate(
            (
                (0, 1, 2, 11, 12, 13),
                (0, 1, 2, 11, 13),
                (0, 1, 2, 11, 12, 13),
                (0, 1, 2, 11, 12, 13),
                (0, 1, 2, 8, 11, 12, 13, 15),
                (0, 1, 2, 6, 8, 12, 13, 15),
            )
        )
        for foreign_stream in (False, True)
        for failure in (
            (*failures, 16, 17, 18, 19) if foreign_stream and kind >= 4 else failures
        )
    ],
)
def test_identical_spin_witness_is_revoked_by_every_source_transition(
    publication_probe: Path, kind: int, failure: int, foreign_stream: bool
) -> None:
    result = subprocess.run(
        [
            str(publication_probe),
            "source",
            str(kind),
            str(failure),
            str(int(foreign_stream)),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_identical_spin_copy_failure_cannot_publish_a_lease(
    publication_probe: Path,
) -> None:
    result = subprocess.run(
        [str(publication_probe), "copy-failure"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
