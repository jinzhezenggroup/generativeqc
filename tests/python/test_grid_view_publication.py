"""Host-compile the production publication tail and native view-export gate.

CUDA computation/transport is replaced by a deterministic fault-injection
stand-in. This checks publication order, not CUDA arithmetic or synchronization.
"""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

STAND_IN = r"""
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace vibeqc::dft {
struct GridTaskView {
  unsigned version;
  std::uint64_t generation;
  std::size_t npoint, nao, nactive, jets;
  const std::size_t* ao_ids;
  const double *points, *ao, *features;
  double *local_potential, *potential;
  void* stream;
  int* error;
};
}
struct Context {
  std::mutex mutex;
  struct { double output_ms{}; } metrics;
  int error_value{};
  int* error = &error_value;
  void* stream{};
  bool fail_section{};
  int sections{};
  void check_device() {}
  template<class F> void section(bool, double&, F f) {
    ++sections;
    f();
    if (fail_section) throw std::runtime_error("injected output failure");
  }
};
struct GridPlan {
  Context context;
  bool local = true, view_ready{}, features_ready = true;
  bool density_jets_ready{}, use_orbitals{}, last_identity_map{};
  std::uint64_t generation{};
  std::size_t last_points = 1, nao = 1, last_active = 1, jets = 1;
  std::size_t id = 0;
  std::size_t* ao_ids = &id;
  double storage[32]{};
  double *points = storage, *ao = storage, *features = storage;
  double *local_potential = storage, *potential = storage;
};
static GridPlan plan;
constexpr int cudaMemcpyDeviceToHost = 0;
int cudaMemcpyAsync(void* destination, const void* source,
                    std::size_t size, int, void*) {
  std::memcpy(destination, source, size);
  return 0;
}
void cuda_check(int status) {
  if (status) throw std::runtime_error("transport failed");
}
template<class F> int guarded(char*, std::size_t, F f) {
  try { f(); return 0; } catch (...) { return 1; }
}
extern "C" int run(int defer_error_to_consumer, int device_error,
                    int transport_failure, int identity) {
  return guarded(nullptr, 0, [&] {
    auto& p = plan;
    auto& ctx = p.context;
    p.view_ready = false;
    ++p.generation;
    p.last_identity_map = identity != 0;
    ctx.error_value = device_error;
    ctx.fail_section = transport_failure != 0;
    ctx.sections = 0;
    const int features = 1;
    const std::size_t npoint = 1, active = 1;
    double *feature_output = nullptr, *jet_output = nullptr;
"""


@pytest.fixture(scope="module")
def publication(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler required for publication regression")
    source = (ROOT / "src/dft/cuda_grid.cu").read_text(encoding="utf-8")
    begin = source.index("static int grid_cuda_run_selected_impl(")
    end = source.index("\nint grid_cuda_run_selected_v1(", begin)
    implementation = source[begin:end]
    tail = implementation[implementation.index("    if (defer_error_to_consumer) {") :]
    view_begin = source.index("int grid_cuda_view_v1(")
    view_end = source.index("\nint grid_cuda_basis_v1(", view_begin)
    view = source[view_begin:view_end]
    directory = tmp_path_factory.mktemp("grid-publication")
    cpp = directory / "publication.cpp"
    library = directory / "publication.so"
    cpp.write_text(
        STAND_IN
        + tail
        + view
        + r"""
extern "C" int view_status() {
  vibeqc::dft::GridTaskView view{};
  return grid_cuda_view_v1(&plan, &view, nullptr, 0);
}
extern "C" int sections() { return plan.context.sections; }
extern "C" int view_error() {
  vibeqc::dft::GridTaskView view{};
  if (grid_cuda_view_v1(&plan, &view, nullptr, 0)) return -1;
  return *view.error;
}
extern "C" int view_identity() {
  vibeqc::dft::GridTaskView view{};
  if (grid_cuda_view_v1(&plan, &view, nullptr, 0)) return -1;
  return view.ao_ids == nullptr;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(cpp),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    native = ct.CDLL(str(library))
    native.run.argtypes = [ct.c_int] * 4
    native.run.restype = ct.c_int
    for name in ("view_status", "sections", "view_error", "view_identity"):
        function = getattr(native, name)
        function.argtypes = []
        function.restype = ct.c_int
    return native


@pytest.mark.parametrize("identity", [0, 1])
@pytest.mark.parametrize("failure", [(1, 0), (2, 0), (0, 1)])
def test_failed_sync_revokes_view_then_success_recovers(
    publication: ct.CDLL, identity: int, failure: tuple[int, int]
) -> None:
    assert publication.run(0, 0, 0, identity) == 0
    assert publication.view_status() == 0
    assert publication.run(0, *failure, identity) != 0
    assert publication.view_status() != 0
    assert publication.run(0, 0, 0, identity) == 0
    assert publication.view_status() == 0
    assert publication.view_identity() == identity


@pytest.mark.parametrize("identity", [0, 1])
def test_deferred_view_retains_error_for_qualified_consumer(
    publication: ct.CDLL, identity: int
) -> None:
    assert publication.run(1, 7, 1, identity) == 0
    assert publication.sections() == 0
    assert publication.view_status() == 0
    assert publication.view_error() == 7
    assert publication.view_identity() == identity
