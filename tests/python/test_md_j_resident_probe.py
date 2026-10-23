"""SDK-free execution tests for the real test-only resident graph bridge.

CUDA and the prepared provider are doubles; these gates establish capture/replay
ownership, host-dispatch exclusion, and failure cleanup. They do not establish
GPU kernel correctness, capture support, or performance on a CUDA device.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def resident_graph_bridge(tmp_path_factory: Any, native_cxx: Any) -> Path:
    folder = tmp_path_factory.mktemp("md-j-resident-graph-bridge")
    (folder / "api").mkdir()
    (folder / "scf").mkdir()
    (folder / "cuda_runtime_api.h").write_text(_CUDA_DOUBLE)
    (folder / "api/handles.hpp").write_text(
        "#pragma once\nnamespace generativeqc::core { struct System {}; }\n"
        "struct generativeqc_system { generativeqc::core::System data; };\n"
    )
    (folder / "scf/cuda_fock_execution.hpp").write_text(_PROVIDER_DOUBLE)
    source = folder / "probe.cpp"
    source.write_text(
        '#include "'
        + str(ROOT / "tests/native/md_j_resident_probe.cpp")
        + '"\n'
        + _DRIVER
    )
    return native_cxx.build_executable(
        [source],
        folder / "probe",
        compile_args=("-std=c++20", "-Wall", "-Wextra", "-Werror", "-I" + str(folder)),
    )


@pytest.mark.parametrize(
    "section",
    [
        "replay",
        "failed-recapture",
        "negative-inputs",
        "diagnostics",
        "partial-creation",
    ],
)
def test_resident_graph_bridge_lifecycle(
    resident_graph_bridge: Path, section: str
) -> None:
    subprocess.run([str(resident_graph_bridge), section], check=True, timeout=30)


def test_resident_graph_python_rejects_unprepared_or_closed() -> None:
    """Reject before importing CuPy or handing a null native owner to C++."""
    from md_j_resident_probe import ResidentMdJProbe

    probe = ResidentMdJProbe.__new__(ResidentMdJProbe)
    probe._handle = ct.c_void_p(1)
    probe._graph_captured = False
    with pytest.raises(RuntimeError, match="has not been captured"):
        probe.evaluate_graph([[1.0]])
    probe._handle = ct.c_void_p()
    with pytest.raises(RuntimeError, match="closed"):
        probe.capture_graph()
    with pytest.raises(RuntimeError, match="closed"):
        probe.evaluate_graph([[1.0]])


@pytest.fixture
def probe_artifact(tmp_path: Path, monkeypatch: Any) -> Any:
    import md_j_resident_probe as bridge

    source, native = tmp_path / "probe.cpp", tmp_path / "native.so"
    source.write_bytes(b"test bridge source")
    native.write_bytes(b"test native identity")
    monkeypatch.setenv("SOURCE", str(tmp_path))
    monkeypatch.setenv("GENERATIVEQC_MD_J_PROBE_CPP", str(source))
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(native))
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "cuda"))
    monkeypatch.setattr(bridge.tempfile, "gettempdir", lambda: str(tmp_path))
    identity = hashlib.sha256(source.read_bytes() + native.read_bytes()).hexdigest()[
        :20
    ]
    binary = tmp_path / f"md-j-resident-probe-{identity}" / "probe.so"
    bridge._probe_library.cache_clear()
    yield bridge, binary
    bridge._probe_library.cache_clear()


@pytest.mark.parametrize("backend_matches", [True, False])
def test_resident_bridge_artifact_hit_needs_no_compiler_cache(
    probe_artifact: Any, monkeypatch: Any, backend_matches: bool
) -> None:
    from generativeqc_compiler.common import compiler_cache

    bridge, binary = probe_artifact
    binary.parent.mkdir()
    binary.write_bytes(b"completed artifact")
    library = SimpleNamespace(
        generativeqc_get_source_identity=ct.c_void_p(17 if backend_matches else 18),
        **{
            "md_j_probe_" + name: SimpleNamespace()
            for name in (
                "create",
                "execute",
                "propose",
                "capture",
                "replay",
                "graph_nodes",
                "graph_kernels",
                "graph_device_copies",
                "schedule",
                "error",
                "device_bytes",
                "destroy",
            )
        },
    )
    native = SimpleNamespace(generativeqc_get_source_identity=ct.c_void_p(17))

    def reject_discovery(**arguments: Any) -> Any:
        raise AssertionError("artifact hit attempted compiler-cache discovery")

    monkeypatch.setattr(compiler_cache, "resolve_compiler_cache", reject_discovery)
    monkeypatch.setattr(
        bridge.ct, "CDLL", lambda path: library if path == str(binary) else native
    )
    if backend_matches:
        assert bridge._probe_library() is library
    else:
        with pytest.raises(RuntimeError, match="different native backend"):
            bridge._probe_library()


@pytest.mark.parametrize("stage", ["discovery", "compile", "link"])
def test_resident_bridge_failed_build_never_publishes_artifact(
    probe_artifact: Any, monkeypatch: Any, stage: str
) -> None:
    from generativeqc_compiler.common import compiler_cache, compiler_process

    bridge, binary = probe_artifact

    def discover(**arguments: Any) -> object:
        if stage == "discovery":
            raise RuntimeError("no usable compiler cache")
        return object()

    def compile_probe(command: list[str], **arguments: Any) -> Any:
        assert "-c" in command
        Path(command[-1]).write_bytes(b"partial object")
        return SimpleNamespace(
            returncode=int(stage == "compile"), stderr="compile failed"
        )

    def link_probe(command: list[str], **arguments: Any) -> Any:
        assert "-shared" in command
        Path(command[-1]).write_bytes(b"partial shared object")
        return SimpleNamespace(returncode=1, stderr="link failed")

    monkeypatch.setattr(compiler_cache, "resolve_compiler_cache", discover)
    monkeypatch.setattr(compiler_cache, "run_cached_compiler", compile_probe)
    monkeypatch.setattr(compiler_process, "run_compiler", link_probe)
    with pytest.raises(RuntimeError, match="compiler cache|compile failed|link failed"):
        bridge._probe_library()
    assert not binary.exists()
    assert not list(binary.parent.glob("tmp*")), (
        "failed build left an unowned temporary directory"
    )


_CUDA_DOUBLE = r"""
#pragma once
#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal };
enum cudaGraphNodeType { cudaGraphNodeTypeKernel, cudaGraphNodeTypeMemset,
                         cudaGraphNodeTypeHost, cudaGraphNodeTypeMemcpy };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaMemoryType { cudaMemoryTypeHost, cudaMemoryTypeDevice };
struct cudaPitchedPtr { void* ptr; };
struct cudaMemcpy3DParms {
  void* srcArray{};
  void* dstArray{};
  cudaPitchedPtr srcPtr{}, dstPtr{};
  cudaMemcpyKind kind{};
};
struct cudaPointerAttributes { cudaMemoryType type{}; int device{}; };
struct FakeNode { cudaGraphNodeType type; std::function<void()> operation; };
struct FakeStream { bool capturing{}, invalidated{}; std::vector<FakeNode> recording; };
struct FakeGraph { std::vector<FakeNode> nodes; };
struct FakeExec { std::vector<FakeNode> nodes; };
using cudaStream_t = FakeStream*;
using cudaEvent_t = int*;
using cudaGraph_t = FakeGraph*;
using cudaGraphExec_t = FakeExec*;
using cudaGraphNode_t = FakeNode*;
namespace fake {
inline int graphs{}, executables{}, events{}, sources{}, dispatches{}, launches{};
inline int event_creates{}, runtime_error{};
inline std::string failure;
inline int injected_node = -1;
inline cudaMemcpyKind copy_kind = cudaMemcpyDeviceToDevice;
inline cudaMemoryType copy_memory_type = cudaMemoryTypeDevice;
inline int copy_device = 0;
inline bool empty_graph{};
inline FakeStream stream;
inline bool fail(const char* stage) {
  if (failure != stage) return false;
  failure.clear();
  return true;
}
inline void submit(cudaGraphNodeType type, std::function<void()> operation) {
  if (stream.capturing) stream.recording.push_back({type, operation});
  else operation();
}
inline void clean() {
  assert(!stream.capturing && !stream.invalidated && stream.recording.empty());
  assert(graphs == 0 && executables == 0 && events == 0 && sources == 0);
  assert(runtime_error == 0);
}
}
inline cudaError_t cudaGetLastError() {
  const auto result = fake::runtime_error;
  fake::runtime_error = 0;
  return result;
}
inline const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  // The real CUDA runtime rejects synchronization during capture. Check every
  // bridge synchronization, including failure and destructor paths.
  assert(stream == &fake::stream && !stream->capturing);
  return cudaSuccess;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
  ++fake::event_creates;
  if (fake::failure == "second-event" && fake::event_creates == 2) {
    fake::failure.clear();
    return 1;
  }
  *event = new int(0); ++fake::events; return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  assert(!fake::stream.capturing);
  delete event; --fake::events; return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t stream) {
  assert(stream == &fake::stream && !stream->capturing);
  return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float* value, cudaEvent_t, cudaEvent_t) {
  assert(!fake::stream.capturing);
  *value = 1.0F; return cudaSuccess;
}
inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  assert(stream == &fake::stream && !stream->capturing);
  assert(mode == cudaStreamCaptureModeThreadLocal);
  if (fake::fail("begin")) return 1;
  stream->capturing = true;
  return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
  assert(stream == &fake::stream && stream->capturing);
  stream->capturing = false;
  if (stream->invalidated) {
    stream->invalidated = false;
    stream->recording.clear();
    *graph = nullptr;
    fake::runtime_error = 1;
    return 1;
  }
  *graph = new FakeGraph{std::move(stream->recording)};
  stream->recording.clear();
  ++fake::graphs;
  if (fake::empty_graph) { (*graph)->nodes.clear(); fake::empty_graph = false; }
  if (fake::injected_node >= 0) {
    (*graph)->nodes.push_back({static_cast<cudaGraphNodeType>(fake::injected_node), [] {}});
    fake::injected_node = -1;
  }
  return fake::fail("end") ? 1 : cudaSuccess;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  assert(!fake::stream.capturing);
  delete graph; --fake::graphs; return cudaSuccess;
}
inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* n) {
  if (fake::fail("inventory")) return 1;
  if (nodes) {
    assert(*n >= graph->nodes.size());
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) nodes[i] = &graph->nodes[i];
  }
  *n = graph->nodes.size(); return cudaSuccess;
}
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type) {
  if (fake::fail("node-type")) return 1;
  *type = node->type; return cudaSuccess;
}
inline cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* copy) {
  assert(node->type == cudaGraphNodeTypeMemcpy);
  if (fake::fail("copy-params")) return 1;
  copy->kind = fake::copy_kind;
  copy->srcPtr.ptr = &fake::copy_device;
  copy->dstPtr.ptr = &fake::copy_device;
  return cudaSuccess;
}
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* pointer) {
  assert(pointer != nullptr);
  if (fake::fail("copy-pointer")) return 1;
  attributes->type = fake::copy_memory_type;
  attributes->device = fake::copy_device;
  return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* executable, cudaGraph_t graph,
                                               unsigned long long flags) {
  assert(!fake::stream.capturing && flags == 0);
  if (fake::fail("instantiate")) return 1;
  *executable = new FakeExec{graph->nodes}; ++fake::executables; return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t executable) {
  assert(!fake::stream.capturing);
  delete executable; --fake::executables; return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
  assert(stream == &fake::stream && !stream->capturing && fake::sources == 1);
  if (fake::fail("launch")) return 1;
  ++fake::launches;
  for (const auto& node : executable->nodes) node.operation();
  return cudaSuccess;
}
"""

_PROVIDER_DOUBLE = r"""
#pragma once
#include "api/handles.hpp"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <string>
constexpr int GENERATIVEQC_STATUS_SUCCESS = 0;
namespace generativeqc::scf {
enum class FockSpin { Restricted, Unrestricted };
enum class FockBackend { Cuda };
struct Term { bool present = true; };
struct FockBuildSpec { FockSpin spin; Term exchange; int derivative_order = 0; };
inline FockBuildSpec make_hf_fock_spec(FockSpin spin) { return {spin, {}, 0}; }
inline FockBuildSpec resolve_fock_build(FockBuildSpec spec, FockBackend, double) { return spec; }
struct Diagnostic {
  struct { const char* schedule = "md-j-reference/double"; } direct;
  std::size_t device_bytes = 64;
};
struct PreparedFockPlan {
  FockBuildSpec spec;
  Diagnostic info;
  PreparedFockPlan(const core::System&, void*, FockBuildSpec value, int, std::size_t) : spec(value) {
    ++fake::sources;
  }
  ~PreparedFockPlan() {
    // Graph executables and events must be released while their borrowed owner
    // is still alive, including a partially constructed probe.
    assert(fake::graphs == 0 && fake::executables == 0 && fake::events == 0);
    assert(!fake::stream.capturing);
    --fake::sources;
  }
  const Diagnostic& diagnostic() const { return info; }
};
struct PreparedCudaFockBinding {
  int device_id = 0;
  cudaStream_t stream = &fake::stream;
  explicit operator bool() const { return stream != nullptr; }
};
inline PreparedCudaFockBinding prepared_cuda_fock_binding(const PreparedFockPlan&) { return {}; }
template<class... Args> int enqueue_prepared_cuda_fock_proposal(Args&&...) { return 1; }
inline int enqueue_prepared_cuda_fock(const PreparedFockPlan&, const double* density,
                                     const double* beta, std::size_t elements, double* coulomb,
                                     double* alpha_exchange, double* beta_exchange,
                                     int* numerical_error, bool mixed, std::string& detail) {
  ++fake::dispatches;
  assert(!mixed);
  if (!density || !coulomb || !numerical_error || elements != 4) {
    // Production Direct's failure guard attempts to drain its stream. During
    // capture that invalidates the graph: EndCapture must still be called and
    // can return an error with no graph handle to release.
    fake::stream.invalidated = fake::stream.capturing;
    detail = "invalid resident buffers"; return 1;
  }
  fake::submit(cudaGraphNodeTypeMemset, [=] { *numerical_error = 0; });
  fake::submit(cudaGraphNodeTypeMemcpy, [] {});
  if (fake::fail("enqueue")) { detail = "injected provider failure"; return 1; }
  fake::submit(cudaGraphNodeTypeKernel, [=] {
    for (std::size_t i = 0; i < elements; ++i) {
      coulomb[i] = 2.0 * (density[i] + (beta ? beta[i] : 0.0));
      if (alpha_exchange) alpha_exchange[i] = 3.0 * density[i];
      if (beta_exchange) beta_exchange[i] = 3.0 * beta[i];
    }
  });
  if (cudaGetLastError() != cudaSuccess) {
    detail = "unhandled CUDA error survived capture cleanup"; return 1;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
}
"""

_DRIVER = r"""
#include <array>
#include <cmath>

struct Buffers {
  std::array<double, 4> density{1., -2., 3., -4.}, beta{-0.5, 1., -1.5, 2.};
  std::array<double, 4> coulomb{}, alpha_exchange{}, beta_exchange{};
  int error = 93;
  double seconds{};
  int capture(void* probe) {
    return md_j_probe_capture(probe, density.data(), beta.data(), density.size(),
                              coulomb.data(), alpha_exchange.data(), beta_exchange.data(), &error);
  }
  int ordinary(void* probe) {
    return md_j_probe_execute(probe, density.data(), beta.data(), density.size(),
                              coulomb.data(), alpha_exchange.data(), beta_exchange.data(), &error,
                              &seconds);
  }
  void replay(void* probe) {
    // A replay must execute the captured device work, not re-enter the native
    // provider's host dispatch. This observes behavior, not call-name spelling.
    const int dispatched = fake::dispatches;
    coulomb.fill(NAN); alpha_exchange.fill(NAN); beta_exchange.fill(NAN); error = 7;
    assert(md_j_probe_replay(probe, &seconds) == 0);
    assert(fake::dispatches == dispatched && seconds > 0.0 && error == 0);
    for (std::size_t i = 0; i < density.size(); ++i) {
      assert(coulomb[i] == 2.0 * (density[i] + beta[i]));
      assert(alpha_exchange[i] == 3.0 * density[i]);
      assert(beta_exchange[i] == 3.0 * beta[i]);
    }
  }
};

void* create() {
  static generativeqc_system system;
  void* probe = md_j_probe_create(&system, 1, 1, 0.0, 0);
  assert(probe && fake::sources == 1);
  return probe;
}
void retained(void* probe, Buffers& buffers) {
  assert(!fake::stream.capturing && fake::stream.recording.empty());
  assert(fake::graphs == 1 && fake::executables == 1 && fake::sources == 1);
  buffers.density[0] += 0.25;
  buffers.replay(probe);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  unsetenv("GENERATIVEQC_MD_J_WORK_COUNTS");
  const std::string section = argv[1];
  if (section == "partial-creation") {
    fake::failure = "second-event";
    generativeqc_system system;
    assert(!md_j_probe_create(&system, 0, 0, 0.0, 0));
    assert(std::string(md_j_probe_error()).find("injected CUDA") != std::string::npos);
    fake::clean();
    md_j_probe_destroy(create());
    fake::clean();
    return 0;
  }
  if (section == "diagnostics") setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1", 1);
  void* probe = create();
  Buffers buffers;
  if (section == "replay") {
    assert(buffers.ordinary(probe) == 0);
    assert(buffers.capture(probe) == 0);
    assert(md_j_probe_graph_nodes(probe) >= md_j_probe_graph_kernels(probe));
    assert(md_j_probe_graph_kernels(probe) > 0);
    assert(md_j_probe_graph_device_copies(probe) > 0);
    buffers.replay(probe);
    buffers.density = {-2., 4., -6., 8.};
    buffers.replay(probe);
    buffers.density.fill(0.); buffers.beta.fill(0.);
    buffers.replay(probe);
    buffers.density.back() = 0.83;
    buffers.replay(probe);
    assert(buffers.capture(probe) == 0);
    retained(probe, buffers);
    assert(buffers.ordinary(probe) == 0);
  } else if (section == "failed-recapture") {
    assert(buffers.capture(probe) == 0);
    assert(md_j_probe_capture(probe, buffers.density.data(), buffers.beta.data(), 3,
                              buffers.coulomb.data(), buffers.alpha_exchange.data(),
                              buffers.beta_exchange.data(), &buffers.error) != 0);
    retained(probe, buffers);
    assert(buffers.ordinary(probe) == 0);
    for (const auto* stage : {"begin", "enqueue", "end", "inventory", "node-type", "copy-params", "copy-pointer", "instantiate"}) {
      fake::failure = stage;
      assert(buffers.capture(probe) != 0);
      assert(std::strlen(md_j_probe_error()) > 0);
      retained(probe, buffers);
      assert(buffers.ordinary(probe) == 0);
    }
    fake::injected_node = cudaGraphNodeTypeHost;
    assert(buffers.capture(probe) != 0);
    assert(std::string(md_j_probe_error()).find("non-device") != std::string::npos);
    retained(probe, buffers);
    for (const auto kind : {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDefault}) {
      fake::copy_kind = kind;
      assert(buffers.capture(probe) != 0);
      assert(std::string(md_j_probe_error()).find("non-device") != std::string::npos);
      fake::copy_kind = cudaMemcpyDeviceToDevice;
      retained(probe, buffers);
    }
    fake::copy_device = 1;
    assert(buffers.capture(probe) != 0);
    fake::copy_device = 0;
    retained(probe, buffers);
    fake::copy_memory_type = cudaMemoryTypeHost;
    assert(buffers.capture(probe) != 0);
    fake::copy_memory_type = cudaMemoryTypeDevice;
    retained(probe, buffers);
    fake::empty_graph = true;
    assert(buffers.capture(probe) != 0);
    assert(std::string(md_j_probe_error()).find("no kernels") != std::string::npos);
    retained(probe, buffers);
    fake::failure = "launch";
    assert(md_j_probe_replay(probe, &buffers.seconds) != 0);
    retained(probe, buffers);
    assert(buffers.capture(probe) == 0);
    retained(probe, buffers);
  } else if (section == "negative-inputs") {
    assert(md_j_probe_replay(probe, &buffers.seconds) != 0);
    assert(std::string(md_j_probe_error()).find("not been captured") != std::string::npos);
    assert(md_j_probe_replay(nullptr, &buffers.seconds) != 0);
    assert(md_j_probe_replay(probe, nullptr) != 0);
    assert(md_j_probe_capture(nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr) != 0);
    assert(md_j_probe_capture(probe, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr) != 0);
    assert(!fake::stream.capturing && fake::graphs == 0 && fake::executables == 0);
    assert(!md_j_probe_create(nullptr, 0, 0, 0., 0));
    assert(buffers.capture(probe) == 0);
    retained(probe, buffers);
  } else if (section == "diagnostics") {
    unsetenv("GENERATIVEQC_MD_J_WORK_COUNTS");
    assert(buffers.capture(probe) != 0);
    assert(std::string(md_j_probe_error()).find("WORK_COUNTS disabled") != std::string::npos);
    assert(fake::graphs == 0 && fake::executables == 0 && !fake::stream.capturing);
    assert(buffers.ordinary(probe) == 0);
  } else { assert(false); }
  md_j_probe_destroy(probe);
  md_j_probe_destroy(nullptr);
  fake::clean();
}
"""
