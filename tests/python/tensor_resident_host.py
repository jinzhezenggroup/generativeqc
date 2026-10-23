"""Compile the complete resident wrapper with explicit host dependency seams.

Int64 passthrough plans with an empty floating literal require no kernels/launches.
Their normal production emitters run without being patched; ordinary emission
can either use a dependency fixture or compile unchanged with host dependencies.
The resident exports/span tables come from resident_source; transfer/checked come
from a byte-for-byte copy of the production cuda_resident.cuh. Validation kernels
can also execute serially on the host. None of this establishes real CUDA launch,
concurrency, scientific numerical parity, or production Context lifecycle behavior.
"""

import ctypes
import shutil
import typing
from pathlib import Path

from generativeqc_compiler.tensor import cuda_resident_emit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/tensor_resident_host"


def build_resident_validation_host(
    plan: typing.Any, directory: Path, native_cxx: typing.Any
) -> ctypes.CDLL:
    """Execute emitted validation arithmetic with serial host CUDA lanes.

    Compile complete emitted kernels, without slicing or rewriting their
    bodies. The small wrapper calls the emitter's per-input entry points; it
    does not model CUDA launches, concurrency, or the resident launch sequence.
    Real-device tests remain responsible for those integration contracts.
    """
    validations, _ = cuda_resident_emit._validation_body(plan)
    calls = "\n".join(
        f"resident_validate_{slot}(arena, &status);"
        for slot, i in enumerate(plan.inputs)
        if plan.steps[i].node.spec.dtype != "int64"
    )
    source = directory / "validation.cpp"
    source.write_text(
        "#include <cmath>\n#define __global__\nusing std::isfinite;\n"
        "struct Dim { unsigned x; };\n"
        "static Dim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};\n"
        "int atomicCAS(int* p, int old, int value) {\n"
        "  const int before = *p; if (before == old) *p = value; return before;\n}\n"
        + validations
        + '\nextern "C" int validate(unsigned char* arena) {\n'
        + "int status = 0;\n"
        + calls
        + "\nreturn status;\n}\n",
        encoding="utf-8",
    )
    library = native_cxx.build_shared(
        [source], directory / "validation.so", compile_args=["-std=c++17"]
    )
    lib = ctypes.CDLL(str(library))
    lib.validate.argtypes = [ctypes.c_void_p]
    lib.validate.restype = ctypes.c_int
    return lib


def build_resident_host(
    plan: typing.Any,
    directory: Path,
    native_cxx: typing.Any,
    monkeypatch: typing.Any,
    transform: typing.Callable[[str], str] | None = None,
    *,
    header_transform: typing.Callable[[str], str] | None = None,
    extension: str = "",
    ordinary: bool = False,
    embed_static_data: bool = True,
) -> ctypes.CDLL:
    """Compile emitted ABIs against a host transport, without CUDA execution.

    ``ordinary`` retains the real ordinary emitter as well. Its dependency
    fixture supports control passthrough plans only; no scientific kernels or
    Graph capture are simulated. Extensions pass through the production API.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "src/tensor/cuda_resident.cuh", directory / "cuda_resident.cuh"
    )
    shutil.copyfile(FIXTURE / "cuda_runtime.cuh", directory / "cuda_runtime.cuh")
    shutil.copyfile(
        FIXTURE / "cuda_graph_context.cuh", directory / "cuda_graph_context.cuh"
    )
    shutil.copyfile(ROOT / "src/tensor/metrics.hpp", directory / "metrics.hpp")
    if header_transform:
        # Opt-in damage controls only; normal gates compile the unchanged header.
        header = directory / "cuda_resident.cuh"
        header.write_text(header_transform(header.read_text()), encoding="utf-8")
    with monkeypatch.context() as patch:
        if not ordinary:
            patch.setattr(
                cuda_resident_emit,
                "emit_cuda",
                lambda *a, **k: '#include "cuda_runtime.cuh"\n',
            )
        source = cuda_resident_emit.resident_source(
            plan, extension=extension, embed_static_data=embed_static_data
        )
    if transform:
        source = transform(source)
    cpp = directory / "resident.cpp"
    cpp.write_text(source, encoding="utf-8")
    library = native_cxx.build_shared(
        [cpp],
        directory / "resident.so",
        compile_args=["-std=c++17", "-pthread"],
        link_args=["-pthread"],
    )
    lib = ctypes.CDLL(str(library))
    signatures = {
        "resident_plan_identity": (ctypes.c_char_p, []),
        "resident_abi": (ctypes.c_int, []),
        "resident_upload": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ],
        ),
        "resident_download": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ],
        ),
        "resident_run": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ],
        ),
        "host_owner": (ctypes.c_void_p, [ctypes.c_size_t, ctypes.c_size_t]),
        "host_status_marker": (None, [ctypes.c_int]),
        "host_metrics": (
            None,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
        ),
        "host_destroy": (None, [ctypes.c_void_p]),
        "host_arena": (ctypes.c_void_p, [ctypes.c_void_p]),
        "host_capacity": (None, [ctypes.c_void_p, ctypes.c_size_t]),
        "host_static_ready": (None, [ctypes.c_void_p, ctypes.c_int]),
        "host_failure": (None, [ctypes.c_int]),
        "host_synchronizations": (ctypes.c_int, []),
        "host_pending_work": (ctypes.c_size_t, []),
        "host_clear_copy_kinds": (None, []),
        "host_copy_matches": (ctypes.c_int, [ctypes.c_int]),
        "host_copy_covers": (ctypes.c_int, [ctypes.c_size_t]),
        "host_hold_mutex": (ctypes.c_void_p, [ctypes.c_void_p]),
        "host_mutex_held": (ctypes.c_int, [ctypes.c_void_p]),
        "host_release_mutex": (None, [ctypes.c_void_p]),
        "host_work_sequence": (ctypes.c_size_t, []),
    }
    for name, (result, args) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args
    return lib
