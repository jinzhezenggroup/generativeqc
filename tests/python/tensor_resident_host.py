"""Compile the complete resident wrapper with explicit host dependency seams.

Only ordinary emission is replaced with the host owner/CUDA dependency fixture.
Int64 passthrough plans with an empty floating literal require no kernels/launches;
their normal production emitters run without being patched.
The resident exports/span tables come from resident_source; transfer/checked come
from a byte-for-byte copy of the production cuda_resident.cuh. This is not a CUDA
kernel, numerical, ordinary-emitter, or production Context lifecycle test.
"""

import ctypes
import shutil
import typing
from pathlib import Path

from generativeqc_compiler.tensor import cuda_resident_emit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/tensor_resident_host"


def build_resident_host(
    plan: typing.Any,
    directory: Path,
    native_cxx: typing.Any,
    monkeypatch: typing.Any,
    transform: typing.Callable[[str], str] | None = None,
    *,
    header_transform: typing.Callable[[str], str] | None = None,
) -> ctypes.CDLL:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "src/tensor/cuda_resident.cuh", directory / "cuda_resident.cuh"
    )
    shutil.copyfile(FIXTURE / "cuda_runtime.cuh", directory / "cuda_runtime.cuh")
    shutil.copyfile(ROOT / "src/tensor/metrics.hpp", directory / "metrics.hpp")
    if header_transform:
        # Opt-in damage controls only; normal gates compile the unchanged header.
        header = directory / "cuda_resident.cuh"
        header.write_text(header_transform(header.read_text()), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(
            cuda_resident_emit,
            "emit_cuda",
            lambda *a, **k: '#include "cuda_runtime.cuh"\n',
        )
        source = cuda_resident_emit.resident_source(plan)
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
        "host_owner": (ctypes.c_void_p, [ctypes.c_size_t]),
        "host_destroy": (None, [ctypes.c_void_p]),
        "host_arena": (ctypes.c_void_p, [ctypes.c_void_p]),
        "host_capacity": (None, [ctypes.c_void_p, ctypes.c_size_t]),
        "host_static_ready": (None, [ctypes.c_void_p, ctypes.c_int]),
        "host_failure": (None, [ctypes.c_int]),
        "host_synchronizations": (ctypes.c_int, []),
        "host_pending_work": (ctypes.c_size_t, []),
        "host_clear_copy_kinds": (None, []),
        "host_copy_matches": (ctypes.c_int, [ctypes.c_int]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args
    return lib
