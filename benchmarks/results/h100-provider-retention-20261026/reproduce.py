"""Public reproduction derivative; recorded source identity is in provenance.json."""

# Preserve original helper signatures, dict construction and failure/cleanup catches.
# ruff: noqa: ANN001, ANN002, ANN201, BLE001, C408

import ctypes as C
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

if not __debug__:
    raise RuntimeError(
        "reproduction requires unoptimized Python for original assertion guards"
    )

started = time.monotonic()
signal.signal(
    signal.SIGALRM,
    lambda *_: (_ for _ in ()).throw(TimeoutError("finite 150s diagnostic expired")),
)
signal.alarm(150)
permit = {"max_observed_provider_delta_bytes": 536870912}
events = []
record = {
    "schema": "root.h100-provider-initialization-diagnostic.v1",
    "scope": "standalone vendor handles; no GenerativeQC, geometry, SCF, matrix operation or scientific request",
    "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "events": events,
}
stream, blas, solver, params = C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_void_p()
runtime = bl = so = None
cleanup = []


def gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    rows = [
        line.strip().split(", ") for line in result.stdout.splitlines() if line.strip()
    ]
    assert len(rows) == 1 and "H100" in rows[0][0], "expected one visible H100"
    name, uuid, pci, driver, memory = rows[0]
    return dict(
        name=name,
        uuid_sha256=hashlib.sha256(uuid.encode()).hexdigest(),
        pci=pci,
        driver=driver,
        memory=memory,
    )


def bind(library, name, args):
    fn = getattr(library, name)
    fn.argtypes = args
    fn.restype = C.c_int
    return fn


def call(library, name, args, *values):
    status = int(bind(library, name, args)(*values))
    events.append(
        dict(api=name, status=status, elapsed_seconds=time.monotonic() - started)
    )
    if status:
        raise RuntimeError(name + " status=" + str(status))


def memory(label):
    call(runtime, "cudaStreamSynchronize", [C.c_void_p], stream)
    free, total = C.c_size_t(), C.c_size_t()
    call(
        runtime,
        "cudaMemGetInfo",
        [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)],
        C.byref(free),
        C.byref(total),
    )
    row = dict(label=label, free_bytes=free.value, total_bytes=total.value)
    record.setdefault("memory_snapshots", []).append(row)
    if label != "before_providers":
        delta = record["memory_snapshots"][0]["free_bytes"] - free.value
        row["signed_free_byte_difference_from_baseline"] = delta
        if delta > permit["max_observed_provider_delta_bytes"]:
            raise RuntimeError("engineering provider delta stop cap exceeded")
    return row


try:
    record["device_before"] = gpu()
    libroot = Path("/opt/generativeqc-reuse/cuda-12.9/targets/x86_64-linux/lib")
    names = ["libcudart.so.12", "libcublas.so.12", "libcusolver.so.11"]
    libraries = {}
    for name in names:
        p = (libroot / name).resolve(strict=True)
        st = p.stat()
        assert st.st_size < (1 << 30), "bounded explicit library hashing only"
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        after = p.stat()
        assert (st.st_ino, st.st_size, st.st_mtime_ns) == (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        libraries[name] = dict(
            path=str(p),
            bytes=st.st_size,
            sha256=digest.hexdigest(),
            mode=oct(st.st_mode & 0o777),
        )
    record["explicitly_loaded_libraries"] = libraries
    record["transitive_dependency_byte_hashes"] = (
        "UNAVAILABLE_NOT_HASHED; exact locator is retained separately"
    )
    runtime = C.CDLL(libraries["libcudart.so.12"]["path"], mode=C.RTLD_GLOBAL)
    bl = C.CDLL(libraries["libcublas.so.12"]["path"], mode=C.RTLD_GLOBAL)
    so = C.CDLL(libraries["libcusolver.so.11"]["path"], mode=C.RTLD_GLOBAL)
    call(runtime, "cudaSetDevice", [C.c_int], 0)
    call(runtime, "cudaStreamCreate", [C.POINTER(C.c_void_p)], C.byref(stream))
    runtime_version = C.c_int()
    call(
        runtime, "cudaRuntimeGetVersion", [C.POINTER(C.c_int)], C.byref(runtime_version)
    )
    record["cuda_runtime_version"] = runtime_version.value
    memory("before_providers")
    call(bl, "cublasCreate_v2", [C.POINTER(C.c_void_p)], C.byref(blas))
    call(bl, "cublasSetStream_v2", [C.c_void_p, C.c_void_p], blas, stream)
    call(bl, "cublasSetPointerMode_v2", [C.c_void_p, C.c_int], blas, 0)
    memory("after_cublas")
    call(so, "cusolverDnCreate", [C.POINTER(C.c_void_p)], C.byref(solver))
    call(so, "cusolverDnSetStream", [C.c_void_p, C.c_void_p], solver, stream)
    call(so, "cusolverDnCreateParams", [C.POINTER(C.c_void_p)], C.byref(params))
    memory("after_cublas_and_cusolver")
    record["reference_allowance_comparison"] = dict(
        current_source="e1e244197411b58da42aa8a2cbce49014d7ed3ff",
        one_provider_allowance_bytes=96 << 20,
        both_providers_allowance_bytes=192 << 20,
        observed_signed_difference=record["memory_snapshots"][-1][
            "signed_free_byte_difference_from_baseline"
        ],
        scope="raw synchronized free-byte difference in this standalone diagnostic, not a historical CBS measurement or causal allocation/peak ledger",
    )
    record["status"] = "OBSERVED_NO_SCIENTIFIC_QUALIFICATION"
except BaseException as exc:
    record["status"] = "FAILED_STOP_NO_RETRY"
    record["error"] = dict(type=type(exc).__name__, message=str(exc))
finally:
    for library, name, handle in [
        (so, "cusolverDnDestroyParams", params),
        (so, "cusolverDnDestroy", solver),
        (bl, "cublasDestroy_v2", blas),
        (runtime, "cudaStreamDestroy", stream),
    ]:
        if library is not None and handle.value:
            try:
                cleanup.append(
                    dict(
                        api=name, status=int(bind(library, name, [C.c_void_p])(handle))
                    )
                )
            except BaseException as exc:
                cleanup.append(dict(api=name, error=repr(exc)))
    record["handle_cleanup"] = cleanup
    try:
        record["device_after"] = gpu()
        record["same_device_before_after"] = (
            record["device_before"] == record["device_after"]
        )
    except BaseException as exc:
        record["device_after_error"] = repr(exc)
    record["elapsed_seconds"] = time.monotonic() - started
    record["scientific_requests"] = 0
    record["actual_internal_provider_work_peak_billing"] = "UNAVAILABLE"
    raw = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    assert len(raw) < (1 << 20)
    print(raw.decode(), end="")
    signal.alarm(0)
if (
    record["status"] != "OBSERVED_NO_SCIENTIFIC_QUALIFICATION"
    or not record.get("same_device_before_after")
    or any(r.get("status") != 0 for r in cleanup)
):
    raise SystemExit(125)
