"""Explicit artifact/work receipts from the executed native automatic KS owner.

This diagnostic reads a current converged token and process-lifetime compiled
descriptors. It never resolves compiler IR, generates source or executes science.
Binary identity comes from the module containing each actual native descriptor,
not from a caller-provided library filename or an emitted-source hash alone.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any

from . import _native


def _loaded_artifact(
    address: int | None,
    memo: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Resolve a live descriptor to its loaded POSIX module and exact file bytes."""
    if not address:
        return None

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("symbol_address", ctypes.c_void_p),
        ]

    query = getattr(ctypes.CDLL(None), "dladdr", None)
    if query is None:
        raise NotImplementedError("loaded native artifact identity requires dladdr")
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    query.restype = ctypes.c_int
    info = DlInfo()
    if not query(address, ctypes.byref(info)) or not info.filename:
        raise RuntimeError("native descriptor has no authenticated loaded module")
    path = Path(info.filename.decode()).resolve(strict=True)
    before = path.stat()
    # Authenticate the ELF base mapping returned by dladdr, rather than the
    # descriptor's page: dynamically initialized static registrations can live
    # in anonymous BSS pages with inode zero. The loader still associates those
    # addresses with this module. An atomic package replacement must not relabel
    # the old loaded module with the new library's hash. The mapped inode itself
    # must remain immutable while loaded, as in ordinary library deployment.
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            start, end = (int(value, 16) for value in fields[0].split("-"))
            if info.base is not None and start <= info.base < end:
                major, minor = (int(value, 16) for value in fields[3].split(":"))
                if (int(fields[4]), os.makedev(major, minor)) != (
                    before.st_ino,
                    before.st_dev,
                ):
                    raise RuntimeError("loaded native artifact file was replaced")
                break
        else:
            raise RuntimeError("native descriptor has no live process mapping")
    # One diagnostic can bind several descriptors/callbacks from the same large
    # core. Reuse its digest only within this read, after independently checking
    # each live mapping and the complete file stat. Never cache across queries.
    key = (
        str(path),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    previous = memo.get(key) if memo is not None else None
    if previous is None:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    else:
        digest = previous["sha256"]
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("native artifact changed while its receipt was read")
    result = {"path": str(path), "sha256": digest, "bytes": before.st_size}
    if memo is not None:
        memo[key] = result.copy()
    return result


def read_automatic_artifact_diagnostic(batch: Any, index: int = 0) -> dict[str, Any]:
    """Read actual AOT identities and cumulative force work for a live KS item.

    A later energy replay retains its compiled artifacts but reports no force
    at the new token. A failed/nonconverged solve cannot borrow an old receipt.
    Register/local/shared-memory measurements are separate binary-qualified
    offline data; these runtime workspace/work fields do not invent them.
    """
    batch._ensure_open()
    library = batch._library
    query = getattr(library, "generativeqc_ks_batch_automatic_artifact_v2", None)
    version = 2 if query is not None else 1
    if query is None:
        query = getattr(library, "generativeqc_ks_batch_automatic_artifact_v1", None)
    if query is None:
        raise NotImplementedError(
            "native automatic artifact diagnostics are unavailable"
        )
    if type(index) is not int or not 0 <= index <= 0xFFFFFFFF:
        raise ValueError("automatic artifact item index must fit uint32")
    identities = (ctypes.c_char_p * (11 if version == 2 else 7))()
    owners = (ctypes.c_void_p * (4 if version == 2 else 2))()
    work = (ctypes.c_uint64 * (43 if version == 2 else 40))()
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    _native.check(
        library,
        query(
            batch._batch,
            index,
            identities,
            len(identities),
            owners,
            len(owners),
            work,
            len(work),
        ),
    )
    names = (
        "identifier",
        "point_expression_identity",
        "point_source_identity",
        "force_profile",
        "force_plan_identity",
        "force_source_identity",
        "fallback_reason",
    ) + (
        (
            "force_consumer_identity",
            "force_consumer_profile",
            "polarized_executable_identity",
            "restricted_executable_identity",
        )
        if version == 2
        else ()
    )
    result = dict(
        zip(
            names,
            (value.decode() if value else "" for value in identities),
            strict=True,
        )
    )
    memo: dict[tuple[Any, ...], dict[str, Any]] = {}
    point_artifact = _loaded_artifact(owners[0], memo)
    force_artifact = _loaded_artifact(owners[1], memo)
    result.update(
        schema=f"generativeqc.native-automatic-artifact.v{version}",
        point_artifact=point_artifact,
        force_artifact=force_artifact,
        force_completed=bool(work[0]),
        domain_version=int(work[1]),
        spin_blocks=int(work[2]),
        source_count=int(work[3]),
        work_scope="cumulative retained force owner at the current final token",
        capacity=dict(
            zip(
                (
                    "device_upper_bytes",
                    "minimum_device_bytes",
                    "grid_numeric_bytes",
                    "stationary_numeric_bytes",
                    "grid_provider_allowance_bytes",
                    "grid_provider_retained_bytes",
                ),
                work[4:10],
                strict=True,
            )
        ),
        stationary_work_v1=list(work[10:34]),
        grid_work=dict(
            zip(
                (
                    "evaluation_points",
                    "evaluation_tiles",
                    "ao_jet_values",
                    "projection_fma_pairs",
                    "identical_spin_copy_bytes",
                    "feature_passes",
                ),
                work[34:40],
                strict=True,
            )
        ),
    )
    if version == 2:
        result["staged_point_work"] = dict(
            zip(("tile_bytes", "launches", "points"), work[40:43], strict=True)
        )
        result["point_executable_artifact"] = _loaded_artifact(owners[2], memo)
        result["force_executable_artifact"] = _loaded_artifact(owners[3], memo)
        result["force_schedule"] = (
            "shared-staged"
            if result["force_consumer_profile"].startswith("shared_")
            else "static-fused"
        )
    return result
