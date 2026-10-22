"""Predeclared, PID-bound Nsight diagnostics rules; never GPU/science acceptance.

The wrapper is independently qualified as NVTX-only. Do not initialize CUDA in
it to silence warnings. Tool/source identities and the two exact warnings are
frozen before capture; a native-child, global or unknown warning stays fatal.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

try:
    from benchmarks.cc_tile_environment import _profiler_recipe, canonical_hash
except ModuleNotFoundError:
    from cc_tile_environment import _profiler_recipe, canonical_hash

SCHEMA = "generativeqc.cc-nsys-diagnostics-policy.v1"
PARENT_WARNINGS = (
    "CUDA profiling might have not been started correctly.",
    "No CUDA events collected. Does the process use CUDA?",
)
SEVERITIES = {"UnknownLevel", "Info", "Warning", "Error", "Verbose"}
SOURCES = {"UnknownSource", "Injection", "Daemon", "Analysis"}


def _codes(value: Any, names: set[str]) -> None:
    if (
        not isinstance(value, dict)
        or value.keys() != names
        or any(type(code) is not int or code < 0 for code in value.values())
        or len(set(value.values())) != len(names)
    ):
        raise ValueError("complete unique Nsight diagnostic enum map required")


def diagnostics_policy(
    profiler: dict, parent_source_sha256: str, severity_codes: dict, source_codes: dict
) -> dict:
    """Freeze only the qualified tool, wrapper source and fixed parent rules."""
    _profiler_recipe(profiler)
    _codes(severity_codes, SEVERITIES)
    _codes(source_codes, SOURCES)
    if (
        not isinstance(parent_source_sha256, str)
        or len(parent_source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in parent_source_sha256)
    ):
        raise ValueError("qualified NVTX-only parent source SHA256 required")
    return {
        "schema": SCHEMA,
        "profiler": deepcopy(profiler),
        "parent_role": "nvtx-only-wrapper",
        "parent_source_sha256": parent_source_sha256,
        "severity_codes": deepcopy(severity_codes),
        "source_codes": deepcopy(source_codes),
        "parent_warnings": list(PARENT_WARNINGS),
    }


def validate_diagnostics_policy(
    policy: Any, expected_hash: Any, profiler: dict
) -> None:
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {
            "schema",
            "profiler",
            "parent_role",
            "parent_source_sha256",
            "severity_codes",
            "source_codes",
            "parent_warnings",
        }
        or policy
        != diagnostics_policy(
            profiler,
            policy["parent_source_sha256"],
            policy["severity_codes"],
            policy["source_codes"],
        )
        or canonical_hash(policy) != expected_hash
    ):
        raise ValueError("original frozen Nsight diagnostics policy required")


def _process(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"pid", "starttime"}
        or type(value["pid"]) is not int
        or value["pid"] <= 0
        or not isinstance(value["starttime"], str)
        or not value["starttime"].isascii()
        or not value["starttime"].isdigit()
        or len(value["starttime"]) > 32
    ):
        raise ValueError("actual process PID/starttime required")


def _rows(rows: Any, severity_codes: dict, source_codes: dict) -> list[dict]:
    _codes(severity_codes, SEVERITIES)
    _codes(source_codes, SOURCES)
    if not isinstance(rows, list) or not rows or len(rows) > 65536:
        raise ValueError("retained Nsight diagnostics unavailable")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"timestamp", "timestampType", "source", "severity", "text", "globalPid"}
            or any(
                type(row[key]) is not int
                for key in (
                    "timestamp",
                    "timestampType",
                    "source",
                    "severity",
                    "globalPid",
                )
            )
            or row["timestamp"] < 0
            or row["timestampType"] not in (1, 2)
            or not isinstance(row["text"], str)
            or not row["text"]
            or "\0" in row["text"]
            or row["severity"] not in severity_codes.values()
            or row["source"] not in source_codes.values()
            or row["severity"] == severity_codes["UnknownLevel"]
            or row["source"] == source_codes["UnknownSource"]
        ):
            raise ValueError("unknown/malformed Nsight diagnostic record")
    return rows


def strict_diagnostics(rows: Any, severity_codes: dict, source_codes: dict) -> dict:
    """Without a prior bound parent exception, every warning/error is rejected."""
    _rows(rows, severity_codes, source_codes)
    if any(
        row["severity"] not in (severity_codes["Info"], severity_codes["Verbose"])
        for row in rows
    ):
        raise ValueError("Nsight warning/error without a bound parent exception")
    return {"classification": "info-or-verbose-only", "records": deepcopy(rows)}


def classify_diagnostics(
    rows: Any,
    policy: dict,
    policy_hash: str,
    *,
    profiler: dict,
    severity_codes: dict,
    source_codes: dict,
    binding: dict,
    binding_hash: str,
    parent: dict,
    native: dict,
    parent_global_tid: int,
    native_global_pid: int,
    parent_cuda_activities: int,
    native_cuda_activities: int,
    parent_source: Path,
) -> dict:
    """Admit at most one of each exact warning from the proven NVTX-only parent.

    Callers derive activities and NVTX process selection from the actual export,
    and independently validate the session binding and source contract. This
    classification does not prove a lossless capture, CUDA coverage or science.
    All original rows, including admitted warnings, remain in the result.
    """
    validate_diagnostics_policy(policy, policy_hash, profiler)
    _rows(rows, severity_codes, source_codes)
    _process(parent)
    _process(native)
    if (
        policy["severity_codes"] != severity_codes
        or policy["source_codes"] != source_codes
        or not isinstance(binding, dict)
        or canonical_hash(binding) != binding_hash
        or binding.get("profiler") != profiler
        or binding.get("parent", {}).get("process") != parent
        or parent == native
        or parent["pid"] == native["pid"]
        or type(parent_global_tid) is not int
        or parent_global_tid < 0
        or (parent_global_tid // 0x1000000) % 0x1000000 != parent["pid"]
        or type(native_global_pid) is not int
        or native_global_pid < 0
        or native_global_pid % 0x1000000 != 0
        or (native_global_pid // 0x1000000) % 0x1000000 != native["pid"]
        or type(parent_cuda_activities) is not int
        or parent_cuda_activities != 0
        or type(native_cuda_activities) is not int
        or native_cuda_activities <= 0
        or parent_source.is_symlink()
        or not parent_source.is_file()
        or hashlib.sha256(parent_source.read_bytes()).hexdigest()
        != policy["parent_source_sha256"]
    ):
        raise ValueError("actual NVTX-only parent/native/tool/source binding required")
    parent_global_pid = parent_global_tid - parent_global_tid % 0x1000000
    accepted = []
    seen = set()
    for row in rows:
        if row["severity"] in (severity_codes["Info"], severity_codes["Verbose"]):
            continue
        if (
            row["severity"] != severity_codes["Warning"]
            or row["source"] != source_codes["Analysis"]
            or row["text"] not in PARENT_WARNINGS
            or row["globalPid"] != parent_global_pid
            or row["text"] in seen
        ):
            raise ValueError("native/global/unknown/extra Nsight warning or error")
        seen.add(row["text"])
        accepted.append(deepcopy(row))
    return {
        "classification": "bound-nvtx-only-parent-warnings"
        if accepted
        else "info-or-verbose-only",
        "policy_sha256": policy_hash,
        "accepted_parent_warnings": accepted,
        "accepted_parent_warning_count": len(accepted),
        "records": deepcopy(rows),
    }
