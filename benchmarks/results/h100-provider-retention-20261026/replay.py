"""Replay one retained vendor-init diagnostic using only CPU/stdlib data reads."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path

MIB = 1 << 20
ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse(data: bytes | str) -> dict:
    def pairs(items: list) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError("nonfinite JSON: " + value)

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(isinstance(result, dict), "expected JSON object")
    return result


def unpack(name: str) -> bytes:
    packed = (ROOT / name).read_bytes()
    require(len(packed) <= MIB, "stored member exceeds one MiB")
    decoder = lzma.LZMADecompressor(memlimit=96 * MIB)
    raw = decoder.decompress(packed, max_length=MIB + 1)
    require(
        len(raw) <= MIB and decoder.eof and not decoder.unused_data,
        "XZ extent/EOF/trailing bytes",
    )
    return raw


def integer(value: object) -> int:
    require(type(value) is int and 0 <= value < 2**64, "expected uint64 integer")
    return value


def replay() -> dict:
    manifest = parse((ROOT / "retention-manifest.json").read_bytes())
    require(
        manifest["schema"] == "generativeqc.provider-initialization-retention.v1",
        "retention schema",
    )
    members = [item["path"] for item in manifest["files"]]
    require(len(members) == len(set(members)), "duplicate retained member")
    require(
        set(members)
        == {path.name for path in ROOT.iterdir() if path.is_file()}
        - {"retention-manifest.json"},
        "retention manifest does not cover all public files",
    )
    for item in manifest["files"]:
        path = item["path"]
        require(isinstance(path, str) and Path(path).name == path, "unsafe member path")
        data = (ROOT / path).read_bytes()
        require(
            len(data) == item["bytes"] and digest(data) == item["sha256"],
            "retained byte binding: " + path,
        )
    provenance = parse((ROOT / "provenance.json").read_bytes())
    raw = unpack("observation.json.xz")
    spec = provenance["raw_observation"]
    require(
        len(raw) == spec["raw_bytes"] and digest(raw) == spec["raw_sha256"],
        "original observation bytes",
    )
    record = parse(raw)
    stdout = unpack("provider-stdout.utf8.xz")
    spec = provenance["raw_provider_stdout"]
    require(
        len(stdout) == spec["raw_bytes"] and digest(stdout) == spec["raw_sha256"],
        "original stdout bytes",
    )
    text = stdout.decode("utf-8")
    value, offset = json.JSONDecoder().raw_decode(text)
    require(
        value == record and text[offset:].strip() == "OK",
        "actual stdout observation/tail",
    )
    require(
        record["schema"] == "root.h100-provider-initialization-diagnostic.v1",
        "raw schema",
    )
    require(
        record["status"] == "OBSERVED_NO_SCIENTIFIC_QUALIFICATION",
        "diagnostic did not finish",
    )
    require(
        type(record["scientific_requests"]) is int
        and record["scientific_requests"] == 0,
        "scientific calls",
    )
    require(record["visibility"] is None, "retain actual unset visibility")
    require(
        record["device_before"] == record["device_after"]
        and record["same_device_before_after"] is True,
        "device drift",
    )
    device = record["device_before"]
    require(
        "H100" in device["name"] and device["driver"] == "595.58.03", "device cohort"
    )
    require(
        device["pci"] == "00000000:3B:00.0"
        and device["uuid_sha256"]
        == "977070d342fac93a326dafacea887da87b42ff849e2c413a449f065e0c4ed666",
        "selected device identity",
    )
    require(record["cuda_runtime_version"] == 12090, "runtime version")
    events = record["events"]
    expected = [
        "cudaSetDevice",
        "cudaStreamCreate",
        "cudaRuntimeGetVersion",
        "cudaStreamSynchronize",
        "cudaMemGetInfo",
        "cublasCreate_v2",
        "cublasSetStream_v2",
        "cublasSetPointerMode_v2",
        "cudaStreamSynchronize",
        "cudaMemGetInfo",
        "cusolverDnCreate",
        "cusolverDnSetStream",
        "cusolverDnCreateParams",
        "cudaStreamSynchronize",
        "cudaMemGetInfo",
    ]
    require(
        [event["api"] for event in events] == expected, "actual diagnostic event order"
    )
    elapsed = record["elapsed_seconds"]
    require(
        type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed < 150,
        "probe clock",
    )
    times = []
    for event in events:
        require(type(event["status"]) is int and event["status"] == 0, "API failure")
        timestamp = event["elapsed_seconds"]
        require(
            type(timestamp) in (int, float)
            and math.isfinite(timestamp)
            and 0 <= timestamp <= elapsed,
            "event clock",
        )
        times.append(timestamp)
    require(times == sorted(times), "nonmonotonic event clock")
    cleanup = record["handle_cleanup"]
    require(
        [item["api"] for item in cleanup]
        == [
            "cusolverDnDestroyParams",
            "cusolverDnDestroy",
            "cublasDestroy_v2",
            "cudaStreamDestroy",
        ],
        "incomplete cleanup",
    )
    require(
        all(
            type(item.get("status")) is int and item["status"] == 0 for item in cleanup
        ),
        "cleanup failure",
    )
    snapshots = record["memory_snapshots"]
    require(
        [row["label"] for row in snapshots]
        == ["before_providers", "after_cublas", "after_cublas_and_cusolver"],
        "snapshot sequence",
    )
    free = [integer(row["free_bytes"]) for row in snapshots]
    totals = [integer(row["total_bytes"]) for row in snapshots]
    require(
        len(set(totals)) == 1 and all(0 < n <= totals[0] for n in free),
        "memory extents",
    )
    blas_delta, combined_delta = free[0] - free[1], free[0] - free[2]
    require(
        snapshots[1]["signed_free_byte_difference_from_baseline"] == blas_delta
        and snapshots[2]["signed_free_byte_difference_from_baseline"] == combined_delta,
        "stored difference disagrees",
    )
    allowance = 2 * (96 * MIB)
    policy = record["reference_allowance_comparison"]
    require(
        policy["current_source"]
        == provenance["source_allowance"]["revision"]
        == "e1e244197411b58da42aa8a2cbce49014d7ed3ff",
        "comparison source identity",
    )
    require(
        policy["one_provider_allowance_bytes"] == 96 * MIB
        and policy["both_providers_allowance_bytes"] == allowance
        and policy["observed_signed_difference"] == combined_delta,
        "frozen allowance/delta",
    )
    require(
        blas_delta == 68 * MIB
        and combined_delta == 358 * MIB
        and allowance == 192 * MIB,
        "retained claim not reproduced",
    )
    libraries = record["explicitly_loaded_libraries"]
    require(
        set(libraries) == {"libcudart.so.12", "libcublas.so.12", "libcusolver.so.11"},
        "explicit library set",
    )
    for item in libraries.values():
        require(0 < integer(item["bytes"]) < 1 << 30, "library size")
        require(
            item["mode"] == "0o644"
            and len(item["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in item["sha256"]),
            "library byte identity",
        )
    locator = unpack("library-locator-stdout.utf8.xz")
    require(digest(locator) == provenance["locator"]["stdout_sha256"], "locator stdout")
    require(
        record["actual_internal_provider_work_peak_billing"] == "UNAVAILABLE"
        and record["transitive_dependency_byte_hashes"].startswith(
            "UNAVAILABLE_NOT_HASHED"
        ),
        "unknowns relabelled",
    )
    transport = provenance["actual_transport"]
    require(
        transport["returncode"] == 0
        and type(transport["scientific_requests"]) is int
        and transport["scientific_requests"] == 0
        and type(transport["retries"]) is int
        and transport["retries"] == 0,
        "transport outcome/call scope",
    )
    require(transport["controller_elapsed_seconds"] >= elapsed, "distinct clock scopes")
    prep = parse((ROOT / "preparation-history.json").read_bytes())
    require(
        prep["CPU_stage"]["errno"] == 122
        and prep["CPU_stage"]["staged_file_bytes"] == 0
        and prep["CPU_stage"]["provider_process_started"] is False
        and prep["CPU_stage"]["quota_resolved"] is False,
        "preparatory failure erased",
    )
    require(
        "[Errno 122] Disk quota exceeded"
        in prep["CPU_stage"]["exact_traceback_excerpt"],
        "missing original quota error",
    )
    return {
        "status": "PASS_RETAINED_STANDALONE_ENGINEERING_REPLAY_ONLY",
        "free_bytes": free,
        "cublas_signed_difference_bytes": blas_delta,
        "both_signed_difference_bytes": combined_delta,
        "both_signed_difference_MiB": combined_delta // MIB,
        "source_allowance_bytes": allowance,
        "source_allowance_MiB": allowance // MIB,
        "excess_bytes": max(combined_delta, 0) - allowance,
        "excess_MiB": (max(combined_delta, 0) - allowance) // MIB,
        "provider_events_success": len(events),
        "cleanup_success": len(cleanup),
        "same_device_before_after": True,
        "scientific_requests": 0,
        "probe_seconds": elapsed,
        "transport_controller_seconds": transport["controller_elapsed_seconds"],
        "clock_relationship": "nested scopes; never add or call endpoint timing/billing",
        "explicit_library_byte_identities": libraries,
        "unknown": [
            "historical CBS before/after/delta/allowance",
            "transitive dependency hashes",
            "causal per-provider allocations",
            "internal work/peak/billing",
            "global or hardware capacity requirement",
        ],
        "scientific_device_endpoint_acceptance": "NOT_GRANTED",
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
