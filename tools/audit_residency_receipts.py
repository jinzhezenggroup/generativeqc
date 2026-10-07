"""Audit declared CUDA transfer and synchronization receipts for issue #1629.

This consumer does not instrument CUDA. A DF component trace adapter exposes its
actual aggregate counters, with incomplete coverage rather than a zero-work pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "generativeqc.residency_receipt"
IDENTITY = (
    "source_commit",
    "native_source",
    "build_sha256",
    "backend",
    "device",
    "endpoint",
    "problem",
    "resource",
    "owner",
    "domain",
)
ROLES = frozenset(
    {"prepare", "replay", "iteration", "tile", "publication", "oracle", "compatibility"}
)
HOT_ROLES = frozenset({"replay", "iteration", "tile"})
COUNTERS = ("h2d_bytes", "d2h_bytes", "d2d_bytes", "syncs", "event_waits")
MAX_COUNT = (1 << 64) - 1
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _decode_json(value: str | bytes) -> Any:
    return json.loads(value, object_pairs_hook=_unique_object)


def _count(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError(f"{label} must be a uint64")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _identity(value: Any) -> dict[str, str]:
    data = _object(value, "identity")
    if set(data) != set(IDENTITY):
        raise ValueError("identity fields differ from the contract")
    if any(type(data[key]) is not str or not data[key] for key in IDENTITY):
        raise ValueError("identity fields must be nonempty strings")
    if not HEX40.fullmatch(data["source_commit"]):
        raise ValueError("source_commit must be a lowercase Git SHA")
    if not HEX64.fullmatch(data["native_source"]):
        raise ValueError("native_source must be a lowercase source SHA-256")
    if not HEX64.fullmatch(data["build_sha256"]):
        raise ValueError("build_sha256 must be a lowercase SHA-256")
    if data["backend"] != "cuda":
        raise ValueError("residency audit requires the CUDA backend")
    return data


def _regions(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = contract.get("regions")
    if type(rows) is not list or not rows:
        raise ValueError("contract requires declared regions")
    result = {}
    for item in rows:
        row = _object(item, "region")
        if set(row) != {"name", "role", "owner", "domain", "hot", "ratchet"}:
            raise ValueError("region fields differ from the contract")
        name = row["name"]
        if type(name) is not str or not name or name in result:
            raise ValueError("region names must be distinct nonempty strings")
        if row["role"] not in ROLES or type(row["hot"]) is not bool:
            raise ValueError("invalid region role or hot flag")
        if row["hot"] != (row["role"] in HOT_ROLES):
            raise ValueError("hot status must agree with the region role")
        if any(
            type(row[key]) is not str or not row[key] for key in ("owner", "domain")
        ):
            raise ValueError("region owner/domain must be explicit")
        ratchet = _object(row["ratchet"], "ratchet")
        if set(ratchet) != set(COUNTERS) | {"round_trips"}:
            raise ValueError("ratchet must cover every transfer and wait category")
        for key, value in ratchet.items():
            _count(value, f"ratchet.{key}")
            if not row["hot"] and value:
                raise ValueError("non-hot regions cannot grant a hot ratchet")
        result[name] = row
    return result


def _totals(
    events: list[dict[str, Any]], regions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    totals = {name: dict.fromkeys(COUNTERS, 0) for name in regions}
    by_role = {role: dict.fromkeys(COUNTERS, 0) for role in ROLES}
    downloads: dict[int, tuple[str, str, str, str]] = {}
    pairs: list[dict[str, Any]] = []
    for expected_seq, item in enumerate(events):
        event = _object(item, "event")
        if _count(event.get("seq"), "seq") != expected_seq:
            raise ValueError("event stream has a gap, duplicate, or reorder")
        region_name = event.get("region")
        if region_name not in regions:
            raise ValueError("unknown event region")
        region = regions[region_name]
        if any(event.get(key) != region[key] for key in ("role", "owner", "domain")):
            raise ValueError("event role/owner/domain differs from declared region")
        kind = event.get("kind")
        if kind == "transfer":
            if set(event) != {
                "seq",
                "region",
                "role",
                "owner",
                "domain",
                "kind",
                "direction",
                "bytes",
                "payload",
                "dependency_domain",
                "derived_from",
            }:
                raise ValueError("transfer event fields are incomplete or unknown")
            direction = event["direction"]
            if direction not in ("h2d", "d2h", "d2d"):
                raise ValueError("unknown transfer direction")
            amount = _count(event["bytes"], "bytes")
            if not amount:
                raise ValueError("zero-byte transfer is not an event")
            if any(
                type(event[key]) is not str or not event[key]
                for key in ("payload", "dependency_domain")
            ):
                raise ValueError("payload and dependency domain are required")
            source = event["derived_from"]
            if direction == "d2h":
                if source is not None:
                    raise ValueError("D2H cannot claim an upstream D2H")
                key = (
                    event["payload"],
                    event["dependency_domain"],
                    region["owner"],
                    region["domain"],
                )
                downloads[expected_seq] = key
            elif direction == "h2d" and source is not None:
                _count(source, "derived_from")
                key = (
                    event["payload"],
                    event["dependency_domain"],
                    region["owner"],
                    region["domain"],
                )
                if downloads.get(source) != key or source >= expected_seq:
                    raise ValueError(
                        "H2D round-trip link lacks the same prior payload/domain"
                    )
                pairs.append(
                    {
                        "d2h_seq": source,
                        "h2d_seq": expected_seq,
                        "payload": event["payload"],
                        "dependency_domain": event["dependency_domain"],
                    }
                )
            elif source is not None:
                raise ValueError("D2D cannot claim a host round-trip link")
            category = f"{direction}_bytes"
        elif kind == "sync":
            if set(event) != {
                "seq",
                "region",
                "role",
                "owner",
                "domain",
                "kind",
                "sync_type",
            }:
                raise ValueError("sync event fields are incomplete or unknown")
            if event["sync_type"] not in ("stream", "device", "event_wait"):
                raise ValueError("unknown sync type")
            category = "event_waits" if event["sync_type"] == "event_wait" else "syncs"
            amount = 1
        else:
            raise ValueError("unknown event kind")
        for bucket in (totals[region_name], by_role[region["role"]]):
            if bucket[category] > MAX_COUNT - amount:
                raise ValueError("event aggregate overflows uint64")
            bucket[category] += amount
    return {"by_region": totals, "by_role": by_role, "round_trips": pairs}


def audit(contract: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Return PASS, FAIL, or INCOMPLETE; malformed/uncovered evidence never passes."""
    try:
        contract = _object(contract, "contract")
        receipt = _object(receipt, "receipt")
        if contract.get("schema") != SCHEMA or contract.get("version") != 1:
            raise ValueError("unsupported contract schema/version")
        identity = _identity(contract.get("identity"))
        regions = _regions(contract)
        if receipt.get("schema") != SCHEMA or receipt.get("version") != 1:
            raise ValueError("unsupported receipt schema/version")
        if _identity(receipt.get("identity")) != identity:
            raise ValueError(
                "receipt identity differs from expected source/build/endpoint"
            )
        if receipt.get("completed") is not True or receipt.get("execution") != "stream":
            raise ValueError("receipt is unfinished or only graph construction")
        coverage = _object(receipt.get("coverage"), "coverage")
        if set(coverage) != {
            "regions",
            "transfers",
            "synchronizations",
            "payload_links",
        }:
            raise ValueError("coverage fields differ from the contract")
        if (
            type(coverage["regions"]) is not list
            or set(coverage["regions"]) != set(regions)
            or len(coverage["regions"]) != len(regions)
        ):
            raise ValueError("missing or duplicate declared region coverage")
        if any(
            coverage[key] is not True
            for key in ("transfers", "synchronizations", "payload_links")
        ):
            raise ValueError("partial transfer/sync/payload coverage")
        events = receipt.get("events")
        if type(events) is not list or _count(
            receipt.get("event_count"), "event_count"
        ) != len(events):
            raise ValueError("event count or event stream is incomplete")
        totals = _totals(events, regions)
        paired = totals["round_trips"]
        failures = []
        for name, region in regions.items():
            if not region["hot"]:
                continue
            actual = totals["by_region"][name]
            actual = {
                **actual,
                "round_trips": sum(
                    events[pair["d2h_seq"]]["region"] == name
                    or events[pair["h2d_seq"]]["region"] == name
                    for pair in paired
                ),
            }
            for category, limit in region["ratchet"].items():
                if actual[category] > limit:
                    failures.append(f"{name}.{category}: {actual[category]} > {limit}")
        return {
            "status": "FAIL" if failures else "PASS",
            "identity": identity,
            "totals": totals,
            "ratchet_failures": failures,
            "candidate_replay_syncs": sum(
                totals["by_region"][name]["syncs"]
                + totals["by_region"][name]["event_waits"]
                for name, row in regions.items()
                if row["role"] == "replay"
            ),
            "publication": totals["by_role"]["publication"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "INCOMPLETE", "reason": str(exc)}


def audit_df_trace(
    contract: dict[str, Any], path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read existing native JSONL counters without inventing per-transfer events."""
    try:
        contract = _object(contract, "contract")
        manifest = _object(manifest, "manifest")
        identity = _identity(contract.get("identity"))
        if contract.get("schema") != SCHEMA or contract.get("version") != 1:
            raise ValueError("unsupported contract schema/version")
        _regions(contract)
        selection = _object(contract.get("historical_trace"), "historical_trace")
        if set(selection) != {"retained", "sha256"}:
            raise ValueError("historical_trace fields differ from the contract")
        if type(selection["retained"]) is not str or not selection["retained"]:
            raise ValueError("historical_trace.retained must be a nonempty string")
        if type(selection["sha256"]) is not str or not HEX64.fullmatch(
            selection["sha256"]
        ):
            raise ValueError("historical_trace.sha256 must be a lowercase SHA-256")
        if manifest.get("schema") != "vibeqc.issue206.practical-auxiliary.retention.v1":
            raise ValueError("unexpected historical retention manifest schema")
        if (
            manifest.get("measured_commit") != identity["source_commit"]
            or manifest.get("native_source_identity") != identity["native_source"]
            or manifest.get("library_sha256") != identity["build_sha256"]
        ):
            raise ValueError(
                "historical trace source/build identity differs from contract"
            )
        relative = path.as_posix().split(
            "benchmarks/results/issue206-practical-auxiliary/", 1
        )[-1]
        if relative != selection["retained"]:
            raise ValueError("historical trace path differs from the contract")
        retained = next(
            row for row in manifest["records"] if row["retained"] == relative
        )
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != selection["sha256"]:
            raise ValueError("historical trace SHA-256 differs from the contract")
        if digest != retained["stored_sha256"] or digest != retained["original_sha256"]:
            raise ValueError("retained trace SHA-256 mismatch")
        totals = dict.fromkeys(COUNTERS, 0)
        diagnostic_events = 0
        records = 0
        for line in raw.splitlines():
            if not line:
                raise ValueError("empty trace record")
            row = _object(_decode_json(line), "DF trace record")
            if row.get("schema") != "vibeqc.df_trace" or row.get("version") != 1:
                raise ValueError("unexpected historical DF trace schema")
            if (
                row.get("valid") is not True
                or row.get("cuda_error") != 0
                or row.get("dropped_regions") != 0
                or row.get("dropped_tiles") != 0
            ):
                raise ValueError("invalid or truncated DF trace")
            if row.get("execution") != "stream":
                raise ValueError("capture is construction, not executed work")
            regions = row.get("regions")
            if (
                type(regions) is not list
                or not regions
                or row.get("profiler_event_count") != 2 * len(regions)
            ):
                raise ValueError("invalid diagnostic event pairing")
            diagnostic_count = _count(
                row["profiler_event_count"], "profiler_event_count"
            )
            if diagnostic_events > MAX_COUNT - diagnostic_count:
                raise ValueError("diagnostic event aggregate overflows uint64")
            diagnostic_events += diagnostic_count
            counters = _object(row.get("counters"), "counters")
            for source, target in (
                ("host_to_device_bytes", "h2d_bytes"),
                ("device_to_host_bytes", "d2h_bytes"),
                ("explicit_synchronizations", "syncs"),
                ("stream_synchronizations", "syncs"),
                ("raw_panel_event_synchronizations", "event_waits"),
            ):
                amount = _count(counters.get(source, 0), source)
                if totals[target] > MAX_COUNT - amount:
                    raise ValueError("DF counter aggregate overflows uint64")
                totals[target] += amount
            records += 1
        if not records:
            raise ValueError("empty DF trace")
        return {
            "status": "INCOMPLETE",
            "reason": "DF component trace aggregates selected owner counters; D2D, all transfers/waits, payload links, and hot-region coverage are unobserved",
            "historical": True,
            "trace_sha256": digest,
            "records": records,
            "observed_counters": totals,
            "diagnostic_profiler_events": diagnostic_events,
            "diagnostic_final_waits": records,
            "identity": identity,
        }
    except (
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        return {"status": "INCOMPLETE", "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt", type=Path)
    source.add_argument("--df-trace", type=Path)
    parser.add_argument("--trace-manifest", type=Path)
    args = parser.parse_args()
    if args.df_trace and not args.trace_manifest:
        parser.error("--df-trace requires --trace-manifest")
    try:
        contract = _decode_json(args.contract.read_text(encoding="utf-8"))
        if args.receipt:
            result = audit(
                contract, _decode_json(args.receipt.read_text(encoding="utf-8"))
            )
        else:
            result = audit_df_trace(
                contract,
                args.df_trace,
                _decode_json(args.trace_manifest.read_text(encoding="utf-8")),
            )
    except (OSError, ValueError) as exc:
        result = {"status": "INCOMPLETE", "reason": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
