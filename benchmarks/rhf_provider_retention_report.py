"""Fail-closed report for diagnostic-only RHF provider-retention events."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

SCHEMA = "generativeqc.rhf-provider-probe.v1"
REPORT_SCHEMA = "generativeqc.rhf-provider-report.v1"
SUPPORTED_PROVIDERS = ("cublas", "cusolver")


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_identity(policy: dict, identity: dict) -> None:
    source = identity.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("invalid source identity")
    if policy.get("source_commit") != source:
        raise ValueError("source identity mismatch")
    for name in (
        "library_sha256",
        "endpoint_sha256",
        "shim_sha256",
        "event_log_sha256",
    ):
        value = identity.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid identity: {name}")
    if not isinstance(identity.get("device_uuid"), str) or not identity["device_uuid"]:
        raise ValueError("device identity is required")
    if not isinstance(identity.get("job_id"), str) or not identity["job_id"]:
        raise ValueError("job identity is required")


def _providers(policy: dict) -> dict[str, dict]:
    raw = policy.get("providers")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("source policy requires provider allowances")
    result = {}
    for name, record in raw.items():
        if name not in SUPPORTED_PROVIDERS or not isinstance(record, dict):
            raise ValueError("unsupported source provider policy")
        allowance = _integer(record.get("allowance_bytes"), f"{name} allowance")
        if not allowance:
            raise ValueError(f"{name} allowance must be positive")
        required = record.get("required_events")
        if (
            not isinstance(required, list)
            or not required
            or len(set(required)) != len(required)
            or any(
                not isinstance(event, str) or not event.startswith(f"{name}_")
                for event in required
            )
        ):
            raise ValueError(f"{name} required events are invalid")
        result[name] = {"allowance_bytes": allowance, "required_events": required}
    return result


def _ordered(records: list[dict]) -> list[dict]:
    if not isinstance(records, list) or not records:
        raise ValueError("event records are required")
    seen = set()
    result = []
    for record in records:
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError("unsupported event schema")
        sequence = _integer(record.get("sequence"), "event sequence")
        if sequence in seen:
            raise ValueError("duplicate event sequence")
        seen.add(sequence)
        if not isinstance(record.get("event"), str) or not record["event"]:
            raise ValueError("event name is required")
        if record.get("phase") not in ("begin", "end"):
            raise ValueError("event phase must be begin or end")
        _integer(record.get("pid"), "event pid")
        _integer(record.get("tid"), "event tid")
        _integer(record.get("status"), "event status")
        if record["event"] == "cuda_mem_get_info":
            if record.get("origin") not in ("application", "provider_internal"):
                raise ValueError("memory query origin is invalid")
            if type(record.get("observed")) is not bool:
                raise ValueError("memory query observed flag must be Boolean")
        result.append(record)
    result.sort(key=lambda item: item["sequence"])
    return result


def _mem_query(record: dict) -> bool:
    return (
        record["event"] == "cuda_mem_get_info"
        and record.get("phase") == "end"
        and record["status"] == 0
        and record.get("observed") is True
        and record.get("origin") == "application"
    )


def summarize(records: list[dict], policy: dict, identity: dict) -> dict:
    """Reconcile one exact provider interval without inferring missing flags."""
    _validate_identity(policy, identity)
    provider_policy = _providers(policy)
    allowances = {
        name: record["allowance_bytes"] for name, record in provider_policy.items()
    }
    ordered = _ordered(records)
    issues = []
    sequences = [record["sequence"] for record in ordered]
    if sequences != list(range(sequences[-1] + 1)):
        issues.append("event sequence contains gaps")
    terminal = [record for record in ordered if record["event"] == "terminal"]
    ordinary = [record for record in ordered if record["event"] != "terminal"]
    if not terminal:
        issues.append("missing terminal record")
    elif len(terminal) != 1:
        issues.append(f"terminal record count is {len(terminal)}, expected 1")
    else:
        final = terminal[0]
        if final is not ordered[-1]:
            issues.append("terminal record is not last")
        if _integer(final.get("events_written"), "terminal events_written") != len(
            ordinary
        ):
            issues.append("terminal event count does not match records")
        dropped = _integer(final.get("dropped_events"), "terminal dropped_events")
        resolution_errors = _integer(
            final.get("resolution_errors"), "terminal resolution_errors"
        )
        write_errors = _integer(final.get("write_errors", 0), "terminal write_errors")
        if dropped:
            issues.append(f"capture dropped {dropped} event(s)")
        if resolution_errors:
            issues.append(f"interposer resolution failed {resolution_errors} time(s)")
        if write_errors:
            issues.append(f"interposer write failed {write_errors} time(s)")

    candidates = []
    by_thread: dict[tuple[int, int], list[dict]] = {}
    for record in ordinary:
        if _mem_query(record):
            _integer(record.get("free_bytes"), "observed free bytes")
            _integer(record.get("total_bytes"), "observed total bytes")
            by_thread.setdefault((record["pid"], record["tid"]), []).append(record)
    for thread_queries in by_thread.values():
        for before, after in pairwise(thread_queries):
            between = [
                record
                for record in ordinary
                if before["sequence"] < record["sequence"] < after["sequence"]
                and record["pid"] == before["pid"]
                and record["tid"] == before["tid"]
            ]
            if any(
                record["event"] in ("cublas_create", "cusolver_create")
                for record in between
            ):
                candidates.append((before, after, between))
    if not candidates:
        issues.append("no complete provider measurement pair")
    elif len(candidates) != 1:
        issues.append(f"ambiguous provider measurement pairs: {len(candidates)}")

    flags = {name: None for name in SUPPORTED_PROVIDERS}
    provider_errors = []
    unexpected_providers = []
    candidate = candidates[0] if len(candidates) == 1 else None
    if candidate:
        _, _, between = candidate
        for record in between:
            if (
                record["event"].startswith(("cublas_", "cusolver_"))
                and record["status"]
            ):
                provider_errors.append(
                    {"event": record["event"], "status": record["status"]}
                )
        if provider_errors:
            issues.append("provider call returned an error")
        for provider in SUPPORTED_PROVIDERS:
            successful_creates = [
                record
                for record in between
                if record["event"] == f"{provider}_create"
                and record["phase"] == "end"
                and record["status"] == 0
            ]
            if successful_creates and provider not in provider_policy:
                flags[provider] = True
                unexpected_providers.append(provider)
                issues.append(
                    f"observed provider absent from source policy: {provider}"
                )
        for provider, provider_record in provider_policy.items():
            creates = [
                record
                for record in between
                if record["event"] == f"{provider}_create" and record["phase"] == "end"
            ]
            if len(creates) != 1:
                issue = (
                    f"missing required provider event: {provider}"
                    if not creates
                    else f"duplicate provider event: {provider}"
                )
                issues.append(issue)
                continue
            complete = True
            end_sequences = []
            pairs = []
            for required_event in provider_record["required_events"]:
                begins = [
                    record
                    for record in between
                    if record["event"] == required_event and record["phase"] == "begin"
                ]
                observed = [
                    record
                    for record in between
                    if record["event"] == required_event and record["phase"] == "end"
                ]
                if len(observed) != 1:
                    issues.append(
                        f"missing required provider event: {required_event}"
                        if not observed
                        else f"duplicate provider event: {required_event}"
                    )
                    complete = False
                elif observed[0]["status"] != 0:
                    complete = False
                else:
                    end_sequences.append(observed[0]["sequence"])
                if len(begins) != 1:
                    issues.append(
                        f"missing provider begin: {required_event}"
                        if not begins
                        else f"duplicate provider begin: {required_event}"
                    )
                    complete = False
                elif begins[0]["status"] != 0:
                    complete = False
                elif len(observed) == 1:
                    if begins[0]["sequence"] >= observed[0]["sequence"]:
                        issues.append(
                            f"provider event pair is reversed: {required_event}"
                        )
                        complete = False
                    else:
                        pairs.append((begins[0]["sequence"], observed[0]["sequence"]))
            if complete and end_sequences != sorted(end_sequences):
                issues.append(f"provider event order mismatch: {provider}")
                complete = False
            if complete and any(
                previous[1] >= following[0] for previous, following in pairwise(pairs)
            ):
                issues.append(f"provider event pairs overlap: {provider}")
                complete = False
            if complete:
                flags[provider] = True

    measurement = None
    if not issues and candidate:
        before, after, between = candidate
        if before["total_bytes"] != after["total_bytes"]:
            issues.append("device total bytes changed across measurement")
        else:
            delta = max(before["free_bytes"] - after["free_bytes"], 0)
            effective = sum(allowances.values())
            measurement = {
                "pid": before["pid"],
                "tid": before["tid"],
                "before_free_bytes": before["free_bytes"],
                "after_free_bytes": after["free_bytes"],
                "total_bytes": before["total_bytes"],
                "retained_delta_bytes": delta,
                "effective_allowance_bytes": effective,
                "exceeds_allowance": delta > effective,
                "observed_provider_flags": flags,
                "provider_internal_mem_queries": sum(
                    record["event"] == "cuda_mem_get_info"
                    and record.get("origin") == "provider_internal"
                    for record in between
                ),
            }
    if issues:
        measurement = None
    return {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE" if not issues else "INCOMPLETE",
        "identity": identity,
        "source_policy": policy,
        "scope": (
            "one provider-retained free-memory delta; not an allocation ownership census, "
            "scientific qualification, or performance timing"
        ),
        "issues": issues,
        "observed_provider_flags": flags,
        "unexpected_observed_providers": unexpected_providers,
        "provider_errors": provider_errors,
        "measurement": measurement,
    }


def load_events(path: Path) -> list[dict]:
    """Load complete JSONL rows; blank, invalid and non-object rows fail closed."""
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank event row at line {number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid event JSON at line {number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"event row {number} is not an object")  # noqa: TRY004
        result.append(record)
    if not result:
        raise ValueError("event log is empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity must be an object")  # noqa: TRY004
    event_sha = hashlib.sha256(args.events.read_bytes()).hexdigest()
    if identity.get("event_log_sha256") != event_sha:
        raise ValueError("event log identity mismatch")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("source policy must be an object")  # noqa: TRY004
    report = summarize(load_events(args.events), policy, identity)
    report["inputs"] = {
        "events": {"sha256": event_sha, "size": args.events.stat().st_size},
        "policy": {
            "sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            "size": args.policy.stat().st_size,
        },
        "identity": {
            "sha256": hashlib.sha256(args.identity.read_bytes()).hexdigest(),
            "size": args.identity.stat().st_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
