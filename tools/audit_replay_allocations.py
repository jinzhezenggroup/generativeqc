"""Verify phase-separated allocation journals against a caller-pinned contract.

This consumer installs no allocator or runtime hook. PASS applies only to the
named observed ownership domains, not process-wide heap/CUDA activity or science.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "generativeqc.replay-allocations.v1"
SCHEMA_V2 = "generativeqc.replay-allocations.v2"
PHASES = {
    "setup",
    "geometry_rebuild",
    "endpoint",
    "replay",
    "iteration",
    "tile",
    "publication",
}
HOT_PHASES = {"replay", "iteration", "tile"}
IDENTITY_FIELDS = {
    "source_commit": 40,
    "source_tree": 40,
    "library_sha256": 64,
    "artifact_sha256": 64,
    "workload_sha256": 64,
    "toolchain": None,
    "device": None,
    "endpoint": None,
}
MAX_COUNT = (1 << 64) - 1


class InvalidReceipt(ValueError):
    """The supplied evidence cannot support the requested assertion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidReceipt(message)


def _fields(value: Any, keys: set[str], label: str) -> None:
    _require(
        type(value) is dict and set(value) == keys, f"{label}: missing/unknown fields"
    )


def _text(value: Any, label: str) -> str:
    _require(
        type(value) is str and bool(value.strip()), f"{label}: expected nonempty string"
    )
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        type(value) is int and 0 <= value <= MAX_COUNT, f"{label}: expected uint64"
    )
    return value


def _identity(value: Any) -> None:
    _fields(value, set(IDENTITY_FIELDS), "identity")
    for key, length in IDENTITY_FIELDS.items():
        text = _text(value[key], key)
        if length:
            _require(
                bool(re.fullmatch(f"[0-9a-f]{{{length}}}", text)),
                f"{key}: invalid digest",
            )


def _domain(value: Any) -> tuple[str, str, str]:
    _fields(value, {"owner", "space", "counter"}, "domain")
    owner = _text(value["owner"], "owner")
    counter = _text(value["counter"], "counter")
    space = _text(value["space"], "space")
    _require(
        space == "host" or bool(re.fullmatch(r"device:(0|[1-9][0-9]*)", space)),
        "invalid space",
    )
    return owner, space, counter


def _schedule(
    value: Any,
    *,
    schema: str = SCHEMA,
    domains: list[tuple[str, str, str]] | None = None,
) -> None:
    _require(type(value) is list and bool(value), "windows: expected nonempty list")
    ids = set()
    asserted = False
    for window in value:
        assertion = (
            "zero_new_allocations"
            if schema == SCHEMA
            else "zero_new_allocation_domains"
        )
        _fields(window, {"id", "phase", assertion}, "window contract")
        name = _text(window["id"], "window id")
        _require(name not in ids, "duplicate window id")
        ids.add(name)
        phase = _text(window["phase"], "phase")
        _require(phase in PHASES, "unknown phase")
        if schema == SCHEMA:
            _require(type(window[assertion]) is bool, "assertion must be bool")
            zero = window[assertion]
        else:
            _require(type(window[assertion]) is list, "zero domains must be a list")
            selected = [_domain(domain) for domain in window[assertion]]
            _require(
                len(set(selected)) == len(selected), "duplicate zero-allocation domain"
            )
            _require(
                all(domain in (domains or []) for domain in selected),
                "unadmitted zero-allocation domain",
            )
            zero = bool(selected)
        if zero:
            _require(phase in HOT_PHASES, "zero assertion requires hot phase")
            asserted = True
    _require(
        asserted, "contract requires an explicit hot zero-new-allocation assertion"
    )


def _live_map(value: Any) -> dict[str, int]:
    _require(type(value) is dict, "initial_live: expected allocation-id map")
    for key, size in value.items():
        _text(key, "allocation id")
        _integer(size, "live requested bytes")
    _integer(sum(value.values()), "live bytes total")
    return dict(value)


def _observation(
    observed: Any, domain: tuple[str, str, str], zero: bool
) -> tuple[dict[str, int], dict[str, int], int]:
    _fields(
        observed,
        {
            "domain",
            "coverage",
            "initial_live",
            "event_begin",
            "event_end",
            "events",
            "metrics",
        },
        "observation",
    )
    _require(_domain(observed["domain"]) == domain, "wrong observation domain/order")
    coverage = observed["coverage"]
    _fields(
        coverage,
        {
            "complete",
            "started_before_window",
            "ended_after_window",
            "synchronized",
            "dropped_events",
        },
        "coverage",
    )
    for key in (
        "complete",
        "started_before_window",
        "ended_after_window",
        "synchronized",
    ):
        _require(coverage[key] is True, f"unobserved/incomplete window: {key}")
    _require(
        _integer(coverage["dropped_events"], "dropped_events") == 0, "dropped events"
    )
    metrics = observed["metrics"]
    _fields(
        metrics,
        {"allocation_count", "requested_bytes", "peak_live_bytes", "live_bytes"},
        "metrics",
    )
    for key, value in metrics.items():
        _integer(value, key)
    live = _live_map(observed["initial_live"])
    live_bytes = peak = sum(live.values())
    count = requested = 0
    begin = _integer(observed["event_begin"], "event_begin")
    end = _integer(observed["event_end"], "event_end")
    events = observed["events"]
    _require(
        type(events) is list and end >= begin and len(events) == end - begin,
        "lost event range",
    )
    for sequence, event in enumerate(events, begin):
        _fields(
            event, {"sequence", "kind", "allocation_id", "requested_bytes"}, "event"
        )
        _require(
            _integer(event["sequence"], "sequence") == sequence, "lost/reordered event"
        )
        name = _text(event["allocation_id"], "allocation_id")
        size = _integer(event["requested_bytes"], "event requested_bytes")
        kind = event["kind"]
        if kind == "allocate":
            _require(name not in live, "allocation id already live")
            live[name] = size
            count += 1
            requested += size
            live_bytes += size
            peak = max(peak, live_bytes)
        elif kind == "release":
            _require(
                name in live and live[name] == size, "release ownership/size mismatch"
            )
            del live[name]
            live_bytes -= size
        else:
            # A resize site is not an allocation event without capacity evidence.
            raise InvalidReceipt(
                "unknown event kind; resize/capacity uncertainty cannot prove allocation"
            )
    computed = {
        "allocation_count": count,
        "requested_bytes": requested,
        "peak_live_bytes": peak,
        "live_bytes": live_bytes,
    }
    _require(
        metrics == computed,
        "count/requested-bytes/peak/live metrics disagree with journal",
    )
    _require(not zero or count == requested == 0, "hot replay made new allocations")
    return computed, live, end


def verify_receipt(receipt: Any, expected: Any) -> dict[str, Any]:
    """Fail closed; expected identity/schedule must be pinned outside the receipt.

    Event cursors and live owners carry across all declared windows. Producers
    must observe the entire declared sequence, including setup and publication;
    gaps must be explicit windows, never silently dropped from hot measurements.
    """
    try:
        _fields(expected, {"schema", "identity", "domains", "windows"}, "contract")
        _require(
            expected["schema"] in (SCHEMA, SCHEMA_V2), "unsupported contract schema"
        )
        _identity(expected["identity"])
        domains = expected["domains"]
        _require(
            type(domains) is list and bool(domains),
            "expected ownership domains missing",
        )
        keys = [_domain(value) for value in domains]
        _require(
            len({(owner, space) for owner, space, _ in keys}) == len(keys),
            "duplicate ownership domain",
        )
        _schedule(expected["windows"], schema=expected["schema"], domains=keys)
        _fields(receipt, {"schema", "identity", "execution", "windows"}, "receipt")
        _require(receipt["schema"] == expected["schema"], "unsupported receipt schema")
        _identity(receipt["identity"])
        _require(
            receipt["identity"] == expected["identity"],
            "source/build/device/endpoint/workload identity mismatch",
        )
        execution = receipt["execution"]
        _fields(execution, {"kind", "completed", "source_matched"}, "execution")
        _require(
            execution["kind"] == "runtime",
            "static/synthetic evidence is not a runtime receipt",
        )
        _require(
            execution["completed"] is True and execution["source_matched"] is True,
            "endpoint incomplete/not source matched",
        )
        windows = receipt["windows"]
        _require(
            type(windows) is list and len(windows) == len(expected["windows"]),
            "missing/extra phase windows",
        )
        previous: dict[tuple[str, str, str], tuple[dict[str, int], int]] = {}
        output = []
        for window, specification in zip(windows, expected["windows"], strict=True):
            _fields(window, {"id", "phase", "observations"}, "runtime window")
            _require(
                window["id"] == specification["id"]
                and window["phase"] == specification["phase"],
                "wrong phase/window order",
            )
            observations = window["observations"]
            _require(
                type(observations) is list and len(observations) == len(keys),
                "missing ownership observation",
            )
            rows = []
            for observed, domain in zip(observations, keys, strict=True):
                zero = (
                    specification["zero_new_allocations"]
                    if expected["schema"] == SCHEMA
                    else domain
                    in {
                        _domain(value)
                        for value in specification["zero_new_allocation_domains"]
                    }
                )
                metrics, live, end = _observation(observed, domain, zero)
                if domain in previous:
                    old_live, old_end = previous[domain]
                    _require(
                        observed["initial_live"] == old_live
                        and observed["event_begin"] == old_end,
                        "unobserved inter-window events/ownership change",
                    )
                previous[domain] = live, end
                rows.append({"domain": observed["domain"], "metrics": metrics})
            output.append(
                {"id": window["id"], "phase": window["phase"], "observations": rows}
            )
        return {
            "status": "PASS",
            "scope": "declared ownership domains only",
            "windows": output,
            "errors": [],
        }
    except InvalidReceipt as error:
        return {"status": "FAIL", "windows": [], "errors": [str(error)]}


def assess_native_device_ledger(
    observation: Any, *, version: int = 1
) -> dict[str, Any]:
    """Consume NativeDeviceLedger.to_dict without inventing unavailable evidence.

    V1 reports successful allocation count and owned live/peak capacity. It has
    no cumulative requested-byte counter. V2 supplies it, but neither version
    has an event journal or full host coverage. Even allocations == 0 cannot
    upgrade either version to the strict receipt contract.
    """
    try:
        _require(
            type(version) is int and version in (1, 2), "unsupported ledger version"
        )
        _fields(
            observation,
            {
                "device",
                "limit_bytes",
                "live_bytes",
                "peak_bytes",
                "allocations",
                "rejected_allocations",
                "owner",
                "scope",
            }
            | ({"requested_bytes"} if version == 2 else set()),
            f"NativeDeviceLedger v{version}",
        )
        for key in (
            "device",
            "limit_bytes",
            "live_bytes",
            "peak_bytes",
            "allocations",
            "rejected_allocations",
        ):
            _integer(observation[key], key)
        _text(observation["owner"], "owner")
        _text(observation["scope"], "scope")
        requested = None
        if version == 2:
            requested = _integer(observation["requested_bytes"], "requested_bytes")
            _require(
                observation["allocations"] != 0 or requested == 0,
                "requested bytes without successful allocations",
            )
        _require(
            observation["live_bytes"]
            <= observation["peak_bytes"]
            <= observation["limit_bytes"],
            "invalid ledger capacities",
        )
        return {
            "status": "INCOMPLETE",
            "owner": observation["owner"],
            "space": f"device:{observation['device']}",
            "allocation_count": observation["allocations"],
            "requested_bytes": requested,
            "peak_live_bytes": observation["peak_bytes"],
            "live_bytes": observation["live_bytes"],
            "rejected_allocations": observation["rejected_allocations"],
            "missing": (["requested-byte counter"] if version == 1 else [])
            + [
                "event journal",
                "phase/source/build identity",
                "complete host observation",
            ],
        }
    except InvalidReceipt as error:
        return {"status": "FAIL", "errors": [str(error)]}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--expected",
        type=Path,
        help="independently pinned identity, domains and phase schedule",
    )
    modes.add_argument(
        "--native-ledger-v1",
        action="store_true",
        help="assess existing incomplete NativeDeviceLedger export",
    )
    modes.add_argument(
        "--native-ledger-v2",
        action="store_true",
        help="assess owned-device metrics including successful requested bytes",
    )
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(
            args.receipt.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
        if args.native_ledger_v1 or args.native_ledger_v2:
            result = assess_native_device_ledger(
                receipt, version=2 if args.native_ledger_v2 else 1
            )
        else:
            expected = json.loads(
                args.expected.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
            )
            result = verify_receipt(receipt, expected)
    except (OSError, ValueError) as error:
        result = {"status": "FAIL", "errors": [str(error)]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
