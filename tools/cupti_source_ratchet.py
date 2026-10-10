"""Conservative review gate for observed explicit work, not a residency PASS.

Budgets come from independently reviewed workload profiles, never from the
capture being checked. Recompute source/API/activity joins from raw journals;
source loss cannot authorize a within-budget result. Implicit waits, scientific
dependencies and graph execution remain outside this deliberately named gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from tools.audit_replay_allocations import (
    InvalidReceipt,
    _fields,
    _integer,
    _require,
    _text,
)
from tools.audit_residency_receipts import HOT_ROLES, _decode_json
from tools.cupti_source_capture import ROLES, WORK_COUNTERS, summarize_sources

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = "generativeqc.observed-source-work-ratchet.v2"
LEGACY_SCHEMA = "generativeqc.observed-source-work-ratchet.v1"
SCOPE = "verified source copy/fence calls and observed CUPTI memcpy/blocking synchronization activity"
MAX_POLICY_BYTES = 1 << 20


def workload_digest(workload: dict[str, Any]) -> str:
    """Match the entire declared workload, including numerical acceptance gates."""
    _require(type(workload) is dict and bool(workload), "invalid ratchet workload")
    try:
        encoded = json.dumps(
            workload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise InvalidReceipt("invalid ratchet workload JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: Any, width: int, label: str) -> None:
    _require(
        type(value) is str and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is not None,
        f"invalid {label}",
    )


def _profile(profile: Any, schema: str = SCHEMA) -> None:
    """Reject ambiguous selectors, role exemptions and incomplete count limits."""
    _fields(
        profile,
        {
            "name",
            "workload_sha256",
            "baseline",
            "regions",
            "unannotated_activity_limit",
        },
        "ratchet profile",
    )
    _text(profile["name"], "profile name")
    _hex(profile["workload_sha256"], 64, "workload SHA-256")
    baseline = profile["baseline"]
    _fields(
        baseline,
        {"source_tree", "library_sha256", "collector_sha256", "slurm_job_id"},
        "ratchet baseline",
    )
    _hex(baseline["source_tree"], 40, "baseline source tree")
    for name in ("library_sha256", "collector_sha256"):
        _hex(baseline[name], 64, "baseline " + name)
    _text(baseline["slurm_job_id"], "baseline Slurm job")
    _require(
        _integer(profile["unannotated_activity_limit"], "unannotated activity limit")
        == 0,
        "unannotated work cannot receive a role/owner exemption",
    )
    regions = profile["regions"]
    _require(
        type(regions) is list and 1 <= len(regions) <= 64, "invalid ratchet regions"
    )
    names = set()
    for region in regions:
        _fields(region, {"name", "role", "owner_roles"}, "ratchet region")
        name = _text(region["name"], "region name")
        _require(name not in names, "duplicate ratchet region")
        names.add(name)
        _require(
            type(region["role"]) is str and region["role"] in HOT_ROLES,
            "ratchet region must be hot",
        )
        rows = region["owner_roles"]
        _require(type(rows) is list and len(rows) <= 128, "invalid owner/role limits")
        identities = set()
        for row in rows:
            _fields(row, {"owner", "role", "max"}, "owner/role limit")
            owner = _text(row["owner"], "ratchet owner")
            _require(
                type(row["role"]) is str and row["role"] in ROLES, "invalid source role"
            )
            identity = (owner, row["role"])
            _require(identity not in identities, "duplicate owner/role limit")
            identities.add(identity)
            counters = set(WORK_COUNTERS)
            if schema == LEGACY_SCHEMA:
                counters.remove("source_event_fences")
            _fields(row["max"], counters, "work limits")
            for counter, limit in row["max"].items():
                _integer(limit, counter)


def load_work_ratchet(path: Path, workload: dict[str, Any]) -> dict[str, Any]:
    """Select one trusted profile; pin its whole file as well as selected limits."""
    with path.open("rb") as handle:
        contents = handle.read(MAX_POLICY_BYTES + 1)
    _require(len(contents) <= MAX_POLICY_BYTES, "ratchet policy exceeds byte bound")
    try:
        policy = _decode_json(contents)
    except (ValueError, UnicodeDecodeError) as error:
        raise InvalidReceipt(f"invalid ratchet policy JSON: {error}") from error
    _fields(policy, {"schema", "scope", "profiles"}, "ratchet policy")
    _require(
        policy["schema"] in (SCHEMA, LEGACY_SCHEMA) and policy["scope"] == SCOPE,
        "invalid ratchet schema/scope",
    )
    profiles = policy["profiles"]
    _require(
        type(profiles) is list and 1 <= len(profiles) <= 16, "invalid ratchet profiles"
    )
    names, workloads = set(), set()
    for profile in profiles:
        _profile(profile, policy["schema"])
        _require(
            profile["name"] not in names
            and profile["workload_sha256"] not in workloads,
            "duplicate ratchet selector",
        )
        names.add(profile["name"])
        workloads.add(profile["workload_sha256"])
    selected = [
        profile
        for profile in profiles
        if profile["workload_sha256"] == workload_digest(workload)
    ]
    _require(len(selected) == 1, "no ratchet profile matches the complete workload")
    return {
        "schema": policy["schema"],
        "sha256": hashlib.sha256(contents).hexdigest(),
        "profile": selected[0],
    }


def check_work_ratchet(
    selected: dict[str, Any],
    raw_source: dict[str, Any],
    raw_activity: dict[str, Any],
    regions: dict[int, dict[str, str]],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Require intact raw joins and declared hot phases before checking maxima.

    Unlisted owner/role pairs have zero allowance, even preparation/publication
    roles inside a selected public replay. Unowned blocking work requires review.
    Nonblocking queries stay visible separately and are never counted as fences.
    A within-budget result is only an observed-work gate, not a round-trip proof.
    """
    _fields(selected, {"schema", "sha256", "profile"}, "selected ratchet")
    _require(
        selected["schema"] in (SCHEMA, LEGACY_SCHEMA), "invalid selected ratchet schema"
    )
    _hex(selected["sha256"], 64, "policy SHA-256")
    profile = selected["profile"]
    _profile(profile, selected["schema"])
    _require(
        profile["workload_sha256"] == workload_digest(workload),
        "ratchet workload mismatch",
    )
    source = summarize_sources(raw_source, raw_activity, regions)
    names = [region["name"] for region in regions.values()]
    _require(len(names) == len(set(names)), "ambiguous public region names")
    phases = {region["name"]: identity for identity, region in regions.items()}
    limits = {}
    selected_phases = set()
    for region in profile["regions"]:
        _require(region["name"] in phases, "missing declared ratchet region")
        phase = phases[region["name"]]
        _require(
            regions[phase]["role"] == region["role"], "ratchet public role mismatch"
        )
        selected_phases.add(phase)
        for row in region["owner_roles"]:
            # A frozen v1 policy never reviewed event fences. Preserve its bytes
            # and explicit allowances, but require review for every new event API.
            limits[(phase, row["owner"], row["role"])] = {
                **dict.fromkeys(WORK_COUNTERS, 0),
                **row["max"],
            }
    issues = list(source["issues"])
    phase_correlations = {
        row["correlation_id"]: row["external_id"]
        for row in raw_activity["records"]
        if row["kind"] == 39 and row["subtype"] == 3
    }
    observed_phases = {
        phase_correlations.get(row["correlation_id"])
        for row in raw_activity["records"]
        if row["kind"] in (4, 5, 48)
        and row["return_value"] == 0
        and row["start"] > 0
        and row["end"] >= row["start"]
    }
    for phase in sorted(selected_phases - observed_phases):
        issues.append(
            "no observed API provenance for selected region " + regions[phase]["name"]
        )
    counts = []
    violations = []
    for row in source["by_phase_owner_role"]:
        phase = row["region_id"]
        if phase not in selected_phases:
            continue
        maximum = limits.get(
            (phase, row["owner"], row["role"]), dict.fromkeys(WORK_COUNTERS, 0)
        )
        counts.append({"region": regions[phase]["name"], **row, "max": dict(maximum)})
        for counter, observed in row["counters"].items():
            if observed > maximum[counter]:
                violations.append(
                    {
                        "region": regions[phase]["name"],
                        "owner": row["owner"],
                        "role": row["role"],
                        "counter": counter,
                        "observed": observed,
                        "maximum": maximum[counter],
                    }
                )
    unannotated = [
        event
        for event in source["unannotated_production_activity"]
        if event["region_id"] in selected_phases
    ]
    queries = [event for event in unannotated if event["kind"] == "query"]
    blocking = [event for event in unannotated if event["kind"] != "query"]
    if blocking:
        violations.append(
            {
                "counter": "unannotated_selected_activity",
                "observed": len(blocking),
                "maximum": 0,
            }
        )
    return {
        "status": "INCOMPLETE"
        if issues
        else "VIOLATION"
        if violations
        else "WITHIN_OBSERVED_RATCHET",
        "scope": SCOPE,
        "residency_status": "INCOMPLETE",
        "profile": profile["name"],
        "policy_sha256": selected["sha256"],
        "issues": issues,
        "violations": violations,
        "counts": counts,
        "unannotated_selected_blocking_activity": blocking,
        "unannotated_selected_nonblocking_queries": queries,
        "payload_dependency_links_complete": False,
        "implicit_wait_coverage_complete": False,
        "graph_wait_coverage_complete": False,
        "zero_round_trip_assertion": None,
    }
