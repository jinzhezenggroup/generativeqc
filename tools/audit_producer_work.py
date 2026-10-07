"""Strict semantic producer-work receipts and existing DF schedule/trace adapters."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "generativeqc.producer-work.v1"
UINT64_MAX = (1 << 64) - 1
IDENTITY = {
    "scientific_problem",
    "resource_regime",
    "producer_domain",
    "dependency_identity",
    "reuse_owner",
    "invalidation_owner",
    "source_sha256",
    "build_sha256",
    "execution_identity",
}
WORK = {
    "logical_elements",
    "executed_elements",
    "producer_callbacks",
    "outer_consumer_multiplicity",
    "memory_budget_bytes",
    "peak_bytes",
}
EVIDENCE = {
    "kind",
    "phase",
    "coverage_complete",
    "execution_complete",
    "source_matched",
    "reusable_dependency_proven",
    "reuse_proof_sha256",
}
DOMAIN = IDENTITY - {"source_sha256", "build_sha256", "execution_identity"}
SCHEDULE_SOURCE = Path("python/generativeqc_compiler/method/df_exchange_schedule.py")


class ReceiptError(ValueError):
    pass


def _invalid_constant(value: str) -> None:
    raise ReceiptError(f"nonstandard JSON number: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ReceiptError("receipt must be an object")
    return result


def read_trace(path: Path, operation_id: int) -> dict[str, Any]:
    _uint(operation_id, "trace_id")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReceiptError(f"cannot read {path}: {exc}") from exc
    selected = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(
                line, object_pairs_hook=_object, parse_constant=_invalid_constant
            )
        except json.JSONDecodeError as exc:
            raise ReceiptError(f"invalid trace JSONL line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReceiptError(f"trace line {number} must be an object")
        _uint(row.get("id"), f"trace line {number} id")
        if row["id"] == operation_id:
            selected.append(row)
    if len(selected) != 1:
        raise ReceiptError(f"expected exactly one trace operation id {operation_id}")
    return selected[0]


def _fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReceiptError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _uint(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or not (int(positive) <= value <= UINT64_MAX):
        raise ReceiptError(f"{label} must be a bounded unsigned integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReceiptError(f"{label} must be nonempty text")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ReceiptError(f"{label} must be boolean")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def validate(receipt: Any) -> dict[str, Any]:
    row = _fields(receipt, {"schema", "identity", "work", "evidence"}, "receipt")
    if row["schema"] != SCHEMA:
        raise ReceiptError("unknown receipt schema")
    identity = _fields(row["identity"], IDENTITY, "identity")
    for key, value in identity.items():
        (_sha if key.endswith("sha256") else _text)(value, key)
    work = _fields(row["work"], WORK, "work")
    for key in WORK - {"peak_bytes"}:
        _uint(
            work[key],
            key,
            positive=key in {"logical_elements", "outer_consumer_multiplicity"},
        )
    if work["peak_bytes"] is not None:
        _uint(work["peak_bytes"], "peak_bytes")
    evidence = _fields(row["evidence"], EVIDENCE, "evidence")
    for key in EVIDENCE - {"kind", "phase", "reuse_proof_sha256"}:
        _bool(evidence[key], key)
    proof = evidence["reuse_proof_sha256"]
    if evidence["reusable_dependency_proven"]:
        _sha(proof, "reuse_proof_sha256")
    elif proof is not None:
        raise ReceiptError("reuse proof digest requires a proven dependency")
    if _text(evidence["kind"], "kind") not in {"schedule_census", "runtime_trace"}:
        raise ReceiptError("unknown evidence kind")
    if _text(evidence["phase"], "phase") not in {
        "static",
        "stream",
        "graph_capture",
        "submitted",
    }:
        raise ReceiptError("unknown evidence phase")
    if evidence["kind"] == "schedule_census":
        if evidence["phase"] != "static" or evidence["execution_complete"]:
            raise ReceiptError("static schedule cannot claim completed execution")
    elif evidence["phase"] == "static":
        raise ReceiptError("runtime trace cannot be static")
    elif evidence["execution_complete"]:
        raise ReceiptError("submission trace cannot claim completed execution")
    if (
        work["executed_elements"] < work["logical_elements"]
        and evidence["coverage_complete"]
    ):
        raise ReceiptError("complete coverage cannot omit logical elements")
    if evidence["coverage_complete"] and work["producer_callbacks"] == 0:
        raise ReceiptError("complete coverage requires producer callbacks")
    return row


def source_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReceiptError(f"cannot bind source {path}: {exc}") from exc


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_source: Path,
    candidate_source: Path,
) -> dict[str, Any]:
    left, right = validate(baseline), validate(candidate)
    for receipt, source in ((left, baseline_source), (right, candidate_source)):
        if source_digest(source) != receipt["identity"]["source_sha256"]:
            raise ReceiptError(f"stale source identity: {source}")
    if {k: left["identity"][k] for k in DOMAIN} != {
        k: right["identity"][k] for k in DOMAIN
    }:
        raise ReceiptError("scientific/problem/resource/producer/owner domains differ")
    if left["evidence"]["kind"] != right["evidence"]["kind"]:
        raise ReceiptError("evidence kinds differ")
    if left["work"]["memory_budget_bytes"] != right["work"]["memory_budget_bytes"]:
        raise ReceiptError("memory budgets differ")
    if not all(r["evidence"]["source_matched"] for r in (left, right)):
        return {
            "status": "INCOMPLETE",
            "reason": "source/build execution match not established",
        }
    if not all(r["evidence"]["coverage_complete"] for r in (left, right)):
        return {"status": "INCOMPLETE", "reason": "partial producer coverage"}
    if left["evidence"]["kind"] == "runtime_trace":
        return {
            "status": "INCOMPLETE",
            "reason": "runtime execution not proven complete",
        }
    old, new = left["work"], right["work"]
    if old["logical_elements"] != new["logical_elements"]:
        raise ReceiptError("logical domain size differs")
    ratio = Fraction(new["executed_elements"], old["executed_elements"])
    callback_ratio = Fraction(new["producer_callbacks"], old["producer_callbacks"])
    status = "FAIL" if ratio > 1 or callback_ratio > 1 else "PASS"
    return {
        "status": status,
        "evidence": left["evidence"]["kind"],
        "runtime_acceptance": "INCOMPLETE",
        "executed_ratio": f"{ratio.numerator}/{ratio.denominator}",
        "callback_ratio": f"{callback_ratio.numerator}/{callback_ratio.denominator}",
        "baseline_amplification": str(
            Fraction(old["executed_elements"], old["logical_elements"])
        ),
        "candidate_amplification": str(
            Fraction(new["executed_elements"], new["logical_elements"])
        ),
        "outer_consumer_multiplicity": new["outer_consumer_multiplicity"],
        "reusable_dependency_proven": right["evidence"]["reusable_dependency_proven"],
        "classification": "reusable producer"
        if right["evidence"]["reusable_dependency_proven"]
        else "work change; reuse unproven",
        "memory_budget_bytes": new["memory_budget_bytes"],
        "peak_bytes": new["peak_bytes"],
    }


def schedule_receipt(
    *,
    root: Path,
    n: int,
    auxiliaries: int,
    rank: int,
    capacity: int,
    dense_row_blocks: int,
    dense_output_blocks: int,
    triangular: bool,
    scientific_problem: str,
    dependency_identity: str,
    build_sha256: str,
) -> dict[str, Any]:
    # The policy is the production compiler schedule; count visits independently.
    source_package = str((root / "python").resolve())
    if source_package not in sys.path:
        sys.path.insert(0, source_package)
    from generativeqc_compiler.method.df_exchange_schedule import (
        projected_exchange_schedule,
    )

    if (
        Path(inspect.getfile(projected_exchange_schedule)).resolve()
        != (root / SCHEDULE_SOURCE).resolve()
    ):
        raise ReceiptError(
            "loaded production schedule is outside the bound source root"
        )
    for label, value in (
        ("n", n),
        ("auxiliaries", auxiliaries),
        ("rank", rank),
        ("capacity", capacity),
        ("dense_row_blocks", dense_row_blocks),
        ("dense_output_blocks", dense_output_blocks),
    ):
        _uint(value, label, positive=True)
    _bool(triangular, "triangular")
    if capacity > UINT64_MAX // 8:
        raise ReceiptError("memory budget byte count overflows")
    source = source_digest(root / SCHEDULE_SOURCE)
    shape = projected_exchange_schedule(
        n,
        auxiliaries,
        rank,
        capacity,
        dense_row_blocks,
        dense_output_blocks,
        triangular,
    )
    if not shape.rows:
        raise ReceiptError("projected production schedule not admitted for this shape")
    generated = callbacks = 0
    for row in range(0, n, shape.rows):
        generated += min(shape.rows, n - row)
        callbacks += 1
        for column in range(0, row + 1 if triangular else n, shape.rows):
            if column == row or (triangular and column + shape.rows == row):
                continue
            generated += min(shape.rows, n - column)
            callbacks += 1
    if generated != shape.generated_rows:
        raise ReceiptError("production generated_rows disagrees with callback census")
    resource = f"n={n};a={auxiliaries};rank={rank};capacity={capacity};dense={dense_row_blocks},{dense_output_blocks};triangular={int(triangular)}"
    return validate(
        {
            "schema": SCHEMA,
            "identity": {
                "scientific_problem": scientific_problem,
                "resource_regime": resource,
                "producer_domain": "streamed_df_occupied_raw_rows",
                "dependency_identity": dependency_identity,
                "reuse_owner": "visit_projected_exchange:two_projection_slots",
                "invalidation_owner": "exchange_invocation:geometry+basis+occupied_coefficients",
                "source_sha256": source,
                "build_sha256": _sha(build_sha256, "build_sha256"),
                "execution_identity": f"static-schedule:{source}",
            },
            "work": {
                "logical_elements": n,
                "executed_elements": generated,
                "producer_callbacks": callbacks,
                "outer_consumer_multiplicity": shape.blocks,
                "memory_budget_bytes": capacity * 8,
                "peak_bytes": None,
            },
            "evidence": {
                "kind": "schedule_census",
                "phase": "static",
                "coverage_complete": True,
                "execution_complete": False,
                "source_matched": True,
                "reusable_dependency_proven": False,
                "reuse_proof_sha256": None,
            },
        }
    )


def trace_receipt(
    trace: dict[str, Any],
    *,
    source: Path,
    scientific_problem: str,
    resource_regime: str,
    dependency_identity: str,
    build_sha256: str,
    execution_identity: str,
    tile_kind: str,
) -> dict[str, Any]:
    """Read existing trace_tile/counter output; submission is never completion proof."""
    if (
        trace.get("schema") != "generativeqc.df_trace"
        or type(trace.get("version")) is not int
        or trace["version"] != 1
    ):
        raise ReceiptError("unsupported production trace")
    if _text(tile_kind, "tile_kind") not in {
        "raw",
        "transformed",
        "derivative",
        "streamed_rows",
    }:
        raise ReceiptError("unknown tile kind")
    if _text(trace.get("execution"), "execution") not in {"stream", "graph_capture"}:
        raise ReceiptError("unknown trace execution mode")
    for key in ("dropped_tiles", "dropped_regions", "cuda_error"):
        _uint(trace.get(key), key)
    if trace.get("valid") is not True or any(
        trace[key] for key in ("dropped_tiles", "dropped_regions", "cuda_error")
    ):
        raise ReceiptError("invalid or truncated production trace")
    counters = trace.get("counters")
    tiles = trace.get("tiles")
    if not isinstance(counters, dict):
        raise ReceiptError("missing production counters")
    if tile_kind == "streamed_rows":
        if trace.get("source_backed") is not True or trace.get("streamed") is not True:
            raise ReceiptError("not a source-backed streamed DF trace")
        nbf = _uint(trace.get("nbf"), "nbf", positive=True)
        systems = _uint(trace.get("systems"), "systems", positive=True)
        logical = nbf * systems
        executed = _uint(
            counters.get("streamed_occupied_raw_generation_rows"),
            "generated rows",
            positive=True,
        )
        row_blocks = _uint(
            counters.get("streamed_occupied_row_blocks"), "row blocks", positive=True
        )
        callbacks = _uint(
            counters.get("streamed_whitening_factor_gemms"),
            "projection callbacks",
            positive=True,
        )
        visits = _uint(
            counters.get("streamed_occupied_source_first"),
            "source-first visits",
            positive=True,
        )
        if visits != systems or row_blocks % systems or callbacks < row_blocks:
            raise ReceiptError("streamed row counter coverage disagrees with systems")
        outer = row_blocks // systems
    else:
        if not isinstance(tiles, list) or not tiles:
            raise ReceiptError("missing production tiles")
        selected = []
        seen: set[tuple[Any, ...]] = set()
        for tile in tiles:
            _fields(
                tile,
                {
                    "system",
                    "pair_begin",
                    "pair_count",
                    "auxiliary_begin",
                    "auxiliary_count",
                    "derivative_coordinate",
                    "transformed",
                    "productions",
                },
                "tile",
            )
            for key in (
                "system",
                "pair_begin",
                "pair_count",
                "auxiliary_begin",
                "auxiliary_count",
                "productions",
            ):
                _uint(
                    tile[key],
                    key,
                    positive=key in {"pair_count", "auxiliary_count", "productions"},
                )
            if (
                type(tile["derivative_coordinate"]) is not int
                or not -1 <= tile["derivative_coordinate"] <= (1 << 63) - 1
            ):
                raise ReceiptError("invalid derivative coordinate")
            _bool(tile["transformed"], "transformed")
            key = tuple(
                tile[k]
                for k in (
                    "system",
                    "pair_begin",
                    "pair_count",
                    "auxiliary_begin",
                    "auxiliary_count",
                    "derivative_coordinate",
                    "transformed",
                )
            )
            if key in seen:
                raise ReceiptError("duplicate tile key")
            seen.add(key)
            kind = (
                "derivative"
                if tile["derivative_coordinate"] >= 0
                else ("transformed" if tile["transformed"] else "raw")
            )
            if kind == tile_kind:
                selected.append(tile)
        if not selected:
            raise ReceiptError("selected producer has no tiles")
        logical = sum(t["pair_count"] * t["auxiliary_count"] for t in selected)
        executed = sum(
            t["pair_count"] * t["auxiliary_count"] * t["productions"] for t in selected
        )
        callbacks = sum(t["productions"] for t in selected)
        if (
            _uint(counters.get(f"{tile_kind}_tile_productions"), "tile counter")
            != callbacks
        ):
            raise ReceiptError("tile counter disagrees with tile census")
        if (
            _uint(counters.get(f"{tile_kind}_value_bytes"), "value byte counter")
            != executed * 8
        ):
            raise ReceiptError("value byte counter disagrees with tile census")
        outer = 1
    if any(v > UINT64_MAX for v in (logical, executed, callbacks)):
        raise ReceiptError("tile census overflow")
    if tile_kind == "streamed_rows":
        reuse_owner = "visit_projected_exchange:two_projection_slots"
        invalidation_owner = "exchange_invocation:geometry+basis+occupied_coefficients"
    else:
        reuse_owner = "runtime.cuda_component_trace:tile_key"
        invalidation_owner = "trace_operation"
    return validate(
        {
            "schema": SCHEMA,
            "identity": {
                "scientific_problem": scientific_problem,
                "resource_regime": resource_regime,
                "producer_domain": f"df_trace:{tile_kind}",
                "dependency_identity": dependency_identity,
                "reuse_owner": reuse_owner,
                "invalidation_owner": invalidation_owner,
                "source_sha256": source_digest(source),
                "build_sha256": _sha(build_sha256, "build_sha256"),
                "execution_identity": execution_identity,
            },
            "work": {
                "logical_elements": logical,
                "executed_elements": executed,
                "producer_callbacks": callbacks,
                "outer_consumer_multiplicity": outer,
                "memory_budget_bytes": 0,
                "peak_bytes": None,
            },
            "evidence": {
                "kind": "runtime_trace",
                "phase": trace["execution"],
                "coverage_complete": False,
                "execution_complete": False,
                "source_matched": False,
                "reusable_dependency_proven": False,
                "reuse_proof_sha256": None,
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--baseline-source", type=Path, required=True)
    comparison.add_argument("--candidate-source", type=Path, required=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--root", type=Path, required=True)
    for name in (
        "n",
        "auxiliaries",
        "rank",
        "capacity",
        "dense_row_blocks",
        "dense_output_blocks",
    ):
        schedule.add_argument("--" + name.replace("_", "-"), type=int, required=True)
    schedule.add_argument("--triangular", action="store_true")
    schedule.add_argument("--scientific-problem", required=True)
    schedule.add_argument("--dependency-identity", required=True)
    schedule.add_argument("--build-sha256", required=True)
    trace = sub.add_parser("trace")
    trace.add_argument("trace", type=Path)
    trace.add_argument("--trace-id", type=int, required=True)
    trace.add_argument("--source", type=Path, required=True)
    for name in (
        "scientific_problem",
        "resource_regime",
        "dependency_identity",
        "build_sha256",
        "execution_identity",
        "tile_kind",
    ):
        trace.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(
                read_json(args.baseline),
                read_json(args.candidate),
                args.baseline_source,
                args.candidate_source,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
        if args.command == "schedule":
            result = schedule_receipt(
                **{key: value for key, value in vars(args).items() if key != "command"}
            )
        else:
            result = trace_receipt(
                read_trace(args.trace, args.trace_id),
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"command", "trace", "trace_id"}
                },
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except ReceiptError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
