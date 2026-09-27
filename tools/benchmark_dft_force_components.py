"""Aggregate DFT CUDA force benchmark evidence into one component report.

This tool does not run a scientific endpoint and does not manufacture missing
measurements. It normalizes retained stationary/WB97M-V work records, preserves
their source hashes, and leaves unavailable component timings as JSON null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import typing
from collections.abc import Mapping
from pathlib import Path

from benchmarks.dft_force_components import normalize_force_work


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(record: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    keys = (
        "system",
        "method",
        "spin",
        "scenario",
        "repeat",
        "atoms",
        "aos",
        "ao_count",
        "primitive_count",
        "basis",
        "grid",
        "endpoint",
    )
    return {key: record[key] for key in keys if key in record}


def _stationary_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, typing.Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        component = row.get("component_breakdown")
        if not isinstance(component, Mapping):
            work = row.get("work")
            if not isinstance(work, Mapping):
                continue
            timeline = row.get("timeline")
            state_export_seconds = None
            if isinstance(timeline, Mapping):
                phases = timeline.get("exclusive_wall_seconds")
                if isinstance(phases, Mapping):
                    value = phases.get("state_export")
                    if value is not None:
                        state_export_seconds = float(value)
            component = normalize_force_work(
                work, state_export_seconds=state_export_seconds
            )
        result.append(
            {
                "metadata": _metadata(row),
                "components": dict(component),
            }
        )
    return result


def _wb97mv_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    component = payload.get("native_force_components")
    if not isinstance(component, Mapping):
        work = payload.get("native_force_work")
        if work is None:
            return []
        component = normalize_force_work(work)
    metadata = _metadata(payload)
    metadata.setdefault("method", payload.get("method", "WB97M-V/RKS"))
    return [{"metadata": metadata, "components": dict(component)}]


def extract_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    """Extract normalized records from one retained benchmark payload."""

    schema = str(payload.get("schema", ""))
    if schema.startswith("vibeqc.stationary-cuda-force-benchmark."):
        return _stationary_records(payload)
    if schema.startswith("vibeqc.readme-wb97mv."):
        return _wb97mv_records(payload)
    if "component_seconds" in payload or "timeline" in payload:
        return [
            {
                "metadata": _metadata(payload),
                "components": normalize_force_work(payload),
            }
        ]
    raise ValueError(f"unsupported DFT force evidence schema: {schema or '<missing>'}")


def _coverage(records: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    component_names: set[str] = set()
    missing_names: set[str] = set()
    routes: set[str] = set()
    for row in records:
        components = typing.cast(Mapping[str, typing.Any], row["components"])
        routes.add(str(components.get("source_route")))
        coverage = components.get("coverage")
        if isinstance(coverage, Mapping):
            component_names.update(
                str(name) for name in coverage.get("wall_seconds", ())
            )
            missing_names.update(
                str(name) for name in coverage.get("missing_wall_seconds", ())
            )
    return {
        "source_routes": sorted(routes),
        "wall_components_observed": sorted(component_names),
        "wall_components_missing_in_at_least_one_record": sorted(missing_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, typing.Any]] = []
    sources = []
    for path in args.input:
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            parser.error(f"input must contain a JSON object: {path}")
        extracted = extract_records(payload)
        records.extend(extracted)
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "schema": payload.get("schema"),
                "records": len(extracted),
            }
        )
    if not records:
        parser.error("inputs contain no successful DFT force component records")

    report = {
        "schema": "vibeqc.dft-force-component-report.v1",
        "sources": sources,
        "records": records,
        "coverage": _coverage(records),
        "measurement_policy": (
            "clean host-wall timing and profiler/device observations remain separate; "
            "unavailable component measurements are JSON null"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
