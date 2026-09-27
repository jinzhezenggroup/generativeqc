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
        "grid_identity",
        "endpoint",
        "selector",
        "mode",
        "density_fitting",
        "method_identity",
        "ks_options_identity",
        "library_sha256",
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
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "unknown"))
        if status != "ok":
            result.append(
                {
                    "metadata": _metadata(row),
                    "status": status,
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
            )
            continue
        component = row.get("component_breakdown")
        if not isinstance(component, Mapping):
            work = row.get("work")
            if not isinstance(work, Mapping):
                result.append(
                    {
                        "metadata": _metadata(row),
                        "status": "missing_component_work",
                    }
                )
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
                "status": "measured",
                "components": dict(component),
            }
        )
    return result


def _wb97mv_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    metadata = _metadata(payload)
    metadata.setdefault("method", payload.get("method", "WB97M-V/RKS"))
    status = str(payload.get("status", "unknown"))
    component = payload.get("native_force_components")
    if not isinstance(component, Mapping):
        work = payload.get("native_force_work")
        if work is None:
            return [
                {
                    "metadata": metadata,
                    "status": status,
                    "error": payload.get("error"),
                }
            ]
        component = normalize_force_work(work)
    normalized_status = (
        "measured" if status in {"measured", "complete", "unknown"} else status
    )
    return [
        {
            "metadata": metadata,
            "status": normalized_status,
            "components": dict(component),
        }
    ]


def _matrix_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, typing.Any]] = []
    for case in rows:
        if not isinstance(case, Mapping):
            continue
        base = {
            key: case[key]
            for key in (
                "method",
                "selector",
                "system",
                "atoms",
                "basis",
                "density_fitting",
                "method_identity",
                "ks_options_identity",
                "library_sha256",
            )
            if key in case
        }
        case_status = str(case.get("status", "unknown"))
        if case_status != "measured":
            result.append(
                {
                    "metadata": {**base, "scenario": "case"},
                    "status": case_status,
                    "error_type": case.get("error_type"),
                    "error": case.get("error"),
                }
            )
            continue

        samples: list[Mapping[str, typing.Any]] = []
        for key in ("cold", "priming", "changed_geometry"):
            value = case.get(key)
            if isinstance(value, Mapping):
                samples.append(value)
        warm = case.get("warm")
        if isinstance(warm, list):
            samples.extend(value for value in warm if isinstance(value, Mapping))

        for sample in samples:
            metadata = {**base, **_metadata(sample)}
            metadata["scenario"] = sample.get("scenario")
            sample_status = str(sample.get("status", "measured"))
            if sample_status != "measured":
                result.append(
                    {
                        "metadata": metadata,
                        "status": sample_status,
                        "error_type": sample.get("error_type"),
                        "error": sample.get("error"),
                    }
                )
                continue
            components = sample.get("force_components")
            if isinstance(components, Mapping):
                result.append(
                    {
                        "metadata": metadata,
                        "status": "measured",
                        "components": dict(components),
                    }
                )
                continue
            force_status = sample.get("force_status")
            if force_status not in (None, "ok"):
                result.append(
                    {
                        "metadata": metadata,
                        "status": force_status,
                        "error": sample.get("force_error"),
                    }
                )

        scf_profile = case.get("scf_profile")
        if not isinstance(scf_profile, Mapping):
            continue
        profile_status = str(scf_profile.get("status", "unknown"))
        if profile_status == "measured" and isinstance(
            scf_profile.get("profile"), Mapping
        ):
            result.append(
                {
                    "metadata": {**base, "scenario": "diagnostic_scf_profile"},
                    "status": "measured",
                    "scf_profile": dict(
                        typing.cast("Mapping[str, typing.Any]", scf_profile["profile"])
                    ),
                    "trace": scf_profile.get("trace"),
                }
            )
        else:
            result.append(
                {
                    "metadata": {**base, "scenario": "diagnostic_scf_profile"},
                    "status": profile_status,
                    "error": scf_profile.get("reason"),
                }
            )
    return result


def extract_records(
    payload: Mapping[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    """Extract normalized records from one retained benchmark payload."""

    schema = str(payload.get("schema", ""))
    if schema.startswith("vibeqc.stationary-cuda-force-benchmark."):
        return _stationary_records(payload)
    if schema.startswith("vibeqc.readme-wb97mv."):
        return _wb97mv_records(payload)
    if schema.startswith("vibeqc.dft-force-matrix."):
        return _matrix_records(payload)
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
    scf_profiled: set[str] = set()
    scf_missing: set[str] = set()
    outcomes: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        outcomes[status] = outcomes.get(status, 0) + 1
        components = row.get("components")
        if isinstance(components, Mapping):
            routes.add(str(components.get("source_route")))
            coverage = components.get("coverage")
            if isinstance(coverage, Mapping):
                component_names.update(
                    str(name) for name in coverage.get("wall_seconds", ())
                )
                missing_names.update(
                    str(name) for name in coverage.get("missing_wall_seconds", ())
                )
        scf_profile = row.get("scf_profile")
        if isinstance(scf_profile, Mapping):
            measured = scf_profile.get("profiled_ms")
            if isinstance(measured, Mapping):
                scf_profiled.update(
                    str(name) for name, value in measured.items() if value is not None
                )
            scf_missing.update(
                str(name) for name in scf_profile.get("missing_expected_components", ())
            )
    return {
        "outcomes": dict(sorted(outcomes.items())),
        "source_routes": sorted(routes),
        "wall_components_observed": sorted(component_names),
        "wall_components_missing_in_at_least_one_record": sorted(missing_names),
        "scf_profiled_components_observed": sorted(scf_profiled),
        "scf_expected_components_missing_in_at_least_one_record": sorted(scf_missing),
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
        parser.error("inputs contain no DFT force evidence records")

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
