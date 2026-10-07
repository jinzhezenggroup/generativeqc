"""Map retained hybrid evidence onto the frozen DFT-MP-v1 FP64 rows.

Legacy benchmark reports are diagnostic candidates, never acceptance receipts.
Only a source-bound installed-production receipt accepted by the shared validator
can contribute a passing row.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tools.dft_mp_v1 import validate

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name("hybrid_1187_candidates.json")
METHODS = frozenset(("pbe0", "b3lyp"))
LEVELS = frozenset(("fp64_energy", "fp64_energy_forces"))


def _read_report(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("report root must be an object")
    return value


def _legacy_candidate(
    path: Path, data: dict, sha256: str, contract: dict, row: dict
) -> dict | None:
    protocol = data.get("protocol", {})
    case = contract["cases"][row["case"]]
    if not isinstance(protocol, dict):
        raise TypeError("protocol must be an object")
    method = f"{row['method'].upper()}/{row['spin'].upper()}"
    if (protocol.get("method"), protocol.get("atoms"), protocol.get("aos")) != (
        method,
        case["atom_count"],
        case["ao_count_spherical"],
    ):
        return None

    reasons = []
    geometries = protocol.get("geometries_bohr")
    expected_geometry = json.loads(
        (ROOT / "tools/dft_mp_v1" / case["input"]).read_text(encoding="utf-8")
    )["atoms"]
    expected_changed = json.loads(
        (ROOT / "tools/dft_mp_v1" / case["changed_input"]).read_text(encoding="utf-8")
    )["atoms"]
    if not isinstance(geometries, list) or geometries != [
        expected_geometry,
        expected_changed,
    ]:
        reasons.append("frozen_geometry_mismatch")
    if protocol.get("grid") != contract["model"]["grid_spec"]:
        reasons.append("frozen_grid_spec_mismatch")
    if protocol.get("basis_identity") != contract["basis"]["basis_pack_sha256"]:
        reasons.append("frozen_basis_pack_mismatch")
    scf = contract["model"]["scf"]
    if (protocol.get("energy_tolerance"), protocol.get("density_tolerance")) != (
        scf["energy_tolerance_eh"],
        scf["density_tolerance"],
    ):
        reasons.append("frozen_scf_target_mismatch")
    reasons.append("not_installed_production_receipt")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256,
        "reasons": reasons,
    }


def _accepted_rows(
    path: Path, contract: dict
) -> tuple[set[str], dict[str, str], dict | None, str | None]:
    try:
        snapshot = path.read_bytes()
        validate.audit(path)
        validate.require(path.read_bytes() == snapshot, "receipt changed during audit")
        receipt = json.loads(snapshot)
        base = path.resolve().parent
        campaign = receipt["campaign"]
        identity = {
            "source_commit": campaign["source_commit"],
            "hardware": campaign["hardware"],
            **{
                f"{key}_sha256": campaign[key]["sha256"]
                for key in (
                    "library",
                    "artifact",
                    "adapter",
                    "build_record",
                    "conditions",
                )
            },
        }
        expected = {row["id"]: row for row in contract["rows"]}
        accepted = set()
        rejected = {}
        for entry in receipt["rows"]:
            row = expected[entry["id"]]
            if (
                row["method"] not in METHODS
                or row["level"] not in LEVELS
                or entry["status"] != "pass"
            ):
                continue
            try:
                validate.checked_capture(base, entry["capture"])
                validate.require(
                    entry["evidence"] == entry["capture"]["stdout"],
                    "pass evidence is not captured stdout",
                )
                evidence = validate.checked_file(base, entry["evidence"])
                validate._check_run(
                    validate.read_json(evidence),
                    campaign,
                    row,
                    contract,
                    evidence.parent,
                    entry["capture"],
                )
                accepted.add(row["id"])
            except (
                validate.InvalidEvidence,
                KeyError,
                TypeError,
                ValueError,
                OSError,
            ) as error:
                rejected[row["id"]] = str(error)
        validate.require(
            path.read_bytes() == snapshot, "receipt changed during row checks"
        )
        return accepted, rejected, identity, None
    except (
        validate.InvalidEvidence,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        return set(), {}, None, str(error)


def map_rows(catalog: Path = CATALOG, receipts: tuple[Path, ...] = ()) -> dict:
    contract = validate.manifest()
    selected = [
        row
        for row in contract["rows"]
        if row["method"] in METHODS and row["level"] in LEVELS
    ]
    report_paths = json.loads(catalog.read_text(encoding="utf-8"))["legacy_reports"]
    candidates = []
    for name in report_paths:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates.append(
            (path, _read_report(path), hashlib.sha256(path.read_bytes()).hexdigest())
        )
    receipt_results = {}
    accepted = set()
    # Freeze and deduplicate path identities before producers can replace receipts
    # between audits. Keep the first supplied spelling for the public audit key.
    receipt_paths = {}
    for path in receipts:
        receipt_paths.setdefault(path.resolve(), path)
    for resolved, path in receipt_paths.items():
        passing, rejected, identity, error = _accepted_rows(resolved, contract)
        receipt_results[str(path)] = {
            "accepted_rows": sorted(passing),
            "rejected_rows": rejected,
            "campaign_identity": identity,
            "error": error,
        }
        accepted.update(passing)
    identities = {
        json.dumps(result["campaign_identity"], sort_keys=True)
        for result in receipt_results.values()
        if result["accepted_rows"]
    }
    aggregation_error = None
    if len(identities) > 1:
        accepted.clear()
        aggregation_error = (
            "incompatible campaign identities; audit each campaign separately"
        )
    output_rows = []
    matched = set()
    for row in selected:
        legacy_candidates = []
        for path, data, sha256 in candidates:
            candidate = _legacy_candidate(path, data, sha256, contract, row)
            if candidate is not None:
                legacy_candidates.append(candidate)
                matched.add(path)
        output_rows.append(
            {
                "id": row["id"],
                "required": row["required"],
                "status": "pass" if row["id"] in accepted else "missing_exact_receipt",
                "legacy_candidates": legacy_candidates,
            }
        )
    return {
        "schema": "dft-mp-1187-hybrid-row-map-v1",
        "contract_sha256": contract["contract_sha256"],
        "required_rows": sum(row["required"] for row in output_rows),
        "passed_required_rows": sum(
            row["required"] and row["status"] == "pass" for row in output_rows
        ),
        "optional_rows": sum(not row["required"] for row in output_rows),
        "unmatched_legacy_reports": [
            path.relative_to(ROOT).as_posix()
            for path, _, _ in candidates
            if path not in matched
        ],
        "receipt_audits": receipt_results,
        "aggregation_error": aggregation_error,
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            map_rows(args.catalog, tuple(args.receipt)), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
