"""Map validated DFT-MP-v1 receipts onto #1186's frozen FP64 rows.

An unrelated benchmark is never treated as a row pass. The optional receipt must
first pass the shared campaign and per-row evidence validator; incomplete rows
remain visible so a partial campaign can guide the next run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .freeze_contract import digest
from .validate import InvalidEvidence, audit, manifest, require

METHODS = frozenset({"lda", "pbe", "r2scan"})
LEVELS = frozenset({"fp64_energy", "fp64_energy_forces"})


def _rows(contract: dict) -> list[dict]:
    rows = [
        row
        for row in contract["rows"]
        if row["required"] and row["method"] in METHODS and row["level"] in LEVELS
    ]
    require(len(rows) == 44, "#1186 frozen FP64 row inventory drift")
    return rows


def build_coverage(receipt_path: Path | None = None) -> dict:
    contract = manifest()
    rows = _rows(contract)
    entries: dict[str, dict] = {}
    receipt_record = None
    invalid_passes: set[str] = set()
    if receipt_path is not None:
        receipt_path = receipt_path.resolve()
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        validated = audit(receipt_path)
        require(
            receipt_path.read_bytes() == raw,
            "receipt changed during validation; retry after the writer finishes",
        )
        entries = {entry["id"]: entry for entry in receipt["rows"]}
        invalid_passes = {
            failure.partition(":")[0]
            for failure in validated["failures"]
            if ": invalid pass:" in failure
        }
        valid_passes = {
            row_id
            for row_id, entry in entries.items()
            if entry["status"] == "pass" and row_id not in invalid_passes
        }
        require(
            len(valid_passes) == validated["passed_rows"],
            "shared validator pass inventory could not be reconciled",
        )
        receipt_record = {
            "path": str(receipt_path),
            "sha256": digest(raw),
            "source_commit": validated["source_commit"],
            "hardware": receipt["campaign"]["hardware"],
        }

    mapped = []
    for row in rows:
        case = contract["cases"][row["case"]]
        entry = entries.get(row["id"])
        status = "missing"
        reason = "no validated installed-production receipt supplied"
        if entry is not None:
            status = entry["status"]
            if status == "pass" and row["id"] in invalid_passes:
                status = "invalid"
                reason = "shared validator rejected pass evidence"
            elif status == "pass":
                reason = "shared validator accepted captured row evidence"
            else:
                reason = entry.get("reason", "no accepted row evidence")
        mapped.append(
            {
                "id": row["id"],
                "method": row["method"],
                "spin": row["spin"],
                "case": row["case"],
                "product": row["product"],
                "atom_count": case["atom_count"],
                "ao_count_spherical": case["ao_count_spherical"],
                "input_sha256": case["input_sha256"],
                "changed_input_sha256": case["changed_input_sha256"],
                "grid_identity": case["grid_identity"],
                "changed_grid_identity": case["changed_grid_identity"],
                "status": status,
                "reason": reason,
            }
        )

    passed = sum(row["status"] == "pass" for row in mapped)
    return {
        "schema_version": 1,
        "issue": 1186,
        "contract_sha256": contract["contract_sha256"],
        "receipt": receipt_record,
        "required_fp64_rows": len(mapped),
        "validated_pass_rows": passed,
        "missing_or_blocked_rows": len(mapped) - passed,
        "rows": mapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.receipt is not None and args.output is not None:
        same_path = args.receipt.resolve() == args.output.resolve()
        same_file = (
            args.receipt.is_file()
            and args.output.is_file()
            and args.receipt.samefile(args.output)
        )
        if same_path or same_file:
            parser.error("--output must not overwrite the input --receipt")
    try:
        result = build_coverage(args.receipt)
    except (InvalidEvidence, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
