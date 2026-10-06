"""Frozen hybrid rows must not inherit incompatible historical PBE0 reports."""

import json
from pathlib import Path

import pytest
from test_dft_mp_v1_contract import _campaign, _file, _run_record

from tools.dft_mp_v1 import validate
from tools.dft_mp_v1.freeze_contract import canonical
from tools.dft_mp_v1.map_1187_hybrid_rows import map_rows


def test_historical_campaigns_do_not_pass_frozen_hybrid_rows() -> None:
    report = map_rows()
    rows = report["rows"]
    assert report["required_rows"] == 26
    assert report["optional_rows"] == 2
    assert report["passed_required_rows"] == 0
    assert len(rows) == 28
    assert len(report["unmatched_legacy_reports"]) == 2
    assert all(row["status"] == "missing_exact_receipt" for row in rows)
    assert all(
        "frozen_grid_spec_mismatch" in candidate["reasons"]
        for row in rows
        for candidate in row["legacy_candidates"]
    )
    assert not next(
        row for row in rows if row["id"] == "b3lyp/rks/benzene/fp64_energy_forces"
    )["legacy_candidates"]


def test_claimed_pass_without_installed_production_campaign_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_sha256": "de6c847b1ed93e537c1422679ae3df53a4cca3cd13e7268483e419afbf2faa00",
                "rows": [{"id": "pbe0/rks/water/fp64_energy", "status": "pass"}],
            }
        ),
        encoding="utf-8",
    )
    report = map_rows(receipts=(receipt,))
    assert report["passed_required_rows"] == 0
    assert report["receipt_audits"][str(receipt)]["error"] == (
        "not installed production evidence"
    )


def _receipt(directory: Path, row_ids: tuple[str, ...]) -> Path:
    directory.mkdir()
    contract = validate.manifest()
    campaign = _campaign(directory)
    entries = []
    for row in contract["rows"]:
        if row["id"] not in row_ids:
            entries.append({"id": row["id"], "status": "not-run", "reason": "fixture"})
            continue
        run_dir = directory / row["case"] / row["level"]
        run_dir.mkdir(parents=True)
        value = _run_record(run_dir, contract, row, campaign)
        precision = value["precision"]
        precision.update(requested_mode="fp64", mixed_stage_fock_builds=0)
        precision["native_provenance"]["mixed_stage_fock_builds"] = 0
        events = [
            event
            for event in precision["scf_fock_timeline"]
            if event["kind"] != "mixed_fock"
        ]
        for sequence, event in enumerate(events):
            event["sequence"] = sequence
        precision["scf_fock_timeline"] = events
        precision["operators"][0].update(compute="fp64", arithmetic_mode="strict")
        value["checks"]["finite_difference"].update(
            step_bohr=[0.01, 0.005], reconverged_each_displacement=True
        )
        value["checks"]["grid_convergence"]["independent_finer_grid"] = True
        case = contract["cases"][row["case"]]
        value["checks"]["changed_geometry"].update(
            input_sha256=case["changed_input_sha256"],
            grid_identity=case["changed_grid_identity"],
            complete_energy_forces=True,
        )
        stdout = _file(run_dir / "stdout.json", canonical(value))
        capture = {
            "stdout": stdout,
            "stderr": _file(run_dir / "stderr.txt"),
            "progress": _file(run_dir / "progress.json"),
        }
        for record in capture.values():
            record["path"] = str((run_dir / record["path"]).resolve())
        entries.append(
            {"id": row["id"], "status": "pass", "evidence": stdout, "capture": capture}
        )
    path = directory / "receipt.json"
    path.write_bytes(
        canonical(
            {
                "schema_version": 1,
                "contract_sha256": contract["contract_sha256"],
                "campaign": campaign,
                "rows": entries,
            }
        )
    )
    return path


def test_valid_row_import_and_invalid_row_isolation(tmp_path: Path) -> None:
    ids = ("pbe0/rks/water/fp64_energy", "pbe0/rks/water/fp64_energy_forces")
    path = _receipt(tmp_path / "campaign", ids)
    report = map_rows(receipts=(path,))
    assert report["passed_required_rows"] == 2
    assert report["receipt_audits"][str(path)]["accepted_rows"] == sorted(ids)
    receipt = json.loads(path.read_bytes())
    broken = next(entry for entry in receipt["rows"] if entry["id"] == ids[1])
    Path(broken["evidence"]["path"]).write_bytes(b"{}")
    report = map_rows(receipts=(path,))
    assert report["passed_required_rows"] == 1
    assert ids[1] in report["receipt_audits"][str(path)]["rejected_rows"]


@pytest.mark.parametrize("phase", ("audit", "row"))
def test_receipt_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    path = _receipt(tmp_path / "campaign", ("pbe0/rks/water/fp64_energy",))
    function = "audit" if phase == "audit" else "_check_run"
    original = getattr(validate, function)
    audited = False
    if phase == "row":
        original_audit = validate.audit

        def audit(*args: object, **kwargs: object) -> dict:
            nonlocal audited
            result = original_audit(*args, **kwargs)
            audited = True
            return result

        monkeypatch.setattr(validate, "audit", audit)

    def mutate(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        if phase == "audit" or audited:
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(validate, function, mutate)
    report = map_rows(receipts=(path,))
    assert report["passed_required_rows"] == 0
    assert report["receipt_audits"][str(path)]["error"] == (
        "receipt changed during audit"
        if phase == "audit"
        else "receipt changed during row checks"
    )


@pytest.mark.parametrize("field", ("device_uuid", "adapter", "conditions", "source"))
def test_incompatible_campaigns_cannot_aggregate(tmp_path: Path, field: str) -> None:
    first = _receipt(tmp_path / "first", ("pbe0/rks/water/fp64_energy",))
    second = _receipt(tmp_path / "second", ("pbe0/rks/water/fp64_energy_forces",))
    compatible = map_rows(receipts=(first, second))
    assert compatible["passed_required_rows"] == 2
    assert compatible["aggregation_error"] is None
    receipt = json.loads(second.read_bytes())
    campaign = receipt["campaign"]
    if field == "device_uuid":
        campaign["hardware"][field] = "another-device"
    elif field == "source":
        for key in ("source_commit", "build_source_commit", "library_source_commit"):
            campaign[key] = "b" * 40
        build = json.loads((second.parent / "build.json").read_bytes())
        build["source_commit"] = campaign["source_commit"]
        campaign["build_record"] = _file(second.parent / "build.json", canonical(build))
        entry = next(e for e in receipt["rows"] if e["status"] == "pass")
        evidence = Path(entry["evidence"]["path"])
        run = json.loads(evidence.read_bytes())
        run["source_commit"] = campaign["source_commit"]
        record = _file(evidence, canonical(run))
        record["path"] = str(evidence)
        entry["evidence"] = entry["capture"]["stdout"] = record
    else:
        campaign[field] = _file(
            second.parent / f"different-{field}.bin", b"different\n"
        )
        if field == "conditions":
            campaign["conditions_sha256"] = campaign[field]["sha256"]
    second.write_bytes(canonical(receipt))
    for paths in ((first, second), (second, first)):
        report = map_rows(receipts=paths)
        assert report["passed_required_rows"] == 0
        assert report["aggregation_error"]
        assert all(
            audit["accepted_rows"] for audit in report["receipt_audits"].values()
        )
