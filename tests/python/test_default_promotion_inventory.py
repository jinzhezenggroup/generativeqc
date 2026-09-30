"""Regression coverage for the #1598 default-promotion inventory."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from tools.check_default_promotion_inventory import (
    DEFAULT_INVENTORY,
    ROOT,
    load_and_validate,
    validate_inventory,
)


def _payload() -> dict:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    return payload


def test_current_default_promotion_inventory_is_complete() -> None:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    assert payload["tracking_issue"] == 1598
    assert len(payload["entries"]) >= 10


def test_inventory_requires_owner_rationale_and_revisit_condition() -> None:
    payload = _payload()
    for field, value in (
        ("owner_issues", []),
        ("rationale", ""),
        ("revisit_condition", ""),
    ):
        candidate = copy.deepcopy(payload)
        candidate["entries"][0][field] = value
        errors = validate_inventory(candidate, root=ROOT, check_sources=False)
        assert any(field in error for error in errors)


def test_inventory_rejects_duplicate_control_registration() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    control = candidate["entries"][0]["controls"][0]
    candidate["entries"][1]["controls"].append(control)
    errors = validate_inventory(candidate, root=ROOT, check_sources=False)
    assert any("registered by both" in error for error in errors)


def test_inventory_rejects_missing_audited_control() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    target = "tensor-execution:cuda-graph"
    for entry in candidate["entries"]:
        if target in entry["controls"]:
            entry["controls"].remove(target)
            break
    errors = validate_inventory(candidate, root=ROOT)
    assert any(target in error and "unregistered" in error for error in errors)


def test_new_tensor_schedule_opt_in_must_be_registered(tmp_path: Path) -> None:
    payload = _payload()
    audited = payload["scope"]["audited_sources"]
    for relative in audited:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    plan = tmp_path / "python/generativeqc_compiler/tensor/cuda_plan.py"
    source = plan.read_text()
    source = source.replace(
        "    direct_gemm: bool = True\n",
        "    new_default_off_path: bool = False\n    direct_gemm: bool = True\n",
        1,
    )
    assert "new_default_off_path" in source
    plan.write_text(source)

    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        "tensor-schedule:new_default_off_path" in error and "unregistered" in error
        for error in errors
    )
