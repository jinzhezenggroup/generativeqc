"""Runtime call-path evidence must not turn into a blanket capability claim."""

from __future__ import annotations

import copy
import json

import pytest

from tools.check_native_runtime_ownership import (
    DELIVERY,
    MANIFEST,
    ROOT,
    validate,
    validate_delivery,
)


@pytest.fixture
def inventory() -> dict:
    return json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))


def test_repository_baseline_source_anchors(inventory: dict) -> None:
    validate(inventory)


@pytest.mark.parametrize(
    "axis",
    ["properties", "method_selectors", "spins", "representations", "approximations"],
)
def test_missing_matrix_axis_is_not_complete(inventory: dict, axis: str) -> None:
    inventory["rows"][0][axis] = []
    with pytest.raises(ValueError, match=axis):
        validate(inventory)


def test_wrapper_parity_is_not_native_parity(inventory: dict) -> None:
    inventory["rows"][0]["qualification"] = "native-parity"
    with pytest.raises(ValueError, match="not native parity"):
        validate(inventory)


def test_source_inventory_is_not_numerical_acceptance(inventory: dict) -> None:
    inventory["coverage"] = "complete"
    with pytest.raises(ValueError, match="numerical qualification"):
        validate(inventory)


def test_python_runtime_cannot_be_declared_retired(inventory: dict) -> None:
    row = next(row for row in inventory["rows"] if row["id"] == "ks-python-cuda-force")
    row["remaining_python_runtime"] = []
    with pytest.raises(ValueError, match="incorrectly marked retired"):
        validate(inventory)


def test_source_symbols_are_not_comment_matches(inventory: dict) -> None:
    inventory["rows"][0]["python_sites"][0]["symbol"] = "not_a_real_source_symbol"
    with pytest.raises(ValueError, match="missing Python symbol"):
        validate(inventory)


def test_native_and_python_sites_keep_separate_classification(inventory: dict) -> None:
    inventory["rows"][0]["python_sites"][0]["classification"] = "native"
    with pytest.raises(ValueError, match="unclassified Python"):
        validate(inventory)


def test_cutover_scope_cannot_shrink_to_delivered_slice(inventory: dict) -> None:
    inventory["required_slices"] = ["R0", "R4"]
    with pytest.raises(ValueError, match="complete R0-R6"):
        validate(inventory)


def test_baseline_needs_exact_source_revision(inventory: dict) -> None:
    inventory["source_sha"] = "master"
    with pytest.raises(ValueError, match="exact source SHA"):
        validate(inventory)


def test_duplicate_ids_are_not_additional_coverage(inventory: dict) -> None:
    inventory["rows"].append(copy.deepcopy(inventory["rows"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate(inventory)


@pytest.fixture
def delivery() -> dict:
    return json.loads((ROOT / DELIVERY).read_text(encoding="utf-8"))


def test_delivery_accounts_for_the_entire_baseline(
    delivery: dict, inventory: dict
) -> None:
    validate_delivery(delivery, inventory)


def test_delivery_cannot_omit_an_undelivered_domain(
    delivery: dict, inventory: dict
) -> None:
    delivery["rows"].pop()
    with pytest.raises(ValueError, match="every baseline row"):
        validate_delivery(delivery, inventory)


def test_delivery_cannot_replace_a_domain_with_unrelated_work(
    delivery: dict, inventory: dict
) -> None:
    delivery["rows"][-1]["id"] = "unrelated"
    with pytest.raises(ValueError, match="every baseline row"):
        validate_delivery(delivery, inventory)


def test_delivery_rejects_duplicate_coverage(delivery: dict, inventory: dict) -> None:
    delivery["rows"].append(copy.deepcopy(delivery["rows"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_delivery(delivery, inventory)


@pytest.mark.parametrize("coverage", ["complete", "native-parity", "qualified"])
def test_delivery_is_not_a_capability_or_scientific_qualification(
    delivery: dict, inventory: dict, coverage: str
) -> None:
    delivery["coverage"] = coverage
    with pytest.raises(ValueError, match="numerical qualification"):
        validate_delivery(delivery, inventory)


def test_delivery_retains_the_original_required_slices(
    delivery: dict, inventory: dict
) -> None:
    delivery["required_slices"] = ["R0", "R1"]
    with pytest.raises(ValueError, match="complete R0-R6"):
        validate_delivery(delivery, inventory)


def test_delivery_requires_its_own_exact_source_revision(
    delivery: dict, inventory: dict
) -> None:
    delivery["source_sha"] = "master"
    with pytest.raises(ValueError, match="exact audit source SHA"):
        validate_delivery(delivery, inventory)


@pytest.mark.parametrize("audited_on", ["today", "2026-11-31", "20261104", None])
def test_delivery_requires_an_unambiguous_audit_date(
    delivery: dict, inventory: dict, audited_on: str | None
) -> None:
    delivery["audited_on"] = audited_on
    with pytest.raises(ValueError, match="ISO audit date"):
        validate_delivery(delivery, inventory)


@pytest.mark.parametrize("field", ["delivered_scope", "source_anchors", "tests"])
def test_delivery_requires_scope_and_evidence_pointers(
    delivery: dict, inventory: dict, field: str
) -> None:
    delivery["rows"][0][field] = []
    with pytest.raises(ValueError, match=field):
        validate_delivery(delivery, inventory)


@pytest.mark.parametrize("field", ["source_anchors", "tests"])
@pytest.mark.parametrize("anchor", ["../AGENTS.md", "/etc/passwd", "missing.py"])
def test_delivery_rejects_missing_or_nonlocal_evidence(
    delivery: dict, inventory: dict, field: str, anchor: str
) -> None:
    delivery["rows"][0][field] = [anchor]
    with pytest.raises(ValueError, match="source anchor"):
        validate_delivery(delivery, inventory)


def test_delivery_cannot_retire_a_remaining_scientific_controller(
    delivery: dict, inventory: dict
) -> None:
    row = next(row for row in delivery["rows"] if row["state"] == "partial-cutover")
    row["state"] = "native-dispatch"
    with pytest.raises(ValueError, match="Python science is not retired"):
        validate_delivery(delivery, inventory)


def test_delivery_cannot_hide_remaining_science_as_a_wrapper(
    delivery: dict, inventory: dict
) -> None:
    row = next(row for row in delivery["rows"] if row["state"] == "partial-cutover")
    row["python_roles"] = ["wrapper"]
    with pytest.raises(ValueError, match="needs classification"):
        validate_delivery(delivery, inventory)


def test_partial_cutover_retains_its_unmet_acceptance(
    delivery: dict, inventory: dict
) -> None:
    row = next(row for row in delivery["rows"] if row["state"] == "partial-cutover")
    row["remaining_acceptance"] = []
    with pytest.raises(ValueError, match="remaining acceptance"):
        validate_delivery(delivery, inventory)


@pytest.mark.parametrize("role", ["compiler-ad", "reference-oracle"])
def test_compiler_and_independent_oracle_work_are_not_retired(
    delivery: dict, inventory: dict, role: str
) -> None:
    row = next(row for row in delivery["rows"] if row["python_roles"] == [role])
    row["retired_python_runtime"] = ["delete this intentional Python owner"]
    with pytest.raises(ValueError, match="retain independent compiler/oracle"):
        validate_delivery(delivery, inventory)
