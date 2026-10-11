"""Runtime call-path evidence must not turn into a blanket capability claim."""

from __future__ import annotations

import copy
import json

import pytest

from tools.check_native_runtime_ownership import MANIFEST, ROOT, validate


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
