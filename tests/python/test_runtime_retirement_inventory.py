"""Retired source locations remain navigable history, not live dispatch gates."""

import json

import pytest

from tools.check_native_runtime_ownership import MANIFEST, ROOT, validate


@pytest.fixture
def inventory() -> dict:
    return json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))


def retirement(inventory: dict) -> dict:
    """Select a historical entry without prescribing production source spelling."""
    return next(
        row["retired_python_sites"][0]
        for row in inventory["rows"]
        if row.get("retired_python_sites")
    )


def test_retirement_does_not_require_the_removed_file(inventory: dict) -> None:
    assert not (ROOT / retirement(inventory)["path"]).exists()
    validate(inventory)


@pytest.mark.parametrize("path", ["../old.py", "/old.py", "old.cpp"])
def test_retired_locations_are_repository_local(inventory: dict, path: str) -> None:
    retirement(inventory)["path"] = path
    with pytest.raises(ValueError, match="retired Python path must be local"):
        validate(inventory)


def test_retirement_records_the_last_live_revision(inventory: dict) -> None:
    retirement(inventory)["last_present_source_sha"] = "master"
    with pytest.raises(ValueError, match="retirement needs an exact source SHA"):
        validate(inventory)


def test_retirement_retains_a_live_replacement(inventory: dict) -> None:
    retirement(inventory)["replacement"] = "missing.py"
    with pytest.raises(ValueError, match="source anchor"):
        validate(inventory)


@pytest.mark.parametrize("field", ["reason", "symbols"])
def test_retirement_has_an_explanation_and_historical_location(
    inventory: dict, field: str
) -> None:
    retirement(inventory)[field] = [] if field == "symbols" else ""
    with pytest.raises(ValueError, match="retirement needs"):
        validate(inventory)
