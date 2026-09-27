"""Structural tests for retained public Libxc CPU evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeqc_compiler.xc import public_inventory


def test_installed_inventory_is_explicitly_empty_until_requalified() -> None:
    assert public_inventory.available_public_functionals() == ()


def test_inventory_rejects_incomplete_stage_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "schema": public_inventory.INVENTORY_SCHEMA,
                "revision": "deadbeef",
                "functionals": {
                    "GGA_X_FAKE": {
                        "capability_identity": "0" * 64,
                        "stages": {"public-method": {}},
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete stage evidence"):
        public_inventory.load_public_inventory(path)


def test_inventory_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"schema": "wrong", "revision": "x", "functionals": {}}))
    with pytest.raises(ValueError, match="unsupported retained"):
        public_inventory.load_public_inventory(path)
