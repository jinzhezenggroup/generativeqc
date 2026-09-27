from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.method.libxc_public_evidence import (
    installed_public_evidence,
    installed_public_evidence_provenance,
    installed_public_functionals,
)

from tools.render_libxc_public_evidence import collect_public_evidence

if TYPE_CHECKING:
    from pathlib import Path


def test_empty_generated_inventory_is_fail_closed() -> None:
    assert installed_public_evidence("LDA_C_BR78") is None
    provenance = installed_public_evidence_provenance()
    assert provenance["schema"] == "vibeqc.installed-libxc-public-evidence/v1"
    assert provenance["source"] is None


def test_renderer_rejects_stale_capability_identity(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "functionals": [
                    {
                        "name": "LDA_C_BR78",
                        "status": "pass",
                        "capability_identity": "stale",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="stale for the current compiler identity"):
        collect_public_evidence(tmp_path)



def test_empty_generated_inventory_exposes_no_public_functionals() -> None:
    assert installed_public_functionals() == ()
