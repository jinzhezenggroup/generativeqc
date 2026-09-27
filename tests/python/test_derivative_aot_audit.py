"""Derivative AOT audit reports deterministic inventory/provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path

from vibeqc_compiler.integral.range_separation import CoulombKernel

from tools import audit_derivative_aot_registry as audit


def test_source_statistics_report_translation_units_and_bytes() -> None:
    assert audit._source_statistics(("a", "bc")) == {
        "translation_units": 2,
        "generated_source_bytes": 3,
        "largest_translation_unit_bytes": 2,
    }


def test_audit_schema_keeps_radial_identity_and_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "radials.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "vibeqc.derivative-aot.radials.v1",
                "entries": [
                    {
                        "backend": "cpu",
                        "family": "short_range",
                        "omega": 0.3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    radial = CoulombKernel("short_range", 0.3)
    monkeypatch.setattr(
        audit,
        "derivative_cpu_aot_sources",
        lambda: (((), "full-a"), ((), "full-bb")),
    )
    monkeypatch.setattr(
        audit,
        "_range_programs",
        lambda radials: (
            [
                {
                    "identity": "program-id",
                    "radial": radial.to_payload(),
                    "angular": [0, 0, 0, 0],
                    "group_index": 0,
                    "component_indices": [0],
                    "entry_prefix": "program",
                    "source_bytes": 7,
                }
            ],
            ("range-x",),
        ),
    )
    monkeypatch.setattr(audit, "file_hash", lambda path: f"hash:{Path(path).name}")

    result = audit.audit(radial_manifest=manifest)
    assert result["schema"] == "vibeqc.derivative-aot.audit.v1"
    assert result["provenance"]["radial_manifest_sha256"] == "hash:radials.json"
    assert result["full_range"]["translation_units"] == 2
    assert result["full_range"]["target"] == "native-host"
    assert result["full_range"]["package_identity"]
    assert result["full_range"]["package_identity_payload"]["spin_contract"] == "spin-neutral"
    assert result["full_range"]["generated_source_bytes"] == len("full-a") + len(
        "full-bb"
    )
    assert result["range"]["radials"] == [radial.to_payload()]
    assert result["range"]["programs"][0]["identity"] == "program-id"
    assert result["package_library"] is None
