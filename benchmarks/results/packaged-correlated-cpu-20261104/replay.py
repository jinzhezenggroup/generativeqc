"""Replay retained numerical evidence without loading a production runtime."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

import numpy as np
from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record


def numerical_errors(samples: list[dict], reference: dict) -> dict:
    """Recompute every full-observable gate against the independent fixture."""
    started, completed = samples
    assert started["event"] == "science_started"
    assert completed["event"] == "science_completed"
    assert started["tracked_source_dirty"] is False
    assert started["source_patch_sha256"] == hashlib.sha256(b"").hexdigest()
    for record in samples:
        assert record["qualification"] is False and record["production_path"] is True
        assert record["complete_endpoint_semantic_work"] is None
        assert record["complete_allocation_peak_bytes"] is None
    assert completed["performance_gate_passed"] is None
    components = completed["components"]
    assert len(components) == 5 and len(completed["execution_groups"]) == 2
    assert completed["convergence"]["requested_policy"] == "standard"
    expected = reference["reference"]
    assert {item["roles"][0] for item in components} == set(expected["components"])
    errors = {}
    for item in [*components, completed]:
        role = item["roles"][0] if "roles" in item else "composite"
        target = expected if role == "composite" else expected["components"][role]
        assert np.asarray(item["forces"]).shape == (2, 3)
        if role != "composite":
            basis = reference["basis_identities"][int(role.endswith("high"))]
            assert item["basis_identity"] == basis
        for field, tolerance in (("energy", 1e-9), ("forces", 2e-6)):
            actual = [item[field]] if field == "energy" else item[field]
            oracle = [target[field]] if field == "energy" else target[field]
            error = block_error(actual, oracle, atol=tolerance, rtol=0)
            assert error["passed"], (role, field, error)
            reported = item["energy_abs" if field == "energy" else "force_max_abs"]
            assert error["max_absolute_error"] == reported
            errors[f"{role}:{field}"] = error
    energy = math.fsum(item["coefficient"] * item["energy"] for item in components)
    forces = sum(
        item["coefficient"] * np.asarray(item["forces"]) for item in components
    )
    assert abs(energy - completed["energy"]) <= 1e-12
    assert np.max(np.abs(forces - completed["forces"])) <= 1e-12
    return errors


def main() -> None:
    """Authenticate the selected packet and refuse broader promotion claims."""
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "publication.json").read_text())
    files = {
        item["path"]: (directory / item["path"]).read_bytes()
        for item in manifest["files"]
    }
    validate_publication(manifest, files)
    retained = load_publication_record(directory, role="samples")
    samples = retained["journal"]
    evidence = load_publication_record(directory)
    provenance = retained["provenance"]
    assert (
        hashlib.sha256(retained["probe_source"].encode()).hexdigest()
        == retained["probe_source_sha256"]
    )
    started = samples[0]
    assert manifest["source"] == {
        "revision": started["source_parent_commit"],
        "dirty": False,
    }
    assert evidence["hashes"]["source"] == started["native_source_identity"]
    assert (
        evidence["stages"]["compilation"]["library_sha256"] == started["library_sha256"]
    )
    assert provenance["native_library_sha256"] == started["library_sha256"]
    reference_path = ROOT / "tests/reference_data/cbs/h2-dz-tz.json"
    assert (
        hashlib.sha256(reference_path.read_bytes()).hexdigest()
        == started["reference_sha256"]
    )
    errors = numerical_errors(samples, json.loads(reference_path.read_text()))
    assert errors == evidence["block_errors"]
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert evidence["memory"]["peak_bytes"] is None
    print(
        json.dumps(
            {"numerical_blocks_passed": len(errors), "performance_qualified": False}
        )
    )


if __name__ == "__main__":
    main()
