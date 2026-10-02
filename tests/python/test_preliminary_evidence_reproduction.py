"""Offline reproduction-admission regressions; never load a native library."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/preliminary-scf-api"


@pytest.fixture
def replay() -> dict:
    return runpy.run_path(str(BUNDLE / "reproduce.py"), run_name="reproduction_test")


def mapping(source_hash: str, *, kind: str = "reviewed-follow-up-source") -> dict:
    return {
        "id": kind,
        "kind": kind,
        "production_source_sha256": source_hash,
        "review_scope": "Explicit untimed source review; does not qualify historical measurements",
        "reviewed_production_changes": [
            {"path": "python/generativeqc/initial_guess.py", "sha256": "d" * 64}
        ],
    }


@pytest.mark.parametrize(
    "kind", ["frozen-measured-source", "reviewed-follow-up-source"]
)
def test_reviewed_exact_mapping_is_admitted(replay: dict, kind: str) -> None:
    entry = mapping("a" * 64, kind=kind)
    result = replay["select_mapping"](
        {"source_hash": "a" * 64},
        {"sources": [entry]},
        {"source": {"source_hash": "a" * 64}},
    )
    assert result == entry


@pytest.mark.parametrize("actual", ["b" * 64, "a" * 63 + "b", "a" * 8])
def test_undeclared_source_hashes_fail_closed(replay: dict, actual: str) -> None:
    with pytest.raises(ValueError, match="no exact reviewed"):
        replay["select_mapping"](
            {"source_hash": actual},
            {"sources": [mapping("a" * 64)]},
            {"source": {"source_hash": "a" * 64}},
        )


def test_followup_cannot_claim_frozen_source(replay: dict) -> None:
    with pytest.raises(ValueError, match="historical source receipt"):
        replay["select_mapping"](
            {"source_hash": "b" * 64},
            {"sources": [mapping("b" * 64, kind="frozen-measured-source")]},
            {"source": {"source_hash": "a" * 64}},
        )


def test_source_mapping_id_does_not_override_hash(replay: dict) -> None:
    with pytest.raises(ValueError, match="no exact reviewed"):
        replay["select_mapping"](
            {"source_hash": "a" * 64},
            {"sources": [mapping("a" * 64)]},
            {"source": {"source_hash": "a" * 64}},
            "unreviewed-name",
        )


def test_new_plan_keeps_actual_source_and_rebuilt_library(
    replay: dict, tmp_path: Path
) -> None:
    actual = {
        "source_hash": "b" * 64,
        "revision": "c" * 40,
        "tree": "d" * 40,
        "dirty": False,
    }
    historical = {
        "source": {"source_hash": "a" * 64},
        "library_sha256": "e" * 64,
        "endpoint_boundary": "test endpoint",
        "checkpoint_boundary": "test export",
    }
    unchanged_historical = json.loads(json.dumps(historical))
    plan = replay["new_reproduction_plan"](
        source=tmp_path / "source",
        library=tmp_path / "new.so",
        output=tmp_path / "new-run",
        revision="c" * 40,
        actual=actual,
        mapping=mapping("b" * 64),
        historical=historical,
        library_hash="f" * 64,
        verified={"mapping_sha256": "0" * 64},
        timeout=300.0,
        bundle=BUNDLE,
    )
    assert plan["scope"].startswith("NEW reproduction")
    assert plan["historical_measurements_reused"] is False
    assert plan["reproduction_library_sha256"] == "f" * 64
    assert plan["historical_library_sha256"] == "e" * 64
    assert plan["verified_production_source_hash"] == "b" * 64
    assert plan["reproduction_source_revision"] == "c" * 40
    assert plan["reproduction_library_build_revision"] is None
    assert plan["caller_declared_build_source"] == {"revision": "c" * 40}
    assert "e" * 64 not in plan["command"]
    assert historical == unchanged_historical


def small_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payloads = {
        "run.py": b"# frozen harness\n",
        "inputs.json": b"{}\n",
        "reproduce.py": b"# verified launcher\n",
        "new_campaign.py": b"# new outer orchestration\n",
        "reproduction-sources.json": b'{"sources": []}\n',
    }
    payloads["provenance.json"] = json.dumps(
        {
            "harness_sha256": hashlib.sha256(payloads["run.py"]).hexdigest(),
            "fixture_sha256": hashlib.sha256(payloads["inputs.json"]).hexdigest(),
        }
    ).encode()
    entries = []
    for name, data in payloads.items():
        (bundle / name).write_bytes(data)
        entries.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (bundle / "publication.json").write_text(json.dumps({"files": entries}))
    return bundle


@pytest.mark.parametrize("name", ["reproduction-sources.json", "run.py", "inputs.json"])
def test_unbound_mapping_or_source_mutation_rejected(
    replay: dict, tmp_path: Path, name: str
) -> None:
    bundle = small_bundle(tmp_path)
    replay["verify_publication"](bundle)
    with (bundle / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        replay["verify_publication"](bundle)


def test_rehashing_a_changed_frozen_harness_is_not_historical_equivalence(
    replay: dict, tmp_path: Path
) -> None:
    bundle = small_bundle(tmp_path)
    replacement = b"# changed harness\n"
    (bundle / "run.py").write_bytes(replacement)
    manifest = json.loads((bundle / "publication.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == "run.py":
            entry.update(
                bytes=len(replacement), sha256=hashlib.sha256(replacement).hexdigest()
            )
    (bundle / "publication.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="historical harness changed"):
        replay["verify_publication"](bundle)
