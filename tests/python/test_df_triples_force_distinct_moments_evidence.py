"""Independent-force and exact work gates for forward-only occupied seed reuse."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import lzma
import statistics
import subprocess
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-triples-force-distinct-moments-20261012"


def test_force_distinct_publication_retains_scope_and_exact_bytes() -> None:
    """No predecessor timing or statistical/global claim is inherited."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        row["path"]: (BUNDLE / row["path"]).read_bytes() for row in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    raw = json.loads(lzma.decompress(files["raw-receipts.json.xz"]))
    for row in raw.values():
        data = (
            row["data"].encode()
            if row["encoding"] == "utf-8"
            else base64.b64decode(row["data"])
        )
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert any(name.endswith("candidate-triples.xml") for name in raw)


def test_force_distinct_samples_reconcile_full_endpoint_and_copy_work() -> None:
    """Independent cotangents are unchanged while all copied forward slots count."""
    evidence = load_publication_record(BUNDLE)
    campaign = evidence["retained_samples"]
    samples = campaign["records"]
    assert [row["arm"] for row in samples] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    occupied, virtuals, auxiliary = 9, 221, 488
    pairs = occupied * (occupied + 1) // 2
    cubes = pairs * occupied
    seeds, copies = 1_953, 477
    saved_work = copies * virtuals**3 * (virtuals + occupied)
    baseline = samples[0]["endpoint"]
    for row in samples:
        actual = row["endpoint"]
        assert row["science"]["passed"] and all(row["science"]["gates"].values())
        assert max(row["science"]["direction_errors"]) <= 3e-7
        assert (
            actual["triples_w_storage_bits"]
            == actual["triples_w_compute_bits"]
            == actual["triples_w_accumulation_bits"]
            == 64
        )
        for name in (
            "numeric_capacity_bytes",
            "ccsd_capacity",
            "ccsd_evaluations",
            "ccsd_contraction_terms",
            "ccsd_gemm_calls",
            "lambda_actions",
            "z_operator_actions",
            "triples_work",
        ):
            assert actual[name] == baseline[name]
        if row["arm"] == "candidate":
            assert actual["fock_response_vector_cubes"] == cubes
            assert actual["fock_response_page_builds"] == pairs
            assert actual["fock_response_w_gemms"] == 2 * seeds
            assert actual["fock_response_w_copy_calls"] == copies
            assert actual["fock_response_w_copy_bytes"] == copies * 8 * virtuals**3
            panel_work = actual["fock_response_panel_gemms"] * auxiliary * virtuals**3
            fock_work = 2 * pairs * (occupied * virtuals**4 + occupied**2 * virtuals**3)
            assert (
                actual["fock_response_work"]
                == panel_work + seeds * virtuals**3 * (virtuals + occupied) + fock_work
            )
            assert (
                baseline["fock_response_work"] - actual["fock_response_work"]
                == saved_work
            )
    medians = {
        arm: statistics.median(
            row["wall_seconds"] for row in samples if row["arm"] == arm
        )
        for arm in ("baseline", "candidate")
    }
    assert medians == campaign["complete_wall_medians"]
    assert (
        100 * (1 - medians["candidate"] / medians["baseline"])
        == campaign["descriptive_wall_saving_percent"]
    )
    assert medians["candidate"] < medians["baseline"]


def test_force_distinct_rebuildable_historical_owner_and_adapter(
    tmp_path: Path,
) -> None:
    """Private result-layout changes require source-matched rebuilt consumers."""
    evidence = load_publication_record(BUNDLE)
    raw = json.loads(lzma.decompress((BUNDLE / "raw-receipts.json.xz").read_bytes()))
    sources = evidence["source_reconstruction"]["compiled_sources"]
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[f"baseline-source/{name}"]["data"].encode())
    subprocess.run(
        ["git", "apply", "-"],
        cwd=tmp_path,
        input=gzip.decompress((BUNDLE / "production-source.patch.gz").read_bytes()),
        check=True,
        capture_output=True,
    )
    for name, expected in sources.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
