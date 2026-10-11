"""Offline independent-force gates for the source-matched replay extension."""

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
BUNDLE = ROOT / "benchmarks/results/df-cc-force-batched-replay-20261012"


def test_force_replay_publication_preserves_exact_receipts_and_scope() -> None:
    """A scoped ABBA observation is not statistical or broad force promotion."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        row["path"]: (BUNDLE / row["path"]).read_bytes() for row in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["settings"]["scope"] == "cuda-df-complete-analytic-forces"
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


def test_force_replay_samples_keep_fresh_physics_and_actual_work() -> None:
    """Recompute all physical work and complete medians, not replay-only wins."""
    evidence = load_publication_record(BUNDLE)
    campaign = evidence["retained_samples"]
    samples = campaign["records"]
    assert [row["arm"] for row in samples] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    fields = {
        "df_auxiliary_slices": "ccsd_q_slices",
        "df_virtual_operations": "ccsd_q_operations",
        "df_accumulation_calls": "ccsd_accumulation_calls",
        "df_contraction_terms": "ccsd_contraction_terms",
        "df_gemm_calls": "ccsd_gemm_calls",
        "df_gemm_summands": "ccsd_gemm_summands",
        "df_packing_bytes": "ccsd_packing_bytes",
        "df_auxiliary_tiles": "ccsd_q_tiles",
        "df_accumulation_bytes": "ccsd_accumulation_bytes",
    }
    for row in samples:
        result = row["endpoint"]
        work = evidence["actual_graph_work"][row["arm"]]
        assert row["science"]["passed"] and all(row["science"]["gates"].values())
        assert max(row["science"]["direction_errors"]) <= 3e-7
        assert result["ccsd_evaluations"] == result["ccsd_pair_evaluations"] == 38
        assert result["ccsd_q_batch_size"] == 8
        assert result["lambda_actions"] == 22 and result["z_operator_actions"] == 13
        for native, published in fields.items():
            expected = (
                result["ccsd_evaluations"] * work["paired_primal"][native]
                + work["physical_replay"][native]
            )
            assert result[published] == expected
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


def test_force_replay_historical_source_reconstructs_exactly(tmp_path: Path) -> None:
    """Future owner changes must not invalidate the historical compiled overlay."""
    evidence = load_publication_record(BUNDLE)
    raw = json.loads(lzma.decompress((BUNDLE / "raw-receipts.json.xz").read_bytes()))
    sources = evidence["source_reconstruction"]["compiled_sources"]
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[f"baseline-source/{name}"]["data"].encode())
    patch = gzip.decompress((BUNDLE / "production-source.patch.gz").read_bytes())
    subprocess.run(
        ["git", "apply", "-"],
        cwd=tmp_path,
        input=patch,
        check=True,
        capture_output=True,
    )
    for name, expected in sources.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
