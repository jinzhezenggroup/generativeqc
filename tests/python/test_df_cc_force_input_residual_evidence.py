"""Offline gates for the source-matched force-consumer qualification."""

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
BUNDLE = ROOT / "benchmarks/results/df-cc-force-input-residual-20261011"


def test_force_input_residual_publication_preserves_scope_and_raw_bytes() -> None:
    """Fresh force qualification must not reinterpret the energy predecessor."""
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
    assert any("loader-attempt0" in name for name in raw)
    assert any("validation-pilot" in name for name in raw)


def test_force_input_residual_samples_preserve_numerics_and_actual_work() -> None:
    """Recompute complete medians and graph-derived counters from raw samples."""
    evidence = load_publication_record(BUNDLE)
    campaign = evidence["retained_samples"]
    samples = campaign["records"]
    assert [row["arm"] for row in samples] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    work = evidence["actual_graph_work"]
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
        assert row["science"]["passed"] and all(row["science"]["gates"].values())
        assert max(row["science"]["direction_errors"]) <= 3e-7
        assert result["ccsd_replay_r1_max"] <= 1e-10
        assert result["ccsd_replay_r2_max"] <= 1e-10
        assert result["ccsd_evaluations"] == (19 if row["arm"] == "candidate" else 38)
        assert result["ccsd_pair_evaluations"] == result["ccsd_evaluations"]
        assert result["numeric_capacity_bytes"] == 19_509_839_897
        assert result["ccsd_capacity"] == 4_392_162_937
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


def test_force_input_residual_historical_source_reconstruction(tmp_path: Path) -> None:
    """Bind the compiled snapshot without freezing unrelated future owner edits."""
    evidence = load_publication_record(BUNDLE)
    raw = json.loads(lzma.decompress((BUNDLE / "raw-receipts.json.xz").read_bytes()))
    source = tmp_path / "src/methods/rccsd_method.cpp"
    source.parent.mkdir(parents=True)
    source.write_bytes(raw["baseline/rccsd_method.cpp"]["data"].encode())
    patch = gzip.decompress((BUNDLE / "production-source.patch.gz").read_bytes())
    # Do not let an in-checkout pytest temp directory discover the parent repo.
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "apply", "-"],
        cwd=tmp_path,
        input=patch,
        check=True,
        capture_output=True,
    )
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == evidence["source_reconstruction"]["compiled_candidate_method_sha256"]
    )
