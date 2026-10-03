"""Keep incremental-KS negative evidence and work units independently auditable."""

import gzip
import json
import runpy
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "benchmarks/results/pbe0-incremental-ks-20261004"


def test_incremental_endpoint_and_work_publication() -> None:
    manifest = json.loads((PUBLICATION / "publication.json").read_text())
    files = {
        entry["path"]: (PUBLICATION / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    verifier = runpy.run_path(str(PUBLICATION / "verify.py"))
    summary = verifier["verify"]()
    assert summary["endpoint_comparisons"] == 720
    assert summary["work_energy_comparisons"] == 36
    assert manifest["decision"]["status"] == "inconclusive"
    evidence = json.loads(gzip.decompress(files["validation.json.gz"]))
    assert not evidence["settings"]["default_promoted"]
    assert not evidence["settings"]["latest_source_large_EF_qualified"]
    assert not evidence["settings"]["broad_uks_qualified"]
    work = json.loads(gzip.decompress(files["observer-work.json.gz"]))
    assert work["summary"]["unavailable"]["primitive_recurrences"] is None
    assert work["cancelled_eager_harness"]["scf_iteration_records_observed"] == 0
    assert all(len(run["rows"]) > 1 for run in work["runs"].values())
