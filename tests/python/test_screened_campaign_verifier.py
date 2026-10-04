"""Publication gates must survive Python optimization and coherent bad evidence."""

import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/results/pbe0-screened-pages-20261003"


@pytest.mark.parametrize("fault", (None, "archive_hash", "qualification", "energy"))
def test_optimized_campaign_verifier_rejects_bad_evidence(
    tmp_path: Path, fault: str | None
) -> None:
    """Run the real verifier under -O, preserving hashes past the targeted gate.

    The numerical counterexample updates the record and summary hashes, so a
    digest mismatch cannot accidentally stand in for the scientific rejection.
    Everything modified here is a temporary copy of the retained publication.
    """
    storage = json.loads((DIRECTORY / "storage.json").read_text())
    members = json.loads(
        lzma.decompress((DIRECTORY / "corrected-campaign.json.xz").read_bytes())
    )
    storage["reference_root"] = str((DIRECTORY / storage["reference_root"]).resolve())
    if fault == "qualification":
        qualification = json.loads(members["native-qualification.json"])
        qualification["status"] = "failed"
        members["native-qualification.json"] = json.dumps(qualification)
    elif fault == "energy":
        record = json.loads(members["baseline/3/native.json"])
        record["records"][0]["energy"] += 0.01
        members["baseline/3/native.json"] = json.dumps(record)
        summary = json.loads(members["summary.json"])
        case = next(case for case in summary["cases"] if case["atoms"] == 3)
        case["variants"]["baseline"]["sha256"] = hashlib.sha256(
            members["baseline/3/native.json"].encode()
        ).hexdigest()
        members["summary.json"] = json.dumps(summary)
    raw = json.dumps(members).encode()
    compressed = lzma.compress(raw)
    storage["sha256"] = hashlib.sha256(compressed).hexdigest()
    storage["uncompressed_sha256"] = hashlib.sha256(raw).hexdigest()
    storage["members"] = {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in members.items()
    }
    if fault == "archive_hash":
        storage["sha256"] = "0" * 64
    (tmp_path / "storage.json").write_text(json.dumps(storage))
    (tmp_path / "corrected-campaign.json.xz").write_bytes(compressed)
    # Import from the checkout for its real dependencies, then point only the
    # evidence directory at the temporary copy. No native/GPU work is executed.
    driver = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("campaign_verifier", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.__file__ = sys.argv[2]
module.main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            driver,
            str(DIRECTORY / "verify.py"),
            str(tmp_path / "verify.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if fault is None:
        assert result.returncode == 0, result.stderr
        assert "PASS: 144 corrected endpoints" in result.stdout
    else:
        expected = {
            "archive_hash": "compressed archive SHA-256 mismatch",
            "qualification": "native qualification did not pass",
            "energy": "independent energy/force acceptance failed",
        }[fault]
        assert result.returncode != 0
        assert expected in result.stderr
        assert "PASS:" not in result.stdout
