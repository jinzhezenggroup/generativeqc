"""Authenticate and independently recompute the retained CUDA schedule proof."""

import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/pbe0-cuda-schedule-20261006"


def _verify(
    directory: Path, output: Path, *, integration: bool = False
) -> subprocess.CompletedProcess[str]:
    """Use the same offline consumer without native libraries or a GPU."""
    return subprocess.run(
        [
            sys.executable,
            str(BUNDLE / "verify.py"),
            str(directory),
            *(["--integration"] if integration else []),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize("integration", [False, True])
def test_publication_recomputes_complete_endpoint_gates(
    tmp_path: Path, integration: bool
) -> None:
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    output = tmp_path / "qualified.json"
    completed = _verify(BUNDLE, output, integration=integration)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text()) == json.loads(
        (
            BUNDLE
            / (
                "integration-qualification.json"
                if integration
                else "qualification.json"
            )
        ).read_text()
    )


@pytest.mark.parametrize(
    "field", ["forces", "iterations", "grid_pair_visits", "runtime_artifact"]
)
@pytest.mark.parametrize("integration", [False, True])
def test_rehashed_bad_samples_cannot_reuse_passing_summary(
    tmp_path: Path, field: str, integration: bool
) -> None:
    """Updating storage checksums must not turn wrong work/vectors into evidence."""
    prefix = "integration-" if integration else ""
    manifest = json.loads((BUNDLE / f"{prefix}manifest.json").read_text())
    payload = json.loads(
        lzma.decompress((BUNDLE / f"{prefix}bundle.json.xz").read_bytes())
    )
    name = "campaign/48-0-adaptive512.json"
    item = payload["artifacts"][name]
    record = json.loads(item["text"])
    if field == "forces":
        record["records"][0]["forces"][0][0] += 1e-4
    elif field == "iterations":
        record["records"][1]["iterations"] += 1
    elif field == "grid_pair_visits":
        record["cuda_schedule_force_work"][0]["grid_pair_visits"] //= 2
    else:
        record["cuda_schedule_force_work"][0]["artifacts"][0]["binary_sha256"] = (
            "0" * 64
        )
    item["text"] = json.dumps(record)
    data = item["text"].encode()
    item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest["artifacts"][name] = {key: item[key] for key in ("bytes", "sha256")}
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    compressed = lzma.compress(raw)
    manifest.update(
        bundle_sha256=hashlib.sha256(compressed).hexdigest(),
        bundle_bytes=len(compressed),
        decoded_bytes=len(raw),
    )
    (tmp_path / f"{prefix}bundle.json.xz").write_bytes(compressed)
    (tmp_path / f"{prefix}manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "forged-qualification.json"
    completed = _verify(tmp_path, output, integration=integration)
    assert completed.returncode != 0
    assert not output.exists()
