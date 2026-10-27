"""Compact identities must preserve downstream production-equivalence checks."""

import hashlib
import json
import lzma
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
XC = ROOT / "benchmarks/results/xc-bounded-cross-20261024/provenance.json.xz"
FAMILIES = ("df-charge-partitions-20261024", "fitted-range-cold-20261024")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mutation", ("delta", "path", "production", "driver"))
def test_changed_source_identity_rejected(
    monkeypatch: pytest.MonkeyPatch, family: str, mutation: str
) -> None:
    directory = ROOT / "benchmarks/results" / family
    target = directory / "provenance.json.xz"
    provenance = json.loads(lzma.decompress(target.read_bytes()))
    previous = json.loads(lzma.decompress(XC.read_bytes()))
    base = previous["source_identities"]["guarded"]
    name = (
        "stage25-test-source-inputs-v2.json"
        if family.startswith("df-")
        else "stage26-test-source-inputs-v2.json"
    )
    delta = provenance["source_manifest_deltas"][name]
    if mutation == "delta":
        delta["manifest_sha256"] = "0" * 64
    elif mutation == "path":
        delta["changed"]["src/unretained.cpp"] = "0" * 64
    elif mutation == "production":
        delta["changed"]["src/dft/cuda_ks.cpp"] = "0" * 64
    else:
        base["downstream_base_files"]["benchmarks/readme_defaults.py"] = "0" * 64
    # Reach the path/production checks even if a changed delta is rehashed.
    if mutation in {"path", "production"}:
        base["downstream_delta_sha256"][name] = hashlib.sha256(
            json.dumps(delta, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        previous["source_identities_sha256"] = hashlib.sha256(
            json.dumps(
                previous["source_identities"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    altered = lzma.compress(json.dumps(provenance).encode())
    altered_xc = lzma.compress(json.dumps(previous).encode())
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            altered if path == target else altered_xc if path == XC else original(path)
        ),
    )
    with pytest.raises(AssertionError):
        runpy.run_path(str(directory / "reproduce.py"))
