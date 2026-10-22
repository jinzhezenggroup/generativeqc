"""Replay targeted convergence acceptance without executing GPU work."""

import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults_qualification import qualify


def main() -> None:
    """Check both solver-only and total-error evidence, preserving known failures."""
    root = Path(__file__).resolve().parent
    raw = (root / "evidence.json.xz").read_bytes()
    summary = json.loads((root / "summary.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == summary["evidence_sha256"]
    evidence = json.loads(lzma.decompress(raw))
    source = evidence["source"]
    assert hashlib.sha256(source["reconstruction_patch"].encode()).hexdigest() == source["reconstruction_patch_sha256"]
    oracles = json.loads(lzma.decompress((root / evidence["oracle_reuse_path"]).read_bytes()))
    reuse = json.loads((root / evidence["reference_reuse_path"]).read_text())
    actual = qualify(evidence["records"], oracles, reference_reuse=reuse)
    assert actual == evidence["qualification"]
    for case in evidence["same_model_oracles"]:
        assert len(case["samples"]) == 4
        for sample in case["samples"]:
            delta = np.asarray(sample["forces"]) - np.asarray(sample["reference_forces"])
            errors = {
                "energy_per_atom": abs(sample["energy"] - sample["reference_energy"]) / len(sample["geometry"]),
                "force_max": float(np.max(np.abs(delta))),
                "force_rms": float(np.sqrt(np.mean(delta**2))),
            }
            assert errors == sample["errors"]
            assert errors["energy_per_atom"] < 2.5e-6
            assert errors["force_max"] < 2.5e-5
            assert errors["force_rms"] < 7.5e-6
    failed = {(q["method"],q["mode"],q["atoms"]) for q in actual if q["status"]=="failed"}
    assert failed == {("hf","df",3),("hf","df",96),("m06-2x","df",3),("m06-2x","direct",3)}
    print(f"{len(actual)} targeted populations: {len(actual)-len(failed)} qualified, {len(failed)} retained failures")
    print(f"{len(evidence['same_model_oracles'])} independent same-model cases passed")


if __name__ == "__main__":
    main()
