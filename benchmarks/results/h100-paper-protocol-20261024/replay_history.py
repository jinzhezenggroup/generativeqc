"""Recheck selected old observables offline; no new reference/device calculation."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import protocol as p


def main() -> None:
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE", "") not in ("", "0"):
        raise RuntimeError("Historical replayers require unoptimized Python")
    sys.path[:0] = [str(p.ROOT), str(p.ROOT / "python")]
    from benchmarks.readme_defaults_qualification import qualify

    manifest = json.loads((p.HERE / "manifest.json").read_text())
    p.validate_manifest(manifest)
    records = []
    targets = {
        (c["input"]["atoms"], c["input"]["method"], c["input"]["mode"])
        for c in manifest["cells"]
        if c["family"] == "hf_dft"
    }
    for arm in ("standard", "fast", "gpu4pyscf"):
        records.extend(
            r
            for r in p.decoded(p.ARCHIVE, p.HF_DIR + arm + ".json.xz")
            if (r["atoms"], r["method"], r["mode"]) in targets
        )
    hf = qualify(
        records,
        p.decoded(p.ARCHIVE, p.HF_DIR + "oracles.json.xz"),
        reference_reuse=p.decoded(p.ARCHIVE, p.HF_DIR + "reference-reuse.json"),
    )
    counts = {
        s: sum(r["status"] == s for r in hf)
        for s in ("qualified", "failed", "unverified")
    }
    p.require(
        counts == {"qualified": 24, "failed": 0, "unverified": 0},
        "historical HF/DFT replay failed",
    )
    verifier = "benchmarks/results/cbs-independent-forces-20261024/verify.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(p.ROOT), str(p.ROOT / "python")])
    result = subprocess.run(
        [sys.executable, str(p.ROOT / verifier), "--revision", p.DIRECTION],
        cwd=p.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    print(
        json.dumps(
            {
                "schema": "generativeqc.h100-paper-history-replay.v1",
                "hf_dft": counts,
                "hf_populations": len(records),
                "hf_samples": sum(len(r["samples"]) for r in records),
                "direction_stdout": result.stdout.strip(),
                "direction_revision": p.DIRECTION,
                "source_authentication": p.authenticate_sources(manifest),
                "reference_inventory": p.reference_inventory(manifest),
                "device_science": "UNAVAILABLE",
                "new_scientific_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
