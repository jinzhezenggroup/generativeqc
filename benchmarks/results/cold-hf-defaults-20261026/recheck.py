"""Recheck retained original endpoint bytes and independent HF qualifications."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

from benchmarks.readme_defaults_qualification import qualify


def main() -> None:
    """Audit historical evidence without a GPU or current-source fingerprints."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    summary = json.loads(lzma.decompress((directory / "summary.json.xz").read_bytes()))
    payloads = json.loads(lzma.decompress((directory / "receipts.json.xz").read_bytes()))
    for name, expected in summary["references"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed independent reference: {name}")
    reference = directory.parent / "readme-final-20261022"
    oracles = json.loads(lzma.decompress((reference / "oracles.json.xz").read_bytes()))
    reuse = json.loads((reference / "reference-reuse.json").read_text())
    calls = 0
    for population in [
        *summary["populations"],
        *summary["supplemental"],
        summary["original"],
    ]:
        payload = payloads[population["file"]].encode()
        if hashlib.sha256(payload).hexdigest() != population["sha256"]:
            raise ValueError(f"Changed endpoint bytes: {population['file']}")
        receipt = json.loads(payload)
        actual = qualify([receipt], oracles, reference_reuse=reuse)[0]
        if actual != population["qualification"]:
            raise ValueError(f"Changed qualification: {population['file']}")
        if population in summary["populations"]:
            if actual["status"] != "qualified":
                raise ValueError("Production HF population failed common qualification")
            for sample in receipt["samples"]:
                calls += 1
                if receipt["arm"] == "standard":
                    work = sample["precision"]
                    if (
                        not work["operator_work_counters_valid"]
                        or work["mixed_stage_fock_builds"] != 0
                        or work["strict_stage_fock_builds"] != sample["iterations"]
                        or sample["fock_builds"]
                        != work["strict_stage_fock_builds"] + work["post_scf_fock_builds"]
                    ):
                        raise ValueError("Incomplete production FP64 work receipt")
    if calls != 24:
        raise ValueError(f"Expected 24 main complete calls, found {calls}")
    print("Rechecked 24 main calls and all retained supplemental qualifications.")


if __name__ == "__main__":
    main()
