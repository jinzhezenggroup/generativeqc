"""Recheck retained endpoint samples on the CPU; this is not GPU qualification."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record


def verify(directory: Path) -> int:
    """Authenticate the publication and independently recompute all 96 gates."""
    manifest = json.loads((directory / "publication.json").read_text())
    files = {
        entry["path"]: (directory / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)

    def original(name: str) -> bytes:
        """Restore original JSON bytes, including the measured oracle digest."""
        selected = [path for path in files if path in {name, name + ".gz"}]
        if len(selected) != 1:
            raise ValueError(f"ambiguous or missing sample: {name}")
        path = selected[0]
        return gzip.decompress(files[path]) if path.endswith(".gz") else files[path]

    specification = importlib.util.spec_from_file_location(
        "retained_matrix", directory / "verify-matched-matrix.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    expected = load_publication_record(directory, role="summary", name="summary.json")
    receipts = load_publication_record(directory, role="summary", name="receipts.json")
    if receipts["matrix_job_exit"] != 0:
        raise ValueError("matrix job did not complete successfully")
    endpoints = 0
    with tempfile.TemporaryDirectory(prefix="direct-force-evidence-") as temporary:
        campaign = Path(temporary)
        (campaign / "job.exit").write_text(str(receipts["matrix_job_exit"]) + "\n")
        for method in ("hf", "pbe0"):
            for atoms in (48, 96):
                for arm in ("reference", "control", "candidate"):
                    name = f"{method}-{atoms}-{arm}"
                    destination = (
                        campaign / name / "results.json"
                        if method == "hf"
                        else campaign / f"{name}.json"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(original(f"samples/{name}.json"))
                reference_path = (
                    campaign / f"{method}-{atoms}-reference" / "results.json"
                    if method == "hf"
                    else campaign / f"{method}-{atoms}-reference.json"
                )
                reference = json.loads(reference_path.read_text())
                for arm in ("control", "candidate"):
                    payload, _ = module.load_native(campaign, method, atoms, arm)
                    for row in payload["records"]:
                        phase = "moved" if row["phase"].startswith("moved") else "cold"
                        oracle = next(
                            item
                            for item in reference["records"]
                            if item["phase"] == phase
                        )
                        energy_error = abs(row["energy"] - oracle["energy"])
                        force_error = max(
                            abs(actual - expected_value)
                            for actual_vector, reference_vector in zip(
                                row["forces"], oracle["forces"], strict=True
                            )
                            for actual, expected_value in zip(
                                actual_vector, reference_vector, strict=True
                            )
                        )
                        if (
                            not math.isfinite(energy_error + force_error)
                            or energy_error > 1e-8
                            or force_error > 1e-7
                            or energy_error != row["energy_error"]
                            or force_error != row["force_error"]
                        ):
                            raise ValueError(
                                f"independent oracle mismatch: {method}/{atoms}/{arm}"
                            )
                        endpoints += 1
        if module.summarize(campaign) != expected:
            raise ValueError("retained samples do not reproduce the reviewed summary")
    if endpoints != 96:
        raise ValueError("incomplete native endpoint matrix")
    return endpoints


def main() -> None:
    """Review local receipts without CUDA, Slurm or a profiler database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "publication", nargs="?", type=Path, default=Path(__file__).parent
    )
    arguments = parser.parse_args()
    endpoints = verify(arguments.publication)
    print(
        f"Publication authenticated; {endpoints} native oracle gates and all medians verified."
    )


if __name__ == "__main__":
    main()
