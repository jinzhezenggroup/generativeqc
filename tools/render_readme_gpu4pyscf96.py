"""Supplement only the two missing GPU4PySCF water-96 timing populations.

Native observations, their historical accuracy labels and every other reference
population stay with their original publications. No electronic calculation is
performed by this evidence reader or renderer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import runpy
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools.render_readme_dft import collect_point

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "benchmarks/results"
PUBLICATION = RESULTS / "readme-gpu4pyscf96-20261018"
SOURCE_COMMIT = "69c4549f260e8c46cdd69f2f59e09b0a1f7d9c4a"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def retain_observations(root: Path) -> None:
    """Losslessly archive just the requested references and their scheduler receipts.

    Failed or partial runs cannot replace a published point. Logs and allocation
    receipts are provenance, never additional timing samples or GPU calculations.
    """
    manifest = {
        "schema": "generativeqc.readme-gpu4pyscf96.v1",
        "benchmark_source_commit": SOURCE_COMMIT,
        "scope": "GPU4PySCF wb97m-v/def2-SVP water-96 Direct and mixed-RSH DF only",
        "process_timeout_seconds": None,
        "launcher_sha256": hashlib.sha256(
            (root / "reproduce.sh").read_bytes()
        ).hexdigest(),
        "basis_file_sha256": hashlib.sha256(
            (root / "inputs/def2-svp.json").read_bytes()
        ).hexdigest(),
        "populations": {},
    }
    for mode, label in (("none", "direct"), ("cuda", "df")):
        directory = root / "measurements" / mode / "wb97m-v/96"
        update = collect_point(directory.parent, 96)
        reference = update["engines"]["reference"]
        if reference["status"] != "measured":
            raise ValueError(f"unfinished or unqualified reference: {mode}")
        payload = (directory / "reference.json").read_bytes()
        filename = f"{label}-reference.json.xz"
        (PUBLICATION / filename).write_bytes(lzma.compress(payload, preset=9))
        job = reference["scheduler"]["SLURM_JOB_ID"]
        manifest["populations"][mode] = {
            "file": filename,
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "outcome": reference["outcome"],
            "slurm_job_id": job,
            "allocation_receipt": (root / "receipts" / f"{job}.job.txt").read_text(),
            "execution_receipt": (root / f"{label}.log").read_text(),
        }
    (PUBLICATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def overlay_reference(
    summary: dict[str, Any], update: dict[str, Any]
) -> dict[str, Any]:
    """Replace one reference, never retroactively requalifying native samples.

    The retained DF/96 protocol includes a native-only workspace allowance.
    GPU4PySCF does not consume that option; every scientific field must match.
    Historical oracle arrays still belong to the historical native gates. The
    new reference's complete arrays are separately retained in its raw archive.
    """
    result = copy.deepcopy(summary)
    point = next(
        point for point in result["methods"]["wb97m-v"] if point["atoms"] == 96
    )
    historical_protocol = copy.deepcopy(point["protocol"])
    reference_protocol = copy.deepcopy(update["protocol"])
    for protocol in (historical_protocol, reference_protocol):
        protocol.pop("density_fitting_memory_budget_bytes", None)
    if historical_protocol != reference_protocol or update["atoms"] != 96:
        raise ValueError("GPU4PySCF/96 scientific protocol differs")
    reference = update["engines"]["reference"]
    if (
        reference["status"] != "measured"
        or Counter(row["phase"] for row in reference["records"]) != PHASE_COUNTS
        or reference["outcome"]["exit_code"] != 0
    ):
        raise ValueError("GPU4PySCF/96 requires all twelve qualified E+F calls")
    if reference["outcome"].get("time_limit_seconds") is not None:
        raise ValueError("GPU4PySCF/96 must not have a process deadline")
    point["engines"]["reference"] = copy.deepcopy(reference)
    if result["density_fitting"]:
        result.setdefault("panel_annotations", {})["wb97m-v"] = (
            "native DF/48,96: accuracy unchecked\n96: native DF 12 GiB; forces 4/4 GiB"
        )
    result["evidence_note"] = (
        "Only GPU4PySCF ωB97M-V/96 is newly measured; see the "
        "[supplement](README.md). All native samples, historical native gates "
        "and other references retain their original publications."
    )
    return result


def read_reference(mode: str) -> dict[str, Any]:
    """Replay the shared scientific and completeness gates on retained raw arrays."""
    manifest = json.loads((PUBLICATION / "manifest.json").read_text())
    observation = manifest["populations"][mode]
    payload = lzma.decompress((PUBLICATION / observation["file"]).read_bytes())
    if hashlib.sha256(payload).hexdigest() != observation["raw_sha256"]:
        raise ValueError("GPU4PySCF/96 raw evidence checksum differs")
    raw = json.loads(payload)
    if raw["engine"] != "reference" or raw["protocol"]["atoms"] != 96:
        raise ValueError("GPU4PySCF/96 observation has a different owner or case")
    if raw["protocol"]["density_fitting"] != (mode == "cuda"):
        raise ValueError("GPU4PySCF/96 Direct and DF evidence cannot be mixed")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "96"
        directory.mkdir()
        (directory / "reference.json").write_bytes(payload)
        (directory / "reference.outcome").write_text(json.dumps(observation["outcome"]))
        return collect_point(Path(temporary), 96)


def combined_summaries() -> dict[str, dict[str, Any]]:
    """Reconstruct the current mixed-source homepage, changing two engines only."""
    from tools.render_readme_r2scan3c import combined_summary

    previous = runpy.run_path(str(RESULTS / "readme-refresh-20261015/refresh.py"))
    return {
        "none": overlay_reference(combined_summary(), read_reference("none")),
        "cuda": overlay_reference(
            previous["combined_summary"]("df-summary.json.xz"), read_reference("cuda")
        ),
    }


def main() -> None:
    """Audit without a GPU and optionally regenerate both homepage assets/tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, help="Retain the two completed raw populations"
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.raw:
        retain_observations(args.raw)
    summaries = combined_summaries()
    if args.render:
        previous = runpy.run_path(str(RESULTS / "readme-wb97mv-20261014/refresh.py"))
        from tools.render_readme_r2scan3c import render_publication

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            render_publication(summaries["none"], destination)
            previous["render_publication"](summaries["cuda"], destination)
            for source, target in (
                ("dft-cold-warm.svg", "assets/readme-dft-performance.svg"),
                ("df-cold-warm.svg", "assets/readme-dft-df-performance.svg"),
                ("timings.md", str(PUBLICATION / "timings.md")),
                ("df-timings.md", str(PUBLICATION / "df-timings.md")),
            ):
                shutil.copyfile(destination / source, REPOSITORY / target)
    print(json.dumps({"updated_reference_atoms": 96, "complete_ef_calls": 24}))


if __name__ == "__main__":
    main()
