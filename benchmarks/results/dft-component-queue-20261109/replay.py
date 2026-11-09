"""Authenticate normal DFT endpoint evidence without GPU, oracle runs or network."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

if not __debug__ or sys.flags.optimize:
    raise RuntimeError("Evidence validation requires assertions; rerun without -O/-OO or PYTHONOPTIMIZE")

from benchmarks._exact_fp64_evidence import load, require

FAMILY = Path(__file__).resolve().parent
ROOT = FAMILY.parents[2]


def materialize(destination: Path, family: Path = FAMILY) -> dict:
    """Recover every authenticated receipt and reproducer into an owned folder."""
    _, files = load(family)
    cohort = json.loads((family / "cohort.json").read_text())
    require(len(files) == cohort["member_count"], "Incomplete retained population")
    for name, text in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe evidence path")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    prototypes = json.loads(files["prototype-receipts.json"])
    require({name: value["archive_sha256"] for name, value in prototypes.items()}
            == cohort["prototype_archives"], "Changed prototype transport provenance")
    shutil.copyfile(family / "qualification.py", destination / "qualification.py")
    return cohort


def qualify_materialized(destination: Path) -> dict:
    """Apply independent science, work, image, dispatch and predeclared timing gates."""
    prototypes = json.loads((destination / "prototype-receipts.json").read_text())
    for prototype in prototypes.values():
        for name, descriptor in prototype["members"].items():
            raw = (destination / name).read_bytes()
            require(len(raw) == descriptor["bytes"] and
                    hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
                    "Prototype receipt qualification failed: " + name)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "python")))
    reports = {}
    for card in ("a100", "5090"):
        output = destination / (card + "-report.json")
        completed = subprocess.run(
            [sys.executable, str(destination / "qualification.py"),
             str(destination / ("native-" + card)), "--card", card, "--output", str(output)],
            cwd=ROOT, env=environment, check=False, capture_output=True, text=True,
        )
        require(completed.returncode == 0, card + " qualification failed: " + completed.stderr)
        report = json.loads(output.read_text())
        require(not report["performance_failures"], card + " performance gate failed: " + str(report["performance_failures"]))
        reports[card] = report
    return reports


def replay(family: Path = FAMILY) -> dict:
    """Qualify this scoped default change, never all-README leadership."""
    with TemporaryDirectory(prefix="dft-component-replay-") as temporary:
        destination = Path(temporary)
        cohort = materialize(destination, family)
        reports = qualify_materialized(destination)
    return {
        "default_promotion_qualified": True,
        "all_readme_leadership_established": False,
        "whole_endpoint_quartet_counters_available": False,
        "scope": cohort["scope"],
        "cards": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.extract:
        args.extract.mkdir(parents=True, exist_ok=False)
        materialize(args.extract)
        print("Authenticated receipts and reproduction sources:", args.extract)
    else:
        result = replay()
        print(json.dumps({
            "default_promotion_qualified": result["default_promotion_qualified"],
            "all_readme_leadership_established": False,
            "cards": {card: {
                name: {key: value[key] for key in ("cold_means", "cold_ratio", "paired_warm_ratios")}
                for name, value in report["populations"].items()
            } for card, report in result["cards"].items()},
        }, indent=2))
