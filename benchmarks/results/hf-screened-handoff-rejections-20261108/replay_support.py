"""Load correlated HF handoff evidence without weakening per-record integrity."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks._exact_fp64_evidence import load


def load_observations(directory: Path, cohort: dict) -> dict[str, str]:
    """Recover exact texts and retain the original per-population integrity gate."""
    entry = cohort["storage"]
    folder = (directory / entry["directory"]).resolve()
    if (
        folder.parent != directory.resolve().parent
        or folder.name != "hf-screened-handoff-rejections-20261108"
    ):
        raise ValueError("Invalid shared observation storage")
    manifest = (folder / "sha256.txt").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != entry["sha256"]:
        raise ValueError("changed observation capsule")
    bundle, retained = load(folder)
    namespace = cohort["capsule_namespace"]
    prefix = namespace + "/"
    files = {
        name.removeprefix(prefix): text
        for name, text in retained.items()
        if name.startswith(prefix)
    }
    members = cohort.get("members", bundle["integrity"].get(namespace))
    if members is None:
        members = {
            item["path"].removeprefix(prefix): item["sha256"]
            for item in bundle["files"]
            if item["path"].startswith(prefix)
        }
    count = cohort.get("member_count", cohort.get("retained_text_files", len(members)))
    if len(members) != count:
        raise ValueError("Missing or incomplete integrity manifest")
    if files.keys() != members.keys():
        raise ValueError("Incomplete observation/provenance population")
    for name, text in files.items():
        if hashlib.sha256(text.encode()).hexdigest() != members[name]:
            raise ValueError("Changed evidence record: " + name)
    return files


def compact_summary(summary: dict) -> dict:
    """Summarize every arm without duplicating raw warms or per-sample errors."""
    result = {name: value for name, value in summary.items() if name != "cards"}
    result["cards"] = {}
    for card, measurements in summary["cards"].items():
        compact = {
            name: value for name, value in measurements.items() if name != "arms"
        }
        compact["arms"] = []
        for arm in measurements["arms"]:
            sample = {
                name: value
                for name, value in arm.items()
                if name not in {"independent_errors", "warm_seconds", "initial_guess"}
            }
            if "initial_guess" in arm:
                sample["initial_guess"] = {
                    name: value
                    for name, value in arm["initial_guess"].items()
                    if name
                    not in {"preparation_numeric_capacity", "preparation_seconds"}
                }
            errors = arm["independent_errors"]
            sample["independent"] = {
                "samples": len(errors),
                "passed": sum(entry["passed"] for entry in errors),
                "energy_max": max(entry["energy"] for entry in errors),
                "force_max": max(entry["force"] for entry in errors),
            }
            compact["arms"].append(sample)
        guesses = [arm.get("initial_guess") for arm in compact["arms"]]
        if guesses[0] is not None and all(guess == guesses[0] for guess in guesses):
            compact["initial_guess_all_arms"] = guesses[0]
            for arm in compact["arms"]:
                del arm["initial_guess"]
        result["cards"][card] = compact
    return result
