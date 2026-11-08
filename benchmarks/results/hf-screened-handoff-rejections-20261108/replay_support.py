"""Load correlated HF handoff evidence without weakening per-record integrity."""

import hashlib
import json
import lzma
from pathlib import Path


def load_observations(directory: Path, cohort: dict) -> dict[str, str]:
    """Authenticate both semantic streams and the complete original population."""
    files = {}
    members = None
    for entry in cohort["capsules"]:
        payload = (directory / entry["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Changed evidence stream: " + entry["path"])
        decoded = json.loads(lzma.decompress(payload))
        if "integrity" in decoded:
            if members is not None:
                raise ValueError("Duplicate integrity manifest")
            members = decoded["integrity"][cohort["capsule_namespace"]]
        stream = decoded[cohort["capsule_namespace"]]
        if files.keys() & stream.keys():
            raise ValueError("Duplicate evidence records across semantic streams")
        files.update(stream)
    if members is None or len(members) != cohort["member_count"]:
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
