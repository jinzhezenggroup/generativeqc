"""Select only unsuccessful first-round CUDA endpoints and retain their diagnoses."""

import argparse
import json
from pathlib import Path

from replay import failure_reason, load_journal, sha
from run_gpu import ATOM_WEIGHT, RECIPES, ROOT


def build_plan(data: dict, first: dict[tuple[str, str], dict]) -> list[dict]:
    expected = {(system, recipe) for system in data["systems"] for recipe in RECIPES}
    if first.keys() != expected:
        raise ValueError("first-round journals must cover the full endpoint population")
    points = []
    for (system, recipe), row in first.items():
        if row["status"] == "ok":
            continue
        points.append(
            {
                "system": system,
                "recipe": recipe,
                "round1_diagnosis": failure_reason(row),
                "round1_error": row.get("error"),
            }
        )
    points.sort(
        key=lambda point: (
            -RECIPES[point["recipe"]][1]
            * sum(
                ATOM_WEIGHT[symbol]
                for symbol, _ in data["systems"][point["system"]]["angstrom"]
            ),
            point["recipe"],
            point["system"],
        )
    )
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = ROOT / "inputs.json"
    raw = sorted((ROOT / "raw").glob("gpu-*.jsonl.xz"))
    first = load_journal(raw, "recipe")
    data = json.loads(inputs.read_text())
    plan = {
        "inputs_sha256": sha(inputs),
        "round1_journal_sha256": {path.name: sha(path) for path in raw},
        "points": build_plan(data, first),
    }
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
