"""Freeze eight independent displaced geometries from committed ENERGY inputs."""

import hashlib
import json
import lzma
import subprocess
from pathlib import Path

import numpy as np

BASE = "334e7f489870dccc22ae8864c01f2353aa9689a4"
ENERGY = "benchmarks/results/cbs-independent-energy-20261023"


def main() -> None:
    raw = subprocess.check_output(["git", "show", f"{BASE}:{ENERGY}/inputs.json"])
    original = json.loads(raw)
    result = {
        "schema": "generativeqc.independent-cbs-displacements.v1",
        "baseline": BASE,
        "central_inputs_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": original["dataset_sha256"],
        "bases": original["bases"],
        "protocols": {str(c): original["protocols"][str(c)] for c in (4, 7)},
        "direction_seed": 709,
        "steps_bohr": [1e-4, 3e-5],
        "tasks": {},
    }
    for carbon in (4, 7):
        atoms = result["protocols"][str(carbon)]["atoms_bohr"]
        coordinates = np.array([a[1] for a in atoms], dtype=np.float64)
        direction = np.random.default_rng(709).normal(size=coordinates.shape)
        direction /= np.linalg.norm(direction)
        for step_index, step in enumerate(result["steps_bohr"]):
            for sign in (-1, 1):
                key = f"C{carbon}-h{step_index}-{'minus' if sign < 0 else 'plus'}"
                result["tasks"][key] = {
                    "carbon": carbon,
                    "step_bohr": step,
                    "sign": sign,
                    "direction": direction.tolist(),
                    "atoms_bohr": [
                        [a[0], r.tolist()]
                        for a, r in zip(
                            atoms, coordinates + sign * step * direction, strict=True
                        )
                    ],
                }
    destination = Path(__file__).with_name("inputs.json.xz")
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(lzma.compress(raw, preset=9))


if __name__ == "__main__":
    main()
