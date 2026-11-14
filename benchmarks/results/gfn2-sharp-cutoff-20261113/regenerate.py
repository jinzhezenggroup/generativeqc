"""Regenerate fresh, model-matched references for the retained campaign inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import lzma
from pathlib import Path

import numpy as np
from tblite import _libtblite
from tblite.interface import Calculator, symbols_to_numbers


def regenerate(output: Path) -> None:
    """Write a new oracle; never overwrite the archived smoothed or sharp results."""
    if output.exists():
        raise ValueError("oracle output already exists")
    if importlib.metadata.version("tblite") != "0.6.0":
        raise ValueError("this campaign requires pinned tblite 0.6.0")
    root = Path(__file__).resolve().parent
    inputs = lzma.decompress((root / "inputs.json.xz").read_bytes())
    report = {
        "status": "running",
        "oracle": "tblite 0.6.0",
        "oracle_library_sha256": hashlib.sha256(
            Path(_libtblite.__file__).read_bytes()
        ).hexdigest(),
        "xtb_config": {"smooth_cutoff": 0.0},
        "accuracy": 1e-4,
        "temperature_hartree": 300 * 3.166808578545117e-6,
        "input_sha256": hashlib.sha256(inputs).hexdigest(),
        "rows": [],
    }
    for case in json.loads(inputs)["cases"]:
        if case["name"] == "water-512":
            continue
        for geometry, coordinates in enumerate(case["geometries"]):
            calculator = Calculator(
                "GFN2-xTB",
                np.asarray(symbols_to_numbers(case["symbols"])),
                np.asarray(coordinates),
                charge=case["charge"],
                uhf=case["multiplicity"] - 1,
                xtb_config=report["xtb_config"],
            )
            for name, value in (
                ("accuracy", report["accuracy"]),
                ("max-iter", 300),
                ("temperature", report["temperature_hartree"]),
                ("mixer-damping", 0.4),
                ("verbosity", 0),
            ):
                calculator.set(name, value)
            result = calculator.singlepoint()
            energy = float(result.get("energy"))
            forces = -np.asarray(result.get("gradient"))
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError("independent oracle returned a nonfinite result")
            report["rows"].append(
                {
                    "case": case["name"],
                    "geometry": geometry,
                    "geometry_sha256": case["geometry_sha256"][geometry],
                    "energy": energy,
                    "forces": forces.tolist(),
                    "charges": result.get("charges").tolist(),
                }
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "measured"
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    regenerate(parser.parse_args().output)
