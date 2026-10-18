"""Recompute every primary endpoint gate from the lossless published arrays."""

import json
import lzma
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

root = Path(__file__).resolve().parent
index = json.loads(lzma.decompress((root / "numeric-index.json.xz").read_bytes()))
with lzma.open(root / "numeric-arrays.npy.xz", "rb") as stream:
    packed = np.load(stream, allow_pickle=False)


def array(name: str) -> NDArray[np.float64]:
    """Read a named, bounded slice without applying any numeric transformation."""
    layout = index["arrays"][name]
    start = layout["offset"]
    return packed[start : start + layout["size"]].reshape(layout["shape"])


count = 0
for case, data in index["cases"].items():
    energy_reference = array(f"{case}/reference/energy")
    force_reference = array(f"{case}/reference/forces")
    for arm, rows in data["arms"].items():
        energies = array(f"{case}/{arm}/energy")
        forces = array(f"{case}/{arm}/forces")
        assert len(rows) == 12
        for position, row in enumerate(rows):
            geometry = row["geometry"]
            energy_error = abs(energies[position] - energy_reference[geometry])
            force_error = np.max(np.abs(forces[position] - force_reference[geometry]))
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert row["gate"] and row["converged"]
            assert row["fock_builds"] > 0 and row["iterations"] > 0
            assert energy_error == row["energy_error"]
            assert force_error == row["force_error"]
            count += 1
    if "--restore-references" in sys.argv:
        method, atoms = case.rsplit("-", 1)
        reference = dict(data["reference_metadata"], protocol=data["protocol"])
        reference["records"] = []
        for metadata in data["reference_records"]:
            geometry = metadata["geometry"]
            reference["records"].append(
                dict(
                    metadata,
                    energy=float(energy_reference[geometry]),
                    forces=force_reference[geometry].tolist(),
                )
            )
        path = root / "references" / "none" / method / atoms / "reference.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reference, indent=2) + "\n")
print(f"All {count} endpoint gates independently recomputed from published arrays.")
