"""Independent dftd4 reference for a stable complete PBE resource-planning task."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import dftd4
import numpy as np
from dftd4.interface import DampingParam, DispersionModel

numbers = np.array([8, 1, 1], dtype=np.int32)
positions = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]])
parameters = {
    "s6": 1.0,
    "s8": 0.95948085,
    "s9": 1.0,
    "a1": 0.38574991,
    "a2": 4.80688534,
    "alp": 16.0,
}
model = DispersionModel(
    numbers, positions, charge=0.0, model="d4", ga=3.0, gc=2.0, wf=6.0
)
model.set_realspace_cutoff(disp2=60.0, disp3=40.0, cn=30.0)
result = model.get_dispersion(DampingParam(**parameters), grad=True)
package = Path(dftd4.__file__).parent
files = sorted(
    p
    for p in package.parent.rglob("*.so*")
    if p.is_file() and ("dftd4" in str(p) or "dftd4.libs" in str(p))
)
receipt = {
    "oracle": "independent dftd4 Python API",
    "version": version("dftd4"),
    "numbers": numbers.tolist(),
    "positions": positions.tolist(),
    "total_charge": 0.0,
    "parameters": parameters,
    "charge_scaling": {"ga": 3.0, "gc": 2.0, "wf": 6.0},
    "cutoffs_bohr": {"disp2": 60.0, "disp3": 40.0, "cn": 30.0},
    "energy": float(result["energy"]),
    "gradient": result["gradient"].tolist(),
    "units": {"positions": "bohr", "energy": "hartree", "gradient": "hartree/bohr"},
    "input_science": "PBE D4(BJ)-EEQ with ATM; no GenerativeQC scientific computation",
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "oracle_binaries": [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in files
    ],
}
print(json.dumps(receipt, indent=2))
