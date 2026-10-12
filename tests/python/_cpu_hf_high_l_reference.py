"""Fail-closed loading of the independently generated fixed high-l HF oracle."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data/cpu_hf_high_l"
MAX_BYTES = 256 * 1024


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("fixed high-l reference exceeds its byte limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("fixed high-l reference must be a JSON object")
    return raw, value


def _array(record: dict[str, Any], shape: list[int]) -> Any:
    values = record["values"]
    if record["dtype"] != "float64" or record["shape"] != shape:
        raise ValueError("fixed high-l reference array shape/dtype mismatch")
    if shape:
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(not isinstance(row, list) or len(row) != 3 for row in values)
        ):
            raise ValueError("fixed high-l reference forces must have shape (2, 3)")
        flat = [value for row in values for value in row]
    else:
        flat = [values]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in flat
    ):
        raise ValueError("fixed high-l reference contains nonfinite/nonreal data")
    packed = struct.pack(f"<{len(flat)}d", *flat)
    if _digest(packed) != record["sha256_le_f64"]:
        raise ValueError("fixed high-l reference numerical checksum mismatch")
    return values


def load_reference(
    *,
    atoms: list[tuple[str, list[float]]],
    basis: dict[str, Any],
    representation: str,
    method: str,
    charge: int,
    spin: int,
    displacement: float,
    root: Path = REFERENCE_DIR,
) -> tuple[float, list[list[float]]]:
    """Match complete explicit inputs, with no live-PySCF fallback."""
    if representation not in ("cartesian", "spherical"):
        raise ValueError("unknown fixed high-l reference representation")
    _, manifest = _read(root / "manifest.json")
    inputs_raw, inputs = _read(root / "inputs.json")
    receipt_raw, receipt = _read(root / "reference.json")
    if manifest["schema"] != "generativeqc.cpu-hf-high-l.frozen.v1":
        raise ValueError("unsupported fixed high-l reference manifest")
    if _digest(inputs_raw) != manifest["inputs_sha256"]:
        raise ValueError("fixed high-l reference input checksum mismatch")
    if (
        _digest(receipt_raw) != manifest["receipt_sha256"]
        or len(receipt_raw) != manifest["receipt_bytes"]
    ):
        raise ValueError("fixed high-l reference receipt checksum/size mismatch")
    if (
        inputs["schema"] != "generativeqc.cpu-hf-high-l.inputs.v1"
        or inputs["unit"] != "Bohr"
        or inputs["precision"] != "float64"
        or inputs["energy_unit"] != "Hartree"
        or inputs["force_unit"] != "Hartree/Bohr"
        or inputs["force_definition"] != "negative analytic nuclear gradient"
    ):
        raise ValueError("fixed high-l reference scientific units/schema mismatch")
    if receipt["schema"] != "generativeqc.cpu-hf-high-l.receipt.v1":
        raise ValueError("unsupported fixed high-l reference receipt")
    provenance = receipt["provenance"]
    if (
        receipt["inputs"] != inputs
        or provenance["inputs_sha256"] != manifest["inputs_sha256"]
        or provenance["generator_sha256"] != manifest["generation"]["generator_sha256"]
    ):
        raise ValueError("fixed high-l reference provenance/input mismatch")
    if {
        name: package["version"] for name, package in provenance["packages"].items()
    } != inputs["packages"]:
        raise ValueError("fixed high-l reference package provenance mismatch")
    generation = manifest["generation"]
    execution = provenance["execution"]
    if (
        execution["checkout_sha"] != generation["commit"]
        or execution["github_sha"] != generation["commit"]
        or execution["github_repository"] != generation["repository"]
        or execution["github_run_id"] != str(generation["run_id"])
        or execution["github_run_attempt"] != str(generation["attempt"])
        or provenance["python"] != generation["python"]
    ):
        raise ValueError("fixed high-l reference acquisition identity mismatch")
    if receipt["replay"]["receipt_sha256"] != generation["replica_sha256"]:
        raise ValueError("fixed high-l reference replay identity mismatch")
    expected = {
        "atom": [[element, list(position)] for element, position in atoms],
        "basis": basis,
        "unit": "Bohr",
        "method": method,
        "charge": charge,
        "spin": spin,
        "cart": representation == "cartesian",
        "displacement_bohr": displacement,
        "scf": {"init_guess": "1e", "conv_tol": 1e-13, "conv_tol_grad": 1e-11},
    }
    records = receipt["records"]
    if len(records) != 8 or len({record["input"]["id"] for record in records}) != 8:
        raise ValueError("incomplete or duplicated fixed high-l reference inventory")
    matches = []
    for record in records:
        case = record["input"]
        canonical = json.dumps(
            case, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if _digest(canonical) != record["input_sha256"]:
            raise ValueError("fixed high-l reference case checksum mismatch")
        if record["converged"] is not True:
            raise ValueError("unconverged independent fixed high-l reference")
        energy = _array(record["energy"], [])
        forces = _array(record["forces"], [2, 3])
        if {key: value for key, value in case.items() if key != "id"} == expected:
            matches.append((float(energy), forces))
    if len(matches) != 1:
        raise ValueError("missing, ambiguous, or changed fixed high-l scientific input")
    return matches[0]
