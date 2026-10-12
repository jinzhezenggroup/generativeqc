"""Read complete authentic fixed ECP matrices without importing PySCF."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data/ecp_raw_matrices"
MAX_BYTES = 256 * 1024
ARRAY_FIELDS = ("overlap", "normalization", "ecp_raw", "ecp_normalized")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("reference exceeds its byte limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    _canonical(value)
    return value, raw


def _array(record: dict[str, Any], shape: list[int]) -> Any:
    if set(record) != {"dtype", "shape", "values", "sha256_le_f64"}:
        raise ValueError("incomplete array metadata")
    if record["dtype"] != "float64" or record["shape"] != shape:
        raise ValueError("incorrect array type or shape")
    values = record["values"]
    if len(shape) == 2:
        if (
            not isinstance(values, list)
            or len(values) != shape[0]
            or any(not isinstance(row, list) or len(row) != shape[1] for row in values)
        ):
            raise ValueError("ragged matrix values")
        flat = [number for row in values for number in row]
    else:
        if not isinstance(values, list) or len(values) != shape[0]:
            raise ValueError("incorrect vector values")
        flat = values
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        for number in flat
    ):
        raise ValueError("nonfinite or nonnumeric FP64 value")
    if _digest(struct.pack(f"<{len(flat)}d", *flat)) != record["sha256_le_f64"]:
        raise ValueError("array checksum mismatch")
    return values


def _validate(
    receipt: dict[str, Any], inputs: dict[str, Any], manifest: dict[str, Any]
) -> None:
    if (
        manifest["schema"] != "generativeqc.ecp-raw-matrices.manifest.v1"
        or inputs["schema"] != "generativeqc.ecp-raw-matrices.inputs.v1"
        or receipt["schema"] != "generativeqc.ecp-raw-matrices.receipt.v1"
        or receipt["inputs"] != inputs
    ):
        raise ValueError("reference schema or input mismatch")
    if (
        inputs["atoms"] != [["Na", [0.13, -0.21, 0.17]], ["H", [0.43, 0.19, 3.2]]]
        or inputs["unit"] != "Bohr"
        or type(inputs["charge"]) is not int
        or inputs["charge"] != 0
        or type(inputs["spin"]) is not int
        or inputs["spin"] != 0
        or inputs["representations"] != ["spherical", "cartesian"]
        or inputs["d_shells"] != [False, True]
        or inputs["basis_libraries"] != {"Na": "lanl2dz", "H": "sto-3g"}
        or inputs["ecp_libraries"] != {"Na": "lanl2dz"}
        or inputs["extra_d_shell"] != [2, [0.35, 1.0]]
        or inputs["normalization"]
        != (
            "ECPscalar / sqrt(diag(int1e_ovlp))[:,None] / "
            "sqrt(diag(int1e_ovlp))[None,:]"
        )
        or inputs["precision"] != "float64"
        or inputs["matrix_unit"] != "Hartree"
        or inputs["overlap_unit"] != "dimensionless"
    ):
        raise ValueError("fixed scientific input mismatch")
    provenance = receipt["provenance"]
    acquisition = manifest["acquisition"]
    if (
        provenance["inputs_sha256"] != manifest["inputs_sha256"]
        or provenance["generator_sha256"] != manifest["generator"]["sha256"]
        or provenance["audit_helper_sha256"] != inputs["audit_helper"]["sha256"]
        or provenance["test_source_sha256"] != inputs["source"]["sha256"]
        or manifest["source"] != inputs["source"]
        or manifest["audit_helper"] != inputs["audit_helper"]
        or provenance["python"] != inputs["python"]
        or provenance["configuration"]
        != {"policy": "explicit-empty-regular-file", "sha256": _digest(b"")}
        or provenance["execution"]
        != {
            "checkout_sha": acquisition["checkout_sha"],
            "github_repository": acquisition["repository"],
            "github_sha": acquisition["checkout_sha"],
            "github_run_id": str(acquisition["run_id"]),
            "github_run_attempt": str(acquisition["run_attempt"]),
        }
        or {
            name: package["version"] for name, package in provenance["packages"].items()
        }
        != inputs["packages"]
    ):
        raise ValueError("historical acquisition provenance mismatch")
    replay = receipt["replay"]
    if (
        replay["schema"] != "generativeqc.independent-process-replay.v1"
        or replay["replicas"] != ["first", "second"]
        or replay["receipt_sha256"]
        != [replica["sha256"] for replica in manifest["replicas"]]
        or len(replay["process_ids"]) != 2
        or any(type(pid) is not int or pid <= 0 for pid in replay["process_ids"])
        or replay["process_ids"][0] == replay["process_ids"][1]
        or receipt["process_id"] != replay["process_ids"][0]
        or receipt["replica"] != "first"
        or replay["absolute_tolerance"] != 1e-13
        or not 0 <= replay["max_absolute_difference"] <= 1e-13
    ):
        raise ValueError("independent process replay mismatch")
    selectors = [
        (f"d{int(d_shell)}-{representation}", representation, d_shell)
        for d_shell in (False, True)
        for representation in ("spherical", "cartesian")
    ]
    records = receipt["records"]
    bindings = manifest["records"]
    if not isinstance(records, list) or len(records) != 4 or len(bindings) != 4:
        raise ValueError("incomplete four-case inventory")
    base_basis = None
    base_ecp = None
    for record, binding, (identity, representation, d_shell) in zip(
        records, bindings, selectors, strict=True
    ):
        case = record["input"]
        if (
            case["id"] != identity
            or binding["id"] != identity
            or case["representation"] != representation
            or type(case["d_shell"]) is not bool
            or case["d_shell"] != d_shell
            or type(case["cart"]) is not bool
            or case["cart"] != (representation == "cartesian")
            or case["atoms"] != inputs["atoms"]
            or case["unit"] != inputs["unit"]
            or type(case["charge"]) is not int
            or case["charge"] != 0
            or type(case["spin"]) is not int
            or case["spin"] != 0
            or record["molecule_atoms"] != case["atoms"]
            or record["molecule_basis"] != case["basis"]
            or record["molecule_ecp"] != case["ecp"]
            or _digest(_canonical(case)) != record["input_sha256"]
        ):
            raise ValueError("resolved scientific input mismatch")
        for key in set(binding) - {"id"}:
            if record[key] != binding[key]:
                raise ValueError("resolved basis/ECP/AO identity mismatch")
        unextended = json.loads(_canonical(case["basis"]))
        if d_shell and unextended["Na"].pop() != inputs["extra_d_shell"]:
            raise ValueError("incorrect d-shell extension")
        if base_basis is None:
            base_basis, base_ecp = unextended, case["ecp"]
        if (
            unextended != base_basis
            or case["ecp"] != base_ecp
            or record["core_electrons"] != [10, 0]
            or record["effective_nuclear_charges"] != [1, 1]
            or record["nelec"] != [1, 1]
        ):
            raise ValueError("inconsistent resolved basis/ECP or electron count")
        nao = record["nao"]
        expected_nao = (15 if representation == "cartesian" else 14) if d_shell else 9
        if type(nao) is not int or nao != expected_nao:
            raise ValueError("incorrect fixed AO count")
        if len(record["ao_labels"]) != nao or len(set(record["ao_labels"])) != nao:
            raise ValueError("incorrect AO ordering")
        matrices = {
            key: _array(record[key], [nao, nao])
            for key in ("overlap", "ecp_raw", "ecp_normalized")
        }
        norms = _array(record["normalization"], [nao])
        for i in range(nao):
            if matrices["overlap"][i][i] <= 0 or norms[i] <= 0:
                raise ValueError("nonpositive AO normalization")
            if abs(norms[i] - math.sqrt(matrices["overlap"][i][i])) > 1e-13:
                raise ValueError("incorrect AO normalization")
            for j in range(nao):
                expected = matrices["ecp_raw"][i][j] / norms[i] / norms[j]
                if abs(matrices["ecp_normalized"][i][j] - expected) > 1e-13:
                    raise ValueError("incorrect normalized ECP matrix")
                if any(
                    abs(matrix[i][j] - matrix[j][i]) > 1e-12
                    for matrix in matrices.values()
                ):
                    raise ValueError("nonsymmetric scalar reference matrix")


def load_reference(
    *, representation: str, d_shell: bool, root: Path = REFERENCE_DIR
) -> tuple[list[Any], dict[str, Any], dict[str, Any], list[list[float]]]:
    """Return newly decoded exact inputs and normalized full matrix; no fallback."""
    if representation not in ("spherical", "cartesian") or type(d_shell) is not bool:
        raise ValueError("unsupported fixed ECP selector")
    manifest, _ = _read(root / "manifest.json")
    inputs, input_raw = _read(root / "inputs.json")
    receipt, receipt_raw = _read(root / "reference.json")
    for prefix, raw in (("inputs", input_raw), ("receipt", receipt_raw)):
        if (
            len(raw) != manifest[f"{prefix}_bytes"]
            or _digest(raw) != manifest[f"{prefix}_sha256"]
        ):
            raise ValueError("fixed reference byte checksum mismatch")
    _validate(receipt, inputs, manifest)
    identity = f"d{int(d_shell)}-{representation}"
    record = next(
        record for record in receipt["records"] if record["input"]["id"] == identity
    )
    case = record["input"]
    return case["atoms"], case["basis"], case["ecp"], record["ecp_normalized"]["values"]
