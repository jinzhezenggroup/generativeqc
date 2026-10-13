"""Read the complete authentic fixed converged Li/UHF Lambda profile without PySCF."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data/li_uhf_lambda"
MAX_BYTES = 256 * 1024
BLOCK_SHAPES = {
    "t1a": (2, 3),
    "t1b": (1, 4),
    "t2aa": (2, 2, 3, 3),
    "t2ab": (2, 1, 3, 4),
    "t2bb": (1, 1, 4, 4),
}
SHAPES = {
    **{
        name: (5, 5)
        for name in (
            "overlap",
            "hcore",
            "density_alpha",
            "density_beta",
            "fock_ao_alpha",
            "fock_ao_beta",
            "coefficients_alpha",
            "coefficients_beta",
            "fock_mo_alpha",
            "fock_mo_beta",
        )
    },
    **{
        name: (5,)
        for name in (
            "orbital_energies_alpha",
            "orbital_energies_beta",
            "occupations_alpha",
            "occupations_beta",
            "eris_energies_alpha",
            "eris_energies_beta",
        )
    },
    "ao_eri": (5, 5, 5, 5),
    **{
        f"{group}.{name}": shape
        for group in (
            "amplitudes",
            "updated_amplitudes",
            "residuals",
            "denominators",
            "multipliers",
            "updated_multipliers",
        )
        for name, shape in BLOCK_SHAPES.items()
    },
    **{name: () for name in ("hf_energy", "correlation_energy", "scf_gradient_norm")},
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in entries:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("fixed reference exceeds byte limit")
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    json.dumps(value, allow_nan=False)
    return value, raw


def _array(record: dict[str, Any], shape: tuple[int, ...]) -> tuple[float, ...]:
    if set(record) != {
        "dtype",
        "shape",
        "order",
        "encoding",
        "bytes",
        "sha256_le_f64",
        "data",
    }:
        raise ValueError("incorrect complete array envelope")
    if (
        record["dtype"] != "float64"
        or record["shape"] != list(shape)
        or not isinstance(record["shape"], list)
        or any(type(size) is not int for size in record["shape"])
        or record["order"] != "C"
        or record["encoding"] != "base64-le-f64"
        or type(record["bytes"]) is not int
        or record["bytes"] != math.prod(shape) * 8
        or not isinstance(record["data"], str)
    ):
        raise ValueError("array shape/type/encoding mismatch")
    raw = base64.b64decode(record["data"], validate=True)
    if (
        len(raw) != record["bytes"]
        or base64.b64encode(raw).decode("ascii") != record["data"]
        or _digest(raw) != record["sha256_le_f64"]
    ):
        raise ValueError("array byte identity mismatch")
    values = struct.unpack(f"<{math.prod(shape)}d", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite decoded array")
    return values


def _validate(
    receipt: dict[str, Any], inputs: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, tuple[float, ...]]:
    if (
        manifest["schema"] != "generativeqc.li-uhf-lambda.manifest.v1"
        or inputs["schema"] != "generativeqc.li-uhf-lambda.inputs.v1"
        or receipt["schema"] != "generativeqc.li-uhf-lambda.receipt.v1"
        or set(receipt)
        != {
            "schema",
            "replica",
            "process_id",
            "inputs",
            "provenance",
            "molecule",
            "arrays",
            "replay",
        }
        or receipt["inputs"] != inputs
        or inputs["atoms"] != [["Li", [0, 0, 0]]]
        or inputs["basis_name"] != "sto-3g"
        or inputs["unit"] != "Angstrom"
        or type(inputs["charge"]) is not int
        or inputs["charge"] != 0
        or type(inputs["spin"]) is not int
        or inputs["spin"] != 1
        or inputs["cart"] is not False
        or inputs["populations"] != {"oa": 2, "ob": 1, "va": 3, "vb": 4}
        or inputs["scf"]
        != {
            "method": "UHF",
            "initial_guess": "minao",
            "conv_tol": 1e-13,
            "polish": {
                "direct_scf": False,
                "conv_tol_grad": 1e-10,
                "max_cycle": 300,
                "dm0": "converged initial density",
                "newton_fallback": True,
            },
        }
        or inputs["coupled_cluster"]
        != {
            "method": "UCCSD",
            "conv_tol": 1e-13,
            "conv_tol_normt": 1e-11,
            "max_cycle": 400,
            "lambda": "solve_lambda(eris=eris)",
        }
        or inputs["block_order"] != list(BLOCK_SHAPES)
        or inputs["precision"] != "float64"
        or inputs["array_encoding"] != "base64 little-endian C-order FP64"
        or inputs["preserved_gates"] != {"lambda_atol": 2e-7, "lambda_rtol": 0}
    ):
        raise ValueError("fixed converged Li Lambda input/profile mismatch")
    provenance, acquisition = receipt["provenance"], manifest["acquisition"]
    if (
        provenance != manifest["provenance"]
        or provenance["inputs_sha256"] != manifest["inputs_sha256"]
        or provenance["generator_sha256"] != manifest["generator"]["sha256"]
        or provenance["source_sha256"]
        != {entry["path"]: entry["sha256"] for entry in inputs["sources"]}
        or provenance["configuration"]
        != {"policy": "explicit-empty-regular-file", "sha256": _digest(b"")}
        or provenance["python"] != inputs["python"]
        or {name: item["version"] for name, item in provenance["packages"].items()}
        != inputs["packages"]
        or provenance["installed_scientific_inputs"]
        != {
            entry["path"]: {
                "bytes": manifest["provenance"]["installed_scientific_inputs"][
                    entry["path"]
                ]["bytes"],
                "sha256": entry["sha256"],
            }
            for entry in [inputs["basis_data"], *inputs["upstream_lambda"]["files"]]
        }
        or provenance["execution"]
        != {
            "checkout_sha": acquisition["checkout_sha"],
            "github_repository": acquisition["repository"],
            "github_sha": acquisition["checkout_sha"],
            "github_run_id": str(acquisition["run_id"]),
            "github_run_attempt": str(acquisition["run_attempt"]),
        }
    ):
        raise ValueError("historical acquisition provenance mismatch")
    molecule = receipt["molecule"]
    if (
        molecule != manifest["molecule"]
        or molecule["basis"] != inputs["resolved_basis"]
        or molecule["ao_locations"] != inputs["ao_identity"]["locations"]
        or molecule["ao_labels_structured"] != inputs["ao_identity"]["labels"]
        or molecule["nao"] != 5
        or type(molecule["nao"]) is not int
        or molecule["nelec"] != [2, 1]
        or any(type(value) is not int for value in molecule["nelec"])
        or molecule["atoms_bohr"] != [["Li", [0, 0, 0]]]
        or type(molecule["charge"]) is not int
        or molecule["charge"] != 0
        or type(molecule["spin"]) is not int
        or molecule["spin"] != 1
        or molecule["cart"] is not False
        or molecule["scf_converged"] is not True
        or len(molecule["ao_labels"]) != 5
        or len(set(molecule["ao_labels"])) != 5
        or molecule["resolved_scf"]["conv_tol"] != 1e-13
        or molecule["resolved_scf"]["init_guess"] != "minao"
    ):
        raise ValueError("resolved Li UHF identity mismatch")
    replay = receipt["replay"]
    replicas = manifest["replicas"]
    if (
        len(replicas) != 2
        or [item["path"] for item in replicas]
        != ["acquisition-first.json", "acquisition-second.json"]
        or [item["replica"] for item in replicas] != ["first", "second"]
        or replay["schema"] != "generativeqc.independent-process-replay.v1"
        or replay["replicas"] != ["first", "second"]
        or replay["receipt_sha256"] != [item["sha256"] for item in replicas]
        or replay["process_ids"] != [item["process_id"] for item in replicas]
        or any(type(pid) is not int or pid <= 0 for pid in replay["process_ids"])
        or replay["process_ids"][0] == replay["process_ids"][1]
        or receipt["replica"] != "first"
        or receipt["process_id"] != replay["process_ids"][0]
        or replay["absolute_tolerance"] != 1e-13
        or type(replay["max_absolute_difference"]) not in (int, float)
        or not 0 <= replay["max_absolute_difference"] <= 1e-13
    ):
        raise ValueError("independent replay identity mismatch")
    if set(receipt["arrays"]) != set(SHAPES) or set(manifest["arrays"]) != set(SHAPES):
        raise ValueError("incomplete scientific array inventory")
    arrays = {}
    for name, shape in SHAPES.items():
        record = receipt["arrays"][name]
        if {key: value for key, value in record.items() if key != "data"} != manifest[
            "arrays"
        ][name]:
            raise ValueError("historical full-array identity mismatch")
        arrays[name] = _array(record, shape)
    if arrays["occupations_alpha"] != (1, 1, 0, 0, 0) or arrays["occupations_beta"] != (
        1,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("incorrect spin occupations")
    if (
        molecule["resolved_scf"]["conv_tol_grad"] != 1e-10
        or molecule["resolved_scf"]["max_cycle"] != 300
        or molecule["resolved_scf"]["direct_scf"] is not False
        or molecule["cc_converged"] is not True
        or molecule["lambda_converged"] is not True
        or molecule["resolved_cc"]["conv_tol"] != 1e-13
        or molecule["resolved_cc"]["conv_tol_normt"] != 1e-11
        or molecule["resolved_cc"]["max_cycle"] != 400
    ):
        raise ValueError("convergence protocol differs from original Lambda fixture")
    for name in (
        "overlap",
        "hcore",
        "density_alpha",
        "density_beta",
        "fock_ao_alpha",
        "fock_ao_beta",
        "fock_mo_alpha",
        "fock_mo_beta",
    ):
        if any(
            abs(arrays[name][5 * i + j] - arrays[name][5 * j + i]) > 1e-12
            for i in range(5)
            for j in range(5)
        ):
            raise ValueError("nonsymmetric one-electron matrix")
    for name in BLOCK_SHAPES:
        for old, updated, denominator, residual in zip(
            arrays[f"amplitudes.{name}"],
            arrays[f"updated_amplitudes.{name}"],
            arrays[f"denominators.{name}"],
            arrays[f"residuals.{name}"],
            strict=True,
        ):
            if (
                abs(residual - denominator * (updated - old)) > 1e-13
                or abs(residual) >= 2e-9
            ):
                raise ValueError("independent converged CC residual mismatch")
        if (
            max(
                abs(a - b)
                for a, b in zip(
                    arrays[f"multipliers.{name}"],
                    arrays[f"updated_multipliers.{name}"],
                    strict=True,
                )
            )
            >= 2e-9
        ):
            raise ValueError("independent Lambda fixed point mismatch")
    for group in (
        "amplitudes",
        "updated_amplitudes",
        "residuals",
        "multipliers",
        "updated_multipliers",
    ):
        if any(value != 0 for value in arrays[f"{group}.t2bb"]):
            raise ValueError("single-beta same-spin channel must be explicit zero")

    return arrays


def load_reference(*, root: Path = REFERENCE_DIR) -> dict[str, tuple[float, ...]]:
    """Load every complete array and prove retained replicas; no import or fallback."""
    manifest, _ = _read(root / "manifest.json")
    inputs, input_raw = _read(root / "inputs.json")
    receipt, receipt_raw = _read(root / "reference.json")
    for prefix, raw in (("inputs", input_raw), ("receipt", receipt_raw)):
        if (
            len(raw) != manifest[f"{prefix}_bytes"]
            or _digest(raw) != manifest[f"{prefix}_sha256"]
        ):
            raise ValueError("fixed reference byte checksum mismatch")
    arrays = _validate(receipt, inputs, manifest)
    for binding in manifest["replicas"]:
        raw_object, raw_bytes = _read(root / binding["path"])
        if (
            len(raw_bytes) != binding["bytes"]
            or _digest(raw_bytes) != binding["sha256"]
        ):
            raise ValueError("raw replica byte identity mismatch")
        expected = {key: value for key, value in receipt.items() if key != "replay"}
        expected["replica"] = binding["replica"]
        expected["process_id"] = binding["process_id"]
        if raw_object != expected:
            raise ValueError("raw replica differs from the validated complete record")
    return arrays


def frozen_problem() -> tuple:
    """Construct fresh GQC inputs and complete independent Lambda multipliers."""
    import numpy as np
    from test_ump2_energy import DenseSource

    from tools.generativeqc_response.uhf import UHFReferenceSnapshot

    arrays = {
        name: np.asarray(values, dtype=np.float64).reshape(SHAPES[name])
        for name, values in load_reference().items()
    }
    snapshot = UHFReferenceSnapshot(
        overlap=arrays["overlap"],
        hcore=arrays["hcore"],
        fock_alpha=arrays["fock_ao_alpha"],
        fock_beta=arrays["fock_ao_beta"],
        coefficients_alpha=arrays["coefficients_alpha"],
        coefficients_beta=arrays["coefficients_beta"],
        orbital_energies_alpha=arrays["orbital_energies_alpha"],
        orbital_energies_beta=arrays["orbital_energies_beta"],
        occupations_alpha=arrays["occupations_alpha"],
        occupations_beta=arrays["occupations_beta"],
        reference_energy=float(arrays["hf_energy"]),
        scf_residual=1e-10,
        geometry_hash="uccsd-test-geometry",
        basis_hash="uccsd-test-basis",
        generation_id="uccsd-test-generation",
        hf_backend="pyscf-test-oracle",
    )
    expected = {
        name.replace("t", "l", 1): arrays[f"multipliers.{name}"]
        for name in BLOCK_SHAPES
    }
    return snapshot, DenseSource(arrays["ao_eri"], snapshot), expected
