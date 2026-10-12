"""Read all four complete authentic NaH orbital-f HF profiles without PySCF."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Reuse the existing bounded stdlib-only complete-FP64 receipt transport.
from _li_uhf_random_reference import _array, _digest, _read

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data/ecp_f_hf"
PROFILES = ("rhf-cartesian", "rhf-spherical", "uhf-cartesian", "uhf-spherical")
# These authenticate immutable external scientific artifacts, not current source.
INPUTS_SHA256 = "20f95e88f94c8598e62a20456290b2ceeaf2d2bdb4715341305c58596ab61676"
MANIFEST_SHA256 = "35a7ef21310483b7293d7da409e1e4ebf49622e40ed741582092c9c798644161"
GENERATOR_SHA256 = "8b1a289607876c50062fa70a2714bed62881c23355160ab985547bb66d654988"
EXECUTION = {
    "checkout_sha": "2c0458410387d211e3b1e5e19bb3694775390e17",
    "github_sha": "2c0458410387d211e3b1e5e19bb3694775390e17",
    "github_repository": "njzjz-bot/vibeqc",
    "github_run_id": "38303387532",
    "github_run_attempt": "1",
}


def shapes(profile: str) -> dict[str, tuple[int, ...]]:
    if profile not in PROFILES:
        raise ValueError("unsupported fixed ECP-f HF profile")
    n = 19 if profile.endswith("cartesian") else 16
    prefix = (2,) if profile.startswith("uhf-") else ()
    return {
        **{key: (n, n) for key in ("overlap", "hcore", "ecp_raw", "ecp_normalized")},
        "normalization": (n,),
        **{
            key: (*prefix, n, n)
            for key in ("density", "fock", "coefficients", "commutator_residual")
        },
        **{key: (*prefix, n) for key in ("orbital_energies", "occupations")},
        "orbital_gradient": (n - 1,),
        "energy": (),
        "nuclear_energy": (),
        "gradient": (2, 3),
        "forces": (2, 3),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate(
    receipt: dict[str, Any],
    inputs: dict[str, Any],
    binding: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, tuple[float, ...]]:
    if (
        set(receipt)
        != {
            "schema",
            "inputs",
            "profile",
            "replica",
            "process_id",
            "provenance",
            "molecule",
            "scf",
            "timings",
            "arrays",
        }
        or receipt["schema"] != "generativeqc.ecp-f-hf.receipt.v1"
        or receipt["profile"] != binding["profile"]
        or receipt["replica"] not in ("first", "second")
        or type(receipt["process_id"]) is not int
        or receipt["process_id"] <= 0
        or _canonical(receipt["inputs"]) != _canonical(inputs)
        or _canonical(receipt["provenance"]) != _canonical(provenance)
        or provenance["inputs_sha256"] != INPUTS_SHA256
        or provenance["generator_sha256"] != GENERATOR_SHA256
        or provenance["execution"] != EXECUTION
        or provenance["configuration"]
        != {"policy": "explicit-empty-regular-file", "sha256": _digest(b"")}
        or provenance["device"] != "CPU"
        or provenance["cuda_visible_devices"] != ""
        or provenance["python"] != inputs["python"]
        or {key: value["version"] for key, value in provenance["packages"].items()}
        != inputs["packages"]
        or provenance["source_sha256"]
        != {entry["path"]: entry["sha256"] for entry in inputs["sources"]}
    ):
        raise ValueError("fixed receipt identity or historical provenance mismatch")
    profile = receipt["profile"]
    array_shapes = shapes(profile)
    n = array_shapes["overlap"][0]
    spin = int(profile.startswith("uhf-"))
    mol, scf = receipt["molecule"], receipt["scf"]
    if (
        _canonical(mol) != _canonical(binding["molecule"])
        or _canonical(scf) != _canonical(binding["scf"])
        or mol["basis"] != inputs["resolved_basis"]
        or mol["ecp"] != inputs["resolved_ecp"]
        or mol["atoms_bohr"] != inputs["atoms"]
        or mol["unit"] != "Bohr"
        or type(mol["nao"]) is not int
        or mol["nao"] != n
        or type(mol["spin"]) is not int
        or mol["spin"] != spin
        or type(mol["charge"]) is not int
        or mol["charge"] != spin
        or mol["cart"] is not profile.endswith("cartesian")
        or mol["nelec"] != ([1, 0] if spin else [1, 1])
        or mol["core_electrons"] != [10, 0]
        or mol["effective_nuclear_charges"] != [1, 1]
        or mol["ao_locations"] != [0, 1, 2, 5, 8, n - 1, n]
        or mol["atom_ao_slices"] != [[0, 5, 0, n - 1], [5, 6, n - 1, n]]
        or len(mol["ao_labels"]) != n
        or len(set(mol["ao_labels"])) != n
        or len(mol["ao_labels_structured"]) != n
        or scf["converged"] is not True
        or scf["method"] != ("UHF" if spin else "RHF")
        or scf["actual_class"] != ("pyscf.scf.uhf.HF1e" if spin else "pyscf.scf.hf.RHF")
        or scf["initial_guess_used"] is not (spin == 0)
        or type(scf["cycles"]) is not int
        or scf["cycles"] != (0 if spin else 8)
        or scf["resolved"]["conv_tol"] != 1e-12
        or scf["resolved"]["conv_tol_grad"] is not None
        or scf["resolved"]["init_guess"] != "minao"
    ):
        raise ValueError("resolved scientific input/AO/SCF identity mismatch")
    if set(receipt["arrays"]) != set(array_shapes) or set(binding["arrays"]) != set(
        array_shapes
    ):
        raise ValueError("incomplete full SCF/gradient array inventory")
    arrays = {}
    for key, shape in array_shapes.items():
        record = receipt["arrays"][key]
        arrays[key] = _array(record, shape)
        if {
            field: value for field, value in record.items() if field != "data"
        } != binding["arrays"][key]:
            raise ValueError("array differs from authenticated manifest")
    if set(receipt["timings"]) != {"scf_seconds", "gradient_seconds"} or any(
        type(t) not in (int, float) or not math.isfinite(t) or t <= 0
        for t in receipt["timings"].values()
    ):
        raise ValueError("invalid observed reference timings")
    if any(
        force != -gradient
        for force, gradient in zip(arrays["forces"], arrays["gradient"], strict=True)
    ):
        raise ValueError("full force is not the negative analytic gradient")
    if any(
        abs(arrays["forces"][axis] + arrays["forces"][3 + axis]) > 2e-7
        for axis in range(3)
    ):
        raise ValueError("full force violates translation invariance")
    occupations = (
        ((1.0,) + (0.0,) * (2 * n - 1)) if spin else ((2.0,) + (0.0,) * (n - 1))
    )
    if arrays["occupations"] != occupations or (
        spin and any(arrays["density"][n * n :])
    ):
        raise ValueError("incorrect complete spin occupations/density")
    for key in ("overlap", "hcore", "ecp_raw", "ecp_normalized", "density", "fock"):
        values = arrays[key]
        for block in range(len(values) // (n * n)):
            offset = block * n * n
            if any(
                abs(values[offset + i * n + j] - values[offset + j * n + i]) > 1e-11
                for i in range(n)
                for j in range(n)
            ):
                raise ValueError("nonsymmetric retained one-electron matrix")
    for i in range(n):
        norm = arrays["normalization"][i]
        if norm <= 0 or abs(norm**2 - arrays["overlap"][i * n + i]) > 1e-12:
            raise ValueError("incorrect retained AO normalization")
        for j in range(n):
            if (
                abs(
                    arrays["ecp_raw"][i * n + j] / norm / arrays["normalization"][j]
                    - arrays["ecp_normalized"][i * n + j]
                )
                > 1e-13
            ):
                raise ValueError("incorrect complete normalized ECP matrix")
    return arrays


def load_reference(
    *,
    representation: str,
    spin: int,
    root: Path = REFERENCE_DIR,
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, tuple[float, ...]]]:
    """Authenticate both originals of all four cases and return fresh exact inputs."""
    if (
        representation not in ("cartesian", "spherical")
        or type(spin) is not int
        or spin not in (0, 1)
    ):
        raise ValueError("unsupported fixed ECP-f HF selector")
    inputs, input_raw = _read(root / "inputs.json")
    manifest, manifest_raw = _read(root / "manifest.json")
    if _digest(input_raw) != INPUTS_SHA256 or _digest(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("immutable external input/manifest byte identity mismatch")
    if (
        set(manifest) != {"schema", "inputs_sha256", "provenance", "profiles"}
        or manifest["schema"] != "generativeqc.ecp-f-hf.manifest.v1"
        or inputs["schema"] != "generativeqc.ecp-f-hf.inputs.v1"
        or manifest["inputs_sha256"] != INPUTS_SHA256
        or [binding["profile"] for binding in manifest["profiles"]] != list(PROFILES)
    ):
        raise ValueError("incomplete authenticated quartet inventory")
    selected = None
    pids = set()
    for binding in manifest["profiles"]:
        profile = binding["profile"]
        if len(binding["replicas"]) != 2:
            raise ValueError("two fresh originals per profile are required")
        originals = []
        for index, original in enumerate(binding["replicas"]):
            replica = ("first", "second")[index]
            filename = f"{profile}-{replica}.json"
            if original["path"] != filename or original["replica"] != replica:
                raise ValueError("unsafe or mismatched original receipt filename")
            receipt, raw = _read(root / filename)
            if len(raw) != original["bytes"] or _digest(raw) != original["sha256"]:
                raise ValueError("original acquisition file byte identity mismatch")
            arrays = _validate(receipt, inputs, binding, manifest["provenance"])
            if (
                receipt["replica"] != replica
                or receipt["process_id"] != original["process_id"]
                or receipt["process_id"] in pids
                or receipt["timings"] != original["timings"]
            ):
                raise ValueError(
                    "independent original process/timing identity mismatch"
                )
            pids.add(receipt["process_id"])
            originals.append(arrays)
        differences = {
            key: max(
                abs(a - b)
                for a, b in zip(originals[0][key], originals[1][key], strict=True)
            )
            for key in shapes(profile)
        }
        if binding["replay"] != {
            "absolute_tolerance": 1e-13,
            "max_absolute_difference_by_array": differences,
        } or any(delta > 1e-13 for delta in differences.values()):
            raise ValueError("complete independent array replay mismatch")
        if profile == f"{'uhf' if spin else 'rhf'}-{representation}":
            selected = originals[0]
    if selected is None or len(pids) != 8:
        raise ValueError("missing requested profile or independent process")
    return inputs["atoms"], inputs["resolved_basis"], inputs["resolved_ecp"], selected
