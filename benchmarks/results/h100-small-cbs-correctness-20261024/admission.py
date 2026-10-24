"""Opt-in, solver-free audit of immutable references for issue #2727.

All hash checks refer to historical inputs, not production function spelling.
No native library, PySCF, CUDA module, or external service is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import subprocess
from pathlib import Path
from typing import Any

from model_contract import archive_expected, coefficients
from model_contract import components as model_components

if not __debug__:
    raise RuntimeError("Reference admission requires non-optimized Python; omit -O/-OO")

ROOT = Path(__file__).resolve().parents[3]
CPU_SOURCE = "bbd41a42764e8d22d15c1348f70a633961d4dcf5"
CBS_SOURCE = "976f756b391bc65edb79f76df1a8c3fe807518a7"
ARCHIVE_REV = "7522994e2e2e477e11c41cb2c2dbe74a2ce600e9"
ARCHIVE_PATH = "benchmarks/results/cbs-fast-memory-20261021/summary.json.xz"
ARCHIVE_SHA = "4e10edc12aff77c0587058b66560e80a63a082c7597634cdc3e63869cebac414"


def blob(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError("missing or nonfinite numeric value")
    return value


def vector(value: Any, atoms: int) -> list:
    if not isinstance(value, list) or len(value) != atoms:
        raise ValueError("incomplete force vector")
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("incomplete force coordinate")
        for x in row:
            finite(x)
    return value


def reconstruct(oracle: dict) -> list:
    """Recover five complete force vectors from all 90 original energies."""
    if oracle.get("step_bohr") != 3e-5:
        raise ValueError("archive displacement step changed")
    rows = oracle.get("displaced_energies", [])
    if len(rows) != 9:
        raise ValueError("missing displaced energies")
    result = [[[0.0] * 3 for _ in range(3)] for _ in range(5)]
    seen = set()
    for row in rows:
        atom, axis = row["atom"], row["axis"]
        if (
            type(atom) is not int
            or type(axis) is not int
            or not 0 <= atom < 3
            or not 0 <= axis < 3
        ):
            raise ValueError("invalid coordinate")
        if (atom, axis) in seen:
            raise ValueError("duplicate coordinate")
        seen.add((atom, axis))
        values = row.get("energies_minus_plus")
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(len(r) != 5 for r in values)
        ):
            raise ValueError("incomplete component energy vector")
        for k, (minus, plus) in enumerate(zip(*values, strict=True)):
            result[k][atom][axis] = -(finite(plus) - finite(minus)) / (
                2 * oracle["step_bohr"]
            )
    if result != oracle.get("component_forces"):
        raise ValueError("stored force vector differs from original displaced energies")
    if len(oracle.get("component_energies", [])) != 5:
        raise ValueError("missing reference constituent energies")
    for x in oracle["component_energies"]:
        finite(x)
    return result


def compare(
    energy: float,
    forces: list,
    expected_e: float,
    expected_f: list,
    gates: dict,
    *,
    strict: bool = True,
) -> dict:
    """Retain the archived driver's strict less-than gates."""
    atoms = len(expected_f)
    vector(forces, atoms)
    vector(expected_f, atoms)
    delta = [
        a - b
        for row, ref in zip(forces, expected_f, strict=True)
        for a, b in zip(row, ref, strict=True)
    ]
    errors = {
        "energy_abs": abs(finite(energy) - finite(expected_e)),
        "force_max_abs": max(map(abs, delta)),
        "force_rms": math.sqrt(sum(d * d for d in delta) / len(delta)),
    }
    if any(
        not math.isfinite(v)
        or (v >= finite(gates[k]) if strict else v > finite(gates[k]))
        for k, v in errors.items()
    ):
        raise ValueError(f"archive scientific gate failed: {errors}")
    return errors


def audit() -> dict:
    archive_bytes = blob(ARCHIVE_REV, ARCHIVE_PATH)
    if digest(archive_bytes) != ARCHIVE_SHA:
        raise ValueError("archive identity changed")
    summary = json.loads(lzma.decompress(archive_bytes))
    water = summary["qualification"]["water"]
    results = []
    for case in water["cases"]:
        force = reconstruct(case["oracle"])
        samples = []
        expected_slots = {
            ("standard", 2048, 0),
            ("standard", 2048, 1),
            ("fast", 2048, 0),
            ("fast", 2048, 1),
            ("fast", 256, 0),
        }
        if (
            len(case["samples"]) != 5
            or {(s["policy"], s["budget_mib"], s["repeat"]) for s in case["samples"]}
            != expected_slots
        ):
            raise ValueError("archive required sample coverage differs")
        for sample in case["samples"]:
            components = model_components(
                {"fixture": "archive-water"},
                sample,
                {
                    "archive_water_expected_basis_identities": [
                        c["basis_identity"]
                        for c in water["cases"][0]["samples"][0]["components"]
                    ]
                },
            )
            expected_e, expected_f = archive_expected(case["oracle"], force)
            gates = sample["convergence"]["budget"] or {
                "energy_abs": 1e-8,
                "force_max_abs": 3e-6,
                "force_rms": 3e-6,
            }
            component_gates = sample["convergence"]["component_budget"] or gates
            errors = compare(
                sample["energy"], sample["forces"], expected_e, expected_f, gates
            )
            component_errors = [
                compare(
                    c["energy"],
                    c["forces"],
                    case["oracle"]["component_energies"][i],
                    force[i],
                    component_gates,
                )
                for i, c in enumerate(components)
            ]
            if sample["executed_backend"] != "cuda" or any(
                c["executed_backend"] != "cuda" or not c["converged"]
                for c in components
            ):
                raise ValueError("archive GPU endpoint did not execute")
            samples.append(
                {
                    "policy": sample["policy"],
                    "budget_mib": sample["budget_mib"],
                    "repeat": sample["repeat"],
                    "gates": gates,
                    "component_gates": component_gates,
                    "errors": errors,
                    "component_errors": component_errors,
                    "triples_energy": components[0]["correlation"][
                        "ccsd_t_triples_energy"
                    ],
                }
            )
        results.append(
            {
                "model": "correlation-only-DF"
                if case["reference"] == "exact"
                else "full-RI-JK",
                "hf_reference": case["reference"],
                "geometry_id": case["geometry_id"],
                "atoms_bohr": case["atoms_bohr"],
                "raw_displaced_energy_count": 90,
                "independent_full_vector_replay": "PASS",
                "samples": samples,
            }
        )
    if {(r["hf_reference"], r["geometry_id"]) for r in results} != {
        (m, g) for m in ("exact", "df") for g in (0, 1)
    }:
        raise ValueError("incomplete model/geometry matrix")
    fixtures = []
    for case in ("h2-dz-tz", "water-algebra"):
        path = f"tests/reference_data/cbs/{case}.json"
        data = blob(CPU_SOURCE, path)
        ref = json.loads(data)
        if data != blob(CBS_SOURCE, path):
            raise ValueError(
                "conventional fixture differs between CPU and frozen H100 source"
            )
        bases = []
        for name, identity in zip(
            (ref["low_basis"], ref["high_basis"]), ref["basis_identities"], strict=True
        ):
            bp = "tests/reference_data/cbs/" + name
            b = blob(CPU_SOURCE, bp)
            if b != blob(CBS_SOURCE, bp):
                raise ValueError("basis input differs between historical sources")
            bases.append({"path": bp, "sha256": digest(b), "basis_identity": identity})
        fixtures.append(
            {
                "fixture": case,
                "model": "conventional-exact",
                "reference_sha256": digest(data),
                "source_equality": "PASS",
                "basis": bases,
                "atoms": ref["atoms"],
                "gates": {"energy_abs_Eh": 1e-9, "force_max_Eh_Bohr": 2e-6},
                "triples_reference_Eh": ref["reference"]["components"][
                    "correction_ccsdt"
                ]["triples_energy"],
                "physical_CBS_basis_pair": case == "h2-dz-tz",
                "DF_or_RIJK_oracle": "UNAVAILABLE; conventional oracle is not a DF oracle",
            }
        )
    c2 = summary["points"][1]["native"]["samples"][0]["components"][0]["result"]
    return {
        "schema": "generativeqc.h100-small-cbs-reference-admission.v3",
        "independent_composite_model": {
            "HF_exponent": "1.63",
            "MP2_exponent": 3,
            "cardinals": [2, 3],
            "component_coefficients": coefficients()[0],
            "HF_weights": coefficients()[1],
            "MP2_weights": coefficients()[2],
            "coefficient_L1": sum(map(abs, coefficients()[0])),
            "coefficient_identity_rounding": "4*ulp(independently derived expected coefficient), separate from unchanged E/F gates",
            "source": "declared archived scientific protocol and retained water driver CBSBasis cardinals; no native weights copied",
            "arithmetic": "80-digit Decimal exp and exact rational MP2 8/19,27/19",
        },
        "status": "HOST_OFFLINE_REFERENCE_PASS_SCI_NOT_AUTHORIZED",
        "CPU_reference_source": CPU_SOURCE,
        "H100_scientific_base": CBS_SOURCE,
        "archive_revision": ARCHIVE_REV,
        "archive_sha256": ARCHIVE_SHA,
        "archive_driver_sha256": digest(water["driver_source"].encode()),
        "archive_verifier_sha256": digest(
            blob(ARCHIVE_REV, "benchmarks/results/cbs-fast-memory-20261021/verify.py")
        ),
        "conventional_fixtures": fixtures,
        "archive_water": results,
        "C2_existing_triples_Eh": c2["correlation"]["ccsd_t_triples_energy"],
        "C2_independent_vector": "UNAVAILABLE; original 24 displaced energies missing, #2705",
        "precision_and_work": "INCOMPLETE; returned zeros do not establish absent work/fallback/peak",
        "new_solver_calls": 0,
        "new_GPU_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new file; existing files are rejected",
    )
    args = parser.parse_args()
    result = audit()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"])
