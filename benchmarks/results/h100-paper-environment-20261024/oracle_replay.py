"""Compare setup returns with existing independent references; never run an oracle.

C2's retained directional derivatives lack their original displaced energies.
Their comparison is reported separately and never promoted to full raw replay.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path

from qz_smoke import checked_file, finite_endpoint, load_json
from replay import recover_sample

HF_REFERENCE_SHA = "d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4"
CBS_REFERENCE_SHA = "4e10edc12aff77c0587058b66560e80a63a082c7597634cdc3e63869cebac414"


def reference(path: Path, checksum: str) -> object:
    checked_file(path, checksum)
    return json.loads(lzma.decompress(path.read_bytes()))


def force_errors(left: list, right: list) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("reference/return force shape mismatch")
    errors = []
    for a, b in zip(left, right, strict=True):
        if len(a) != 3 or len(b) != 3:
            raise ValueError("incomplete force vector")
        errors.extend(x - y for x, y in zip(a, b, strict=True))
    if not all(math.isfinite(x) for x in errors):
        raise ValueError("nonfinite force comparison")
    return {
        "force_max_eh_bohr": max(abs(x) for x in errors),
        "force_rms_eh_bohr": math.sqrt(math.fsum(x * x for x in errors) / len(errors)),
    }


def compare_hf(raw: dict, sample: dict, references: list) -> dict:
    if raw["arm"] not in {"native-hf", "gpu4pyscf-hf"} or not sample.get("converged"):
        raise ValueError("not a converged tiny HF endpoint")
    rows = [r for r in references if r["method"] == "hf" and r["atoms"] == 3]
    if len(rows) != 2 or {r["backend"] for r in rows} != {"cuda", "cpu"}:
        raise ValueError("missing unique HF water-3 CPU/GPU reference pair")
    rows = {r["backend"]: r for r in rows}
    for r in rows.values():
        if r["geometry_bohr"] != raw["geometry_bohr"] or len(r["samples"]) != 1:
            raise ValueError("HF scientific input differs from the frozen oracle")
        if not r["samples"][0]["converged"]:
            raise ValueError("reference did not converge")
    target, spot = rows["cuda"]["samples"][0], rows["cpu"]["samples"][0]
    finite_endpoint(sample["energy"], sample["forces"], 3)
    errors = {
        "energy_eh_per_atom": abs(sample["energy"] - target["energy"]) / 3,
        **force_errors(sample["forces"], target["forces"]),
    }
    spot_errors = {
        "energy_eh_per_atom": abs(spot["energy"] - target["energy"]) / 3,
        **force_errors(spot["forces"], target["forces"]),
    }
    gates = {
        "energy_eh_per_atom": 1e-5,
        "force_max_eh_bohr": 1e-4,
        "force_rms_eh_bohr": 3e-5,
    }
    return {
        "scope": "HF water-3 independent complete E/F comparison",
        "errors": errors,
        "gates": gates,
        "observable_gate": "PASS"
        if all(errors[k] <= v for k, v in gates.items())
        else "FAIL",
        "cpu_gpu_spot_errors": spot_errors,
        "quarter_spot_gate": "PASS"
        if all(spot_errors[k] <= v / 4 for k, v in gates.items())
        else "FAIL",
        "grid_stability": "not applicable: HF has no XC/VV10 grid",
        "reference_sha256": HF_REFERENCE_SHA,
        "scientific_acceptance": "INCOMPLETE: device/source/environment provenance requires separate review",
    }


def cbs_rows(protocol: dict, sample: dict) -> dict:
    if not sample.get("converged") or sample.get("executed_backend") != "cuda":
        raise ValueError("composite did not return a converged CUDA endpoint")
    finite_endpoint(sample["energy"], sample["forces"], len(protocol["atoms_bohr"]))
    identities = protocol["orbital_identities"]
    if len(identities) != 2 or identities[0] == identities[1]:
        raise ValueError("CBS requires two distinct frozen orbital identities")
    h, m = protocol["hf_weights"], protocol["mp2_weights"]
    coefficients = {
        "low/rhf": h[0] - m[0],
        "high/rhf": h[1] - m[1],
        "low/mp2": m[0] - 1,
        "high/mp2": m[1],
        "low/df-ccsd(t)": 1.0,
    }
    rows = {}
    for component in sample["components"]:
        if component["basis_identity"] not in identities:
            raise ValueError("unexpected constituent basis identity")
        level = "low" if component["basis_identity"] == identities[0] else "high"
        key = level + "/" + component["method"]
        if key in rows or key not in coefficients:
            raise ValueError("duplicate/unexpected CBS constituent")
        if abs(component["coefficient"] - coefficients[key]) > 1e-15:
            raise ValueError("CBS constituent coefficient differs from frozen model")
        result = component["result"]
        if not result["converged"] or result["executed_backend"] != "cuda":
            raise ValueError("constituent did not return a converged CUDA endpoint")
        finite_endpoint(result["energy"], result["forces"], len(protocol["atoms_bohr"]))
        rows[key] = result
    if set(rows) != set(coefficients):
        raise ValueError("missing CBS constituent")
    rows["composite"] = sample
    return rows


def compare_cbs(raw: dict, sample: dict, summary: dict) -> dict:
    old = [
        point
        for point in summary["points"]
        if point["native"]["protocol"]["carbon"] == 2
    ]
    if len(old) != 1:
        raise ValueError("missing unique archived C2 point")
    ref = old[0]["reference"]
    protocol = raw["protocol"]
    keys = (
        "carbon",
        "qm7_index_zero_based",
        "dataset_sha256",
        "atoms_bohr",
        "orbital_bases",
        "auxiliary_bases",
        "orbital_identities",
        "auxiliary_identities",
        "precision",
        "frozen_core",
        "hf_reference",
        "hf_weights",
        "mp2_weights",
        "correction_basis",
        "endpoint",
    )
    if any(protocol.get(k) != ref["protocol"].get(k) for k in keys):
        raise ValueError(
            "CBS scientific input/model differs from independent reference"
        )
    if (
        protocol.get("hf_reference") != "df"
        or protocol.get("correlation_memory_budget_bytes") != 34359738368
    ):
        raise ValueError("setup is not full RI-JK at the frozen 32 GiB numeric budget")
    rows = cbs_rows(protocol, sample)
    direction = ref["direction"]
    if len(direction) != len(protocol["atoms_bohr"]) or any(
        len(row) != 3 for row in direction
    ):
        raise ValueError("incomplete retained reference direction")
    results = {}
    for key, result in rows.items():
        forces = result["forces"]
        projection = math.fsum(
            x * y
            for force, vector in zip(forces, direction, strict=True)
            for x, y in zip(force, vector, strict=True)
        )
        translation = max(
            abs(math.fsum(force[axis] for force in forces)) for axis in range(3)
        )
        energy_error = abs(result["energy"] - ref["energies"][key])
        steps = [
            {
                "step_bohr": row["step_bohr"],
                "error_eh_bohr": abs(projection + row["values"][key]),
            }
            for row in ref["derivatives"]
        ]
        if sorted(row["step_bohr"] for row in steps) != [3e-5, 1e-4]:
            raise ValueError("missing retained two-step directional comparison")
        results[key] = {
            "energy_error_eh": energy_error,
            "independent_energy_gate": "PASS" if energy_error <= 5e-8 else "FAIL",
            "translation_max_eh_bohr": translation,
            "translation_gate": "PASS" if translation <= 1e-6 else "FAIL",
            "retained_direction_comparisons": steps,
            "retained_direction_gate": "PASS"
            if all(r["error_eh_bohr"] <= 5e-6 for r in steps)
            else "FAIL",
        }
    return {
        "scope": "CBS C2 constituent/composite comparison against retained independent reference",
        "reference_sha256": CBS_REFERENCE_SHA,
        "constituents_and_composite": results,
        "gates": {
            "energy_eh": 5e-8,
            "direction_eh_bohr": 5e-6,
            "translation_eh_bohr": 1e-6,
        },
        "displaced_energy_originals": "UNAVAILABLE: #2705; derivatives were not reconstructed from nonexistent raw energies",
        "independent_full_force_vector": "UNAVAILABLE for this C2 reference",
        "scientific_acceptance": "INCOMPLETE: retained derivative comparison is not raw displaced-energy replay",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_directory", type=Path)
    parser.add_argument("--hf-oracles", type=Path)
    parser.add_argument("--cbs-summary", type=Path)
    args = parser.parse_args()
    raw = load_json(args.attempt_directory / "raw.json")
    if len(raw["samples"]) != 1 or len(raw["numeric_originals"]) != 1:
        raise ValueError("setup comparison requires exactly one returned sample")
    sample = recover_sample(
        args.attempt_directory, raw["samples"][0], raw["numeric_originals"][0]
    )
    if raw["arm"] == "cbs-c2":
        if args.cbs_summary is None:
            parser.error("CBS comparison requires --cbs-summary")
        result = compare_cbs(
            raw, sample, reference(args.cbs_summary, CBS_REFERENCE_SHA)
        )
    else:
        if args.hf_oracles is None:
            parser.error("HF comparison requires --hf-oracles")
        result = compare_hf(raw, sample, reference(args.hf_oracles, HF_REFERENCE_SHA))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
