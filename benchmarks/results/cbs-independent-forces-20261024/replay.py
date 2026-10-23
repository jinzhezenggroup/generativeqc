"""Replay frozen-source force projections; missing references remain INCOMPLETE."""

import argparse
import hashlib
import json
import lzma
import math
import subprocess
import types
from pathlib import Path

import numpy as np

BASE = "334e7f489870dccc22ae8864c01f2353aa9689a4"
ENERGY = "benchmarks/results/cbs-independent-energy-20261023"
HERE = Path(__file__).parent
FORCE_GATE = 5e-6
TRANSLATION_GATE = 1e-6


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"])


def central_tools() -> types.ModuleType:
    """Reuse delivered read-only source/protocol verification, with Git LF bytes."""
    module = types.ModuleType("delivered_energy_replay")
    exec(  # noqa: S102 -- execute only the immutable, explicitly pinned Git LF source
        compile(blob(f"{ENERGY}/replay.py"), f"{ENERGY}/replay.py", "exec"),
        module.__dict__,
    )
    return module


def validate_inputs(inputs: dict) -> None:
    original_raw = blob(f"{ENERGY}/inputs.json")
    original = json.loads(original_raw)
    assert inputs["baseline"] == BASE
    assert inputs["central_inputs_sha256"] == sha(original_raw)
    assert inputs["dataset_sha256"] == original["dataset_sha256"]
    assert inputs["bases"] == original["bases"]
    assert inputs["protocols"] == {
        str(c): original["protocols"][str(c)] for c in (4, 7)
    }
    assert inputs["direction_seed"] == 709 and inputs["steps_bohr"] == [1e-4, 3e-5]
    expected = set()
    for carbon in (4, 7):
        atoms = inputs["protocols"][str(carbon)]["atoms_bohr"]
        center = np.array([a[1] for a in atoms])
        direction = np.random.default_rng(709).normal(size=center.shape)
        direction /= np.linalg.norm(direction)
        for i, step in enumerate(inputs["steps_bohr"]):
            for sign in (-1, 1):
                key = f"C{carbon}-h{i}-{'minus' if sign < 0 else 'plus'}"
                expected.add(key)
                assert inputs["tasks"][key] == {
                    "carbon": carbon,
                    "step_bohr": step,
                    "sign": sign,
                    "direction": direction.tolist(),
                    "atoms_bohr": [
                        [a[0], r.tolist()]
                        for a, r in zip(
                            atoms, center + sign * step * direction, strict=True
                        )
                    ],
                }
    assert set(inputs["tasks"]) == expected


def vectors(native: dict, protocol: dict, tools: types.ModuleType) -> dict:
    tools.native_values(native, protocol)
    sample = native["samples"][0]
    rows = {}
    for item in sample["components"]:
        level = (
            "low"
            if item["basis_identity"] == protocol["orbital_identities"][0]
            else "high"
        )
        rows[f"{level}/{item['method']}"] = np.asarray(
            item["result"]["forces"], dtype=float
        )
    rows["composite"] = np.asarray(sample["forces"], dtype=float)
    shape = (len(protocol["atoms_bohr"]), 3)
    assert all(v.shape == shape and np.isfinite(v).all() for v in rows.values())
    reconstructed = sum(tools.coefficients()[k] * rows[k] for k in tools.coefficients())
    assert np.max(np.abs(reconstructed - rows["composite"])) < 5e-12
    return rows


def validate_receipt(
    record: dict,
    inputs: dict,
    input_sha: str,
    task_id: str,
    folder: Path,
    tools: types.ModuleType,
) -> dict:
    task = inputs["tasks"][task_id]
    assert record["task_id"] == task_id and record["task"] == task
    assert record["protocol"] == inputs["protocols"][str(task["carbon"])]
    assert record["inputs_sha256"] == input_sha
    assert record["dataset_sha256"] == inputs["dataset_sha256"]
    assert record["driver_sha256"] == sha((HERE / "oracle.py").read_bytes())
    assert record["packages"]["pyscf"] == "2.14.0"
    assert json.loads((folder.parent / "outcome.json").read_bytes())["exit_code"] == 0
    low, high = record["stages"]
    assert [low["cardinal"], high["cardinal"]] == [2, 3]
    protocol = record["protocol"]
    for stage in (low, high):
        i = stage["cardinal"] - 2
        assert stage["orbital"] == protocol["orbital_bases"][i]
        assert stage["auxiliary"] == protocol["auxiliary_bases"][i]
        assert stage["status"] == "completed" and stage["rhf"]["converged"]
        assert stage["mp2"]["frozen"] == 0 and stage["mp2"]["reference_df_shared"]
        assert (
            sha((folder / f"basis-{stage['cardinal']}.log").read_bytes())
            == stage["log_sha256"]
        )
        for role, basis in stage["basis_records"].items():
            records = {int(k): v for k, v in basis["records"].items()}
            expected = inputs["bases"][stage[role]]
            assert sha(json.dumps(records, sort_keys=True).encode()) == basis["sha256"]
            assert basis["sha256"] == expected["pyscf_records_sha256"]
            assert basis["native_identity"] == expected["identity"]
    assert low["ccsd"]["converged"] and low["ccsd"]["frozen"] == 0
    assert low["ccsd"]["reference_df_shared"]
    assert (
        low["triples"]["energy"] == low["ccsd"]["energy"] + low["triples"]["correction"]
    )
    values = record["energies"]
    assert {k: v for k, v in values.items() if k != "composite"} == {
        "low/rhf": low["rhf"]["energy"],
        "low/mp2": low["mp2"]["energy"],
        "low/df-ccsd(t)": low["triples"]["energy"],
        "high/rhf": high["rhf"]["energy"],
        "high/mp2": high["mp2"]["energy"],
    }
    assert all(math.isfinite(v) for v in values.values())
    assert (
        abs(
            sum(tools.coefficients()[k] * values[k] for k in tools.coefficients())
            - values["composite"]
        )
        < 5e-12
    )
    return values


def projection_errors(
    rows: dict, direction: list, minus: dict, plus: dict, step: float
) -> dict:
    """Force is minus the energy derivative; every constituent keeps the same gate."""
    result = {}
    for key, force in rows.items():
        derivative = (plus[key] - minus[key]) / (2 * step)
        projection = float(np.vdot(force, direction))
        assert math.isfinite(derivative) and math.isfinite(projection)
        result[key] = {
            "energy_derivative": derivative,
            "native_force_projection": projection,
            "error_eh_per_bohr": abs(projection + derivative),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    raw_inputs = (HERE / "inputs.json.xz").read_bytes()
    inputs = json.loads(lzma.decompress(raw_inputs))
    validate_inputs(inputs)
    tools = central_tools()
    summary, provenance = tools.native_bundle()
    central = json.loads(blob(f"{ENERGY}/comparison.json"))
    assert central["status"] == "PASS"
    publication = json.loads(blob(f"{ENERGY}/publication.json"))
    for entry in publication["files"]:
        raw = blob(f"{ENERGY}/{entry['path']}")
        assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"]
    result = {
        "schema": "generativeqc.independent-cbs-force-comparison.v1",
        "native_provenance": provenance,
        "force_gate_eh_per_bohr": FORCE_GATE,
        "translation_gate_eh_per_bohr": TRANSLATION_GATE,
        "central_comparison_sha256": sha(blob(f"{ENERGY}/comparison.json")),
        "molecules": [],
    }
    for carbon in (4, 7):
        protocol = inputs["protocols"][str(carbon)]
        central_row = next(r for r in central["molecules"] if r["carbon"] == carbon)
        assert central_row["status"] == "PASS"
        assert all(
            e <= 5e-8
            for errors in central_row["errors_eh"].values()
            for e in errors.values()
        )
        row = {"carbon": carbon, "status": "INCOMPLETE", "policies": {}, "tasks": {}}
        result["molecules"].append(row)
        completed = {}
        failed = False
        for key, task in inputs["tasks"].items():
            if task["carbon"] != carbon:
                continue
            folder = args.raw / key / "science"
            path = folder / "receipt.json"
            outcome_path = folder.parent / "outcome.json"
            outcome = (
                json.loads(outcome_path.read_bytes())
                if outcome_path.is_file()
                else None
            )
            if outcome is not None and outcome["exit_code"] != 0:
                failed = True
            if not path.is_file():
                row["tasks"][key] = {
                    "status": "FAIL"
                    if outcome is not None and outcome["exit_code"] != 0
                    else "INCOMPLETE",
                    "outcome": outcome,
                }
                continue
            receipt = json.loads(path.read_bytes())
            row["tasks"][key] = {
                "status": receipt["status"],
                "outcome": outcome,
                "receipt_sha256": sha(path.read_bytes()),
                "seconds": receipt.get("seconds"),
                "max_rss_kib": receipt.get("max_rss_kib"),
                "error": receipt.get("error"),
            }
            if receipt["status"] == "FAIL":
                failed = True
            elif (
                receipt["status"] == "completed"
                and outcome is not None
                and outcome["exit_code"] == 0
            ):
                completed[key] = validate_receipt(
                    receipt, inputs, sha(raw_inputs), key, folder, tools
                )
        for policy, series in (("standard", summary), ("fast", summary["fast"])):
            point = next(p for p in series["points"] if p["carbon"] == carbon)
            native = point["native"]
            assert point["native_outcome"]["exit_code"] == 0
            assert native["library_sha256"] == provenance["source"]["library_sha256"]
            assert (
                native["driver_sha256"]
                == provenance["source"]["file_sha256"]["benchmarks/readme_cbs.py"]
            )
            forces = vectors(native, protocol, tools)
            translation = {
                k: float(np.max(np.abs(np.sum(v, axis=0)))) for k, v in forces.items()
            }
            values = {"translation_eh_per_bohr": translation, "steps": []}
            row["policies"][policy] = values
            failed |= any(v > TRANSLATION_GATE for v in translation.values())
            for i, step in enumerate(inputs["steps_bohr"]):
                a, b = f"C{carbon}-h{i}-minus", f"C{carbon}-h{i}-plus"
                if a not in completed or b not in completed:
                    continue
                errors = projection_errors(
                    forces,
                    inputs["tasks"][a]["direction"],
                    completed[a],
                    completed[b],
                    step,
                )
                values["steps"].append({"step_bohr": step, "components": errors})
                failed |= any(
                    v["error_eh_per_bohr"] > FORCE_GATE for v in errors.values()
                )
        row["status"] = (
            "FAIL" if failed else "PASS" if len(completed) == 4 else "INCOMPLETE"
        )
    statuses = [r["status"] for r in result["molecules"]]
    result["status"] = (
        "FAIL"
        if "FAIL" in statuses
        else "PASS"
        if statuses == ["PASS", "PASS"]
        else "INCOMPLETE"
    )
    args.output.write_bytes(
        (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "molecules": [
                    {"carbon": r["carbon"], "status": r["status"]}
                    for r in result["molecules"]
                ],
            }
        )
    )
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
