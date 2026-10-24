"""Replay frozen-source force projections; missing references remain INCOMPLETE."""

import argparse
import hashlib
import json
import lzma
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


def require_unoptimized_python() -> None:
    # Assertions are scientific gates; subprocesses also inherit this environment.
    if (
        not __debug__
        or sys.flags.optimize
        or os.environ.get("PYTHONOPTIMIZE", "") not in ("", "0")
    ):
        raise RuntimeError(
            "Scientific verification requires unoptimized Python; "
            "omit -O/-OO and unset PYTHONOPTIMIZE"
        )


require_unoptimized_python()

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


def execution_records(name: str = "execution-initial.json.xz") -> dict:
    bundle = json.loads(lzma.decompress((HERE / name).read_bytes()))
    assert bundle["schema"] == "generativeqc.cbs-force-execution-raw.v1"
    files = {}
    for entry in bundle["files"]:
        raw = entry["utf8"].encode("utf-8")
        assert entry["path"] not in files
        assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"]
        files[entry["path"]] = raw
    return files


def check_terminal(initial: dict, final: dict, inputs: dict) -> dict:
    report = json.loads(final["terminal-postflight.json"])
    freeze = json.loads(initial["source-freeze-final.json"])
    environment = json.loads(initial["environment.json"])
    config = json.loads(initial["run-config.json"])
    assert report["status"] == "PASS"
    assert report["source_revision"] == freeze["revision"] == config["source_revision"]
    assert report["host"] == environment["host"]
    assert report["dataset_sha256"] == inputs["dataset_sha256"]
    assert not report["science_workers"] and not report["mode_errors"]
    assert report["cgroup"] == {
        k: environment["cgroup"][k] for k in ("cpu.max", "memory.max")
    }
    for field, expected in (
        ("source", freeze["files"]),
        ("libraries", environment["libraries"]),
    ):
        assert report[field].keys() == expected.keys()
        for name, original in expected.items():
            observed = report[field][name]
            assert (
                observed["expected_sha256"]
                == observed["observed_sha256"]
                == original["sha256"]
            )
            assert observed["bytes"] == original["bytes"]
    assert report["task_outcomes"].keys() == inputs["tasks"].keys()
    assert all(v["exit_code"] == 0 for v in report["task_outcomes"].values())
    controllers = report["controller_outcomes"]
    assert set(controllers) == {"suite.outcome.json", "launch/outcome.json"}
    assert all(
        v["exit_code"] == 0 and v == json.loads(final[k])
        for k, v in controllers.items()
    )
    return {
        "status": "PASS",
        "source_revision": report["source_revision"],
        "postflight_sha256": sha(final["terminal-postflight.json"]),
        "task_count": len(report["task_outcomes"]),
    }


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
    h, m = protocol["hf_weights"], protocol["mp2_weights"]
    terms = {
        "hf_forces": h[0] * rows["low/rhf"] + h[1] * rows["high/rhf"],
        "mp2_correlation_forces": m[0] * (rows["low/mp2"] - rows["low/rhf"])
        + m[1] * (rows["high/mp2"] - rows["high/rhf"]),
        "delta_ccsdt_forces": rows["low/df-ccsd(t)"] - rows["low/mp2"],
    }
    for key, expected in terms.items():
        actual = np.asarray(sample[key], dtype=float)
        assert actual.shape == shape and np.isfinite(actual).all()
        assert np.max(np.abs(actual - expected)) < 5e-12
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
    driver = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:benchmarks/results/cbs-independent-forces-20261024/oracle.py",
        ]
    )
    assert record["driver_sha256"] == sha(driver)
    assert record["packages"]["pyscf"] == "2.14.0"
    central_bundle = json.loads(
        lzma.decompress(blob(f"{ENERGY}/receipts/C{task['carbon']}.json.xz"))
    )
    central_receipt = json.loads(central_bundle["receipt_json"])
    assert record["source_files"] == central_receipt["source_files"]
    execution = execution_records()
    environment = json.loads(execution["environment.json"])
    config = json.loads(execution["run-config.json"])
    assert record["packages"] == environment["packages"]
    assert record["host"] == environment["host"] == config["host"]
    assert record["python"] == environment["python"]
    assert record["executable"] == environment["executable"]
    assert record["threads"] == config["threads"] == 4
    assert record["max_memory_mb"] == config["pyscf_memory_mb"] == 8000
    assert config["maximum_energy_calls"] == 8 and config["automatic_retries"] == 0
    assert config["driver_sha256"] == record["driver_sha256"]
    assert config["inputs_sha256"] == input_sha
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
    require_unoptimized_python()
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
        "expanded_cbs_coefficients": tools.coefficients(),
        "directional_convention": "native F dot direction = -(E_plus-E_minus)/(2*h)",
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
    result["numerical_status"] = (
        "FAIL"
        if "FAIL" in statuses
        else "PASS"
        if statuses == ["PASS", "PASS"]
        else "INCOMPLETE"
    )
    result["terminal_postflight"] = (
        check_terminal(
            execution_records(), execution_records("execution-final.json.xz"), inputs
        )
        if (HERE / "execution-final.json.xz").is_file()
        else {
            "status": "INCOMPLETE",
            "reason": "Terminal source/environment records unavailable",
        }
    )
    result["status"] = (
        "FAIL"
        if result["numerical_status"] == "FAIL"
        else "PASS"
        if result["numerical_status"]
        == result["terminal_postflight"]["status"]
        == "PASS"
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
