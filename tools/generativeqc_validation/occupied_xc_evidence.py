"""Replay occupied-XC scientific gates without running a GPU or a reference code.

Selected final-source populations retain every returned FP64 force bit. Other
populations are explicitly observational force-gate receipts, not offline force
replays. Profile clocks remain separate from clean endpoint measurements.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np

from tools.generativeqc_validation.xc_diagonal_evidence import REFERENCE

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/results/occupied-xc-20261017"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def summarize(root: Path = ROOT) -> dict[str, Any]:
    """Check force integrity, independent gates, complete populations and clocks.

    Incomplete populations are reported but never compared as complete endpoints.
    Storage anchors come from older native samples; only the separate independent
    geometry oracles can qualify scientific accuracy.
    """
    with lzma.open(root / "evidence.json.xz", "rt") as stream:
        data = json.load(stream)
    assert data["schema"] == "occupied-xc-cold-v2"
    assert set(data["reference_sha256"]) == {"forces.npz", "evidence.json.xz"}
    for name, expected in data["reference_sha256"].items():
        assert (
            hashlib.sha256((REFERENCE["ROOT"] / name).read_bytes()).hexdigest()
            == expected
        )
    encoded = data["forces"]
    assert encoded["encoding"] == "xor-shared-native-fp64-byteplanes"
    raw = base64.b64decode(encoded["data"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == encoded["byteplanes_sha256"]
    planes = np.frombuffer(raw, dtype="u1").reshape(encoded["shape"])
    assert planes.ndim == 3 and planes.shape[0] == 8
    reconstructed: dict[int, bytes] = {}
    output: dict[str, Any] = {
        "populations": {},
        "comparisons": {},
        "profiles": {},
        "observational_populations": {},
        "incomplete_populations": {},
    }

    def reconstruct(stored: dict[str, Any]) -> dict[str, Any]:
        """Authenticate measured provenance and restore the projected native rows."""
        assert stored["force_retention"] == "lossless"
        source = data["sources"][stored["source_key"]]
        build = stored["native_build"]
        assert build["probe"]["source_identity"] == source["source_identity"]
        assert build["library_sha256"] == source["library_sha256"]
        assert (
            build["library_sha256"] == stored["receipts"]["library.sha256"].split()[0]
        )
        case = stored["protocol_key"]
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        assert _digest(oracle["protocol"]) == stored["protocol_sha256"]
        shared = REFERENCE["load"](f"raw/candidate-{case}.json.xz")
        anchors = {
            row["geometry"]: row["forces"]
            for row in shared["records"]
            if row["phase"] in ("cold", "moved")
        }
        native = {
            key: value
            for key, value in stored.items()
            if key not in ("projection_sha256", "receipts")
        }
        native["records"] = []
        atoms = oracle["protocol"]["atoms"]
        for stored_row in stored["records"]:
            row = dict(stored_row)
            column = row.pop("forces_column")
            assert type(column) is int and 0 <= column < planes.shape[1]
            assert column not in reconstructed
            anchor = np.asarray(anchors[row["geometry"]], dtype="<f8").reshape(-1)
            delta = planes[:, column, :].T.copy().reshape(-1).view("<u8")
            assert delta.size >= 3 * atoms
            forces = (delta[: 3 * atoms] ^ anchor.view("<u8")).view("<f8")
            assert np.isfinite(forces).all()
            reconstructed[column] = forces.tobytes()
            row["forces"] = forces.reshape(atoms, 3).tolist()
            native["records"].append(row)
        assert _digest(native) == stored["projection_sha256"]
        native["protocol"] = oracle["protocol"]
        references = {
            row["geometry"]: row
            for row in oracle["records"]
            if row["phase"] in ("cold", "moved")
        }
        for row in native["records"]:
            reference = references[row["geometry"]]
            energy_error = abs(row["energy"] - reference["energy"])
            force_error = float(
                np.max(np.abs(np.asarray(row["forces"]) - reference["forces"]))
            )
            assert row["gate"] is True and row["converged"] is True
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert abs(energy_error - row["energy_error"]) <= 1e-15
            assert abs(force_error - row["force_error"]) <= 1e-15
            assert type(row["iterations"]) is int and row["iterations"] > 0
            assert type(row["fock_builds"]) is int and row["fock_builds"] > 0
        return native

    for key, stored in data["populations"].items():
        cohort, arm, case = key.split("/")
        native = reconstruct(stored)
        if native["stage"] != "complete" or native["status"] != "measured":
            output["incomplete_populations"][key] = {
                "status": native["status"],
                "stage": native["stage"],
                "completed_calls": len(native["records"]),
                "original_json_sha256": stored["original_json_sha256"],
            }
            continue
        assert stored["receipts"]["exit-status.txt"].strip() == "0"
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        result = REFERENCE["qualify"](native, oracle)
        assert result == stored["independent_validation_receipt"]
        output["populations"][key] = result
    for key, profile in data["profiles"].items():
        cohort, arm = key.split("/")
        native = reconstruct(profile["scientific_records"])
        assert native["status"] == "diagnostic_completed"
        assert [row["phase"] for row in native["records"]] == ["cold", "warm"]
        oracle = REFERENCE["load"]("raw/reference-r2scan-3c-96.json.xz")
        reference = oracle["records"][0]
        for row in native["records"]:
            assert row["geometry"] == 0 and row["gate"] and row["converged"]
            energy_error = abs(row["energy"] - reference["energy"])
            force_error = float(
                np.max(np.abs(np.asarray(row["forces"]) - reference["forces"]))
            )
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert abs(energy_error - row["energy_error"]) <= 1e-15
            assert abs(force_error - row["force_error"]) <= 1e-15
            window = profile["windows"][row["phase"]]
            assert all(row[field] == value for field, value in window["record"].items())
            root_seconds = window["public_execute_seconds"]
            for field in ("exclusive_host_seconds", "exclusive_device_window_seconds"):
                assert all(seconds >= 0 for seconds in window[field].values())
                assert abs(sum(window[field].values()) - root_seconds) < 1e-7
            ledger = window["complete_endpoint_wall_ledger"]
            assert all(
                seconds >= 0
                for seconds in ledger["complete_exclusive_wall_seconds"].values()
            )
            assert (
                abs(
                    sum(ledger["complete_exclusive_wall_seconds"].values())
                    - row["complete_seconds"]
                )
                < 1e-7
            )
            assert abs(ledger["complete_reconciliation_error_seconds"]) < 1e-7
            assert abs(window["host_reconciliation_error_seconds"]) < 1e-7
            assert abs(window["device_reconciliation_error_seconds"]) < 1e-7
            assert all(
                seconds >= 0 for seconds in window["kernel_device_seconds"].values()
            )
        output["profiles"][key] = {
            "windows": profile["windows"],
            "force_work": REFERENCE["force_work"](profile["census"]),
            "occupied_witnesses_consumed": sum(
                value != 0 for value in profile["xc_source_witness_generations"]
            ),
        }

    for key, stored in data["observational_populations"].items():
        _, arm, case = key.split("/")
        assert stored["force_retention"] == "original-local-with-gate-receipt"
        source = data["sources"][stored["source_key"]]
        assert (
            stored["native_build"]["probe"]["source_identity"]
            == source["source_identity"]
        )
        assert stored["native_build"]["library_sha256"] == source["library_sha256"]
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        assert _digest(oracle["protocol"]) == stored["protocol_sha256"]
        if stored["stage"] != "complete" or stored["status"] != "measured":
            output["incomplete_populations"][key] = {
                "status": stored["status"],
                "stage": stored["stage"],
                "completed_calls": len(stored["records"]),
                "original_json_sha256": stored["original_json_sha256"],
                "force_retention": stored["force_retention"],
            }
            continue
        assert stored["receipts"]["exit-status.txt"].strip() == "0"
        receipt = stored["independent_validation_receipt"]
        assert receipt["library_sha256"] == source["library_sha256"]
        assert (
            stored["receipts"]["library.sha256"].split()[0] == source["library_sha256"]
        )
        assert len(receipt["calls"]) == len(REFERENCE["PHASES"])
        assert [row["phase"] for row in receipt["calls"]] == REFERENCE["PHASES"]
        assert receipt["maximum_energy_error"] <= 1e-8
        assert receipt["maximum_force_error"] <= 1e-7
        references = {
            row["geometry"]: row
            for row in oracle["records"]
            if row["phase"] in ("cold", "moved")
        }
        for row, checked in zip(stored["records"], receipt["calls"], strict=True):
            assert all(row[field] == value for field, value in checked.items())
            error = abs(row["energy"] - references[row["geometry"]]["energy"])
            assert error <= 1e-8 and abs(error - row["energy_error"]) <= 1e-15
        assert len(bytes.fromhex(stored["original_force_stream_sha256"])) == 32
        output["observational_populations"][key] = receipt
    comparable = {**output["populations"], **output["observational_populations"]}
    for key, candidate in comparable.items():
        cohort, arm, case = key.split("/")
        baseline_key = f"{cohort}/baseline/{case}"
        if arm != "occupied" or baseline_key not in comparable:
            continue
        baseline = comparable[baseline_key]
        output["comparisons"][f"{cohort}/{case}"] = {
            "baseline": baseline["complete_seconds"],
            "occupied": candidate["complete_seconds"],
            "change_percent": {
                phase: 100 * (seconds / baseline["complete_seconds"][phase] - 1)
                for phase, seconds in candidate["complete_seconds"].items()
            },
            "offline_force_replay": {
                "baseline": baseline_key in output["populations"],
                "occupied": key in output["populations"],
            },
        }
    assert set(reconstructed) == set(range(planes.shape[1]))
    assert (
        hashlib.sha256(
            b"".join(reconstructed[index] for index in range(planes.shape[1]))
        ).hexdigest()
        == encoded["reconstructed_sha256"]
    )
    output["qualified_calls"] = sum(
        len(row["calls"]) for row in output["populations"].values()
    )
    output["qualified_profile_calls"] = sum(
        len(profile["windows"]) for profile in output["profiles"].values()
    )
    output["observational_calls"] = sum(
        len(row["calls"]) for row in output["observational_populations"].values()
    )
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, sort_keys=True))
