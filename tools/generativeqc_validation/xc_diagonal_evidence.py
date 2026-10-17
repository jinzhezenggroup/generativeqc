"""Recheck stored complete-endpoint gates without a GPU, network or calculation.

The independent geometry oracles are shared with the prerequisite MD-J packet.
Forces are lossless XOR deltas from shared, already-reviewed FP64 samples, not
rounded residuals. This reader reconstructs every original bit, prints a summary
and never creates or overwrites an output file.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/xc-diagonal-potential-20261016"
)
REFERENCE = runpy.run_path(
    str(ROOT.parent / "md-j-sparse-support-20261016/summarize.py")
)


def summarize() -> dict[str, Any]:
    """Recompute every E/F gate, preserve all samples and separate profile clocks."""
    with lzma.open(ROOT / "evidence.json.xz", "rt") as stream:
        data = json.load(stream)
    with np.load(ROOT / "forces.npz", allow_pickle=False) as archive:
        byteplanes = archive["xor_byteplanes"]
    assert data["schema"] == "xc-diagonal-potential-v3"
    assert byteplanes.dtype == np.dtype("u1") and byteplanes.ndim == 3
    assert byteplanes.shape[0] == 8 and byteplanes.shape[2] == 3 * 96
    for name in ("forces.npz", "evidence.json.xz"):
        shared = ROOT.parent / "md-j-sparse-support-20261016" / name
        assert (
            hashlib.sha256(shared.read_bytes()).hexdigest()
            == data["receipts"][f"shared-anchor-{name}-sha256"]
        )
    force_digest = hashlib.sha256()
    force_values = 0
    output: dict[str, Any] = {"populations": {}, "comparisons": {}, "profiles": {}}

    def source_key(campaign: str, arm: str) -> str:
        return (
            "baseline"
            if arm == "baseline"
            else "stack"
            if campaign == "qualified-stack"
            else "candidate"
        )

    def reconstruct(stored: dict[str, Any], source: str) -> dict[str, Any]:
        """Verify the retained projection, including every reconstructed force bit."""
        nonlocal force_values
        expected = data["receipts"]["final-source-input-comparison.json"]["sources"][
            source
        ]
        assert (
            stored["native_build"]["probe"]["source_identity"]
            == expected["input_merkle_sha256"]
        )
        assert (
            stored["native_build"]["library_sha256"]
            == stored["receipts"]["library.sha256"].split()[0]
        )
        native = dict(stored)
        case = native.pop("protocol_key")
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        shared = REFERENCE["load"](f"raw/candidate-{case}.json.xz")
        anchors = {
            row["geometry"]: row
            for row in shared["records"]
            if row["phase"] in ("cold", "moved")
        }
        native["protocol"] = oracle["protocol"]
        native["records"] = []
        for stored_row in stored["records"]:
            row = dict(stored_row)
            column = row.pop("forces_delta")
            atoms = oracle["protocol"]["atoms"]
            assert 0 <= column < byteplanes.shape[1]
            anchor = np.zeros(3 * 96, dtype="<f8")
            anchor[: 3 * atoms] = np.asarray(
                anchors[row["geometry"]]["forces"], dtype="<f8"
            ).reshape(-1)
            delta = byteplanes[:, column, :].T.copy().reshape(-1).view("<u8")
            actual = (delta ^ anchor.view("<u8")).view("<f8")[: 3 * atoms]
            assert np.isfinite(actual).all()
            force_digest.update(actual.tobytes())
            force_values += actual.size
            row["forces"] = actual.reshape(atoms, 3).tolist()
            native["records"].append(row)
        assert stored["receipts"]["exit-status.txt"].strip() == "0"
        for field in ("receipts", "original_json_sha256", "canonical_json_sha256"):
            native.pop(field)
        canonical = json.dumps(
            native, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        assert (
            hashlib.sha256(canonical.encode()).hexdigest()
            == stored["canonical_json_sha256"]
        )
        return native

    for key, stored in data["populations"].items():
        campaign, arm, case = key.split("/")
        native = reconstruct(stored, source_key(campaign, arm))
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        result = REFERENCE["qualify"](native, oracle)
        result["raw_json_sha256"] = stored["original_json_sha256"]
        output["populations"][key] = result
    for key, candidate in output["populations"].items():
        campaign, arm, case = key.split("/")
        if arm != "candidate":
            continue
        baseline_key = f"{campaign}/baseline/{case}"
        baseline = output["populations"].get(baseline_key)
        if baseline is None:
            baseline = data["observational_controls"][baseline_key][
                "independent_validation_receipt"
            ]
        output["comparisons"][f"{campaign}/{case}"] = {
            "baseline": baseline["complete_seconds"],
            "candidate": candidate["complete_seconds"],
            "change_percent": {
                phase: 100 * (seconds / baseline["complete_seconds"][phase] - 1)
                for phase, seconds in candidate["complete_seconds"].items()
            },
        }
    for key, profile in data["profiles"].items():
        campaign, arm = key.split("/")
        native = reconstruct(profile["scientific_records"], source_key(campaign, arm))
        oracle = REFERENCE["load"]("raw/reference-r2scan-3c-96.json.xz")
        assert native["status"] == "diagnostic_completed"
        assert [row["phase"] for row in native["records"]] == ["cold", "warm"]
        reference = oracle["records"][0]
        assert reference["phase"] == "cold" and reference["geometry"] == 0
        for row in native["records"]:
            energy_error = abs(row["energy"] - reference["energy"])
            force_error = float(
                np.max(np.abs(np.asarray(row["forces"]) - reference["forces"]))
            )
            assert row["geometry"] == 0 and row["gate"] and row["converged"]
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert abs(energy_error - row["energy_error"]) <= 1e-15
            assert abs(force_error - row["force_error"]) <= 1e-15
        output["profiles"][key] = {
            "force_work": REFERENCE["force_work"](profile["census"])
        }
        for phase in ("cold", "warm"):
            row = profile[phase]
            native_row = next(
                record for record in native["records"] if record["phase"] == phase
            )
            assert all(
                value == native_row[field] for field, value in row["record"].items()
            )
            assert abs(row["host_reconciliation_error_seconds"]) < 1e-7
            assert abs(row["device_reconciliation_error_seconds"]) < 1e-7
            ledger = row["complete_endpoint_wall_ledger"]
            assert abs(ledger["complete_reconciliation_error_seconds"]) < 1e-7
            assert (
                abs(
                    ledger["complete_seconds"]
                    - sum(ledger["complete_exclusive_wall_seconds"].values())
                )
                < 1e-7
            )
            device: dict[str, float] = {}
            for kernel in row["kernels"]:
                for scope, seconds in kernel["phase_kernel_seconds"].items():
                    device[scope] = device.get(scope, 0.0) + seconds
            output["profiles"][key][phase] = {
                "exclusive_host_seconds": row["exclusive_host_seconds"],
                "device_window_partition_seconds": row[
                    "exclusive_device_window_seconds"
                ],
                "kernel_device_seconds": device,
                "complete_wall_ledger": row["complete_endpoint_wall_ledger"],
                "profile_record": row["record"],
            }
    output["qualified_calls"] = sum(
        len(row["calls"]) for row in output["populations"].values()
    )
    assert len(output["profiles"]) == 4
    assert len(output["populations"]) in (2, 12, 16)
    output["observational_controls"] = data["observational_controls"]
    assert len(output["observational_controls"]) + len(output["populations"]) == 16
    for key, control in output["observational_controls"].items():
        campaign, arm, case = key.split("/")
        scalar = control["scalar_projection"]
        expected = data["receipts"]["final-source-input-comparison.json"]["sources"][
            source_key(campaign, arm)
        ]
        assert (
            scalar["native_build"]["probe"]["source_identity"]
            == expected["input_merkle_sha256"]
        )
        assert control["exit_status"].strip() == "0"
        oracle = REFERENCE["load"](f"raw/reference-{case}.json.xz")
        references = {
            row["geometry"]: row
            for row in oracle["records"]
            if row["phase"] in ("cold", "moved")
        }
        receipt = control["independent_validation_receipt"]
        assert scalar["native_build"]["library_sha256"] == receipt["library_sha256"]
        assert len(receipt["calls"]) == 12
        assert [row["phase"] for row in receipt["calls"]] == REFERENCE["PHASES"]
        for row, actual in zip(receipt["calls"], scalar["records"], strict=True):
            assert all(value == actual[field] for field, value in row.items())
            energy_error = abs(actual["energy"] - references[row["geometry"]]["energy"])
            assert abs(energy_error - row["energy_error"]) <= 1e-15
            assert row["gate"] and row["converged"]
            assert row["energy_error"] <= 1e-8 and row["force_error"] <= 1e-7
            assert row["iterations"] > 0 and row["fock_builds"] > 0
        assert receipt["complete_seconds"] == {
            phase: statistics.median(
                row["complete_seconds"]
                for row in scalar["records"]
                if row["phase"] == phase
            )
            for phase in dict.fromkeys(REFERENCE["PHASES"])
        }
        if arm == "candidate":
            baseline = data["observational_controls"][f"{campaign}/baseline/{case}"][
                "independent_validation_receipt"
            ]
            output["comparisons"][f"{campaign}/{case}"] = {
                "baseline": baseline["complete_seconds"],
                "candidate": receipt["complete_seconds"],
                "force_evidence": "collection-time independent checks; raw-force SHA receipt, not offline force replay",
                "change_percent": {
                    phase: 100 * (seconds / baseline["complete_seconds"][phase] - 1)
                    for phase, seconds in receipt["complete_seconds"].items()
                },
            }
    output["observational_calls"] = 12 * len(output["observational_controls"])
    assert output["qualified_calls"] + output["observational_calls"] == 192
    output["qualified_profile_calls"] = 8
    output["reconstructed_fp64_force_values"] = force_values
    output["force_stream_sha256"] = force_digest.hexdigest()
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, allow_nan=False))
