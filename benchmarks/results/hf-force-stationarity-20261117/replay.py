"""Recompute the retained scientific gates without loading a native library or GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import numpy as np


def replay_bundle(bundle: dict[str, object]) -> dict[str, object]:
    """Validate complete observations, independent DF derivatives and provenance."""
    assert bundle["schema"] == "generativeqc.hf-force-stationarity-qualification/v1"
    assert bundle["gates"] == {"energy": 1e-9, "forces": 1e-7, "translation": 1e-7}
    recipe = bundle["recipe"]
    assert hashlib.sha256(recipe["text"].encode()).hexdigest() == recipe["sha256"]
    baseline = bundle["baseline"]
    baseline_error = float(
        np.max(
            np.abs(
                np.asarray(baseline["samples"][0]["forces"])
                - np.asarray(baseline["oracle"]["records"]["initial"]["forces"])
            )
        )
    )
    assert baseline_error > bundle["gates"]["forces"]
    assert (
        baseline["native_probe"]["source_identity"]
        != bundle["source"]["native_source_identity"]
    )
    maximum = dict.fromkeys(bundle["gates"], 0.0)
    phases = {"cold", "warm", "moved", "moved_warm", "energy_only", "energy_to_forces"}
    cases = {
        f"{method}-{mode}-{basis}"
        for method, mode in [("rhf", "direct"), ("uhf", "direct"), ("rhf", "df")]
        for basis in ["def2-svp", "def2-tzvp"]
    }
    endpoints = force_endpoints = issue_observations = 0
    issue_force_error = 0.0
    assert set(bundle["nodes"]) == {"n1", "n2"}
    for node, acquisition in bundle["nodes"].items():
        assert acquisition["exit"].strip() == "0"
        assert acquisition["binary_sha256"] == bundle["binary_sha256"]
        observed = set()
        inventory = (
            acquisition["metadata"]["device-inventory.csv"].replace("-", "").lower()
        )
        assert acquisition["device_uuid"] in inventory
        for observation in acquisition["observations"]:
            arm, repeat, case = (observation[key] for key in ("arm", "repeat", "case"))
            group = (arm, repeat, case)
            assert group not in observed
            observed.add(group)
            record = observation["record"]
            assert (
                f"JobId={record['slurm_job_id']} "
                in acquisition["metadata"]["scontrol.txt"]
            )
            assert record["device_uuid"] == acquisition["device_uuid"]
            assert (
                record["native_probe"]["source_identity"]
                == bundle["source"]["native_source_identity"]
            )
            device = record["native_probe"]["device"]
            assert device["registry_status"] == "matched"
            assert (device["major"], device["minor"]) == (12, 0)
            assert ("5090" if node == "n1" else "6000") in device["name"]
            assert device["release_build"] == 1 and device["fast_compile"] == 0
            assert device["official_profile"] == (
                "sm_120" if arm == "auto" else "portable_cuda"
            )
            assert record["public_controls"] == {
                "energy": 1e-9,
                "density": 1e-5,
                "screening": 1e-12,
            }
            assert len(record["samples"]) == 6
            assert {sample["phase"] for sample in record["samples"]} == phases
            if record["mode"] == "df":
                oracle = record["oracle"]
                assert oracle["metric_relative_threshold"] == 1e-10
                assert oracle["auxiliary_representation"] == "cartesian"
                for reference in oracle["records"].values():
                    assert reference["fd_steps"] == [0.01, 0.02]
                    derivatives = []
                    for step, samples in zip(
                        reference["fd_steps"], reference["fd_samples"], strict=True
                    ):
                        forces = np.full_like(reference["forces"], np.nan, dtype=float)
                        assert len(samples) == forces.size
                        positions = set()
                        for sample in samples:
                            position = (sample["atom"], sample["axis"])
                            assert position not in positions
                            positions.add(position)
                            assert len(sample["ranks"]) == 4
                            assert set(sample["ranks"]) == {reference["metric_rank"]}
                            energies = sample["energies"]
                            assert set(energies) == {"-2", "-1", "1", "2"}
                            forces[position] = (
                                energies["2"]
                                - 8 * energies["1"]
                                + 8 * energies["-1"]
                                - energies["-2"]
                            ) / (12 * step)
                        assert np.all(np.isfinite(forces))
                        derivatives.append(forces)
                    assert (
                        float(np.max(np.abs(derivatives[0] - derivatives[1]))) <= 5e-8
                    )
                    np.testing.assert_allclose(
                        (16 * derivatives[0] - derivatives[1]) / 15,
                        reference["forces"],
                        rtol=0,
                        atol=1e-12,
                    )
            for sample in record["samples"]:
                phase = sample["phase"]
                reference = record["oracle"]["records"][
                    "moved" if phase.startswith("moved") else "initial"
                ]
                assert sample["converged"] and sample["executed_backend"] == "cuda"
                if phase in {"warm", "moved_warm", "energy_to_forces"}:
                    assert sample["warm_start_used"]
                errors = {"energy": abs(sample["energy"] - reference["energy"])}
                endpoints += 1
                if phase == "energy_only":
                    assert sample["forces"] is None
                else:
                    actual, expected = (
                        np.asarray(sample["forces"]),
                        np.asarray(reference["forces"]),
                    )
                    assert actual.shape == expected.shape and np.all(
                        np.isfinite(actual)
                    )
                    errors["forces"] = float(np.max(np.abs(actual - expected)))
                    errors["translation"] = float(
                        np.max(np.abs(np.sum(actual, axis=0)))
                    )
                    force_endpoints += 1
                    if case == "rhf-direct-def2-tzvp" and phase in {"cold", "warm"}:
                        issue_observations += 1
                        issue_force_error = max(issue_force_error, errors["forces"])
                for quantity, error in errors.items():
                    assert np.isfinite(error) and error <= bundle["gates"][quantity], (
                        node,
                        group,
                        phase,
                        quantity,
                        error,
                    )
                    np.testing.assert_allclose(
                        error, sample["errors"][quantity], rtol=1e-12, atol=1e-15
                    )
                    maximum[quantity] = max(maximum[quantity], error)
        assert observed == {
            (arm, repeat, case)
            for arm in ["auto", "portable"]
            for repeat in range(1, 5)
            for case in cases
        }
    assert bundle["nodes"]["n1"]["device_uuid"] != bundle["nodes"]["n2"]["device_uuid"]
    assert (endpoints, force_endpoints, issue_observations) == (576, 480, 32)
    return {
        "endpoint_samples": endpoints,
        "force_samples": force_endpoints,
        "maximum_errors": maximum,
        "baseline_force_error": baseline_error,
        "original_issue_samples": issue_observations,
        "original_issue_force_error": issue_force_error,
    }


def replay(path: Path) -> dict[str, object]:
    """Keep historical evidence distinct from separately acquired integration."""
    bundle = json.loads(lzma.decompress(path.read_bytes()))
    historical = {"source": bundle["source"], **replay_bundle(bundle)}
    if "integrated_qualification" not in bundle:
        return historical
    integrated = bundle["integrated_qualification"]
    assert (
        integrated["source"]["native_source_identity"]
        != bundle["source"]["native_source_identity"]
    )
    return {
        "pre_reconciliation": historical,
        "integrated": {"source": integrated["source"], **replay_bundle(integrated)},
    }


if __name__ == "__main__":
    print(
        json.dumps(replay(Path(__file__).with_name("qualification.json.xz")), indent=2)
    )
