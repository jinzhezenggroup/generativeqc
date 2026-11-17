"""Recompute scientific eligibility and balanced paired endpoint timing ratios."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from collections import defaultdict
from pathlib import Path

import numpy as np


def compare_records(records: dict, fixtures: dict) -> dict:
    """Classify every retained process without dropping valid slower endpoints."""
    expected_names = {
        f"{profile}-{case}-{order}-{'master' if order in (1, 4, 6, 7) else 'candidate'}.json"
        for profile in ("auto", "portable")
        for case in fixtures
        for order in range(1, 9)
    }
    assert set(records) == expected_names
    groups = defaultdict(dict)
    for name, record in sorted(records.items()):
        label, order, arm = Path(name).stem.rsplit("-", 2)
        profile, case = label.split("-", 1)
        order = int(order)
        assert record["arm"] == arm and record["profile"] == profile
        expected_source = (
            "1832792c9425c24a67a436fa117928011d1291fc43124e7e2c2edc25d2811101"
            if arm == "master"
            else "1615b290a1e7d706a1e360036b410d490d73b1835d5ac77f1de1ef4d937d8d40"
        )
        assert record["native_probe"]["source_identity"] == expected_source
        assert record["python_source_identity"] == expected_source
        assert arm == ("master" if order in (1, 4, 6, 7) else "candidate")
        fixture = fixtures[case]
        assert len(record["samples"]) == 6
        assert {sample["phase"] for sample in record["samples"]} == {
            "cold",
            "warm",
            "moved",
            "moved_warm",
            "energy_only",
            "energy_to_forces",
        }
        assert record["native_probe"]["device"]["release_build"] == 1
        assert record["native_probe"]["device"]["fast_compile"] == 0
        assert record["native_probe"]["device"]["official_profile"] == (
            "sm_120" if profile == "auto" else "portable_cuda"
        )
        assert record["first_use_ns"] == (
            record["import_ns"]
            + record["construct_ns"]
            + record["samples"][0]["endpoint_ns"]
        )
        for sample in record["samples"]:
            assert sample["geometry"] == (
                "moved" if sample["phase"].startswith("moved") else "initial"
            )
            assert (sample["forces"] is None) == (sample["phase"] == "energy_only")
            assert sample["endpoint_ns"] > 0 and sample["execute_ns"] > 0
            assert sample["endpoint_ns"] == sample["execute_ns"] + (
                record["prepare_ns"] if sample["phase"] == "cold" else 0
            )
            assert sample["iterations"] >= 0 and sample["fock_builds"] is not None
            expected = fixture["oracle"]["records"][sample["geometry"]]
            errors = {"energy": abs(sample["energy"] - expected["energy"])}
            if sample["forces"] is not None:
                forces = np.asarray(sample["forces"])
                errors.update(
                    forces=float(np.max(np.abs(forces - expected["forces"]))),
                    translation=float(np.max(np.abs(forces.sum(axis=0)))),
                )
            valid = (
                sample["converged"]
                and sample["status"] == 0
                and sample["executed_backend"] == "cuda"
            )
            valid = bool(
                valid
                and all(
                    value <= (1e-9 if key == "energy" else 1e-7)
                    for key, value in errors.items()
                )
            )
            assert valid == sample["scientifically_valid"]
            groups[(profile, case, sample["phase"])][order] = {
                **sample,
                "valid": valid,
                "uuid": record["device_uuid"],
            }
        groups[(profile, case, "first_use")][order] = {
            "endpoint_ns": record["first_use_ns"],
            "valid": record["samples"][0]["scientifically_valid"],
            "iterations": record["samples"][0]["iterations"],
            "fock_builds": record["samples"][0]["fock_builds"],
            "uuid": record["device_uuid"],
        }
    results = []
    for key, observations in sorted(groups.items()):
        assert set(observations) == set(range(1, 9)), (key, sorted(observations))
        assert len({item["uuid"] for item in observations.values()}) == 1
        before = [observations[index] for index in (1, 4, 6, 7)]
        after = [observations[index] for index in (2, 3, 5, 8)]
        old = np.array([item["endpoint_ns"] for item in before], dtype=float)
        new = np.array([item["endpoint_ns"] for item in after], dtype=float)
        generator = np.random.default_rng(3092)
        indices = generator.integers(0, 4, size=(20000, 4))
        draws = new[indices].mean(axis=1) / old[indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975]).tolist()
        eligible = all(item["valid"] for item in before + after)
        verdict = (
            "scientific_failure_not_ranked"
            if not eligible
            else "regression"
            if lower > 1
            else "nonregression"
            if upper <= 1
            else "inconclusive"
        )
        results.append(
            {
                "profile": key[0],
                "case": key[1],
                "phase": key[2],
                "master_valid": [item["valid"] for item in before],
                "candidate_valid": [item["valid"] for item in after],
                "master_ms": (old / 1e6).tolist(),
                "candidate_ms": (new / 1e6).tolist(),
                "candidate_over_master_mean": float(new.mean() / old.mean()),
                "candidate_over_master_median": float(np.median(new) / np.median(old)),
                "paired_bootstrap_ratio_95pct": [lower, upper],
                "verdict": verdict,
                "master_work": [
                    [item["iterations"], item["fock_builds"]] for item in before
                ],
                "candidate_work": [
                    [item["iterations"], item["fock_builds"]] for item in after
                ],
            }
        )
    return {
        "schema": "generativeqc.hf-force-stationarity-performance/v1",
        "order": "ABBA then BAAB, A=master, B=candidate; four fresh pairs per group",
        "zero_slowdown_ratio": 1,
        "bootstrap_seed": 3092,
        "bootstrap_draws": 20000,
        "results": results,
        "performance_accepted": bool(
            results
            and any(
                item["verdict"] != "scientific_failure_not_ranked" for item in results
            )
            and all(
                item["verdict"] in ("nonregression", "scientific_failure_not_ranked")
                and all(item["candidate_valid"])
                for item in results
            )
        ),
    }


def compare(root: Path) -> dict:
    """Load one node's raw acquisition and its independently generated fixtures."""
    records = {
        path.name: json.loads(path.read_text())
        for path in root.glob("*.json")
        if not path.name.endswith(".auxiliary.json")
    }
    fixtures = {
        path.stem: json.loads(path.read_text())
        for path in (root.parent / "fixtures").glob("*.json")
    }
    return compare_records(records, fixtures)


def replay_artifact(path: Path) -> dict:
    """Verify archived provenance and recompute all scientific and timing rows."""
    qualification = json.loads(
        lzma.decompress(path.with_name("qualification.json.xz").read_bytes())
    )
    campaign = json.loads(lzma.decompress(path.read_bytes()))
    reference = qualification["integrated_qualification"]
    assert campaign["gates"] == {"energy": 1e-9, "forces": 1e-7, "translation": 1e-7}
    assert campaign["candidate"] == reference["source"]
    assert (
        hashlib.sha256(campaign["recipes"]["endpoint.py"].encode()).hexdigest()
        == campaign["recipe_sha256"]
    )
    fixtures = {}
    for name, stored in campaign["fixtures"].items():
        assert hashlib.sha256(stored["text"].encode()).hexdigest() == stored["sha256"]
        fixtures[name] = json.loads(stored["text"])
    expected_fixtures = {
        observation["case"]: observation["record"]
        for observation in reference["nodes"]["n1"]["observations"]
        if observation["arm"] == "auto" and observation["repeat"] == 1
    }
    assert fixtures == expected_fixtures
    assert len(fixtures) == 6
    assert set(campaign["nodes"]) == {"n1", "n2"}
    candidate_binary_sha256 = {
        Path(line.split()[1]).stem.rsplit("-", 1)[-1]: line.split()[0]
        for line in reference["binary_sha256"].splitlines()
    }
    assert set(candidate_binary_sha256) == {"auto", "portable"}
    results = {}
    uuids = set()
    for node, acquisition in campaign["nodes"].items():
        assert acquisition["driver_exit"] == 0
        metadata = acquisition["metadata"]
        assert metadata["exit-auto-portable.txt"].strip() == "0"
        for name, record in acquisition["records"].items():
            assert (
                f"JobId={record['slurm_job_id']} "
                in metadata["scontrol-auto-portable.txt"]
            )
            assert record["device_uuid"] in metadata[
                "device-inventory-auto-portable.csv"
            ].lower().replace("-", "")
            assert record["recipe_sha256"] == campaign["recipe_sha256"]
            expected_binary = (
                campaign["baseline_binary_sha256"]
                if record["arm"] == "master"
                else candidate_binary_sha256
            )[record["profile"]]
            assert record["library_sha256"] == expected_binary
            case = name.split("-", 1)[1].rsplit("-", 2)[0]
            assert record["fixture_sha256"] == campaign["fixtures"][case]["sha256"]
            uuids.add(record["device_uuid"])
        results[node] = compare_records(acquisition["records"], fixtures)
    assert len(uuids) == 2
    rejected = campaign["rejected_setup"]["records"]
    assert rejected
    assert any(
        record["arm"] == "master"
        and record["native_probe"]["source_identity"]
        != campaign["baseline"]["native_source_identity"]
        for record in rejected.values()
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).with_name("performance.json.xz"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = replay_artifact(args.artifact)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for node, result in summary.items():
        counts = {}
        for row in result["results"]:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
            if row["verdict"] == "regression":
                print(
                    node,
                    row["profile"],
                    row["case"],
                    row["phase"],
                    f"candidate/master={row['candidate_over_master_mean']:.4f}",
                    row["paired_bootstrap_ratio_95pct"],
                )
        print(node, counts, "Performance accepted:", result["performance_accepted"])
