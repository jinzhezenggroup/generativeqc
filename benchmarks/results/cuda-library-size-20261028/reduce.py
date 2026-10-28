"""Replay this one-off binary/endpoint qualification without a compiler or GPU."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from statistics import median


def reduce(data: dict) -> dict:
    artifacts = data["final-artifacts"]["artifacts"]
    result = {
        "library_bytes": {name: item["bytes"] for name, item in artifacts.items()},
        "saved_bytes": artifacts["baseline"]["bytes"] - artifacts["combined"]["bytes"],
        "df_kernels": data["df-kernel-payloads"]["counts"],
        "cohorts": {},
        "calls": 0,
        "maximum_paired_energy_error": 0.0,
        "maximum_paired_force_error": 0.0,
    }
    # This is historical, opt-in executable-image evidence, not a required test
    # of production function/source spelling or a constraint on future kernels.
    assert data["df-kernel-payloads"]["counts"]["changed"] == 0
    assert data["full-build"]["returncode"] == 0
    endpoints = data["endpoints"]
    for kind in ("direct", "df", "dffast"):
        for size in (24, 96):
            cohort = f"{kind}-{size}"
            observations = {arm: [] for arm in ("baseline", "candidate")}
            maximum_oracle_energy = maximum_oracle_force = 0.0
            oracle = endpoints[f"{cohort}-oracle.json"]["oracle"]
            for index in range(5):
                pair = [
                    endpoints[f"{cohort}-{index}-{arm}.json"] for arm in observations
                ]
                for arm, population in zip(observations, pair, strict=True):
                    expected = artifacts[
                        "baseline" if arm == "baseline" else "combined"
                    ]
                    assert population["library_sha256"] == expected["sha256"]
                    observations[arm].append(population["rows"])
                    for row in population["rows"]:
                        reference = oracle[int(row["phase"] == "moved")]
                        maximum_oracle_energy = max(
                            maximum_oracle_energy,
                            abs(row["energy"] - reference["energy"]),
                        )
                        maximum_oracle_force = max(
                            maximum_oracle_force, force_error(row, reference)
                        )
                        result["calls"] += 1
                for before, after in zip(pair[0]["rows"], pair[1]["rows"], strict=True):
                    assert before["phase"] == after["phase"]
                    assert before["iterations"] == after["iterations"]
                    assert before["fock_builds"] == after["fock_builds"]
                    # Mixed arithmetic/retry work must be conserved too.
                    for field in (
                        "attempted_fp32_weighted_tasks",
                        "attempted_fp32_primitive_sources",
                        "weighted_task_retries",
                    ):
                        a = ((before.get("precision") or {}).get("gradient") or {}).get(
                            field
                        )
                        b = ((after.get("precision") or {}).get("gradient") or {}).get(
                            field
                        )
                        assert a == b
                    result["maximum_paired_energy_error"] = max(
                        result["maximum_paired_energy_error"],
                        abs(before["energy"] - after["energy"]),
                    )
                    result["maximum_paired_force_error"] = max(
                        result["maximum_paired_force_error"], force_error(before, after)
                    )
            assert maximum_oracle_energy < 1e-7
            assert maximum_oracle_force < (1e-5 if kind == "dffast" else 1e-6)
            phases = {}
            for phase in ("cold", "warm", "moved"):
                times = {
                    arm: [
                        median(
                            r["seconds"] for r in rows if r["phase"].startswith(phase)
                        )
                        for rows in population
                    ]
                    for arm, population in observations.items()
                }
                ratio = median(times["candidate"]) / median(times["baseline"])
                assert ratio <= 1.02, (cohort, phase, ratio)
                phases[phase] = {"process_seconds": times, "ratio": ratio}
            result["cohorts"][cohort] = {
                "phases": phases,
                "maximum_oracle_energy_error": maximum_oracle_energy,
                "maximum_oracle_force_error": maximum_oracle_force,
            }
    assert result["maximum_paired_energy_error"] <= 1e-9
    assert result["maximum_paired_force_error"] <= 1e-8
    result["maximum_endpoint_ratio"] = max(
        p["ratio"] for c in result["cohorts"].values() for p in c["phases"].values()
    )
    return result


def force_error(a: dict, b: dict) -> float:
    return max(
        abs(x - y)
        for v, w in zip(a["forces"], b["forces"], strict=True)
        for x, y in zip(v, w, strict=True)
    )


if __name__ == "__main__":
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as handle:
        print(json.dumps(reduce(json.load(handle)), indent=2, sort_keys=True))
