"""Recompute native force-map acceptance from retained independent E/F arrays.

This is an opt-in evidence replay, not a permanent source fingerprint or
performance gate. Timings are observations; no new default is promoted here.
"""

from __future__ import annotations

import json
import lzma
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def force_array(value: object, atoms: int) -> list[float]:
    """Reject broadcastable/nonfinite force arrays before computing errors."""
    assert isinstance(value, list) and len(value) == atoms
    assert all(isinstance(row, list) and len(row) == 3 for row in value)
    result = [component for row in value for component in row]
    assert all(
        type(component) in (int, float) and math.isfinite(component)
        for component in result
    )
    return result


def validate(payload: dict) -> dict:
    """Independently check oracle parity, executed work and every retained arm."""
    maximum_energy_error = maximum_force_error = 0.0
    endpoints = 0
    timings = []
    for case in payload["cases"]:
        for metric in case["reference_metrics"]:
            assert type(metric["dimension"]) is int and metric["dimension"] > 0
            assert all(
                math.isfinite(metric[field])
                for field in ("omega", "minimum", "maximum", "relative_threshold")
            )
            assert (
                metric["relative_threshold"]
                == case["input"]["density_fitting_relative_threshold"]
            )
            assert (
                metric["minimum"] > metric["maximum"] * metric["relative_threshold"] > 0
            )
        capacity = case["capacity"]
        for envelope in capacity.values():
            assert len(envelope) == 9
            assert all(type(value) is int and value >= 0 for value in envelope)
            assert 0 < envelope[4] <= envelope[5]
            assert envelope[6] > 0
        assert capacity["automatic"][4] >= capacity["bounded"][4]
        assert capacity["automatic"][6] >= capacity["bounded"][6]
        references = case["references"]
        assert len(references) == 2
        atoms = len(references[0]["forces"])
        for reference in references:
            assert math.isfinite(reference["energy"])
            force_array(reference["forces"], atoms)
        samples = {(row["mode"], row["stage"]): row for row in case["samples"]}
        assert len(samples) == len(case["samples"]) == 12
        for mode in ("dense", "auto", "minimum"):
            for stage in ("cold", "warm", "moved", "moved-warm"):
                row = samples[mode, stage]
                reference = references[int(stage.startswith("moved"))]
                actual = force_array(row["forces"], atoms)
                expected = force_array(reference["forces"], atoms)
                assert math.isfinite(row["energy"])
                energy_error = abs(row["energy"] - reference["energy"])
                force_error = max(
                    abs(left - right)
                    for left, right in zip(actual, expected, strict=True)
                )
                assert energy_error <= 1e-8 and force_error <= 3e-7
                maximum_energy_error = max(maximum_energy_error, energy_error)
                maximum_force_error = max(maximum_force_error, force_error)
                assert math.isfinite(row["physical_residual_rms"])
                assert 0 <= row["physical_residual_rms"] <= 1e-9
                assert (
                    math.isfinite(row["endpoint_seconds"])
                    and row["endpoint_seconds"] > 0
                )
                work = row["work"]
                inputs = case.get("input", {})
                if inputs.get("nonlocal_grid") is not None and inputs.get(
                    "nonlocal_grid"
                ) != inputs.get("grid"):
                    assert work["shared_nonlocal_grid"] == 0
                assert bool(work["indexed_grid"]) == (mode == "auto")
                assert 0 < work["grid_active_point_ao"] <= work["grid_dense_point_ao"]
                assert (work["grid_active_point_ao"] < work["grid_dense_point_ao"]) == (
                    mode == "auto"
                )
                assert (work["grid_discovery_ao_jet_values"] > 0) == (
                    mode == "auto" and stage in ("cold", "moved")
                )
                dense = samples["dense", stage]
                assert abs(row["energy"] - dense["energy"]) <= 1e-9
                assert (
                    max(
                        abs(left - right)
                        for left, right in zip(
                            actual, force_array(dense["forces"], atoms), strict=True
                        )
                    )
                    <= 1e-9
                )
                assert (
                    work["grid_dense_point_ao"] == dense["work"]["grid_dense_point_ao"]
                )
                for counter in ("grid_ao_jet_values", "grid_projection_fma_pairs"):
                    assert type(work[counter]) is int and work[counter] >= 0
                    if mode == "auto":
                        assert work[counter] < dense["work"][counter]
                    else:
                        assert work[counter] == dense["work"][counter]
                endpoints += 1
        ratios = {
            stage: samples["auto", stage]["endpoint_seconds"]
            / samples["dense", stage]["endpoint_seconds"]
            for stage in ("cold", "warm", "moved", "moved-warm")
        }
        timings.append({"case": case["id"], "observed_auto_over_dense": ratios})
    assert endpoints
    return {
        "cases": len(payload["cases"]),
        "complete_native_endpoints": endpoints,
        "maximum_energy_error": maximum_energy_error,
        "maximum_force_error": maximum_force_error,
        "median_observed_warm_ratio": median(
            row["observed_auto_over_dense"]["warm"] for row in timings
        ),
        "timing_observations_not_promotion": timings,
    }


def validate_endpoints(payload: dict, acceptance: dict) -> dict:
    """Recheck fresh-process pairs without turning observations into promotion.

    Every pair shares the exact model, native binary, allocation and SCF branch.
    Warm samples may legitimately reuse the frozen converged density with zero
    SCF/Fock work. Cold and moved samples must perform their own Fock work.
    """
    references = {
        json.dumps(case["input"], sort_keys=True): case for case in acceptance["cases"]
    }
    pairs = defaultdict(dict)
    for run in payload["runs"]:
        key = (
            run["device"],
            run["input"]["method"],
            run["input"]["fitted"],
            run["pair"],
        )
        assert run["mode"] not in pairs[key]
        pairs[key][run["mode"]] = run
    observations = defaultdict(list)
    endpoints = 0
    for key, arms in pairs.items():
        assert set(arms) == {"auto", "dense"}
        auto, dense = arms["auto"], arms["dense"]
        assert auto["input"] == dense["input"]
        assert auto["process"] != dense["process"]
        for identity in ("host", "job", "visible_devices"):
            assert auto[identity] == dense[identity]
        ratios = {}
        durations = {mode: defaultdict(list) for mode in arms}
        assert len(auto["samples"]) == len(dense["samples"]) >= 6
        repeats = (len(auto["samples"]) - 2) // 2
        assert [sample["stage"] for sample in auto["samples"]] == (
            ["cold"] + ["warm"] * repeats + ["moved"] + ["moved-warm"] * repeats
        )
        for selected, baseline in zip(auto["samples"], dense["samples"], strict=True):
            assert selected["stage"] == baseline["stage"]
            stage = selected["stage"]
            assert stage in ("cold", "warm", "moved", "moved-warm")
            for counter in ("iterations", "fock_builds"):
                assert type(selected[counter]) is int and selected[counter] >= 0
                assert selected[counter] == baseline[counter]
            if stage in ("cold", "moved"):
                assert selected["fock_builds"] > 0
            for mode, sample in (("auto", selected), ("dense", baseline)):
                reference = references[json.dumps(arms[mode]["input"], sort_keys=True)][
                    "references"
                ][int(stage.startswith("moved"))]
                atoms = len(reference["forces"])
                actual = force_array(sample["forces"], atoms)
                expected = force_array(reference["forces"], atoms)
                assert math.isfinite(sample["energy"])
                assert abs(sample["energy"] - reference["energy"]) <= 1e-8
                assert (
                    max(
                        abs(left - right)
                        for left, right in zip(actual, expected, strict=True)
                    )
                    <= 3e-7
                )
                assert math.isfinite(sample["physical_residual_rms"])
                assert 0 <= sample["physical_residual_rms"] <= 1e-9
                assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
                work = sample["work"]
                inputs = arms[mode]["input"]
                if inputs.get("nonlocal_grid") is not None and inputs.get(
                    "nonlocal_grid"
                ) != inputs.get("grid"):
                    assert work["shared_nonlocal_grid"] == 0
                assert bool(work["indexed_grid"]) == (mode == "auto")
                assert (work["grid_discovery_ao_jet_values"] > 0) == (
                    mode == "auto" and stage in ("cold", "moved")
                )
                durations[mode][stage].append(sample["seconds"])
                endpoints += 1
            assert abs(selected["energy"] - baseline["energy"]) <= 1e-9
            assert (
                max(
                    abs(left - right)
                    for left, right in zip(
                        force_array(selected["forces"], atoms),
                        force_array(baseline["forces"], atoms),
                        strict=True,
                    )
                )
                <= 1e-9
            )
            assert (
                selected["work"]["grid_dense_point_ao"]
                == baseline["work"]["grid_dense_point_ao"]
            )
            for counter in (
                "grid_active_point_ao",
                "grid_ao_jet_values",
                "grid_projection_fma_pairs",
            ):
                assert 0 <= selected["work"][counter] < baseline["work"][counter]
        for mode in arms:
            assert set(durations[mode]) == {"cold", "warm", "moved", "moved-warm"}
            assert len(durations[mode]["cold"]) == len(durations[mode]["moved"]) == 1
            assert (
                len(durations[mode]["warm"]) == len(durations[mode]["moved-warm"]) >= 2
            )
        for stage in durations["auto"]:
            ratios[stage] = median(durations["auto"][stage]) / median(
                durations["dense"][stage]
            )
        observations[key[:3]].append(
            {
                "pair": key[3],
                "ratios": ratios,
                "seconds": {
                    mode: {stage: median(values) for stage, values in stages.items()}
                    for mode, stages in durations.items()
                },
            }
        )
    domains = []
    for domain, samples in sorted(observations.items()):
        assert len(samples) >= 4
        domains.append(
            {
                "device": domain[0],
                "method": domain[1],
                "fitted": domain[2],
                "pairs": len(samples),
                "median_auto_over_dense": {
                    stage: median(sample["ratios"][stage] for sample in samples)
                    for stage in ("cold", "warm", "moved", "moved-warm")
                },
                "samples": samples,
            }
        )
    assert domains
    return {
        "complete_native_endpoints": endpoints,
        "observations_not_global_promotion": domains,
    }


if __name__ == "__main__":
    with lzma.open(Path(__file__).with_name("acceptance.json.xz"), "rt") as handle:
        acceptance = json.load(handle)
    print(json.dumps(validate(acceptance), indent=2, sort_keys=True))
    timings = Path(__file__).with_name("endpoints.json.xz")
    if timings.exists():
        with lzma.open(timings, "rt") as handle:
            print(
                json.dumps(
                    validate_endpoints(json.load(handle), acceptance),
                    indent=2,
                    sort_keys=True,
                )
            )
