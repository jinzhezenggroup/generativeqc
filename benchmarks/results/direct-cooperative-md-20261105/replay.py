"""Replay radial-only endpoint gates without loading either engine or a GPU.

Run with the published evidence directory as the optional positional argument.
Cross-device reference comparisons qualify observables only, never timing.
"""

import itertools
import json
import lzma
import math
import statistics
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

NATIVE_GATES = (1e-9, 1e-7, 3e-8)
COMMON_GATES = (1e-5, 1e-4, 3e-5)
TASK_KEYS = (
    "method",
    "mode",
    "atoms",
    "geometry_bohr",
    "orbital_basis",
    "basis_representation",
    "charge",
    "spin",
    "runner_sha256",
)
WORK_KEYS = (
    "iterations",
    "fock_builds",
    "warm_start_used",
    "warm_start_fallback",
)
INCREMENTAL_KEYS = (
    "policy_version",
    "requested",
    "active",
    "quartet_work_counters_valid",
    "anchor_full_builds",
    "delta_builds",
    "periodic_rebuilds",
    "bypass_full_builds",
    "post_scf_full_builds",
    "anchor_updates",
    "full_candidate_shell_quartets",
    "full_rejected_shell_quartets",
    "full_admitted_shell_quartets",
    "full_admitted_quartet_tiles",
    "delta_candidate_shell_quartets",
    "delta_rejected_shell_quartets",
    "delta_admitted_shell_quartets",
    "delta_admitted_quartet_tiles",
)


def load_population(folder: Path, arm: str) -> dict[str, Any]:
    """Require the full declared population and retained process success."""
    assert (folder / f"{arm}.exit-code.txt").read_text().strip() == "0"
    record = json.loads(lzma.decompress((folder / f"{arm}.json.xz").read_bytes()))
    assert record["status"] == "complete" and record["warm_repeats"] == 3
    assert len(record["samples"]) == 4 and record["atoms"] == 96
    assert record["method"] == "r2scan-3c" and record["mode"] == "direct"
    assert record["occupied_xc_env"] is None
    for index, sample in enumerate(record["samples"]):
        assert (sample["phase"], sample["repeat"]) == (
            ("cold", 0) if index == 0 else ("warm", index - 1)
        )
        assert sample["converged"] and sample["iterations"] > 0
        assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
        assert math.isfinite(sample["energy"])
        assert len(sample["forces"]) == record["atoms"]
        assert all(len(row) == 3 for row in sample["forces"])
        assert all(math.isfinite(value) for row in sample["forces"] for value in row)
    if arm != "reference":
        identity = json.loads((folder / f"{arm}-identity.json").read_text())
        assert identity["source_identity"] == record["source_revision"]
        for key in ("source_identity", "library_sha256"):
            assert identity[key] == record["native_build"][key]
        assert identity["scheduler"] == record["scheduler"]
    return record


def compare_observables(
    actual: dict[str, Any],
    expected: dict[str, Any],
    gates: tuple[float, float, float],
) -> dict[str, Any]:
    """Compare every cold/warm pairing so no sample is cherry-picked."""
    for key in TASK_KEYS:
        assert actual[key] == expected[key], key
    worst = [0.0, 0.0, 0.0]
    pairs = 0
    for actual_sample, expected_sample in itertools.product(
        actual["samples"], expected["samples"]
    ):
        differences = [
            left - right
            for actual_row, expected_row in zip(
                actual_sample["forces"], expected_sample["forces"], strict=True
            )
            for left, right in zip(actual_row, expected_row, strict=True)
        ]
        errors = (
            abs(actual_sample["energy"] - expected_sample["energy"]) / actual["atoms"],
            max(map(abs, differences)),
            math.sqrt(statistics.mean(value * value for value in differences)),
        )
        assert all(error <= gate for error, gate in zip(errors, gates, strict=True)), (
            errors
        )
        worst = [
            max(previous, current)
            for previous, current in zip(worst, errors, strict=True)
        ]
        pairs += 1
    return {"pairs": pairs, "maximum_errors": worst, "gates": gates}


def summarize_pair(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Qualify identical work and report complete populations, not best samples."""
    result = {"observables": compare_observables(candidate, baseline, NATIVE_GATES)}
    for baseline_sample, candidate_sample in zip(
        baseline["samples"], candidate["samples"], strict=True
    ):
        assert baseline_sample["settings"] == candidate_sample["settings"]
        for key in WORK_KEYS:
            assert baseline_sample[key] == candidate_sample[key], key
        for key in INCREMENTAL_KEYS:
            assert (
                baseline_sample["incremental_direct_jk"][key]
                == candidate_sample["incremental_direct_jk"][key]
            ), key
    for phase in ("cold", "warm"):
        before = [
            sample["seconds"]
            for sample in baseline["samples"]
            if sample["phase"] == phase
        ]
        after = [
            sample["seconds"]
            for sample in candidate["samples"]
            if sample["phase"] == phase
        ]
        result[phase] = {
            "baseline_seconds": before,
            "candidate_seconds": after,
            "baseline_median": statistics.median(before),
            "candidate_median": statistics.median(after),
            "median_ratio": statistics.median(after) / statistics.median(before),
            "corresponding_ratios": [
                right / left for left, right in zip(before, after, strict=True)
            ],
        }
    result["work"] = [
        {key: sample[key] for key in (*WORK_KEYS, "incremental_direct_jk")}
        for sample in candidate["samples"]
    ]
    return result


def main() -> None:
    """Fail closed on incomplete process, numerical, work, or 5090 evidence."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    populations = {}
    report = {
        "scope": "one static geometry; compiler/artifact caches reused; no A100 reference timing"
    }
    for device in ("5090", "a100"):
        folder = root / device
        assert (folder / "exit-code.txt").read_text().strip() == "0"
        baseline = load_population(folder, "baseline")
        candidate = load_population(folder, "candidate")
        assert (
            baseline["accelerator"]["pci_bus_id"]
            == candidate["accelerator"]["pci_bus_id"]
        )
        assert baseline["scheduler"] == candidate["scheduler"]
        report[device] = summarize_pair(baseline, candidate)
        populations[device] = (baseline, candidate)
    assert all(
        ratio <= 1.05 for ratio in report["5090"]["warm"]["corresponding_ratios"]
    )
    reference = load_population(root / "5090", "reference")
    report["reference_observables"] = {
        f"{device}-{arm}": compare_observables(record, reference, COMMON_GATES)
        for device, records in populations.items()
        for arm, record in zip(("baseline", "candidate"), records, strict=True)
    }
    independent = root / "5090" / "independent.xml"
    assert (independent.parent / "independent.exit-code.txt").read_text().strip() == "0"
    xml_bytes = independent.read_bytes()
    assert len(xml_bytes) <= 1 << 20
    assert (
        b"<!DOCTYPE" not in xml_bytes.upper() and b"<!ENTITY" not in xml_bytes.upper()
    )
    document = element_tree.fromstring(xml_bytes)  # noqa: S314 - bounded pytest XML, DTD/entities rejected
    cases = list(document.iter("testcase"))
    assert len(cases) == 24
    assert all(
        not list(case.iter("skipped"))
        and not list(case.iter("failure"))
        and not list(case.iter("error"))
        for case in cases
    )
    report["independent_libcint_cases"] = len(cases)
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
