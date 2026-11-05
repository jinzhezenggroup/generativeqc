"""Fail closed on incomplete LR endpoint populations or changed semantic work."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
import sys
from pathlib import Path
from typing import Any

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
    "protocol",
    "thread_controls",
    "occupied_xc_env",
)
WORK_KEYS = ("iterations", "fock_builds", "warm_start_used", "warm_start_fallback")
GATES = (1e-9, 1e-7, 3e-8)


def population(publication: dict[str, Any], device: str, name: str) -> dict[str, Any]:
    """Authenticate the full cold+three-warm cohort and its actual loaded ELF."""
    key = f"{device}/{name}"
    record = publication["records"][key]
    identity = publication["identities"][key]
    assert record["status"] == "complete" and record["warm_repeats"] == 3
    assert record["atoms"] == 96 and record["mode"] == "direct"
    assert len(record["samples"]) == 4
    assert identity["source_identity"] == record["source_revision"]
    assert identity["scheduler"] == record["scheduler"]
    for key in ("source_identity", "library_sha256"):
        assert identity[key] == record["native_build"][key]
    if name.endswith("previous"):
        assert (
            identity["source_identity"]
            == "4c54ccfb3d38e5ec86b147a019d8ef99e87c8b8247ca5a42bf50a84027dc61ed"
        )
        assert (
            identity["library_sha256"]
            == "3142176ecd915a1fe5eaaced6aa5cb4b3da3af6a1ff7b71b4cf6a2255f199799"
        )
    else:
        assert identity["source_identity"] == publication["source_identity"]
        assert (
            identity["library_sha256"]
            == publication["candidate_image"]["library_sha256"]
        )
    for index, sample in enumerate(record["samples"]):
        assert (sample["phase"], sample["repeat"]) == (
            ("cold", 0) if index == 0 else ("warm", index - 1)
        )
        assert sample["converged"] and sample["iterations"] > 0
        assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
        assert math.isfinite(sample["energy"])
        assert len(sample["forces"]) == 96
        assert all(len(row) == 3 for row in sample["forces"])
        assert all(math.isfinite(value) for row in sample["forces"] for value in row)
        assert sample["warm_start_used"] == bool(index)
    return record


def compare(
    before: dict[str, Any], after: dict[str, Any], *, legacy_image: bool = False
) -> dict[str, Any]:
    """Use paired observables and all timings, never a selected fast repeat."""
    for key in TASK_KEYS:
        assert before[key] == after[key], key
    worst = [0.0, 0.0, 0.0]
    work = []
    schedule_changes = []
    for baseline, candidate in zip(before["samples"], after["samples"], strict=True):
        settings_before = dict(baseline["settings"])
        settings_after = dict(candidate["settings"])
        if legacy_image:
            previous_tile = settings_before.pop("resolved_force_tile_points", None)
            current_tile = settings_after.pop("resolved_force_tile_points", None)
            if previous_tile != current_tile:
                schedule_changes.append(
                    {"resolved_force_tile_points": [previous_tile, current_tile]}
                )
        assert settings_before == settings_after, "mathematical settings"
        for key in WORK_KEYS:
            assert baseline[key] == candidate[key], key
        delta = [
            left - right
            for left_row, right_row in zip(
                candidate["forces"], baseline["forces"], strict=True
            )
            for left, right in zip(left_row, right_row, strict=True)
        ]
        errors = [
            abs(candidate["energy"] - baseline["energy"]) / 96,
            max(map(abs, delta)),
            math.sqrt(statistics.mean(value * value for value in delta)),
        ]
        assert all(error <= gate for error, gate in zip(errors, GATES, strict=True)), (
            errors
        )
        worst = [max(left, right) for left, right in zip(worst, errors, strict=True)]
        if before["method"] == "wb97m-v":
            previous = baseline["ks_diagnostic"]
            current = candidate["ks_diagnostic"]
            assert previous["nonlocal_work"] == current["nonlocal_work"]
            for key in ("coulomb_applications", "exchange_applications"):
                assert (
                    previous["coulomb_exchange_work"][key]
                    == current["coulomb_exchange_work"][key]
                ), key
            assert current["coulomb_exchange_work"]["complete"]
            precision = candidate["precision"]
            assert precision["complete"] and precision["operator_inventory_complete"]
            assert precision["operator_work_counters_valid"]
        work.append({key: candidate[key] for key in WORK_KEYS})
    report = {
        "maximum_errors": worst,
        "gates": GATES,
        "work": work,
        "upstream_execution_schedule_delta": schedule_changes,
        "lr_only_attribution": not legacy_image,
    }
    for phase in ("cold", "warm"):
        initial = [row["seconds"] for row in before["samples"] if row["phase"] == phase]
        final = [row["seconds"] for row in after["samples"] if row["phase"] == phase]
        report[phase] = {
            "before_seconds": initial,
            "after_seconds": final,
            "before_median": statistics.median(initial),
            "after_median": statistics.median(final),
            "ratio": statistics.median(final) / statistics.median(initial),
        }
    return report


def test_gate(
    name: str, receipt: dict[str, Any], expected: tuple[int, int, int]
) -> None:
    """Keep initial fixture/environment failures distinct from accepted gates."""
    assert len(receipt["original_xml_sha256"]) == 64
    cases = receipt["cases"]
    failures = [case for case in cases if case["outcome"] in {"failure", "error"}]
    skipped = [case for case in cases if case["outcome"] == "skipped"]
    assert all(
        case["outcome"] in {"passed", "failure", "error", "skipped"} for case in cases
    )
    assert (
        len(cases) - len(failures) - len(skipped),
        len(skipped),
        len(failures),
    ) == expected
    if name == "host-including-native-missing":
        assert all(
            "test_hf_resources" in case["attributes"]["classname"] for case in failures
        )
        assert all(
            "native library was not found" in json.dumps(case["details"])
            for case in failures
        )


def reference_gate(
    root: Path,
    publication: dict[str, Any],
    *,
    receipt_name: str = "reference-reuse.json",
    expected_pairs: int = 48,
) -> None:
    """Reuse the original independent reference only for static observables."""
    receipt = json.loads((root / receipt_name).read_text())
    repository = root.parents[2]
    path = repository / receipt["reference_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["reference_sha256"]
    previous = json.loads(lzma.decompress(path.read_bytes()))
    reference = previous["records"]["A100/reference"]
    assert reference["status"] == "complete" and len(reference["samples"]) == 2
    worst = [0.0, 0.0, 0.0]
    pairs = 0
    for key, record in publication["records"].items():
        if "/wb-" not in key:
            continue
        for field, expected in previous["cohort"].items():
            assert record[field] == expected, field
        for sample in record["samples"]:
            for target in reference["samples"]:
                delta = [
                    actual - expected
                    for actual_row, expected_row in zip(
                        sample["forces"], target["forces"], strict=True
                    )
                    for actual, expected in zip(actual_row, expected_row, strict=True)
                ]
                errors = [
                    abs(sample["energy"] - target["energy"]) / 96,
                    max(map(abs, delta)),
                    math.sqrt(statistics.mean(value * value for value in delta)),
                ]
                assert all(
                    error <= gate
                    for error, gate in zip(errors, receipt["gates"], strict=True)
                )
                worst = [
                    max(actual, expected)
                    for actual, expected in zip(worst, errors, strict=True)
                ]
                pairs += 1
    assert (
        pairs == receipt["pairs"] == expected_pairs
        and worst == receipt["maximum_errors"]
    )
    assert not receipt["timing_reused"]


def integration_gate(root: Path, measured: dict[str, Any]) -> dict[str, Any]:
    """Authenticate later default-only execution, never relabel old paired data."""
    integration = json.loads(
        lzma.decompress((root / "integration.json.xz").read_bytes())
    )
    assert integration["schema"] == "generativeqc.direct-lr-weighted-integration.v1"
    assert integration["status"] == "qualified"
    assert integration["source_identity"] == (
        "ff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa"
    )
    assert integration["base_commit"] == "9a72d770a8d32d0491a51161fd477d3891d8d86d"
    assert integration["master_commit"] == "1943a24b7c851a9c2bf4f68a1026c9f637c98f12"
    assert integration["lr_selector"] == "unset"
    delivery = json.loads((root / "delivery.json").read_text())
    assert delivery["source_identity"] == integration["source_identity"]
    assert delivery["qualified_base"]["oid"] == integration["base_commit"]
    assert (
        delivery["latest_master"]["oid"] == "863359f4bc8eb019d69e05623d992ba4cad51ad7"
    )
    trees = []
    for key in ("qualified_base", "latest_master"):
        raw = delivery[key]["raw_commit"].encode()
        header = b"commit " + str(len(raw)).encode() + b"\0"
        assert (
            hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()
            == delivery[key]["oid"]
        )
        trees.append(raw.decode().splitlines()[0])
    assert trees[0] == trees[1] and trees[0].startswith("tree ")
    assert integration["candidate_image"]["library_sha256"] == (
        "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a"
    )
    build = json.loads((root / "integration-build.json").read_text())
    assert build["exit-code.txt"].strip() == "0"
    assert build["identity"] == integration["candidate_image"]
    assert build["default_policy"]["status"] == 0
    assert build["default_policy"]["policy_version"] == 2
    failed_build = build["initial_mixed_object_build"]
    assert (
        json.loads(failed_build["failed-image-default-policy.json"])["policy_version"]
        == 1
    )
    assert json.loads(failed_build["identity.json"])["library_sha256"] == (
        "d5d4181ffdb0b6a09ad50a5aeed7b5858a075c408e245e5afaa2b116ac262160"
    )
    report = {}
    for device in ("A100", "RTX5090"):
        assert integration["cohort_exit_codes"][device] == 0
        assert integration["device_policy_receipts"][device]["status"] == 0
        assert integration["device_policy_receipts"][device]["policy_version"] == 2
        current = population(integration, device, "wb-default")
        scheduler_key = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
        assert current["scheduler"][scheduler_key]
        assert current["scheduler"]["CUDA_VISIBLE_DEVICES"]
        result = {"wb": compare(population(measured, device, "wb-on"), current)}
        if device == "RTX5090":
            result["semilocal"] = compare(
                population(measured, device, "r2-on"),
                population(integration, device, "r2-default"),
            )
        for comparison in result.values():
            comparison["lr_only_attribution"] = False
            assert comparison["warm"]["ratio"] <= 1.05
        report[device] = result
    tests = json.loads(
        lzma.decompress(
            (root / "science" / "integration-test-cases.json.xz").read_bytes()
        )
    )
    test_gate("integration-device-libcint", tests["device-libcint"], (8, 0, 0))
    test_gate("integration-native-resources", tests["native-resources"], (27, 0, 0))
    test_gate("initial-policy-mismatch", tests["initial-policy-mismatch"], (0, 0, 1))
    assert "native SCF default policy returned an invalid receipt" in json.dumps(
        tests["initial-policy-mismatch"]
    )
    reference_gate(
        root,
        integration,
        receipt_name="integration-reference-reuse.json",
        expected_pairs=16,
    )
    expected = json.loads((root / "integration-summary.json").read_text())
    assert json.dumps(report, sort_keys=True) == json.dumps(expected, sort_keys=True)
    return report


def main() -> None:
    """Replay every sample and failure boundary without loading a GPU runtime."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    publication = json.loads(
        lzma.decompress((root / "measurements.json.xz").read_bytes())
    )
    assert publication["schema"] == "generativeqc.direct-lr-weighted-endpoints.v1"
    assert (
        publication["source_identity"]
        == "c313decb8884e69c3b44b6fe2e72c9491869b64313db3e4fcdd56f8e5f52a2a1"
    )
    assert (
        publication["candidate_image"]["library_sha256"]
        == "d9b33f54155186cc3240ecd325a290bcd1fecadcc136ed9c8ce72107f88f3486"
    )
    report = {}
    for device in ("A100", "RTX5090"):
        assert publication["cohort_exit_codes"][device] == 0
        records = {
            name: population(publication, device, name)
            for name in ("wb-previous", "wb-off", "wb-on")
        }
        result = {
            "same_image": compare(records["wb-off"], records["wb-on"]),
            "previous_image": compare(
                records["wb-previous"], records["wb-on"], legacy_image=True
            ),
            "upstream_image_without_lr": compare(
                records["wb-previous"], records["wb-off"], legacy_image=True
            ),
        }
        assert result["same_image"]["warm"]["ratio"] < 1.0
        assert result["previous_image"]["warm"]["ratio"] <= 1.05
        if device == "RTX5090":
            result["semilocal_nonregression"] = compare(
                population(publication, device, "r2-previous"),
                population(publication, device, "r2-on"),
                legacy_image=True,
            )
            assert result["semilocal_nonregression"]["warm"]["ratio"] <= 1.05
        report[device] = result
    tests = json.loads(
        lzma.decompress((root / "science" / "test-cases.json.xz").read_bytes())
    )
    for name, counts in (
        ("host-including-native-missing", (2076, 164, 11)),
        ("initial-fault-fixture", (106, 0, 8)),
        ("fixed-fault-fixture", (114, 0, 0)),
        ("contracts", (154, 0, 0)),
        ("device-libcint", (8, 0, 0)),
        ("native-resources", (27, 0, 0)),
    ):
        test_gate(name, tests[name], counts)
    reference_gate(root, publication)
    expected = json.loads((root / "summary.json").read_text())
    assert json.dumps(report, sort_keys=True) == json.dumps(expected, sort_keys=True)
    if (root / "integration.json.xz").exists():
        report["latest_master_default_integration"] = integration_gate(
            root, publication
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
