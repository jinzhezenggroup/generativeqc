"""Recheck the integrated PR image and all 72 independent scientific endpoints."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import runpy
import struct
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
HISTORICAL = ROOT.parent / "r2scan3c-final-validation-20261015"
SELECTOR = "GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"
FORCE_SELECTOR = "GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS"
PHASES = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def require(condition: bool, message: str) -> None:
    """Keep evidence checks active even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def read_compressed(path: Path) -> dict:
    """Read immutable JSON evidence without recreating benchmark measurements."""
    return json.loads(lzma.decompress(path.read_bytes()))


def restore_dictionary(baseline: object, delta: dict) -> object:
    """Restore the complete observation, including unchanged historical metadata."""
    if "value" in delta:
        return delta["value"]
    restored = dict(baseline)
    for key in delta["removed"]:
        del restored[key]
    for key, change in delta["fields"].items():
        restored[key] = restore_dictionary(restored.get(key), change)
    return restored


def verify() -> None:
    """Recompute provenance, complete endpoint gates and independent force errors."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        payload = (ROOT / name).read_bytes()
        require(
            len(payload) == expected["bytes"]
            and hashlib.sha256(payload).hexdigest() == expected["sha256"],
            f"checksum mismatch: {name}",
        )
    qualification = read_compressed(ROOT / "qualification.json.xz")
    observations = qualification["observations"]
    receipts = qualification["receipts"]
    for name, document in receipts["documents"].items():
        payload = document["content"].encode()
        require(
            len(payload) == document["bytes"]
            and hashlib.sha256(payload).hexdigest() == document["sha256"],
            f"retained document checksum mismatch: {name}",
        )
    frozen = observations["frozen_source"]
    archive = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "archive", frozen["head"], "upstream"]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        assets = {
            member.name: hashlib.sha256(tree.extractfile(member).read()).hexdigest()
            for member in tree.getmembers()
            if member.isfile()
        }
    asset_identity = hashlib.sha256(
        (json.dumps(assets, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    require(
        len(assets) == frozen["upstream_assets"]["file_count"]
        and asset_identity == frozen["upstream_assets"]["inventory_sha256"],
        "upstream compiler input mismatch",
    )
    delta = frozen["native_input_delta"]
    baseline = REPOSITORY / delta["base"]
    require(
        hashlib.sha256(baseline.read_bytes()).hexdigest() == delta["base_sha256"],
        "historical inventory changed",
    )
    historical = read_compressed(baseline)
    old_by_shape = {
        (point["density_fitting"], point["atoms"]): point
        for point in historical["comparisons"]
    }
    observations["comparisons"] = [
        restore_dictionary(
            old_by_shape[point["density_fitting"], point["atoms"]], point["delta"]
        )
        for point in observations.pop("comparison_deltas")
    ]
    files = dict(historical["frozen_source"]["files"])
    for name in delta["removed"]:
        del files[name]
    files.update(delta["changed"])
    identity = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode()
    ).hexdigest()
    require(
        len(files) == delta["file_count"]
        and identity == frozen["source_identity"] == manifest["native_source_identity"],
        "source inventory/image mismatch",
    )
    native_archive = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "archive", frozen["head"], "--", *files]
    )
    with tarfile.open(fileobj=io.BytesIO(native_archive)) as tree:
        committed_inputs = {
            member.name: hashlib.sha256(tree.extractfile(member).read()).hexdigest()
            for member in tree.getmembers()
            if member.isfile()
        }
    require(
        committed_inputs == files, "native inputs differ from the qualified Git head"
    )
    require(
        observations["native_library_sha256"] == manifest["native_library_sha256"],
        "native image mismatch",
    )
    verifier = runpy.run_path(str(HISTORICAL / "refresh.py"))["verify_comparison"]
    old_by_size = {
        point["atoms"]: point
        for point in historical["comparisons"]
        if point["density_fitting"] == "none"
    }
    try:
        verifier(old_by_size[96])
    except ValueError as error:
        require(
            "none/96/moved: complete-endpoint nonregression gate failed" in str(error),
            "unexpected historical failure",
        )
    else:
        raise ValueError("historical Direct/96 moved failure was waived")
    require(
        {
            (point["density_fitting"], point["atoms"])
            for point in observations["comparisons"]
        }
        == {("none", 48), ("none", 96)},
        "missing Direct qualification shape",
    )
    calls = 0
    for comparison in observations["comparisons"]:
        verifier(comparison)
        atoms = comparison["atoms"]
        require(
            comparison["arms"]["control"]["protocol"]
            == comparison["arms"]["candidate"]["protocol"]
            == old_by_size[atoms]["arms"]["candidate"]["protocol"],
            "scientific protocol or tolerance changed",
        )
        for arm, point in comparison["arms"].items():
            native = point["engines"]["native"]
            build = native["native_build"]
            require(
                build["probe"]["source_identity"] == identity
                and build["library_sha256"] == manifest["native_library_sha256"],
                "stale comparison image",
            )
            require(
                native["source_file_sha256"] == frozen["consumer_source_file_sha256"],
                "consumer provenance changed",
            )
            settings = native["native_schedule_settings"]
            require(
                FORCE_SELECTOR not in settings
                and (
                    settings.get(SELECTOR) == "0"
                    if arm == "control"
                    else SELECTOR not in settings
                ),
                "not an unset-default versus explicit-0 comparison",
            )
            rows = native["records"]
            require(
                Counter(row["phase"] for row in rows) == PHASES,
                "incomplete phase population",
            )
            require(
                point["independent_references"]
                == old_by_size[atoms]["arms"]["candidate"]["independent_references"],
                "independent oracle changed",
            )
            references = {
                row["geometry"]: row for row in point["independent_references"]
            }
            previous_forces = {
                geometry: reference["forces"]
                for geometry, reference in references.items()
            }
            force_receipt = receipts["forces"][point["population"]]
            require(
                force_receipt["raw_sha256"] == native["raw_sha256"]
                and len(force_receipt["records"]) == len(rows),
                "raw force receipt mismatch",
            )
            for row, force_row in zip(rows, force_receipt["records"], strict=True):
                require(
                    all(
                        row[key] == force_row[key]
                        for key in ("phase", "geometry", "repeat")
                    ),
                    "force population reordered",
                )
                reference = references[row["geometry"]]
                expected = reference["forces"]
                prior_values = [
                    value
                    for vector in previous_forces[row["geometry"]]
                    for value in vector
                ]
                prior_bits = struct.unpack(
                    f"<{atoms * 3}Q", struct.pack(f"<{atoms * 3}d", *prior_values)
                )
                encoded = bytes.fromhex(force_row["forces_xor64le_hex"])
                require(len(encoded) == atoms * 3 * 8, "force encoding shape mismatch")
                xor_bits = struct.unpack(f"<{atoms * 3}Q", encoded)
                values = struct.unpack(
                    f"<{atoms * 3}d",
                    struct.pack(
                        f"<{atoms * 3}Q",
                        *(
                            value ^ prior
                            for value, prior in zip(xor_bits, prior_bits, strict=True)
                        ),
                    ),
                )
                actual = [
                    values[index : index + 3] for index in range(0, len(values), 3)
                ]
                previous_forces[row["geometry"]] = actual
                require(
                    len(actual) == len(expected) == atoms
                    and all(len(vector) == 3 for vector in actual),
                    "force shape mismatch",
                )
                require(
                    all(math.isfinite(value) for vector in actual for value in vector),
                    "nonfinite force",
                )
                energy_error = abs(row["energy"] - reference["energy"])
                force_error = max(
                    abs(value - target)
                    for vector, oracle in zip(actual, expected, strict=True)
                    for value, target in zip(vector, oracle, strict=True)
                )
                require(
                    math.isfinite(energy_error)
                    and energy_error <= 1e-8
                    and force_error <= 1e-7,
                    "independent scientific gate failed",
                )
                require(
                    math.isclose(row["energy_error"], energy_error, abs_tol=1e-15)
                    and math.isclose(row["force_error"], force_error, abs_tol=1e-15),
                    "retained numerical error differs from forces",
                )
                require(
                    row["gate"] is True
                    and row["converged"]
                    and row["status"] == 0
                    and math.isfinite(row["complete_seconds"])
                    and row["complete_seconds"] > 0,
                    "incomplete endpoint",
                )
                calls += 1
    require(
        calls
        == observations["complete_native_calls"]
        == manifest["complete_comparison_calls"]
        == 48,
        "wrong complete-call count",
    )
    require(
        {point["atoms"] for point in receipts["small_systems"]} == {3, 12},
        "missing small-system regression",
    )
    require(
        sum(len(point["records"]) for point in receipts["small_systems"]) == 24
        and all(
            row["gate"] is True
            and row["energy_error"] <= 1e-8
            and row["force_error"] <= 1e-7
            for point in receipts["small_systems"]
            for row in point["records"]
        ),
        "small-system numerical failure",
    )
    for point in receipts["small_systems"]:
        atoms = point["atoms"]
        require(
            point["protocol"] == old_by_size[atoms]["arms"]["candidate"]["protocol"]
            and point["native_build"]["probe"]["source_identity"] == identity
            and point["native_build"]["library_sha256"]
            == manifest["native_library_sha256"]
            and point["source_file_sha256"] == frozen["consumer_source_file_sha256"]
            and SELECTOR not in point["native_schedule_settings"]
            and FORCE_SELECTOR not in point["native_schedule_settings"],
            "small-system source, default or scientific protocol changed",
        )
        require(
            Counter(row["phase"] for row in point["records"]) == PHASES,
            "incomplete small-system phase population",
        )
        references = {
            reference["geometry"]: reference
            for reference in old_by_size[atoms]["arms"]["candidate"][
                "independent_references"
            ]
        }
        for row in point["records"]:
            reference = references[row["geometry"]]
            actual = row["forces"]
            require(
                len(actual) == atoms and all(len(vector) == 3 for vector in actual),
                "small-system force shape mismatch",
            )
            require(
                all(math.isfinite(value) for vector in actual for value in vector),
                "nonfinite small-system force",
            )
            energy_error = abs(row["energy"] - reference["energy"])
            force_error = max(
                abs(value - target)
                for vector, oracle in zip(actual, reference["forces"], strict=True)
                for value, target in zip(vector, oracle, strict=True)
            )
            require(
                math.isfinite(energy_error)
                and energy_error <= 1e-8
                and force_error <= 1e-7,
                "small-system independent scientific gate failed",
            )
            require(
                math.isclose(row["energy_error"], energy_error, abs_tol=1e-15)
                and math.isclose(row["force_error"], force_error, abs_tol=1e-15),
                "small-system errors differ from independent force arrays",
            )
            require(
                row["converged"]
                and row["status"] == 0
                and math.isfinite(row["complete_seconds"])
                and row["complete_seconds"] > 0,
                "incomplete small-system endpoint",
            )
    for node in ("n4", "n5"):
        outcome = json.loads(receipts["documents"][f"outcome-{node}.json"]["content"])
        require(
            outcome["exit_code"] == 0 and outcome["job"],
            "Slurm qualification job failed",
        )
        scheduler = receipts["documents"][f"scheduler-{node}.txt"]["content"]
        require(
            f"JobId={outcome['job']} " in scheduler and "JobState=RUNNING" in scheduler,
            "missing live qualification allocation",
        )
    device_outcome = json.loads(
        receipts["documents"]["outcome-compensated-tests.json"]["content"]
    )
    device_log = receipts["documents"]["compensated-tests-n4.log"]["content"]
    require(
        device_outcome["exit_code"] == 0
        and device_outcome["all_53_cases_executed"] is True
        and device_outcome["job"]
        and "\n53 passed in" in device_log
        and f"verified_source_identity={identity}" in device_log
        and f"verified_library_sha256={manifest['native_library_sha256']}"
        in device_log,
        "device predicate or compensated-XC execution is incomplete",
    )
    scheduler = receipts["documents"]["scheduler-compensated-tests.txt"]["content"]
    require(
        f"JobId={device_outcome['job']} " in scheduler
        and "JobState=COMPLETED" in scheduler
        and "ExitCode=0:0" in scheduler
        and "TimeLimit=00:10:00" in scheduler,
        "missing successful finite device-test allocation",
    )
    require(
        set(receipts["diagnostics"]) == {"none-48", "none-96"},
        "missing resident-route diagnostic",
    )
    for diagnostic in receipts["diagnostics"].values():
        require(
            diagnostic["native_build"]["probe"]["source_identity"] == identity
            and SELECTOR not in diagnostic["native_schedule_settings"],
            "diagnostic source/default mismatch",
        )
        require(
            bool(diagnostic["resident_proofs"])
            and all(
                event["valid"] and event["cuda_error"] == 0
                for event in diagnostic["resident_proofs"]
            ),
            "resident proof not executed",
        )
    print(
        "48 complete matched-image endpoints and 24 small-system calls pass; historical Direct/96 moved remains rejected"
    )


if __name__ == "__main__":
    verify()
