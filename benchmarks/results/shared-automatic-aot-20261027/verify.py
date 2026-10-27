"""Explicit offline checks for retained #1123 evidence; not a required CI test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parent
LIMITS = {"REG": 255, "STACK": 4096, "LOCAL": 4096, "SHARED": 49152}


def close(actual, expected, tolerance):
    """Keep the original absolute scientific gates, with no relative widening."""
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def receipt(row, library):
    """Bind completed forces and all callback owners to the executed library."""
    value = row["artifact"]
    assert value["force_completed"] and value["source_count"] == 7
    assert value["fallback_reason"] in ("none", "minimum-workspace")
    kinds = ["point_artifact", "force_artifact"]
    if value["schema"].endswith(".v2"):
        kinds += ["point_executable_artifact", "force_executable_artifact"]
        assert value["force_schedule"] == "shared-staged"
        assert all(
            value["staged_point_work"][key] > 0
            for key in ("tile_bytes", "points", "launches")
        )
    for kind in kinds:
        assert value[kind]["sha256"] == library["sha256"]
        assert value[kind]["bytes"] == library["bytes"]
    assert (
        value["capacity"]["device_upper_bytes"]
        >= value["capacity"]["minimum_device_bytes"]
        > 0
    )
    return value


def finite_differences(reference):
    """Recheck original or freshly reconverged independent full/directional FDs."""
    forces = np.asarray(reference["forces"])
    projected = (
        forces.ravel()
        if len(reference["independent_fd"]) == forces.size
        else [np.dot(forces[-1], [0.2, -0.3, 0.4])]
    )
    for key in ("native_fd", "independent_fd", "analytic_directions"):
        close(projected, reference[key], 3e-5)
    close(forces.sum(axis=0), 0, 2e-8)
    return len(projected)


def endpoint_matrix(actual, references, library):
    """Compare every coordinate and center oracle without rebranding old FDs."""
    assert set(actual) == set(references)
    count = 0
    for name, record in actual.items():
        reference = references[name]
        assert (record["name"], record["spins"]) == (
            reference["name"],
            reference["spins"],
        )
        for row, expected in zip(record["rows"], reference["rows"], strict=True):
            assert row["phase"] == expected["phase"]
            receipt(row, library)
            close(row["energy"], row["reference_energy"], 3e-9)
            close(row["energy"], sum(expected["scf"]["components"].values()), 3e-9)
            close(row["forces"], expected["forces"], 3e-8)
            close(np.sum(row["forces"], axis=0), 0, 2e-8)
            count += 1
    return count


def audit(records, execution, library):
    """Join retained object/source receipts to matching executed callback IDs."""
    assert records["library"]["sha256"] == library["sha256"]
    assert records["limits"] == {
        "registers": 255,
        "stack": 4096,
        "local": 4096,
        "static_shared": 49152,
    }
    count = 0
    for record in execution.values():
        for row in record["rows"]:
            value = receipt(row, library)
            profile = records["profiles"][value["force_profile"]]
            assert profile["source_identity"] == value["force_source_identity"]
            assert profile["point_source_identity"] == value["point_source_identity"]
            for resources in profile["scopes"].values():
                assert resources
                for resource in resources.values():
                    for key, maximum in LIMITS.items():
                        assert resource[key] <= maximum
            count += 1
    assert count == len(records["executions"])
    assert all(
        row["dynamic_becke_shared_bytes"] <= LIMITS["SHARED"]
        for row in records["executions"]
    )


def compilation(records):
    """Check all scoped resource distributions and actual cold/hit counters."""
    for name, record in records.items():
        assert record["linked_device_image"] == record["installed_device_image"]
        assert not record["exported_point_workers"]
        assert (
            record["verified_launcher"]["commands"]
            == record["verified_launcher"]["verified"]
            > 0
        )
        for phase in ("cold", "hit"):
            measurement = record["compilation"][phase]
            assert measurement["returncode"] == 0
            assert (
                measurement["seconds"] > 0
                and measurement["aggregate_peak_rss_bytes"] > 0
            )
            stats = measurement["cache_after"]["stats"]
            assert not stats["cache_errors"]["counts"]
            if phase == "hit":
                assert not stats["cache_misses"]["counts"]
                assert sum(stats["cache_hits"]["counts"].values()) > 0
        for scope, resources in record["resources"].items():
            histogram = resources["histogram"]
            assert (
                sum(row["count"] for row in histogram)
                == resources["count"]
                == record["kernel_counts"][scope]
            )
            for key, maximum in resources["maxima"].items():
                assert maximum == max(row[key] for row in histogram)
            for key, maximum in LIMITS.items():
                assert resources["maxima"][key] <= maximum
            assert resources["spilling_kernels"] == sum(
                row["count"]
                for row in histogram
                if row["spill_load_bytes"] or row["spill_store_bytes"]
            )
        assert len(record["registrations"]) == (197 if name.endswith("full") else 11)
    assert records["baseline-full"]["groups"]["force_consumers"]["count"] == 394
    assert records["candidate-full"]["groups"]["force_consumers"]["count"] == 6
    assert (
        records["candidate-full"]["kernel_counts"]["automatic_force"]
        == records["candidate-narrow"]["kernel_counts"]["automatic_force"]
        == 180
    )


def performance(scope):
    """Replay the exact ABBA/BAAB work and <=5% gates with every stored sample."""
    record = load_json(ROOT / f"performance-{scope}.json.xz")
    spec = importlib.util.spec_from_file_location(
        "shared_aot_performance", ROOT / "drivers/analyze-performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"performance-{scope}"
        path.mkdir()
        for name, run in record["runs"].items():
            (path / name).write_text(json.dumps(run, indent=2) + "\n")
        result = module.analyze(path)
    assert json.loads(json.dumps(result)) == record["summary"] and result["passed"]
    return result


def process_cold(scope, compiler):
    """Keep fresh-process timing separate from the SDK's initialized context."""
    data = load_json(ROOT / f"cli-cold-{scope}.json.gz")
    assert (
        data["complete"]
        and data["passed"]
        and data["no_toolchain"]
        and data["ptx_jit_disabled"]
    )
    assert tuple(r["arm"] for r in data["records"]) == tuple(data["order"])
    assert len(data["records"]) == 8
    for record in data["records"]:
        libraries = [
            digest
            for path, digest in record["identities"].items()
            if Path(path).name.startswith("libgenerativeqc.so")
        ]
        assert libraries == [
            compiler[f"{record['arm']}-{scope}"]["installed_library"]["sha256"]
        ]
    for index, gate in enumerate(data["gates"]):
        samples = {"baseline": [], "candidate": []}
        anchor = data["records"][0]["rows"][index]["result"]
        for record in data["records"]:
            row = record["rows"][index]
            if row["result"]["iterations"] != anchor["iterations"]:
                assert (gate["system"], gate["name"], gate["spin"]) == (
                    "h4",
                    "MGGA_X_LTA",
                    2,
                )
                assert record["arm"] == "candidate"
                assert row["result"]["iterations"] == anchor["iterations"] + 1
            close(row["result"]["energy_hartree"], anchor["energy_hartree"], 3e-9)
            close(
                row["result"]["forces_hartree_per_bohr"],
                anchor["forces_hartree_per_bohr"],
                3e-8,
            )
            samples[record["arm"]].append(row["seconds"])
        assert samples == gate["samples"]
        ratio = statistics.median(samples["candidate"]) / statistics.median(
            samples["baseline"]
        )
        assert ratio == gate["ratio"] and ratio <= 1.05 and gate["passed"]


def main():
    """Authenticate the retained bundle before checking scientific acceptance."""
    assert __debug__, "run without Python -O"
    manifest = load_json(ROOT / "manifest.json")
    assert manifest["files"], "bind the completed publication before verification"
    assert manifest["source"]["candidate_verification"]["mismatches"] == []
    for name, expected in manifest["files"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
    for name, expected in manifest["references"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
    compiler = load_json(ROOT / "compilation.json.gz")
    compilation(compiler)
    science = load_json(ROOT / "science.json.xz")
    references = load_json(
        ROOT / "../automatic-native-cuda-20261027/references.json.gz"
    )
    assert (
        sum(
            finite_differences(row)
            for record in references.values()
            for row in record["rows"]
        )
        == 300
    )
    for name, key in (
        ("narrow", "candidate-narrow"),
        ("full", "candidate-full"),
        ("incumbent", "baseline-narrow"),
    ):
        assert (
            endpoint_matrix(science[name], references, compiler[key]["library"]) == 58
        )
    for name, key in (("narrow", "candidate-narrow"), ("full", "candidate-full")):
        audit(
            load_json(ROOT / f"audit-{name}.json.gz"),
            science[name],
            compiler[key]["library"],
        )
    assert len(science["aliases"]) == 8 and len(science["repair_fd"]) == 2
    assert (
        sum(
            finite_differences(row)
            for records in (science["aliases"], science["repair_fd"])
            for record in records.values()
            for row in record["rows"]
        )
        == 120
    )
    for name, record in science["repair_fd"].items():
        actual = science["narrow"][name]["rows"][0]
        expected = record["rows"][0]
        close(actual["forces"], expected["forces"], 3e-8)
        for key in ("force_source_identity", "point_source_identity"):
            assert actual["artifact"][key] == expected["artifact"][key]
    assert (
        endpoint_matrix(
            science["alias_incumbent"],
            science["aliases"],
            compiler["baseline-full"]["library"],
        )
        == 8
    )
    audit(
        load_json(ROOT / "audit-aliases.json.gz"),
        science["aliases"],
        compiler["candidate-full"]["library"],
    )
    aliases = science["alias_reuse"]
    assert (
        aliases["scientific_registration_count"] == 4
        and aliases["executable_worker_count"] == 5
    )
    assert len(set(aliases["restricted_executable_identities"])) == 1
    assert (
        len(set(aliases["polarized_executable_identities"]))
        == len(set(aliases["scientific_identities"]))
        == 4
    )
    for a, b in zip(
        science["phased"]["baseline"]["rows"],
        science["phased"]["candidate"]["rows"],
        strict=True,
    ):
        close(a["energy"], b["energy"], 1e-9)
        close(a["forces"], b["forces"], 1e-7)
        for row in (a, b):
            close(row["forces"], row["bounded_forces"], 1e-7)
            for fd in row["finite_differences"]:
                close(
                    -(fd["energies"][1] - fd["energies"][0]) / (2 * fd["step"]),
                    row["analytic"],
                    1e-7,
                )
            assert row["bounded_artifact"]["fallback_reason"] == "minimum-workspace"
            assert (
                row["artifact"]["stationary_work_v1"][0]
                - row["bounded_artifact"]["stationary_work_v1"][0]
                == 10487488
            )
        assert (
            a["artifact"]["stationary_work_v1"][3]
            == b["artifact"]["stationary_work_v1"][3]
        )
    for scope in ("narrow", "full"):
        installed = science["installed"][scope]
        assert len(installed["rows"]) == (22 if scope == "narrow" else 28)
        assert (
            installed["identities"][installed["loaded_library"]]
            == compiler[f"candidate-{scope}"]["installed_library"]["sha256"]
        )
        for row in installed["rows"]:
            filename = f"{row['name']}-{row['spins']}-True-False-False-h4.json"
            expected = (references if filename in references else science["aliases"])[
                filename
            ]["rows"][0]
            close(
                row["result"]["energy_hartree"],
                sum(expected["scf"]["components"].values()),
                3e-9,
            )
            close(row["result"]["forces_hartree_per_bohr"], expected["forces"], 3e-8)
        result = performance(scope)
        assert result["library_sha256"] == {
            arm: compiler[f"{arm}-{scope}"]["library"]["sha256"]
            for arm in ("baseline", "candidate")
        }
        process_cold(scope, compiler)
        print(
            scope,
            result["endpoint_count"],
            "SDK samples; maximum median ratio",
            max(r["ratio"] for r in result["gates"]),
        )
    math = load_json(ROOT / "point-math.json.gz")
    assert (
        math["registrations"] == 197
        and math["programs"] == 394
        and math["mathematical_changes"] == 0
    )
    capabilities = load_json(ROOT / "capabilities.json.gz")
    assert len(capabilities) == 197
    from generativeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

    for name, entries in capabilities.items():
        assert set(entries) == {"compiled-cuda"}
        assert entries["compiled-cuda"]["qualification"]["numerical_promotion"] is False
        capability = functional_capability(name, evidence=entries)
        assert (
            "compiled-cuda" in capability.qualified_stages
            and "forces" not in capability.qualified_stages
        )
    print(
        "shared automatic AOT numerical, artifact, resource and performance evidence passed"
    )


if __name__ == "__main__":
    main()
