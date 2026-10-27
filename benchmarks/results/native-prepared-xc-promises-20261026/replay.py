"""Replay retained scientific, resource, iteration-work and timing gates offline.

The original reviewed checker is restored from its hash-bound source, while
the shared publication reader owns plain/gzip storage. This never loads a CUDA
library or launches an endpoint. It recomputes all 100 performance decisions;
failed comparisons stay failures, even when numerical checks pass.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from types import ModuleType


def repository_root() -> Path:
    """Locate the shared reader in either a reviewed bundle or staging area."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools/generativeqc_validation/record.py").is_file():
            return parent
    raise RuntimeError("Run this script within a GenerativeQC checkout")


sys.path.insert(0, str(repository_root()))
from tools.generativeqc_validation.record import load_publication_record, load_record


def load(directory: Path, name: str, role: str) -> dict:
    if (directory / "publication.json").exists():
        return load_publication_record(directory, name=name, role=role)
    return load_record(directory / name)


def load_references(directory: Path) -> dict:
    """Resolve shared immutable oracle values once without duplicating datasets."""
    refs = load(directory, "references.json", "input")
    inputs = load(directory, "frozen-inputs.json", "samples")
    for row in refs["oracles"].values():
        stored = inputs["files"][row["frozen_input"]]
        assert stored["kind"] == "json" and stored["sha256"] == row["sha256"]
        raw = (json.dumps(stored["value"], indent=2, allow_nan=False) + "\n").encode()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        row["value"] = stored["value"]
    return refs


def extract_frozen(directory: Path, destination: Path) -> int:
    """Restore hash-identical source/inputs without executing or overwriting them."""
    sources = load(directory, "frozen-sources.json", "reproduction")
    inputs = load(directory, "frozen-inputs.json", "samples")
    files = {}
    for name, item in sources["files"].items():
        files[name] = (
            sources["text_by_sha256"][item["sha256"]].encode(),
            item["sha256"],
        )
    for name, item in inputs["files"].items():
        assert name not in files
        text = (
            (json.dumps(item["value"], indent=2, allow_nan=False) + "\n")
            if item["kind"] == "json"
            else item["text"]
        )
        files[name] = (text.encode(), item["sha256"])
    planned = {}
    for name, (raw, expected) in files.items():
        assert hashlib.sha256(raw).hexdigest() == expected
        path = Path(name.replace("historical/", "timing/historical-assets/", 1))
        assert not path.is_absolute() and ".." not in path.parts
        planned[path] = raw
    destination.mkdir(parents=True, exist_ok=False)
    for path, raw in planned.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return len(planned)


def restore_checker(sources: dict, directory: Path) -> ModuleType:
    """Import the unchanged historical science/work gates without its runner."""
    row = sources["files"]["historical/ks_cli_qualification_v5.py"]
    raw = sources["text_by_sha256"][row["sha256"]].encode()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    path = directory / "ks_cli_qualification_v5.py"
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("retained_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def references(inputs: dict, checker: ModuleType) -> dict:
    """Use separate explicit-grid SDK and current-default CLI oracles."""
    original = inputs["oracles"]
    historical = original["ks-cli-protocol-v5.json"]["value"]
    protocol = inputs["protocol"]
    for key in (
        "cases",
        "orders",
        "samples_per_block",
        "warmups_per_block",
        "ratio_gate",
        "numeric_cap_bytes",
        "energy_absolute_gate",
        "force_absolute_gate",
    ):
        assert protocol[key] == historical[key]
    assert protocol["samples_per_block"] == 21 and protocol["warmups_per_block"] == 3
    assert protocol["ratio_gate"] == 1.05
    assert protocol["numeric_cap_bytes"] == 2 << 30
    assert protocol["energy_absolute_gate"] == 1e-8
    assert protocol["force_absolute_gate"] == 1e-6
    old = {}
    for kind, receipt in historical["references"].items():
        row = original[receipt["file"]]
        assert row["sha256"] == receipt["sha256"]
        ref = row["value"]
        assert ref["completed"] and ref["version"] == "2.14.0"
        assert ref["basis_source_sha256"] == historical["basis_source_sha256"]
        assert ref["generator_sha256"] == receipt["generator_sha256"]
        old[kind] = {case["definition"]["name"]: case for case in ref["cases"]}
    fresh_row = original["cli-default-pbe-reference.json"]
    assert fresh_row["sha256"] == protocol["default_grid_reference_sha256"]
    fresh = fresh_row["value"]
    assert fresh["completed"] and fresh["pyscf_version"] == "2.14.0"
    assert not fresh["production_modules_imported"]
    assert fresh["basis_source_sha256"] == historical["basis_source_sha256"]
    current = {row["name"]: row for row in fresh["cases"]}
    result = {}
    for case in protocol["cases"]:
        for api in case["apis"]:
            if api == "cli" and case["method"].startswith("pbe-"):
                row = current[case["reference_name"]]
                assert all(case[k] == v for k, v in row["definition"].items())
                for phase in row["phases"]:
                    assert phase["converged"] and phase["gradient_fd_max_error"] <= 2e-7
                expected = [
                    dict(p, cuda_requested_properties=case["cuda_requested_properties"])
                    for p in row["phases"]
                ]
            else:
                expected = checker.expected_phases(case, api, old)
            result[api, case["name"]] = expected
    return result


def promise(resource: dict, cap: int, closed: bool = False) -> None:
    """Unused owner credit is separate from numeric and provider charges."""
    unused = resource["capacity_reserved_bytes"]
    assert type(unused) is int and unused >= 0
    assert (
        resource["numeric_live_bytes"] + resource["provider_reserved_bytes"] + unused
        <= cap
    )
    if closed:
        assert unused == 0


def observation(
    value: dict,
    case: dict,
    api: str,
    variant: str,
    cap: int,
    expected: list,
    protocol: dict,
    checker: ModuleType,
    errors: dict,
    native_promise: bool = True,
) -> None:
    if api == "sdk":
        checker.check_sdk(value, expected, protocol, variant, "cuda", cap)
        if variant == "candidate":
            for resource in value["preparation_native_admission"]:
                promise(resource, cap)
                if native_promise:
                    assert resource["capacity_reserved_bytes"] > 0
            for phase in value["phases"]:
                promise(phase["native_admission"], cap)
            for resource in value["post_close_native_admission"]:
                promise(resource, cap, closed=True)
    else:
        assert value["normal_close_included"] and value["fresh_process_per_geometry"]
        assert (
            not value["prepared_warm_replay"] and value["requested_precision"] == "fp64"
        )
        assert len(value["phases"]) == 2
        for phase, ref in zip(value["phases"], expected, strict=True):
            checker.check_cli(
                phase["raw_json"], case, ref, protocol, variant, "cuda", cap
            )
            checker.check_publication(phase, ref, protocol, "cuda")
            if variant == "candidate":
                resource = phase["raw_json"]["owned_device_resources"]
                assert resource["admission"] == "numeric-cap"
                promise(resource, cap)
    for phase, ref in zip(value["phases"], expected, strict=True):
        errors["energies"].append(phase["energy"] - ref["energy"])
        if phase["forces"] is not None:
            errors["forces"].extend(
                a - b
                for ar, br in zip(phase["forces"], ref["forces"], strict=True)
                for a, b in zip(ar, br, strict=True)
            )


def controls(inputs: dict, record: dict, checker: ModuleType) -> dict:
    """Replay every fixed boundary and explicit unsupported-property refusal."""
    protocol = inputs["protocol"]
    expected = references(inputs, checker)
    cases = {c["name"]: c for c in protocol["cases"]}
    boundaries = record["boundaries"]
    search = record["cap_search"]
    assert boundaries["completed"] and search["completed"]
    assert len(boundaries["cases"]) == len(search["cases"]) == 50
    assert [(r["api"], r["definition"]["name"]) for r in boundaries["cases"]] == [
        (api, c["name"]) for c in protocol["cases"] for api in c["apis"]
    ]
    errors = {"energies": [], "forces": []}
    refused = successes = 0
    for row, bound in zip(boundaries["cases"], search["cases"], strict=True):
        assert row["completed"] and bound["completed"]
        case, api, cap = row["definition"], row["api"], row["qualified_cap"]
        assert case == cases[case["name"]] == bound["definition"]
        assert api == bound["api"] and cap == bound["qualified_cap"]
        cursor = 0
        full_work = None
        for label, limit in (
            ("full-cap", protocol["numeric_cap_bytes"]),
            ("exact", cap),
            ("one-below", cap - 1),
            ("one-byte", 1),
        ):
            failed = False
            work = []
            for geometry in range(1 if api == "sdk" else 2):
                call = row["calls"][cursor]
                cursor += 1
                assert (call["label"], call["cap"], call["geometry"]) == (
                    label,
                    limit,
                    geometry,
                )
                value = call["publication"]
                if call["returncode"]:
                    assert call["returncode"] == 1 and value is None
                    assert (
                        "native_status=7:" if api == "sdk" else "(status 7)"
                    ) in call["stderr"]
                    failed = True
                    refused += 1
                    break
                successes += 1
                if api == "sdk":
                    observation(
                        value,
                        case,
                        api,
                        "candidate",
                        limit,
                        expected[api, case["name"]],
                        protocol,
                        checker,
                        errors,
                        native_promise=False,
                    )
                    work.extend(p["iterations"] for p in value["phases"])
                else:
                    checker.check_cli(
                        value,
                        case,
                        expected[api, case["name"]][geometry],
                        protocol,
                        "candidate",
                        "cuda",
                        limit,
                    )
                    promise(value["owned_device_resources"], limit)
                    work.append(value["iterations"])
            assert failed == (label in ("one-below", "one-byte"))
            if not failed:
                if full_work is None:
                    full_work = work
                else:
                    assert work == full_work
        assert cursor == len(row["calls"])
    preflight = record["preflight"]
    assert preflight["completed"] and len(preflight["cases"]) == 60
    assert [(r["variant"], r["definition"]["name"]) for r in preflight["cases"]] == [
        (v, c["name"]) for v in ("baseline", "candidate") for c in protocol["cases"]
    ]
    force_refusals = 0
    for row in preflight["cases"]:
        case = row["definition"]
        for call in row["calls"]:
            kind, value = call["kind"], call["publication"]
            if kind.startswith("cli-force-refusal-"):
                assert case["cuda_requested_properties"] == ["energy"]
                assert call["exit_code"] == 2 and value is None
                assert call["stderr"].startswith(
                    "generativeqc: native CLI DFT forces are not exposed for this prepared context\n"
                )
                force_refusals += 1
            elif kind == "sdk-force-refusal":
                assert case["cuda_requested_properties"] == ["energy"]
                assert call["exit_code"] == 1 and value is None
                assert "native_status=3:" in call["stderr"]
                force_refusals += 1
            else:
                assert kind in ("cli-0", "cli-1") and call["exit_code"] == 0
                geometry = int(kind[-1])
                checker.check_cli(
                    value,
                    case,
                    expected["cli", case["name"]][geometry],
                    protocol,
                    row["variant"],
                    "cuda",
                    protocol["numeric_cap_bytes"],
                )
    assert force_refusals == 96
    smokes = record["sdk_smokes"]
    assert len(smokes) == 60
    assert {(r["variant"], r["case"], r["metadata"]) for r in smokes} == {
        (variant, c["name"], metadata)
        for variant in ("baseline", "candidate")
        for c in protocol["cases"]
        if "sdk" in c["apis"]
        for metadata in (("retain",) if variant == "baseline" else ("retain", "close"))
    }
    for row in smokes:
        assert row["exit_code"] == 0
        checker.check_sdk(
            row["publication"],
            expected["sdk", row["case"]],
            protocol,
            row["variant"],
            "cuda",
            protocol["numeric_cap_bytes"],
            metadata=row["metadata"],
        )
    return {
        "boundary_cases": 50,
        "typed_budget_refusals": refused,
        "successful_boundary_commands": successes,
        "unsupported_force_refusals": force_refusals,
        "sdk_metadata_smokes": len(smokes),
    }


def replay(inputs: dict, samples: dict, sources: dict, checker: ModuleType) -> dict:
    protocol = inputs["protocol"]
    expected = references(inputs, checker)
    blocks = samples["blocks"]
    required = [
        (api, order, slot)
        for order, slots in protocol["orders"].items()
        for slot in slots
        for api in ("sdk", "cli")
    ]
    assert [(b["api"], b["order"], b["slot"]) for b in blocks] == required
    errors = {"energies": [], "forces": []}
    grouped = {}
    observations = warmups = 0
    for block in blocks:
        assert block["completed"]
        api, order, slot, variant = (
            block[k] for k in ("api", "order", "slot", "variant")
        )
        assert variant == ("baseline" if slot.startswith("a") else "candidate")
        cases = [c for c in protocol["cases"] if api in c["apis"]]
        assert [r["name"] for r in block["cases"]] == [c["name"] for c in cases]
        for row, case in zip(block["cases"], cases, strict=True):
            assert len(row["observations"]) == 21 and row["warmups_completed"] == 3
            assert len(row["warmup_observations"]) == (0 if api == "sdk" else 3)
            env = row["environment"]
            assert (
                env["PATH"] == "/nonexistent" and env["GENERATIVEQC_PROFILE"] == "off"
            )
            assert env["CUDA_VISIBLE_DEVICES"] == block["cuda_visible_devices"]
            assert all(
                env[k] == "1"
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            )
            assert env["GENERATIVEQC_DF_VALUE_STORAGE"] == case["storage"]
            assert env["GENERATIVEQC_DF_EXCHANGE"] == case["exchange"]
            assert len(row["calls"]) == (1 if api == "sdk" else 48)
            for call in row["calls"]:
                assert call["returncode"] == 0 and call["elapsed_ns"] > 0
                assert (
                    block["started_unix_ns"]
                    <= call["started_unix_ns"]
                    < call["finished_unix_ns"]
                    <= block["finished_unix_ns"]
                )
            for index, value in enumerate(row["observations"]):
                if api == "sdk":
                    assert (
                        value["sample_index"] == index
                        and value["samples_requested"] == 21
                        and value["warmups_completed"] == 3
                    )
            for value in row["warmup_observations"] + row["observations"]:
                assert math.isfinite(value["seconds"]) and value["seconds"] > 0
                observation(
                    value,
                    case,
                    api,
                    variant,
                    protocol["numeric_cap_bytes"],
                    expected[api, case["name"]],
                    protocol,
                    checker,
                    errors,
                )
            observations += len(row["observations"])
            warmups += len(row["warmup_observations"])
            grouped.setdefault((order, api, case["name"]), []).extend(
                (variant, v) for v in row["observations"]
            )
    comparisons = []
    for key, values in grouped.items():
        medians = {
            v: statistics.median(o["seconds"] for kind, o in values if kind == v)
            for v in ("baseline", "candidate")
        }
        row = {
            "order": key[0],
            "api": key[1],
            "case": key[2],
            "median_seconds": medians,
            "ratio": medians["candidate"] / medians["baseline"],
            "passed": False,
        }
        try:
            row.update(checker.compare_work_and_times(values, protocol))
            row["passed"] = True
        except AssertionError as error:
            row["failure"] = repr(error)
        comparisons.append(row)
    assert len(comparisons) == 100
    actual = {(r["order"], r["api"], r["case"]): r for r in comparisons}
    assert samples["audit"]["completed"] and samples["audit"]["all_publications_passed"]
    assert actual == {
        (r["order"], r["api"], r["case"]): r for r in samples["audit"]["comparisons"]
    }
    quantitative = {}
    for key, values in errors.items():
        gate = protocol[
            "energy_absolute_gate" if key == "energies" else "force_absolute_gate"
        ]
        maximum = max(map(abs, values))
        quantitative[key] = {
            "passed": maximum <= gate,
            "atol": gate,
            "rtol": 0,
            "max_absolute_error": maximum,
            "max_scaled_error": maximum / gate,
            "rms_error": math.sqrt(sum(v * v for v in values) / len(values)),
            "shape": [len(values)],
        }
    return {
        "observations": observations,
        "saved_warmup_observations": warmups,
        "comparisons": comparisons,
        "block_errors": quantitative,
        "all_timing_work_gates_passed": all(r["passed"] for r in comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extract-frozen", type=Path)
    args = parser.parse_args()
    if args.extract_frozen:
        count = extract_frozen(args.directory, args.extract_frozen)
        print(f"Restored {count} frozen source/input files; no execution performed")
        return
    inputs = load_references(args.directory)
    samples = load(args.directory, "samples.json", "samples")
    sources = load(args.directory, "frozen-sources.json", "reproduction")
    control_record = load(args.directory, "controls.json", "samples")
    with tempfile.TemporaryDirectory() as temporary:
        checker = restore_checker(sources, Path(temporary))
        result = replay(inputs, samples, sources, checker)
        result["controls"] = controls(inputs, control_record, checker)
    if args.output:
        with args.output.open("x") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    if not result["all_timing_work_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
