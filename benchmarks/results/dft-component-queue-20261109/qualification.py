"""Gate the frozen normal-image cohort; never discard an inconvenient sample."""

import argparse
import csv
import json
import lzma
import math
from pathlib import Path
import re
from statistics import mean, median
import sys

from benchmarks.dft_becke_qualification import _paired_work_and_science
from benchmarks.dft_streaming_qualification import qualify_independent_record
from benchmarks.readme_defaults_accuracy import errors

if not __debug__ or sys.flags.optimize:
    raise RuntimeError("Evidence validation requires assertions; rerun without -O/-OO or PYTHONOPTIMIZE")

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--card", choices=("a100", "5090"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
owner = Path(__file__).resolve().parent
directory = args.directory
assert (directory / "exit-code.txt").read_text().strip() == "0"
package = json.loads((owner / "normal-payload.receipt.json").read_text())
images = package["images"]
reference_root = Path("benchmarks/results/readme-refresh-20261026")
oracles = json.loads(lzma.decompress((reference_root / "oracles.json.xz").read_bytes()))
contracts = json.loads((reference_root / "reference-reuse.json").read_text())["entries"]
report = {"normal_images": images, "card": args.card, "populations": {}, "profiles": {}, "other_hybrid_guards": {},
          "performance_failures": [], "whole_endpoint_quartet_counters_available": False}
expected = set()
observed_device = None


def load(name):
    expected.add(name + ".json")
    return json.loads((directory / (name + ".json")).read_text())


def science(value):
    return qualify_independent_record(value, oracles=oracles, reference_contracts=contracts)


def native(name, arm):
    global observed_device
    value = load(name)
    identity = load("identity-" + name)
    assert all(identity[key] == images[arm][key] for key in ("source_identity", "library_sha256"))
    assert all(value["native_build"][key] == images[arm][key] for key in ("source_identity", "library_sha256"))
    assert value["arm"] == "standard" and value["mode"] == "direct"
    assert "component_queue_pilot" not in value and value["warm_repeats"] == 3
    accelerator = value["accelerator"]
    expected_gpu = {"a100": ("NVIDIA A100-PCIE-40GB", [8, 0]),
                    "5090": ("NVIDIA GeForce RTX 5090", [12, 0])}[args.card]
    assert accelerator["backend"] == "cuda"
    assert (accelerator["name"], accelerator["compute_capability"]) == expected_gpu
    device = (accelerator["device_id"], accelerator["pci_bus_id"], accelerator["nvidia_smi"]["uuid"])
    assert all(item is not None for item in device)
    if observed_device is None:
        observed_device = device
    assert device == observed_device
    defaults = {"energy_tolerance": 1e-9, "density_tolerance": 1e-5,
                "screening_tolerance": 1e-12, "max_iterations": 100, "diis_history": 8,
                "force_grid_response": True, "requested_scf_tile_points": None,
                "force_tile_points": None, "nonlocal_grid_inherits_xc": True}
    for sample in value["samples"]:
        assert all(sample["settings"][key] == setting for key, setting in defaults.items())
    return value


def raw_k(text):
    rows = re.findall(r"^PASS .*?/([^/ ]+) predicates=48 independent_max_abs=([0-9.e+-]+) unsupported_refusals=3$", text, re.MULTILINE)
    assert len(rows) == 18 and "ALL_PASS fixture_files_passed=18" in text
    manifest = json.loads((owner / "fixture-manifest.json").read_text())
    assert len({name for name, _ in rows}) == 18
    assert {name for name, _ in rows} == {fixture["file"] for fixture in manifest["fixtures"]}
    errors = [float(error) for _, error in rows]
    assert all(math.isfinite(error) and error >= 0 for error in errors)
    return {"definitions": 18, "repeated_predicates": 864, "unsupported_refusals": 54,
            "max_abs_error": max(errors)}


if args.card == "a100":
    for arm in ("control", "candidate"):
        report[f"raw_k_{arm}"] = raw_k((directory / f"raw-k-{arm}.log").read_text())
if args.card == "5090":
    assert "Normal shared-XY ABI contains only its SM80 cubin" in (directory / "raw-k-coverage-boundary.txt").read_text()
    # Transport archives stay ignored; replay authenticates their exact members.
    prototype = owner / "stream-5090-16921"
    assert (prototype / "exit-code.txt").read_text().strip() == "0"
    report["raw_k_actual_queue_fatbin_not_normal_elf"] = raw_k((prototype / "raw-k-stream.log").read_text())
    prototype = owner / "sanitizers-5090-16923"
    assert (prototype / "exit-code.txt").read_text().strip() == "0"
    logs = {tool: (prototype / (tool + ".log")).read_text()
            for tool in ("memcheck", "racecheck", "synccheck")}
    report["sanitizers"] = {}
    for tool in ("memcheck", "racecheck", "synccheck"):
        text = logs[tool]
        assert text.count("predicates=6 ") == 6 and "ALL_PASS fixture_files_passed=6" in text
        marker = "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if tool == "racecheck" else "ERROR SUMMARY: 0 errors"
        assert marker in text
        report["sanitizers"][tool] = "actual queue fatbin, not normal ELF: six classes x six graph phases/reset, zero errors/hazards/warnings"

for atoms in (96, 48):
    for method in ("b3lyp", "pbe0"):
        names = [f"{method}-{atoms}-{ordinal}-{arm}" for ordinal, arm in enumerate(
            ("control", "candidate", "candidate", "control"), 1)]
        records = {name: native(name, "candidate" if name.endswith("-candidate") else "control") for name in names}
        independent = {name: science(value) for name, value in records.items()}
        baseline = records[names[0]]
        parity = [_paired_work_and_science(value, baseline) for value in records.values()]
        for value in records.values():
            for current, reference in zip(value["samples"], baseline["samples"], strict=True):
                assert current["ks_diagnostic"]["coulomb_exchange_work"] == reference["ks_diagnostic"]["coulomb_exchange_work"]
                for key, expected_value in reference["incremental_direct_jk"].items():
                    if key != "max_abs_delta_density":
                        assert current["incremental_direct_jk"][key] == expected_value
        colds = {arm: [value["samples"][0]["seconds"] for name, value in records.items()
                       if name.endswith("-" + arm)] for arm in ("control", "candidate")}
        means = {arm: mean(values) for arm, values in colds.items()}
        relative_mads = [median(abs(value - median(values)) for value in values) / median(values)
                         for values in colds.values()]
        floor = max(0.02, 2 * sum(relative_mads))
        ratio = means["candidate"] / means["control"]
        warms = [current["seconds"] / reference["seconds"]
                 for candidate, control in ((names[1], names[0]), (names[2], names[3]))
                 for current, reference in zip(records[candidate]["samples"][1:], records[control]["samples"][1:], strict=True)]
        if ratio > (1 - floor if atoms == 96 and args.card == "a100" else 1.03):
            report["performance_failures"].append(f"{method}-{atoms}: cold/noise gate")
        if max(warms) > 1.03:
            report["performance_failures"].append(f"{method}-{atoms}: paired warm gate")
        population = {
            "cold_samples": colds, "cold_means": means, "cold_ratio": ratio, "noise_floor": floor,
            "paired_warm_ratios": warms,
            "strict_native_parity": {key: max(item[key] for item in parity) for key in parity[0]},
            "independent_science": independent,
            "semantic_work": {name: [{key: sample[key] for key in ("iterations", "fock_builds", "initial_guess", "incremental_direct_jk")}
                | {"j_k_applications": sample["ks_diagnostic"]["coulomb_exchange_work"]}
                for sample in value["samples"]] for name, value in records.items()},
        }
        if atoms == 96:
            references = [load(f"{method}-reference-{ordinal}") for ordinal in (1, 2)]
            for reference in references:
                accelerator = reference["accelerator"]
                assert accelerator["backend"] == "cuda"
                assert (accelerator["device_id"], accelerator["pci_bus_id"], accelerator["nvidia_smi"]["uuid"]) == observed_device
            population["fresh_gpu4pyscf_science"] = [science(value) for value in references]
            population["fresh_gpu4pyscf_colds"] = [value["samples"][0]["seconds"] for value in references]
            population["candidate_over_gpu4pyscf_cold"] = means["candidate"] / mean(population["fresh_gpu4pyscf_colds"])
        report["populations"][f"{method}-{atoms}"] = population
        print(args.card, method, atoms, "cold", means, "ratio", ratio, "floor", floor,
              "max_warm", max(warms), "over_gpu4pyscf", population.get("candidate_over_gpu4pyscf_cold"))

for method in ("m06-2x", "wb97m-v"):
    for atoms in (96, 48):
        names = {arm: f"{method}-{atoms}-guard-{arm}" for arm in ("control", "candidate")}
        records = {arm: native(name, arm) for arm, name in names.items()}
        independent = {arm: science(value) for arm, value in records.items()}
        parity = _paired_work_and_science(records["candidate"], records["control"])
        for current, reference in zip(records["candidate"]["samples"], records["control"]["samples"], strict=True):
            assert current["ks_diagnostic"]["coulomb_exchange_work"] == reference["ks_diagnostic"]["coulomb_exchange_work"]
        ratios = [current["seconds"] / reference["seconds"]
                  for current, reference in zip(records["candidate"]["samples"], records["control"]["samples"], strict=True)]
        if max(ratios) > 1.03:
            report["performance_failures"].append(f"{method}-{atoms}: single-pair cold/warm regression guard")
        report["other_hybrid_guards"][f"{method}-{atoms}"] = {
            "single_pair_not_abba_promotion": True, "candidate_over_control_ratios": ratios,
            "strict_native_parity": parity, "independent_science": independent,
            "fock_builds": {arm: [sample["fock_builds"] for sample in value["samples"]] for arm, value in records.items()},
        }
        print(args.card, method, atoms, "single-pair guard ratios", ratios)

for method in ("b3lyp", "pbe0"):
    baseline = json.loads((directory / f"{method}-96-1-control.json").read_text())
    for arm in ("control", "candidate"):
        name = f"{method}-{arm}-profile"
        value = load(name)
        assert value["status"] == "complete" and value["warm_repeats"] == 3 and len(value["samples"]) == 4
        assert all(value["native_build"][key] == images[arm][key] for key in ("source_identity", "library_sha256"))
        _paired_work_and_science(value, baseline)
        target = next(item for item in oracles if item["method"] == method and item["atoms"] == 96 and item["backend"] == "cuda")
        assert value["geometry_bohr"] == target["geometry_bohr"]
        sample = value["samples"][0]
        diagnostic_science = errors(sample["energy"], sample["forces"], target["samples"][-1], 96)
        assert diagnostic_science["passed"]
        all_diagnostic_science = science(value)
        lines = (directory / f"{method}-{arm}-cold-kernels.tsv").read_text().splitlines()
        header = dict(field.split("=", 1) for field in lines[0].removeprefix("# ").split())
        assert set(header) == {"version", "dropped", "starved", "errors", "unknown", "flush", "disable"}
        assert int(header["version"]) in (27, 28) and all(int(value) == 0 for key, value in header.items() if key != "version")
        rows = list(csv.DictReader(lines[1:], delimiter="\t"))
        shared = [row for row in rows if "shared_rys_xy_streaming" in row["name"] and "shell_class_fock" in row["name"]]
        focks = sample["fock_builds"]
        assert bool(shared) == (args.card == "a100")
        if shared:
            assert len(shared) == 6 and all(int(row["count"]) == focks for row in shared)
        report["profiles"][name] = {"collector": header, "actual_shared_rows": shared,
                                   "independent_science": all_diagnostic_science, "cold_science": diagnostic_science,
                                   "focks": focks, "warm_kernels_excluded_by_cold_retain_callback": True}
assert {path.name for path in directory.glob("*.json")} == expected
args.output.write_text(json.dumps(report, indent=2) + "\n")
print("performance_failures", report["performance_failures"])
