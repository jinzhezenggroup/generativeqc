"""Recompute prospective #2727 returns from IEEE originals and frozen inputs.

This reader never loads the native library, PySCF or CUDA. Numerical acceptance,
complete telemetry, input authentication and protected delivery remain axes.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

from admission import compare, finite, reconstruct
from model_contract import ROLES, archive_expected
from model_contract import components as model_components
from transport import checked, read, sha, write_new


def recover(directory: Path) -> dict:
    if not (directory / "return.json").exists():
        if (directory / "enriched/return.json").exists():
            binding = read(directory / "observables-binding.json")
            for name, expected in binding.items():
                checked(directory / "observables" / name, expected)
            return recover(directory / "enriched")
        if (directory / "observables/return.json").exists():
            return recover(directory / "observables")
    sample = read(directory / "return.json")
    binary = (directory / "values.bin").read_bytes()
    index = json.loads((directory / "values.index.json").read_bytes())
    offset = 0
    fields = []
    floating_paths = []

    def collect(value: Any, path: tuple) -> None:
        if isinstance(value, float) or (
            isinstance(value, dict) and set(value) == {"raw_nonfinite"}
        ):
            floating_paths.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                collect(v, (*path, k))
        elif isinstance(value, list):
            for k, v in enumerate(value):
                collect(v, (*path, k))

    collect(sample, ())

    def reshape(values: list, shape: list) -> Any:
        if not shape:
            return values[0]
        if len(shape) == 1:
            return values
        stride = math.prod(shape[1:])
        return [
            reshape(values[i * stride : (i + 1) * stride], shape[1:])
            for i in range(shape[0])
        ]

    for entry in index:
        field = tuple(entry["field"])
        if not field or any(
            field[: len(p)] == p or p[: len(field)] == field for p in fields
        ):
            raise ValueError("IEEE index field coverage overlaps")
        fields.append(field)
        if (
            entry["offset"] != offset
            or entry["bytes"] < 0
            or offset + entry["bytes"] > len(binary)
        ):
            raise ValueError("IEEE index has gaps, overlaps or out-of-range bytes")
        dtype = entry["dtype"]
        codes = {
            "f8": "d",
            "f4": "f",
            "i8": "q",
            "u8": "Q",
            "i4": "i",
            "u4": "I",
            "i2": "h",
            "u2": "H",
            "i1": "b",
            "u1": "B",
            "b1": "?",
        }
        if dtype[:1] not in ("<", "|", "=") or dtype[1:] not in codes:
            raise ValueError("unsupported retained numeric dtype")
        code = codes[dtype[1:]]
        size = struct.calcsize("<" + code)
        count = math.prod(entry["shape"]) if entry["shape"] else 1
        if entry["bytes"] != count * size:
            raise ValueError("IEEE dtype/shape/size mismatch")
        values = list(
            struct.unpack("<" + code * count, binary[offset : offset + entry["bytes"]])
        )
        recovered = reshape(values, entry["shape"])
        node = sample
        for key in entry["field"][:-1]:
            node = node[key]
        key = entry["field"][-1]
        stored = node[key]

        def same(a: Any, b: Any) -> bool:
            if isinstance(a, list):
                return (
                    isinstance(b, list)
                    and len(a) == len(b)
                    and all(same(x, y) for x, y in zip(a, b, strict=True))
                )
            if isinstance(a, float) and not math.isfinite(a):
                return isinstance(b, dict) and b.get("raw_nonfinite") == str(a)
            return a == b and (
                not isinstance(a, float)
                or math.copysign(1.0, a) == math.copysign(1.0, b)
            )

        if not same(recovered, stored):
            raise ValueError("JSON number differs from retained IEEE original")
        node[key] = recovered
        offset += entry["bytes"]
    if offset != len(binary):
        raise ValueError("unindexed original numeric bytes")
    if any(not any(p[: len(f)] == f for f in fields) for p in floating_paths):
        raise ValueError("JSON contains floating values absent from IEEE index")
    gate = directory / "gate.json"
    if gate.exists():
        for name, expected in read(gate)["raw_hashes"].items():
            checked(directory / name, expected)
    return sample


def qualification(call: dict, sample: dict, inputs: dict) -> dict:
    if (
        sample["call"] != call
        or not sample["converged"]
        or sample["executed_backend"] != "cuda"
    ):
        raise ValueError("returned call or actual CUDA execution differs")
    if len(sample["components"]) != 5 or any(
        not c["converged"] or c["executed_backend"] != "cuda"
        for c in sample["components"]
    ):
        raise ValueError("missing actual CUDA constituents")
    meta = sample["basis_metadata"]
    if (
        meta["density_fitting"] != (call["fixture"] == "archive-water")
        or meta["hf_reference"] != call.get("hf_reference", "exact")
        or meta["frozen_core"] != 0
        or meta["representation"] != "spherical"
        or meta["hf_exponent"] != 1.63
        or meta["correlation_exponent"] != 3.0
        or sample["convergence"]["requested_policy"] != call["policy"]
    ):
        raise ValueError("returned scientific model or policy differs")
    if call["kind"] != "EF":
        model_components(call, sample, inputs)
        finite(sample["energy"])
        return {
            "kind": "native_energy_only",
            "energy": sample["energy"],
            "forces_present": sample["forces"] is not None,
            "independent_full_vector": False,
        }
    if call["fixture"] == "archive-water":
        case = next(
            c
            for c in inputs["water_cases"]
            if c["reference"] == call["hf_reference"]
            and c["geometry_id"] == call["geometry_id"]
        )
        forces = reconstruct(case["oracle"])
        ordered = model_components(call, sample, inputs)
        e, f = archive_expected(case["oracle"], forces)
        composite = compare(
            sample["energy"], sample["forces"], e, f, call["composite_gates"]
        )
        components = [
            compare(
                c["energy"],
                c["forces"],
                case["oracle"]["component_energies"][i],
                forces[i],
                call["constituent_gates"],
            )
            for i, c in enumerate(ordered)
        ]
    else:
        reference = json.loads(
            inputs["conventional"][f"tests/reference_data/cbs/{call['fixture']}.json"][
                "content_utf8"
            ]
        )["reference"]
        limits = {"energy_abs": 1e-9, "force_max_abs": 2e-6, "force_rms": 2e-6}
        composite = compare(
            sample["energy"],
            sample["forces"],
            reference["energy"],
            reference["forces"],
            limits,
            strict=False,
        )
        components = []
        for role, c in zip(ROLES, model_components(call, sample, inputs), strict=True):
            oracle = reference["components"][role]
            components.append(
                compare(
                    c["energy"],
                    c["forces"],
                    oracle["energy"],
                    oracle["forces"],
                    limits,
                    strict=False,
                )
            )
            if (
                c["method"] == "ccsd(t)"
                and abs(
                    finite(c["correlation"]["ccsd_t_triples_energy"])
                    - oracle["triples_energy"]
                )
                > 1e-9
            ):
                raise ValueError("independent triples energy gate failed")
        for field in ("hf_energy", "mp2_correlation_energy", "delta_ccsdt_energy"):
            if abs(finite(sample[field]) - reference[field]) > 1e-9:
                raise ValueError("independent algebra term gate failed")
        if any(
            abs(sum(row[axis] for row in sample["forces"])) > 2e-7 for axis in range(3)
        ):
            raise ValueError("translation gate failed")
    return {
        "kind": "independent_full_vector",
        "composite_errors": composite,
        "constituent_errors": components,
    }


def replay(plan_path: Path, output: Path) -> dict:
    plan = read(plan_path)
    inputs = read(
        checked(
            plan_path.parent / plan["input_bundle_file"], plan["input_bundle_sha256"]
        )
    )
    lease = read(output / "exclusive-batch-lease.json")
    if lease["prepared_plan_sha256"] != sha(plan_path):
        raise ValueError("actual batch lease binds another plan")
    rows = []
    samples = {}
    attempts = 0
    retained_returns = 0
    for call in plan["calls"]:
        folder = output / call["process"]
        attempt = folder / f"attempt-{call['index']:03d}.json"
        receipt = folder / f"call-{call['index']:03d}"
        if not attempt.exists():
            rows.append({"index": call["index"], "status": "NOT_RUN"})
            continue
        if read(attempt)["call"] != call:
            raise ValueError("attempt differs from frozen inventory")
        attempts += 1
        if not any(
            (receipt / path).exists()
            for path in (
                "return.json",
                "enriched/return.json",
                "observables/return.json",
            )
        ):
            rows.append(
                {
                    "index": call["index"],
                    "status": "NO_RETURN; inspect original process outcome/streams",
                }
            )
            continue
        sample = recover(receipt)
        retained_returns += 1
        if sample.get("schema") == "generativeqc.h100-small-cbs-observables.v1":
            rows.append(
                {
                    "index": call["index"],
                    "status": "TRANSPORT_INCOMPLETE",
                    "retained_observables": sample,
                    "metadata_enrichment": "UNAVAILABLE",
                }
            )
            continue
        samples[call["index"]] = sample
        try:
            value = qualification(call, sample, inputs)
            rows.append({"index": call["index"], "status": "NUMERIC_PASS", **value})
        except (ValueError, KeyError, TypeError) as error:
            rows.append(
                {"index": call["index"], "status": "NUMERIC_FAIL", "error": str(error)}
            )
    directions = []
    for process in plan["processes"][:2]:
        calls = [c for c in plan["calls"] if c["process"] == process["id"]]
        if not all(c["index"] in samples for c in calls):
            directions.append({"process": process["id"], "status": "INCOMPLETE"})
            continue
        group = [samples[c["index"]] for c in calls]
        reference = json.loads(
            inputs["conventional"][
                f"tests/reference_data/cbs/{calls[0]['fixture']}.json"
            ]["content_utf8"]
        )
        projection = -sum(
            f * d
            for row, direction in zip(
                group[0]["forces"], reference["direction"], strict=True
            )
            for f, d in zip(row, direction, strict=True)
        )
        native = [
            {
                "step_bohr": p["call"]["step_bohr"],
                "error": abs(
                    (p["energy"] - m["energy"]) / (2 * p["call"]["step_bohr"])
                    - projection
                ),
            }
            for m, p in ((group[2], group[3]), (group[4], group[5]))
        ]
        independent = [
            {"step_bohr": r["step_bohr"], "error": abs(projection - r["derivative"])}
            for r in reference["finite_difference"]
        ]
        parity = abs(group[1]["energy"] - group[0]["energy"])
        passed = (
            all(
                math.isfinite(r["error"]) and r["error"] < 2e-6
                for r in native + independent
            )
            and parity <= 1e-9
        )
        directions.append(
            {
                "process": process["id"],
                "status": "PASS" if passed else "FAIL",
                "native_self_consistency": native,
                "independent_stored_PySCF_directions": independent,
                "native_energy_parity_error": parity,
            }
        )
    complete = all(r["status"] == "NUMERIC_PASS" for r in rows) and all(
        d["status"] == "PASS" for d in directions
    )
    return {
        "status": "NUMERIC_PASS" if complete else "FAIL_OR_INCOMPLETE",
        "attempted_calls": attempts,
        "retained_native_returns": retained_returns,
        "returned_EF_requests_with_metadata": sum(
            s["call"]["kind"] == "EF" for s in samples.values()
        ),
        "returned_energy_only_requests_with_metadata": sum(
            s["call"]["kind"] != "EF" for s in samples.values()
        ),
        "calls": rows,
        "directional_and_parity": directions,
        "batch_outcome": read(output / "batch-outcome.json"),
        "operator_precision_inventory": "INCOMPLETE",
        "semantic_work_inventory": "INCOMPLETE",
        "full_scientific_acceptance": "INCOMPLETE",
        "publication_and_merge": "NOT_EVALUATED",
        "input_authentication": "separate root originals/publication closure required; arithmetic replay is not source authentication",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.plan, args.output_root)
    write_new(args.report, report)
    print(report["status"])
