"""Authenticate XC unroll evidence, independent gates and complete timing pairs."""

from __future__ import annotations

import hashlib
import json
import lzma
import re
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks._exact_fp64_evidence import load
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]


def require(condition: bool, detail: str) -> None:
    """Keep every integrity/scientific gate active under python -O as well."""
    if not condition:
        raise ValueError(detail)


def checked(raw: bytes, digest: str, name: str) -> bytes:
    require(hashlib.sha256(raw).hexdigest() == digest, f"checksum: {name}")
    return raw


def same_task(left: dict, right: dict) -> None:
    """Compare scientific controls and physical-device locators before timing."""
    for key in (
        "method",
        "mode",
        "atoms",
        "geometry_bohr",
        "orbital_basis",
        "basis_representation",
        "charge",
        "spin",
        "auxiliary_input_sha256",
        "protocol",
        "thread_controls",
    ):
        require(left[key] == right[key], f"task differs: {key}")
    require(
        left["environment"]["host"]["node"] == right["environment"]["host"]["node"],
        "node differs",
    )
    for key in ("name", "pci_bus_id"):
        require(
            left["accelerator"][key] == right["accelerator"][key],
            f"device differs: {key}",
        )
    uuids = [
        (r["accelerator"].get("nvidia_smi") or {}).get("uuid") for r in (left, right)
    ]
    if all(uuids):
        require(uuids[0] == uuids[1], "GPU UUID differs")
    else:
        require(
            all(r["scheduler"].get("LSB_JOBID") for r in (left, right)),
            "missing device UUID without ECNU locator",
        )
        require(
            left["scheduler"]["CUDA_VISIBLE_DEVICES"]
            == right["scheduler"]["CUDA_VISIBLE_DEVICES"],
            "LSF visibility differs",
        )


def times(record: dict, phase: str) -> list[float]:
    require(record["status"] == "complete", "cannot time incomplete population")
    values = [s["seconds"] for s in record["samples"] if s["phase"] == phase]
    require(
        len(values) == (1 if phase == "cold" else record["warm_repeats"]),
        "repeat census differs",
    )
    return values


def paired_errors(left: dict, right: dict, *, large: bool = False) -> list[dict]:
    """Qualify outputs and semantic work, preserving every sampled phase."""
    if not large:
        same_task(left, right)
    else:
        for key in (
            "method",
            "atoms",
            "geometry_bohr",
            "moved_positions_bohr",
            "memory_budget_bytes",
        ):
            require(left[key] == right[key], f"large task differs: {key}")
    if large and left["schema"] == "generativeqc.xc-unroll-large-control.v2":
        require(right["schema"] == left["schema"], "larger control version differs")
        for record in (left, right):
            original = [xyz for _, xyz in record["geometry_bohr"]]
            for sample in record["samples"]:
                expected_positions = (
                    record["moved_positions_bohr"]
                    if sample["phase"] in ("moved", "moved-warm")
                    else original
                )
                require(
                    sample["evaluated_positions_bohr"] == expected_positions,
                    "larger replay coordinates differ",
                )
    require(
        left["status"] == right["status"] == "complete", "incomplete paired endpoint"
    )
    require(
        left["scheduler"] == right["scheduler"], "paired arms must share allocation"
    )
    require(
        len(left["samples"]) == len(right["samples"]), "paired repeat census differs"
    )
    errors = []
    for a, b in zip(left["samples"], right["samples"], strict=True):
        for key in (
            "phase",
            "iterations",
            "fock_builds",
            "settings",
            "warm_start_used",
            "warm_start_fallback",
        ):
            require(a[key] == b[key], f"paired work differs: {key}")
        if not large:
            require(a["repeat"] == b["repeat"], "repeat index differs")
        for key in a["initial_guess"]:
            if key != "preparation_seconds":
                require(
                    a["initial_guess"][key] == b["initial_guess"][key],
                    f"initial guess differs: {key}",
                )
        for key in (
            "grid_points",
            "tile_points",
            "ao_order",
            "scf_domain",
            "coulomb_exchange_work",
            "nonlocal_work",
            "nonlocal_history_scopes",
        ):
            require(
                a["ks_diagnostic"].get(key) == b["ks_diagnostic"].get(key),
                f"diagnostic work differs: {key}",
            )
        error = {
            "phase": a["phase"],
            "energy_abs": abs(a["energy"] - b["energy"]),
            "force_max": float(
                abs(np.asarray(a["forces"]) - np.asarray(b["forces"])).max()
            ),
        }
        if not large:
            error["repeat"] = a["repeat"]
        require(
            error["energy_abs"] < 1e-7 and error["force_max"] < 1e-7,
            "paired energy/force gate failed",
        )
        errors.append(error)
    return errors


def main() -> None:
    folder = Path(__file__).parent
    bundle, files = load(folder)
    references = {
        name: json.loads(
            lzma.decompress(checked((ROOT / name).read_bytes(), digest, name))
        )
        for name, digest in bundle["references"].items()
    }
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]
    seed = references[
        "benchmarks/results/cuda-ks-seed-products-20261101/receipts.json.xz"
    ]
    prior = references[seed["prior_path"]]
    analysis = json.loads(files["analysis.json"])
    records = {}
    for row in analysis["checks"]:
        name = row["file"]
        record = records[name] = json.loads(files[name])
        expected = bundle[
            "candidate_identity" if "/candidate-" in name else "parent_identity"
        ]
        if record["arm"] == "standard":
            require(
                record["native_build"]["source_identity"] == expected,
                f"source identity: {name}",
            )
        else:
            require(record["arm"] == "gpu4pyscf", "unexpected measurement arm")
        actual = qualify(
            [record],
            oracles,
            reference_reuse=reuse[row["device"]]["analysis"]["reference_reuse"],
        )[0]
        require(
            actual == row["qualification"] and actual["status"] == "qualified",
            f"independent gate: {name}",
        )
    rejected = {row["file"]: row for row in analysis["rejected"]}
    require(
        not (records.keys() & rejected.keys()), "record both qualified and rejected"
    )
    observed = set()
    for name, raw in files.items():
        if name.endswith(".json") and name.startswith(("a100-", "5090-", "pro6000-")):
            record = json.loads(raw)
            if record.get("schema") != "generativeqc.readme-defaults.v2":
                continue
            observed.add(name)
            if name in rejected:
                require(
                    record.get("status") != "complete"
                    or re.fullmatch(r"[0-9a-f]{40}", record.get("source_revision", ""))
                    is None,
                    "complete valid population was excluded",
                )
    require(observed == records.keys() | rejected.keys(), "measurement census differs")
    for pair in analysis["pairs"]:
        actual = paired_errors(records[pair["candidate"]], records[pair["parent"]])
        require(actual == pair["paired_errors"], "saved paired gate differs")
        for phase in ("cold", "warm"):
            n, p = (
                median(times(records[pair[arm]], phase))
                for arm in ("candidate", "parent")
            )
            require(
                pair[phase]
                == {"candidate": n, "parent": p, "change_percent": 100 * (n / p - 1)},
                "paired timing differs",
            )
    require(len(analysis["summaries"]) == 21, "missing final functional/device")
    require(
        {(row["device"], row["method"]) for row in analysis["summaries"]}
        == {
            (device, method)
            for device in ("a100", "5090", "pro6000")
            for method in (
                "pbe",
                "pbe0",
                "b3lyp",
                "r2scan",
                "r2scan-3c",
                "m06-2x",
                "wb97m-v",
            )
        },
        "final cohort differs",
    )
    for row in analysis["summaries"]:
        key = row["gpu4_reference"]
        reference = json.loads(
            (
                prior["records"][key[6:]]
                if key.startswith("prior/")
                else seed["records"][key]
            )["json"]
        )
        q = qualify(
            [reference],
            oracles,
            reference_reuse=reuse[row["device"]]["analysis"]["reference_reuse"],
        )[0]
        require(q["status"] == "qualified", "GPU4 reference qualification failed")
        require(
            len(row["pairs"]) == len(set(row["pairs"])) == 2,
            "expected two distinct complete counterbalanced pairs",
        )
        require(
            {name.rsplit("-", 1)[-1] for name in row["pairs"]}
            == {"r1.json", "r2.json"},
            "matched rounds differ",
        )
        for name in row["pairs"]:
            same_task(records[name], reference)
        for phase in ("cold", "warm"):
            populations = {
                arm: [
                    value
                    for name in row["pairs"]
                    for value in times(
                        records[
                            name
                            if arm == "candidate"
                            else name.replace("/candidate-", "/parent-")
                        ],
                        phase,
                    )
                ]
                for arm in ("candidate", "parent")
            }
            n, p = (median(populations[arm]) for arm in ("candidate", "parent"))
            g = median(times(reference, phase))
            expected = {
                "samples": populations,
                "candidate_median": n,
                "parent_median": p,
                "change_percent": 100 * (n / p - 1),
                "gpu4_median": g,
                "gpu4_overhead_percent": 100 * (n / g - 1),
            }
            require(row[phase] == expected, "aggregate timing differs")
            require(
                expected["change_percent"] <= 1.0,
                "prespecified performance regression review threshold exceeded",
            )
            print(
                row["device"],
                row["method"],
                phase,
                f"{p:.3f} -> {n:.3f}s ({100 * (n / p - 1):+.2f}%); vs retained GPU4 {100 * (n / g - 1):+.2f}%",
            )
    require(
        len(analysis["fresh_rankings"]) == 2,
        "missing fresh Blackwell 3c reference controls",
    )
    require(
        {row["device"] for row in analysis["fresh_rankings"]} == {"5090", "pro6000"},
        "fresh reference device census differs",
    )
    for row in analysis["fresh_rankings"]:
        require(
            len(row["candidates"]) == len(row["references"]) == 2,
            "fresh reference repeat census differs",
        )
        for a, b in zip(row["candidates"], row["references"], strict=True):
            same_task(records[a], records[b])
            require(
                records[a]["scheduler"] == records[b]["scheduler"],
                "fresh reference allocation differs",
            )
        for phase in ("cold", "warm"):
            a = [
                value
                for name in row["candidates"]
                for value in times(records[name], phase)
            ]
            b = [
                value
                for name in row["references"]
                for value in times(records[name], phase)
            ]
            expected = {
                "candidate_samples": a,
                "reference_samples": b,
                "overhead_percent": 100 * (median(a) / median(b) - 1),
            }
            require(row[phase] == expected, "fresh reference timing differs")
            print(
                row["device"],
                "r2scan-3c",
                phase,
                "fresh same-allocation GPU4",
                f"{row[phase]['overhead_percent']:+.2f}%",
            )
    # A successful tile32 launch does not establish a useful endpoint schedule.
    # Recompute the rejected experiment independently, including its separate
    # fresh-reference cache-conditioning controls on n1.
    configuration = json.loads(files["tile32-followup/binary-configurations.json"])
    followup = {}
    for name, raw in files.items():
        if not name.startswith("tile32-followup/5090-") or not name.endswith(".json"):
            continue
        record = json.loads(raw)
        if record.get("schema") != "generativeqc.readme-defaults.v2":
            continue
        require(record["status"] == "complete", "incomplete tile32 followup")
        if record["arm"] == "standard":
            variant = "candidate" if "/candidate-" in name else "parent"
            require(
                record["native_build"]["source_identity"]
                == bundle["candidate_identity"],
                "tile32 experiment must share the unroll4 source",
            )
            require(
                record["native_build"]["library_sha256"]
                == configuration[variant]["binary_sha256"],
                "tile configuration binary mismatch",
            )
        qualified = qualify(
            [record],
            oracles,
            reference_reuse=reuse["5090"]["analysis"]["reference_reuse"],
        )[0]
        require(qualified["status"] == "qualified", f"tile32 followup gate: {name}")
        followup[name] = record
    require(len(followup) == 18, "tile32 followup census differs")
    for method in ("pbe", "r2scan", "r2scan-3c", "m06-2x"):
        prefix = "tile32-followup/5090-pilot-15031/"
        left, right = (
            followup[prefix + f"{arm}-{method}-96.json"]
            for arm in ("candidate", "parent")
        )
        paired_errors(left, right)
        print(
            "Rejected tile32",
            method,
            *(
                f"{phase} {100 * (median(times(left, phase)) / median(times(right, phase)) - 1):+.2f}%"
                for phase in ("cold", "warm")
            ),
        )
    require(
        files["tile32-followup/5090-pilot-15031/exit-code.txt"].strip() == "2",
        "original unsupported GPU4 argument failure must remain visible",
    )
    for job in (15032, 15033):
        require(
            files[f"tile32-followup/5090-reference-{job}/exit-code.txt"].strip() == "0",
            "incomplete fresh n1 reference experiment",
        )
    for phase in ("cold", "warm"):
        samples = {}
        for engine in ("standard", "gpu4pyscf"):
            records_by_round = [
                followup[
                    f"tile32-followup/5090-reference-15033/{engine}-r2scan-3c-96-r{round}.json"
                ]
                for round in (1, 2)
            ]
            samples[engine] = [
                value for record in records_by_round for value in times(record, phase)
            ]
        for round in (1, 2):
            left, right = (
                followup[
                    f"tile32-followup/5090-reference-15033/{engine}-r2scan-3c-96-r{round}.json"
                ]
                for engine in ("standard", "gpu4pyscf")
            )
            same_task(left, right)
            require(
                left["scheduler"] == right["scheduler"],
                "n1 reference allocation differs",
            )
        print(
            "Additional conditioned n1 reference",
            phase,
            f"{100 * (median(samples['standard']) / median(samples['gpu4pyscf']) - 1):+.2f}%",
        )
    configuration8 = json.loads(files["tile8-followup/binary-configurations.json"])
    require(
        configuration8["candidate"]["matrix_tile"] == 8
        and configuration8["parent"]["matrix_tile"] == 16,
        "tile8 configuration differs",
    )
    prefix8 = "tile8-followup/5090-pilot-15034/"
    for method in ("pbe", "r2scan", "r2scan-3c", "m06-2x"):
        pair = []
        for arm in ("candidate", "parent"):
            record = json.loads(files[prefix8 + f"{arm}-{method}-96.json"])
            require(
                record["native_build"]["source_identity"]
                == bundle["candidate_identity"],
                "tile8 experiment source differs",
            )
            require(
                record["native_build"]["library_sha256"]
                == configuration8[arm]["binary_sha256"],
                "tile8 configuration binary differs",
            )
            q = qualify(
                [record],
                oracles,
                reference_reuse=reuse["5090"]["analysis"]["reference_reuse"],
            )[0]
            require(q["status"] == "qualified", "tile8 independent gate failed")
            pair.append(record)
        left, right = pair
        paired_errors(left, right)
        print(
            "Rejected tile8",
            method,
            *(
                f"{phase} {100 * (median(times(left, phase)) / median(times(right, phase)) - 1):+.2f}%"
                for phase in ("cold", "warm")
            ),
        )
    require(
        files[prefix8 + "exit-code.txt"].strip() == "0", "tile8 pilot did not complete"
    )
    require(
        len(bundle["large_controls"]) == 6, "larger coordinate-control census differs"
    )
    for pair in bundle["large_controls"]:
        left, right = (json.loads(files[pair[k]]) for k in ("candidate", "parent"))
        require(
            left["native_build"]["source_identity"] == bundle["candidate_identity"],
            "larger candidate source differs",
        )
        require(
            right["native_build"]["source_identity"] == bundle["parent_identity"],
            "larger parent source differs",
        )
        require(
            paired_errors(left, right, large=True) == pair["paired_errors"],
            "large control gate differs",
        )
    for name, tokens in bundle["required_checks"].items():
        for token in tokens:
            require(token in files[name], f"missing check {token}: {name}")
    print(
        len(analysis["checks"]),
        "independently qualified populations; all original member/reference hashes verified.",
    )


if __name__ == "__main__":
    main()
