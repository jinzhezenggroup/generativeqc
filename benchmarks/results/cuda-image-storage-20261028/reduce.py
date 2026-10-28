"""Replay retained storage, scientific and endpoint gates without CUDA.

Instruction inventories are collector attestations: the collector compares every
section and retains their canonical multiset digest, not executable payloads.
The complete collection recipe and raw-inventory identity accompany the receipt.
"""

from __future__ import annotations

import base64
import struct
import json
import lzma
from pathlib import Path
from statistics import median


def force_error(a: dict, b: dict) -> float:
    return max(
        abs(x - y)
        for v, w in zip(a["forces"], b["forces"], strict=True)
        for x, y in zip(v, w, strict=True)
    )


def decode_samples(samples: dict, references: dict) -> dict:
    """Restore every energy/force bit from the lossless column-delta encoding."""
    for sample in samples.values():
        encoded = sample.pop("energy_force_ieee754_xor", None)
        if encoded is None:
            continue
        width = 1 + 3 * sample.pop("force_atoms")
        planes = base64.b64decode(encoded)
        count = len(planes) // 8
        raw = bytes(planes[byte * count + i] for i in range(count) for byte in range(8))
        bits = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
        assert len(bits) == width * len(sample["rows"])
        for i, row in enumerate(sample["rows"]):
            reference = struct.unpack("<" + "Q" * width, base64.b64decode(
                references[sample["cohort"]][row["phase"]]))
            values = [a ^ b for a, b in zip(
                bits[i * width:(i + 1) * width], reference, strict=True)]
            numbers = struct.unpack("<" + "d" * width,
                                    struct.pack("<" + "Q" * width, *values))
            row["energy"] = numbers[0]
            row["forces"] = [list(numbers[j:j + 3]) for j in range(1, width, 3)]
    return samples


def endpoints(samples: dict, expected: dict, rounds: int) -> dict:
    """Keep every timing failure while enforcing all scientific/work gates."""
    result = {"calls": 0, "failures": [], "cohorts": {}, "paired_energy": 0.0,
              "paired_force": 0.0, "oracle_energy": 0.0, "oracle_force": 0.0}
    for kind in ("direct", "df", "dffast", "pbe", "pbedf"):
        for size in (24, 96):
            cohort = f"{kind}-{size}"
            oracle = samples[f"{cohort}-oracle.json"]["oracle"]
            arms = ["baseline", "candidate"]
            if f"{cohort}-0-compressed.json" in samples:
                arms.append("compressed")
            observations = {arm: [] for arm in arms}
            for index in range(rounds):
                pair = []
                for arm in arms:
                    population = samples[f"{cohort}-{index}-{arm}.json"]
                    assert population["library_sha256"] == expected[arm]
                    assert population["job"] and population["module_loading"] == "LAZY"
                    rows = population["rows"]
                    assert [r["phase"] for r in rows] == [
                        "cold", "warm1", "warm2", "warm3", "moved"]
                    observations[arm].append(rows)
                    pair.append(rows)
                    for row in rows:
                        reference = oracle[int(row["phase"] == "moved")]
                        energy = abs(row["energy"] - reference["energy"])
                        force = force_error(row, reference)
                        assert energy <= 1e-7
                        assert force <= (1e-5 if kind == "dffast" else 1e-6)
                        result["oracle_energy"] = max(result["oracle_energy"], energy)
                        result["oracle_force"] = max(result["oracle_force"], force)
                        result["calls"] += 1
                for other in pair[1:]:
                    for before, after in zip(pair[0], other, strict=True):
                        for field in ("iterations", "fock_builds"):
                            assert before[field] == after[field]
                        for field in ("attempted_fp32_weighted_tasks",
                                      "attempted_fp32_primitive_sources",
                                      "weighted_task_retries"):
                            a = ((before.get("precision") or {}).get("gradient") or {}).get(field)
                            b = ((after.get("precision") or {}).get("gradient") or {}).get(field)
                            assert a == b
                        result["paired_energy"] = max(
                            result["paired_energy"], abs(before["energy"] - after["energy"]))
                        result["paired_force"] = max(
                            result["paired_force"], force_error(before, after))
            phases = {}
            for phase in ("cold", "warm", "moved"):
                times = {
                    arm: [median(row["seconds"] for row in rows
                                 if row["phase"].startswith(phase))
                          for rows in population]
                    for arm, population in observations.items()
                }
                ratio = median(times["candidate"]) / median(times["baseline"])
                phases[phase] = {"process_seconds": times, "ratio": ratio}
                if ratio > 1.02:
                    result["failures"].append([cohort, phase, ratio])
                if "compressed" in times:
                    phases[phase]["compression_ratio"] = (
                        median(times["compressed"]) / median(times["candidate"]))
            result["cohorts"][cohort] = phases
    assert result["paired_energy"] <= 1e-9
    assert result["paired_force"] <= 1e-8
    result["maximum_ratio"] = max(
        p["ratio"] for c in result["cohorts"].values() for p in c.values())
    return result


def reduce(data: dict) -> dict:
    census = data["census"]
    result = {"sizes": {}, "campaigns": {}}
    for scope, after in (("df", "compressed"), ("package", "final")):
        before, candidate = census[scope]["baseline"], census[scope][after]
        assert before.keys() == candidate.keys()
        assert len(before) == (80 if scope == "df" else 32)
        for name in before:
            assert before[name]["instructions"] == candidate[name]["instructions"], name
        result["sizes"][scope] = {
            "before": sum(x["bytes"] for x in before.values()),
            "after": sum(x["bytes"] for x in candidate.values()),
            "artifacts": len(before),
        }
    assert census["native"]["baseline"]["instructions"] == census["native"]["compressed"]["instructions"]
    for arm in ("baseline", "final", "default"):
        witness = data["records"][f"remote/package-load-{arm}.json"]
        assert witness["mode"] == "EAGER" and witness["job"]
        assert len(witness["rows"]) == 32
        expected = census["package"]["baseline" if arm == "baseline" else "final"]
        for row in witness["rows"]:
            assert row["sha256"] == expected[row["library"]]["sha256"]
    result["sizes"]["native_historical"] = {
        "before": data["artifacts"]["historical-original"]["bytes"],
        "default": census["native"]["baseline"]["bytes"],
        "compressed": census["native"]["compressed"]["bytes"],
        "instruction_sections": census["native"]["baseline"]["instructions"]["count"],
    }
    for name, campaign in data["campaigns"].items():
        if name == "mainline-default":
            assert campaign["hashes"]["baseline"] == data["artifacts"]["mainline-original"]["sha256"]
            assert campaign["hashes"]["candidate"] == data["artifacts"]["mainline-default"]["sha256"]
        else:
            assert campaign["hashes"]["baseline"] == data["artifacts"]["historical-original"]["sha256"]
            expected = "compressed" if name == "compression" else "baseline"
            assert campaign["hashes"]["candidate"] == census["native"][expected]["sha256"]
        result["campaigns"][name] = endpoints(
            decode_samples(campaign["samples"], campaign["numeric_references"]), campaign["hashes"], campaign["rounds"])
    # Earlier failures are retained negative evidence, not promoted acceptance.
    assert result["campaigns"]["compression"]["failures"]
    final = result["campaigns"]["mainline-default"]
    assert final["calls"] == 1000 and not final["failures"], final["failures"]
    before, after = (data["artifacts"][name] for name in
                     ("mainline-original", "mainline-default"))
    assert before["instructions"] == after["instructions"]
    result["sizes"]["native_mainline"] = {
        "before": before["bytes"], "default": after["bytes"],
        "instruction_sections": after["instructions"]["count"],
    }
    return result


if __name__ == "__main__":
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as handle:
        print(json.dumps(reduce(json.load(handle)), indent=2, sort_keys=True))
