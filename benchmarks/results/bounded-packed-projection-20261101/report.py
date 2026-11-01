"""Authenticate exact original receipts and replay independent endpoint gates.

All floating values are FP64 XOR deltas keyed by their JSON location and method.
Byte planes improve compression without rounding. Original JSON formatting and
SHA-256 are reconstructed before any scientific or timing comparison is made.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
import struct
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
PRIOR = runpy.run_path(
    str(ROOT / "benchmarks/results/xc-potential-unroll4-20261101/report.py")
)
require = PRIOR["require"]


def load() -> tuple[dict, dict[str, str]]:
    """Reconstruct and authenticate every original receipt, even under -O."""
    folder = Path(__file__).parent
    for line in (folder / "sha256.txt").read_text().splitlines():
        digest, name = line.split()
        require(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
        )
    bundle = json.loads(lzma.decompress((folder / "receipts.json.xz").read_bytes()))
    require(bundle["schema"] == "generativeqc.bounded-projection-evidence.v1", "schema")
    planes = lzma.decompress((folder / "fp64.xz").read_bytes())
    count = bundle["float_count"]
    require(len(planes) == 8 * count, "FP64 storage size")
    last: dict[tuple, int] = {}
    cursor = 0

    def decode(value: object, path: tuple) -> object:
        nonlocal cursor
        if isinstance(value, dict) and "$fp64" in value:
            require(value == {"$fp64": cursor} and cursor < count, "FP64 ordering")
            delta = int.from_bytes(
                bytes(planes[lane * count + cursor] for lane in range(8)), "little"
            )
            word = delta ^ last.get(path, 0)
            last[path] = word
            cursor += 1
            return struct.unpack("<d", word.to_bytes(8, "little"))[0]
        if isinstance(value, list):
            return [decode(v, path + (i,)) for i, v in enumerate(value)]
        if isinstance(value, dict):
            return {k: decode(v, path + (k,)) for k, v in value.items()}
        return value

    files = {}
    for row in bundle["files"]:
        text = (
            (json.dumps(decode(row["json"], (row["anchor"],)), indent=2) + "\n")
            if "json" in row
            else row["text"]
        )
        require(hashlib.sha256(text.encode()).hexdigest() == row["sha256"], row["path"])
        require(row["path"] not in files, "duplicate receipt")
        files[row["path"]] = text
    require(cursor == count, "unconsumed FP64 values")
    return bundle, files


def main() -> None:
    """Check complete populations, scheduler/device identity, work and E/F gates."""
    bundle, files = load()
    references = {}
    for name, digest in bundle["references"].items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, name)
        references[name] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]["a100"]["analysis"]["reference_reuse"]
    results = []
    population_count = call_count = 0
    for folder in bundle["populations"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", "allocation failed")
        identities = {
            v: json.loads(files[f"{folder}/{v}-identity.json"])
            for v in ("baseline", "binaries")
        }
        records = {}
        for name, text in files.items():
            path = Path(name)
            if (
                path.parent.as_posix() != folder
                or not path.name.startswith(("conditioning-", "slot-"))
                or path.suffix != ".json"
            ):
                continue
            r = json.loads(text)
            variant = "baseline" if "-baseline-" in path.name else "binaries"
            identity = identities[variant]
            for key in ("source_identity", "library_sha256"):
                require(r["native_build"][key] == identity[key], "native " + key)
            for key in ("LSB_JOBID", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"):
                if key in identity["scheduler"]:
                    require(
                        r["scheduler"].get(key) == identity["scheduler"][key],
                        "scheduler " + key,
                    )
            gate = qualify([r], oracles, reference_reuse=reuse)[0]
            require(gate["status"] == "qualified", str(gate))
            population_count += 1
            for s in r["samples"]:
                require(s["converged"], "convergence")
                p = s["precision"]
                for kind, key in (
                    ("strict_fock", "strict_stage_fock_builds"),
                    ("mixed_fock", "mixed_stage_fock_builds"),
                    ("post_scf_fock", "post_scf_fock_builds"),
                ):
                    require(
                        sum(
                            e["count"]
                            for e in p["scf_fock_timeline"]
                            if e["kind"] == kind
                        )
                        == p[key],
                        "Fock timeline",
                    )
                require(
                    sum(
                        p[k]
                        for k in (
                            "strict_stage_fock_builds",
                            "mixed_stage_fock_builds",
                            "post_scf_fock_builds",
                        )
                    )
                    == s["fock_builds"],
                    "Fock census",
                )
                call_count += 1
            records[path.name] = r
        for method in sorted({r["method"] for r in records.values()}):
            group = {n: r for n, r in records.items() if r["method"] == method}
            require(len(group) == 6, "incomplete conditioning/ABBA population")
            errors = PRIOR["paired_errors"](
                *(
                    group[f"conditioning-{v}-{method}.json"]
                    for v in ("baseline", "binaries")
                )
            )
            arms = {
                v: [group[f"slot-{slot}-{v}-{method}.json"] for slot in slots]
                for v, slots in (("baseline", (1, 4)), ("binaries", (2, 3)))
            }
            for a, b in zip(arms["baseline"], arms["binaries"], strict=True):
                errors += PRIOR["paired_errors"](a, b)
            seconds = {
                v: {
                    phase: [
                        s["seconds"]
                        for r in pops
                        for s in r["samples"]
                        if s["phase"] == phase
                    ]
                    for phase in ("cold", "warm")
                }
                for v, pops in arms.items()
            }
            results.append(
                {
                    "allocation": folder,
                    "method": method,
                    "seconds": seconds,
                    "percent_change": {
                        phase: 100
                        * (
                            median(seconds["binaries"][phase])
                            / median(seconds["baseline"][phase])
                            - 1
                        )
                        for phase in ("cold", "warm")
                    },
                    "paired_max_energy": max(e["energy_abs"] for e in errors),
                    "paired_max_force": max(e["force_max"] for e in errors),
                    "work": {
                        v: [
                            [[s["iterations"], s["fock_builds"]] for s in r["samples"]]
                            for r in pops
                        ]
                        for v, pops in arms.items()
                    },
                }
            )
    print(
        json.dumps(
            {
                "qualified_populations": population_count,
                "qualified_calls": call_count,
                "comparisons": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
