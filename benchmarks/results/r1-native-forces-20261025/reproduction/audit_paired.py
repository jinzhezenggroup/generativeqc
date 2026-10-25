"""Audit complete R1 endpoint receipts without treating partial blocks as passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
methods = ("lda", "pbe", "pbe0", "b3lyp", "r2scan", "wb97m-v", "m06-2x", "mn15")
cases = {f"{m}-{s}-{d}" for m in methods for s in (0, 1) for d in (0, 1)}
cases.update(("pbe-water36-df-d", "pbe-water48-df-d"))
unsupported = {"mn15-0-1", "mn15-1-1"}
revisions = {
    "baseline": "df2ded411074d3b04b5c236b83d760ca1aafc295",
    "current": "685478ca1111407573a5cbfa813d63ae0a5713ce",
}
failures, incomplete, receipts, metadata = [], [], [], []
groups = defaultdict(list)
for order in ("ABBA", "BAAB"):
    for block, letter in enumerate(order):
        variant = "baseline" if letter == "A" else "current"
        path = args.directory / f"paired-{order}-{block}-{variant}.jsonl"
        if not path.exists():
            incomplete.append(f"Missing {path.name}")
            continue
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        header = rows.pop(0)["metadata"]
        metadata.append(dict(order=order, block=block, variant=variant, **header))
        assert header["source_revision"] == revisions[variant], path
        assert header["samples"] == 11 and header["warmups"] == 3, path
        assert header["tracing"] is False, path
        seen, errors = defaultdict(list), set()
        for row in rows:
            if "error" in row:
                errors.add(row["case"])
                if variant != "baseline" or row["case"] not in unsupported:
                    failures.append(dict(file=path.name, **row))
                continue
            assert row["case"] in cases, row
            assert row["executed_backend"] == "cuda", row
            assert math.isfinite(row["seconds"]) and row["seconds"] > 0, row
            assert math.isfinite(row["energy"]), row
            assert all(math.isfinite(x) for v in row["forces"] for x in v), row
            if variant == "current" and row["phase"] == "cold":
                assert row["native"] is True, row
            seen[row["case"], row["phase"]].append(row["sample"])
            groups[order, variant, row["case"], row["phase"]].append(row)
        expected_errors = unsupported if variant == "baseline" else set()
        for name in cases - expected_errors:
            for phase in ("cold", "warm", "moved"):
                if sorted(seen[name, phase]) != list(range(11)):
                    incomplete.append(f"{path.name}: incomplete {name}/{phase}")
        if errors != expected_errors:
            incomplete.append(f"{path.name}: expected errors {expected_errors}, got {errors}")
        receipts.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows)))

comparisons = []
for order in ("ABBA", "BAAB"):
    for name in sorted(cases - unsupported):
        for phase in ("cold", "warm", "moved"):
            baseline = groups[order, "baseline", name, phase]
            current = groups[order, "current", name, phase]
            if len(baseline) != 22 or len(current) != 22:
                continue
            medians = [statistics.median(r["seconds"] for r in samples) for samples in (baseline, current)]
            max_energy = max_force = 0.0
            work_mismatches = []
            # All-to-all comparison at the same geometry catches block-dependent drift.
            for left in baseline:
                for right in current:
                    if phase == "moved" and left["sample"] % 2 != right["sample"] % 2:
                        continue
                    max_energy = max(max_energy, abs(left["energy"] - right["energy"]))
                    max_force = max(max_force, max(abs(x - y) for a, b in zip(left["forces"], right["forces"], strict=True) for x, y in zip(a, b, strict=True)))
                    if any(left[k] != right[k] for k in ("iterations", "fock_builds", "warm_start_used")):
                        work_mismatches.append({k: [left[k], right[k]] for k in ("iterations", "fock_builds", "warm_start_used")})
            row = dict(order=order, case=name, phase=phase, samples_per_variant=22,
                       baseline_median_seconds=medians[0], current_median_seconds=medians[1],
                       ratio=medians[1]/medians[0], max_energy_difference=max_energy,
                       max_force_difference=max_force, work_mismatch_pairs=len(work_mismatches),
                       work_mismatch_examples=work_mismatches[:2])
            comparisons.append(row)
            if row["ratio"] > 1.05 or max_energy > 2e-8 or max_force > 3e-7 or work_mismatches:
                failures.append(row)

result = dict(status="INCOMPLETE" if incomplete else ("FAIL" if failures else "PASS"),
              gate=dict(max_ratio=1.05, max_energy_difference=2e-8, max_force_difference=3e-7,
                        equal_iterations_fock_builds_warm_start=True),
              source_revisions=revisions, metadata=metadata, receipts=receipts,
              incomplete=incomplete, failures=failures, comparisons=comparisons,
              new_endpoints_without_baseline=sorted(unsupported))
args.output.write_text(json.dumps(result, indent=2)+"\n")
print(result["status"], "comparisons", len(comparisons), "failures", len(failures), "incomplete", len(incomplete))
for row in sorted(comparisons, key=lambda r:r["ratio"], reverse=True)[:12]:
    print(row["order"], row["case"], row["phase"], f'{row["ratio"]:.4f}',
          "work mismatches", row["work_mismatch_pairs"])
