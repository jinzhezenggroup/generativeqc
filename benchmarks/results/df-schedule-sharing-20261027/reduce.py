"""Recompute the opt-in historical machine-code and complete-endpoint gates.

This receipt audit is not a required source-fingerprint or exact-kernel-count CI
test. It uses the measured schedule mapping, independent of future refactors.
"""

from __future__ import annotations

import json
import lzma
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)
    before, after = (data["census"][arm] for arm in ("baseline", "candidate"))
    checked = 0
    for old, new in zip(before["objects"], after["objects"], strict=True):
        lookup = {kernel["key"]: kernel for kernel in new["kernels"]}
        for kernel in old["kernels"]:
            a, b, c, variant = map(int, re.findall(r"Lj(\d+)E", kernel["key"]))
            representative = data["schedule_aliases"][f"{a},{b},{c}"][variant]
            key = re.sub(
                r"(shell_(?:panel|packet)I(?:Lj\d+E){3})Lj\d+E",
                rf"\g<1>Lj{representative}E",
                kernel["key"],
            )
            assert kernel["resources"] == lookup[key]["resources"]
            assert kernel["instructions"] == lookup[key]["instructions"]
            checked += 1
    maximum_energy_error = maximum_force_error = 0.0
    groups = defaultdict(lambda: defaultdict(list))
    for name, old in data["endpoint_samples"].items():
        if "-baseline-" not in name:
            continue
        new = data["endpoint_samples"][name.replace("-baseline-", "-candidate-")]
        for left, right in zip(old["samples"], new["samples"], strict=True):
            for x, y in zip(left["items"], right["items"], strict=True):
                maximum_energy_error = max(
                    maximum_energy_error, abs(x["energy"] - y["energy"])
                )
                maximum_force_error = max(
                    maximum_force_error,
                    max(
                        abs(a - b)
                        for u, v in zip(x["forces"], y["forces"], strict=True)
                        for a, b in zip(u, v, strict=True)
                    ),
                )
                assert x["iterations"] == y["iterations"]
                assert x["fock_builds"] == y["fock_builds"]
        for x, y in zip(old["trace"], new["trace"], strict=True):
            assert x["operation"] == y["operation"]
            assert x["tiles"] == y["tiles"]
            for key in (x["counters"].keys() | y["counters"].keys()) - {
                "signature_packet_preparation_ns"
            }:
                assert x["counters"].get(key) == y["counters"].get(key), key
        group = "/".join(
            str(old[key])
            for key in ("case", "schedule", "buckets", "budget_bytes", "batch")
        )
        for arm, record in (("baseline", old), ("candidate", new)):
            for phase in ("cold", "warm", "moved"):
                groups[group][arm, phase].append(
                    statistics.median(
                        row["seconds"]
                        for row in record["samples"]
                        if row["phase"].startswith(phase)
                    )
                )
    assert maximum_energy_error < 3e-10
    assert maximum_force_error < 3e-9
    endpoints = {}
    for group, values in groups.items():
        endpoints[group] = {}
        for phase in ("cold", "warm", "moved"):
            medians = [
                statistics.median(values[arm, phase])
                for arm in ("baseline", "candidate")
            ]
            ratio = medians[1] / medians[0]
            assert ratio <= 1.05, (group, phase, ratio)
            endpoints[group][phase] = {
                "baseline_seconds": medians[0],
                "candidate_seconds": medians[1],
                "ratio": ratio,
            }
    print(
        json.dumps(
            {
                "identical_mapped_kernels": checked,
                "kernel_counts": [before["kernel_count"], after["kernel_count"]],
                "object_bytes": [before["object_bytes"], after["object_bytes"]],
                "maximum_energy_error": maximum_energy_error,
                "maximum_force_error": maximum_force_error,
                "endpoints": endpoints,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
