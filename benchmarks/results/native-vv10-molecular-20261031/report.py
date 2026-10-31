"""Report every retained complete endpoint alongside its independent gate."""

import json
import lzma
from pathlib import Path
from statistics import median


def main() -> None:
    """Keep default-grid measurements distinct from cross-binary controls."""
    data = json.loads(
        lzma.decompress(Path(__file__).with_name("observations.json.xz").read_bytes())
    )
    for name in ("a100", "5090", "pro6000"):
        analysis = data["analyses"][name]
        gates = {q["receipt_sha256"]: q["status"] for q in analysis["qualification"]}
        comparisons = {row["case"]: row for row in analysis["comparisons"]}
        print(name, "case cold_seconds warm_seconds reference_warm_seconds gate")
        for case, record in sorted(data["cohorts"][name]["records"].items()):
            if record["status"] != "complete":
                print(case, record["status"])
                continue
            samples = record["samples"]
            cold = next(s["seconds"] for s in samples if s["phase"] == "cold")
            warm = median(s["seconds"] for s in samples if s["phase"] == "warm")
            reference = comparisons[case + ".json"].get("reference_warm_seconds")
            print(
                case,
                f"{cold:.6f}",
                f"{warm:.6f}",
                f"{median(reference):.6f}" if reference else "unmeasured",
                gates[record["receipt_sha256"]],
            )
    controls = data["provenance"].get("abba-analysis.json", {})
    for device, control in controls.get("devices", {}).items():
        print(device, "accepted ABBA controls (warm median, min, max seconds)")
        for arm, result in control["arms"].items():
            print(
                arm,
                *(
                    f"{result[key]:.6f}"
                    for key in (
                        "warm_median_seconds",
                        "warm_min_seconds",
                        "warm_max_seconds",
                    )
                ),
            )
        for rejected in control["excluded_populations"]:
            print("excluded", rejected["path"], rejected.get("reason", rejected["status"]))


if __name__ == "__main__":
    main()
