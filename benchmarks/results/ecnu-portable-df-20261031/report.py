"""Report recorded complete endpoint populations without hiding failed gates."""

import json
import lzma
from pathlib import Path
from statistics import median


def main() -> None:
    """Keep measured candidate and automatic-integration artifacts separate."""
    data = json.loads(
        lzma.decompress(Path(__file__).with_name("observations.json.xz").read_bytes())
    )
    columns = data["sample_columns"]
    phase, seconds = columns.index("phase"), columns.index("seconds")
    for name in ("a100-clean", "a100-promotion", "5090-auto", "pro6000-auto"):
        cohort = data["cohorts"][name]
        comparisons = {row["case"]: row for row in data["analyses"][name]["comparisons"]}
        print(name, "case warm_seconds reference_seconds")
        for case, record in sorted(cohort["records"].items()):
            if not case.endswith("-96"):
                continue
            warm = median(s[seconds] for s in record["samples"] if s[phase] == "warm")
            reference = comparisons[case + ".json"].get("reference_warm_seconds")
            print(case, f"{warm:.6f}", f"{median(reference):.6f}" if reference else "unmeasured")
        print("Failed gates:", [
            (q["method"], q["mode"], q["atoms"])
            for q in data["analyses"][name]["qualification"] if q["status"] != "qualified"
        ])


if __name__ == "__main__":
    main()
