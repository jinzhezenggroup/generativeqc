"""Report complete response-workspace controls without selecting fast repeats."""

import json
import lzma
from pathlib import Path


def main() -> None:
    """Print every retained method/device population summary and gate count."""
    data = json.loads(
        lzma.decompress(Path(__file__).with_name("observations.json.xz").read_bytes())
    )
    for device, rows in data["analysis"]["devices"].items():
        print(device)
        for method, result in rows["methods"].items():
            print(method, "arm cold_seconds warm_median warm_stdev warm_seconds")
            for arm, values in result["arms"].items():
                print(
                    arm,
                    values["cold_seconds"],
                    values["warm_median"],
                    values["warm_stdev"],
                    values["warm_seconds"],
                )
            print("candidate/baseline", result.get("candidate_over_baseline"))
        print(
            "qualified",
            sum(
                p.get("qualification", {}).get("status") == "qualified"
                for p in rows["populations"]
            ),
            "/",
            len(rows["populations"]),
        )
    if "diagnostic" in data:
        print("Experimental grid diagnostic; not production defaults")
        for tile, record in data["diagnostic"]["arms"].items():
            warm = next(s["seconds"] for s in record["samples"] if s["phase"] == "warm")
            response = record["warm_response"][0]
            print(
                tile,
                "warm_seconds",
                warm,
                "occupied_response_seconds",
                response["host_completion_ms"] / 1000,
                "metric_bytes_avoided",
                response["counters"]["response_metric_workspace_bytes_avoided"],
                "fitted_panel_gemms",
                response["counters"].get("response_fitted_panel_gemms", 0),
                "gate",
                record["qualification"]["status"],
            )


if __name__ == "__main__":
    main()
