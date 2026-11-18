"""Collect trusted local native GPU test output without retaining bulky XML."""

from __future__ import annotations

import argparse
import json
import lzma
import runpy
from pathlib import Path
from xml.etree import ElementTree


def main() -> None:
    """Preserve all measured arrays and require successful independent replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", nargs="+", type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--endpoints", nargs="*", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("acceptance.json.xz")
    )
    arguments = parser.parse_args()
    cases = []
    for path in arguments.junit:
        root = ElementTree.parse(path)  # noqa: S314 -- trusted local pytest receipts only
        for case in root.iter("testcase"):
            assert not any(
                case.find(status) is not None
                for status in ("failure", "error", "skipped")
            )
            properties = {
                item.attrib["name"]: item.attrib["value"]
                for item in case.findall("properties/property")
            }
            cases.append(
                {
                    "id": f"{path.stem}:{case.attrib['name']}",
                    "input": json.loads(properties["native_force_map_input"]),
                    "references": json.loads(properties["native_force_map_references"]),
                    "samples": json.loads(properties["native_force_map_endpoints"]),
                    "capacity": json.loads(properties["native_force_map_capacity"]),
                    "reference_metrics": json.loads(
                        properties["native_force_map_reference_metrics"]
                    ),
                }
            )
    payload = {
        "schema": "generativeqc.native-force-ao-acceptance.v1",
        "provenance": json.loads(arguments.provenance.read_text()),
        "cases": cases,
    }
    replay = runpy.run_path(Path(__file__).with_name("replay.py"))
    summary = replay["validate"](payload)
    with lzma.open(arguments.output, "wt") as handle:
        json.dump(
            payload, handle, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.endpoints:
        endpoints = {
            "runs": [json.loads(path.read_text()) for path in arguments.endpoints]
        }
        summary = replay["validate_endpoints"](endpoints, payload)
        with lzma.open(arguments.output.with_name("endpoints.json.xz"), "wt") as handle:
            json.dump(
                endpoints,
                handle,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
