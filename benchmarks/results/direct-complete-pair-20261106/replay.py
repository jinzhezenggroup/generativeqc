"""Replay complete-pair endpoint/science gates; racecheck remains inconclusive."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import math
import runpy
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

IMAGES = {
    "baseline": (
        "b78434b9416f124db8bf73b3b339bae894027959ed0220f6731e1bb1b10c2293",
        "05319ea1eada9df72daa64fe05d131c553de582211b6b594d4d2f44f5fba8871",
    ),
    "candidate": (
        "4c54ccfb3d38e5ec86b147a019d8ef99e87c8b8247ca5a42bf50a84027dc61ed",
        "3142176ecd915a1fe5eaaced6aa5cb4b3da3af6a1ff7b71b4cf6a2255f199799",
    ),
}


def population(folder: Path, arm: str) -> dict[str, Any]:
    """Require complete populations and the actual fail-fast runner's zero exit."""
    assert (folder / "exit-code.txt").read_text().strip() == "0"
    record = json.loads(lzma.decompress((folder / f"{arm}.json.xz").read_bytes()))
    assert record["status"] == "complete" and record["warm_repeats"] == 3
    assert record["atoms"] == 96 and record["method"] == "r2scan-3c"
    assert record["mode"] == "direct" and record["occupied_xc_env"] is None
    assert len(record["samples"]) == 4
    for index, sample in enumerate(record["samples"]):
        assert (sample["phase"], sample["repeat"]) == (
            ("cold", 0) if index == 0 else ("warm", index - 1)
        )
        assert sample["converged"] and sample["iterations"] > 0
        assert math.isfinite(sample["energy"])
        assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
        assert len(sample["forces"]) == 96
        assert all(len(row) == 3 for row in sample["forces"])
        assert all(math.isfinite(value) for row in sample["forces"] for value in row)
    identity = json.loads((folder / f"{arm}-identity.json").read_text())
    assert identity["source_identity"] == record["source_revision"] == IMAGES[arm][0]
    assert identity["scheduler"] == record["scheduler"]
    for key, expected in zip(
        ("source_identity", "library_sha256"), IMAGES[arm], strict=True
    ):
        assert identity[key] == record["native_build"][key] == expected
    return record


def independent_xml(path: Path, expected: int, skipped: int = 0) -> None:
    """Check every declared case, preserving explicit invalid-signature skips."""
    content = lzma.decompress(path.read_bytes())
    assert len(content) <= 1 << 20
    assert b"<!DOCTYPE" not in content.upper() and b"<!ENTITY" not in content.upper()
    document = element_tree.fromstring(content)  # noqa: S314 - bounded pytest XML; DTD/entities refused
    cases = list(document.iter("testcase"))
    assert len(cases) == expected + skipped
    assert sum(bool(list(case.iter("skipped"))) for case in cases) == skipped
    assert all(
        not list(case.iter("failure")) and not list(case.iter("error"))
        for case in cases
    )


def sanitizer_receipt(root: Path) -> dict[str, Any]:
    """Do not turn a bounded diagnostic timeout into a sanitizer success."""
    folder = root / "safety"
    assert (folder / "exit-code.txt").read_text().strip() == "1"
    assert (folder / "memcheck.exit-code.txt").read_text().strip() == "0"
    assert (folder / "racecheck.exit-code.txt").read_text().strip() == "124"
    memcheck = gzip.decompress((folder / "memcheck.log.gz").read_bytes())
    racecheck = gzip.decompress((folder / "racecheck.log.gz").read_bytes())
    assert b"ERROR SUMMARY: 0 errors" in memcheck
    assert racecheck.strip() == b"========= COMPUTE-SANITIZER"
    independent_xml(folder / "memcheck.xml.xz", 1)
    return {
        "job": "Slurm16008",
        "runner_exit": 1,
        "memcheck": "passed; zero reported errors; all seven classes",
        "racecheck": "inconclusive; 180s cap; exit124; no diagnostic",
        "blanket_sanitizer_pass": False,
    }


def main() -> None:
    """Reuse the unchanged accepted observable/work gates, without either engine."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    prior = root.parent / "direct-cooperative-md-20261105"
    gates = runpy.run_path(str(prior / "replay.py"))
    reference = gates["load_population"](prior / "5090", "reference")
    report: dict[str, Any] = {
        "scope": "one static geometry; compact-MD baseline; reused reference for science only"
    }
    observations = {}
    for device in ("5090", "a100"):
        baseline = population(root / device, "baseline")
        candidate = population(root / device, "candidate")
        assert baseline["scheduler"] == candidate["scheduler"]
        assert baseline["thread_controls"] == candidate["thread_controls"]
        assert baseline["protocol"] == candidate["protocol"]
        assert (
            baseline["accelerator"]["pci_bus_id"]
            == candidate["accelerator"]["pci_bus_id"]
        )
        scheduler = baseline["scheduler"]
        if device == "5090":
            assert scheduler["SLURM_JOB_ID"] == "16007"
        else:
            assert scheduler["LSB_JOBID"] == "225765"
        assert scheduler["CUDA_VISIBLE_DEVICES"]
        report[device] = gates["summarize_pair"](baseline, candidate)
        for arm, record in (("baseline", baseline), ("candidate", candidate)):
            observations[f"{device}-{arm}"] = gates["compare_observables"](
                record, reference, gates["COMMON_GATES"]
            )
    assert all(
        ratio <= 1.05 for ratio in report["5090"]["warm"]["corresponding_ratios"]
    )
    independent_xml(root / "5090/independent.xml.xz", 6)
    independent_xml(root / "host.xml.xz", 585, 164)
    independent_xml(root / "default-policy.xml.xz", 1165)
    report["reference_observables"] = observations
    report["independent_device_cases"] = 6
    report["independent_host_cases"] = 585
    report["invalid_host_signatures_skipped"] = 164
    report["default_policy_host_cases"] = 1165
    report["safety"] = sanitizer_receipt(root)
    report["status"] = "endpoint-and-independent-science-passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
