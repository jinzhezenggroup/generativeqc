"""Replay retained default endpoint qualifications without a GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from pathlib import Path

from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_qualification import qualify


def read(path: Path) -> dict | list:
    """Read a lossless evidence container without changing producer receipts."""
    return json.loads(lzma.decompress(path.read_bytes()))


def main() -> None:
    """Recompute every main gate, cold median and moved-geometry comparison."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    summary = read(directory / "summary.json.xz")
    payloads = read(directory / "receipts.json.xz")
    patch = lzma.decompress((directory / "source.patch.xz").read_bytes())
    if hashlib.sha256(patch).hexdigest() != summary["source"]["patch_sha256"]:
        raise ValueError("Changed frozen-source reconstruction patch")
    for name, expected in summary["references"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed independent reference: {name}")
    reference = directory.parent / "readme-refresh-20261025"
    oracles = read(reference / "oracles.json.xz")
    reuse = json.loads((reference / "reference-reuse.json").read_text())
    groups: dict[str, dict[str, list[float]]] = {}
    calls = 0
    for population in summary["populations"]:
        payload = payloads[population["file"]].encode()
        if hashlib.sha256(payload).hexdigest() != population["sha256"]:
            raise ValueError(f"Changed endpoint bytes: {population['file']}")
        receipt = json.loads(payload)
        actual = qualify([receipt], oracles, reference_reuse=reuse)[0]
        if actual != population["qualification"]:
            raise ValueError(f"Changed qualification: {population['file']}")
        # DF fitting error is part of the retained common-gate result, not an
        # omitted population or a reason to relax the independent target.
        expected_status = (
            "failed"
            if receipt["method"] == "hf" and receipt["mode"] == "df"
            else "qualified"
        )
        if actual["status"] != expected_status:
            raise ValueError("Unexpected common-gate result")
        key = "-".join(
            str(receipt[k]) for k in ("method", "mode", "atoms", "arm")
        )
        group = groups.setdefault(key, {"cold": [], "warm": []})
        for sample in receipt["samples"]:
            calls += 1
            group[sample["phase"]].append(sample["seconds"])
            if not sample["converged"]:
                raise ValueError("Incomplete endpoint")
    if calls != 144 or len(summary["populations"]) != 36:
        raise ValueError("Expected 36 main populations and 144 complete calls")
    for key, group in groups.items():
        for phase in ("cold", "warm"):
            retained = summary["groups"][key]
            if group[phase] != retained[phase] or statistics.median(group[phase]) != (
                retained[f"{phase}_median"]
            ):
                raise ValueError(f"Changed timing reduction: {key}/{phase}")
        if key.endswith("-standard"):
            other = key.removesuffix("-standard") + "-gpu4pyscf"
            if statistics.median(group["cold"]) >= statistics.median(
                groups[other]["cold"]
            ):
                raise ValueError(f"Cold median is not faster: {key}")
    evidence = read(directory / "evidence.json.xz")
    census = evidence["work_census"]
    before = census["r2-default-v1"]["by_fock"]
    after = census["r2-default-v2"]["by_fock"]
    for invocation in (1, *range(8, 14)):
        if before[str(invocation)] != after[str(invocation)]:
            raise ValueError("Physical/warm work changed in the retained census")
    for field in ("uniform_roots", "uniform_summands"):
        if sum(after[str(i)][field] for i in range(2, 8)) >= sum(
            before[str(i)][field] for i in range(2, 8)
        ):
            raise ValueError("Proposal census does not remove primitive work")
    moved = evidence["larger_geometry"]
    references = {
        (r["atoms"], r["geometry"]): r["sample"]
        for r in moved
        if r["kind"] == "reference"
    }
    checked = 0
    for record in moved:
        if record["kind"] != "native":
            continue
        actual = errors(
            record["energy"],
            record["forces"],
            references[(record["atoms"], record["geometry"])],
            record["atoms"],
        )
        if actual != record["errors"] or not actual["passed"]:
            raise ValueError("Changed larger/moved observable gate")
        checked += 1
    if checked != 8:
        raise ValueError("Expected eight independent larger/moved checks")
    print("Rechecked 144 main calls, six cold wins and eight larger/moved gates.")


if __name__ == "__main__":
    main()
