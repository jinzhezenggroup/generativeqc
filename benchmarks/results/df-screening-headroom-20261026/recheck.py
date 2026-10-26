"""Replay retained endpoint, identity, work and independent numerical gates.

No GPU execution is required. Raw endpoint strings remain byte-recoverable;
exploratory summaries and selected trace lines are explicitly distinguished
from the clean three-round qualification populations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
from pathlib import Path

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults_qualification import qualify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = "9ad12bbe8e66f8dfb002e68a2b9d90e2e314f475bfc141dd3701258082407462"
CANDIDATE = "1dc2803d680cfa12a40fff3586be79cd3ad90fc1267de526050fb8847109829c"
INPUTS = (
    "geometry_bohr",
    "orbital_basis",
    "basis_representation",
    "charge",
    "spin",
    "auxiliary_input_sha256",
)


def digest(data: bytes) -> str:
    """Hash original bytes, including whitespace in retained producer receipts."""
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> dict | list:
    """Load plain or losslessly compressed JSON."""
    data = path.read_bytes()
    return json.loads(lzma.decompress(data) if path.suffix == ".xz" else data)


def timing(samples: list[dict], phase: str) -> dict:
    """Retain all observations; summarize pooled warm calls without filtering."""
    values = [s["seconds"] for s in samples if s["phase"] == phase]
    return {
        "seconds": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def errors(first: list[dict], second: list[dict]) -> dict:
    """Compare every baseline/candidate pair, including every warm repeat."""
    return {
        "energy_max_abs": max(
            abs(a["energy"] - b["energy"]) for a in first for b in second
        ),
        "force_max_abs": max(
            float(np.max(np.abs(np.asarray(a["forces"]) - b["forces"])))
            for a in first
            for b in second
        ),
    }


def main() -> None:
    """Verify the historical evidence and optionally compare its exact summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = read(HERE / "manifest.json.xz")
    bundles = {}
    for name, expected in manifest["bundles"].items():
        data = (HERE / name).read_bytes()
        assert len(data) == expected["bytes"] and digest(data) == expected["sha256"]
        members = json.loads(lzma.decompress(data))
        assert members.keys() == expected["members"].keys()
        for key, value in members.items():
            raw = value.encode()
            assert digest(raw) == expected["members"][key]["sha256"]
            assert len(raw) == expected["members"][key]["bytes"]
        bundles[name] = members
    assert (
        digest(lzma.decompress((HERE / "source.patch.xz").read_bytes()))
        == (manifest["source"]["patch_sha256"])
    )
    for path, expected in manifest["references"].items():
        assert digest((ROOT / path).read_bytes()) == expected
    endpoints = {k: json.loads(v) for k, v in bundles["endpoints.json.xz"].items()}
    diagnostics = bundles["diagnostics.json.xz"]
    build = bundles["build.json.xz"]
    relink = json.loads(build["incremental-build/parent-relink.json"])
    assert relink["sha256"] == relink["expected"] == BASELINE
    assert (
        json.loads(build["incremental-build/libraries.json"])["libgenerativeqc.so"]
        == CANDIDATE
    )
    reuse = json.loads(build["incremental-build/reuse-summary.json"])
    assert reuse["parent_library_sha256"] == BASELINE
    assert len(reuse["reused_objects"]) == 478
    assert reuse["audited_source_file_count"] == 2314
    oracles = read(HERE.parent / "readme-refresh-20261026/oracles.json.xz")
    reference_reuse = read(HERE.parent / "readme-refresh-20261026/reference-reuse.json")
    summary = {"cards": {}, "qualified_populations": 0, "qualified_calls": 0}
    for cohort in ("qualification-0", "qualification-1"):
        allocation = json.loads(diagnostics[f"{cohort}/allocation.json"])
        commands = json.loads(diagnostics[f"{cohort}/commands.json"])
        assert len(commands) == 9 and all(r["returncode"] == 0 for r in commands)
        card = {"pci": allocation["pci"], "slurm_job": allocation["job"], "arms": {}}
        populations = {}
        for arm in ("baseline", "candidate", "gpu4pyscf"):
            rows = [endpoints[f"{cohort}/{i}-{arm}.json"] for i in range(3)]
            populations[arm] = rows
            for r in rows:
                assert r["status"] == "complete" and len(r["samples"]) == 4
                assert (r["method"], r["mode"], r["atoms"]) == ("pbe0", "df", 96)
                assert r["slurm_job_id"] == allocation["job"]
                assert (
                    r["environment"]["runtime"]["cuda_visible_devices"]
                    == (allocation["visibility"])
                )
                if arm != "baseline":
                    assert r["accelerator"]["pci_bus_id"] == allocation["pci"]
                if arm != "gpu4pyscf":
                    assert r["native_build"]["library_sha256"] == (
                        BASELINE if arm == "baseline" else CANDIDATE
                    )
                result = qualify([r], oracles, reference_reuse=reference_reuse)[0]
                assert result["status"] == "qualified", result
                summary["qualified_populations"] += 1
                summary["qualified_calls"] += len(r["samples"])
            samples = [s for r in rows for s in r["samples"]]
            card["arms"][arm] = {p: timing(samples, p) for p in ("cold", "warm")}
            card["arms"][arm]["work"] = [
                {
                    "phase": s["phase"],
                    "iterations": s["iterations"],
                    "focks": s.get("fock_builds", s.get("effective_potential_builds")),
                    "preliminary_focks": (s.get("initial_guess") or {}).get(
                        "preliminary_fock_builds"
                    ),
                }
                for s in rows[0]["samples"]
            ]
        reference = populations["baseline"][0]
        for rows in populations.values():
            for r in rows:
                assert all(r.get(f) == reference.get(f) for f in INPUTS)
                assert r["thread_controls"] == reference["thread_controls"]
        for baseline, candidate in zip(
            populations["baseline"], populations["candidate"], strict=True
        ):
            for a, b in zip(baseline["samples"], candidate["samples"], strict=True):
                for field in ("phase", "iterations", "fock_builds", "settings"):
                    assert a[field] == b[field]
                assert (
                    a["initial_guess"]["preliminary_fock_builds"]
                    == (b["initial_guess"]["preliminary_fock_builds"])
                )
        card["parity"] = {}
        for phase in ("cold", "warm"):
            selected = {
                arm: [s for r in rows for s in r["samples"] if s["phase"] == phase]
                for arm, rows in populations.items()
            }
            difference = errors(selected["baseline"], selected["candidate"])
            assert difference["energy_max_abs"] < 1e-9
            assert difference["force_max_abs"] < 1e-10
            card["parity"][phase] = difference
            assert (
                card["arms"]["candidate"][phase]["maximum"]
                < (card["arms"]["gpu4pyscf"][phase]["minimum"])
            )
        summary["cards"][cohort] = card
    traces = {}
    for path in manifest["trace_sources"]:
        r = json.loads(diagnostics[path + ".force-response"])
        assert r["operation"] == "force_response" and r["valid"]
        c = r["counters"]
        assert c["response_auxiliary_blocks"] == 9
        assert (
            c["response_probe_total_device_bytes"]
            <= c["response_declared_device_budget_bytes"]
        )
        if path.startswith("qualification"):
            assert c["screening_shell_enabled"] == 1
            assert c["screening_shell_primitive_products_skipped"] > 0
        if path.startswith("qualification-0"):
            assert c["screening_shell_tile_reduced"] == 1
        if path.startswith("trace-b"):
            assert c["screening_shell_capacity_fallback"] == 1
        traces[path] = {
            "host_completion_ms": r["host_completion_ms"],
            "counters": {
                k: v
                for k, v in c.items()
                if k.startswith("screening_shell")
                or k
                in (
                    "shell_primitive_products",
                    "response_auxiliary_blocks",
                    "response_available_device_headroom_bytes",
                    "response_probe_total_device_bytes",
                )
            },
        }
    summary["diagnostic_traces"] = traces
    summary["auxiliary_parity"] = {}
    for mode in ("hf", "larger"):
        a = endpoints[f"moved-regression/{mode}-baseline.json"]["samples"]
        b = endpoints[f"moved-regression/{mode}-candidate.json"]["samples"]
        values = []
        for x, y in zip(a, b, strict=True):
            assert (x["phase"], x["iterations"], x["fock_builds"]) == (
                y["phase"],
                y["iterations"],
                y["fock_builds"],
            )
            delta = errors([x], [y])
            assert delta["energy_max_abs"] < 1e-9 and delta["force_max_abs"] < 1e-10
            values.append({"phase": x["phase"], **delta})
        summary["auxiliary_parity"][mode] = values
    if args.verify:
        assert summary == read(HERE / "summary.json.xz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("Verified 18 populations / 72 independently qualified calls, two cards,")
    print("all-repeat parity, response work/budget, HF96 and moved PBE0-96 parity.")


if __name__ == "__main__":
    main()
