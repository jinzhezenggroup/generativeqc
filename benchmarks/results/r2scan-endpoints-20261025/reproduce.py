"""Replay the complete #2786 default-only figures and numerical qualification.

Run from the repository with PYTHONPATH=python:. and an ignored output directory.
Historical timings are used only in the explicit before/after comparison; every
point in either figure comes from the new, single-revision population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import METHODS, SIZES
from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_report import LABELS, load_records, reduce_record

ROOT = Path(__file__).resolve().parent
ARMS = ("standard", "gpu4pyscf")
LABEL = {"standard": "GenerativeQC", "gpu4pyscf": "GPU4PySCF"}
INPUT_FIELDS = (
    "geometry_bohr",
    "orbital_basis",
    "basis_representation",
    "charge",
    "spin",
    "auxiliary_input_sha256",
)


def read(path: Path) -> dict | list:
    """Decode retained lossless JSON without rewriting producer identities."""
    data = path.read_bytes()
    return json.loads(lzma.decompress(data) if path.suffix == ".xz" else data)


def key(record: dict) -> tuple:
    """Identify a whole cold/three-warm population, never an individual sample."""
    return tuple(record[k] for k in ("arm", "method", "mode", "atoms"))


def work(sample: dict, arm: str) -> str:
    """Keep preliminary guess work and incremental work distinct from totals."""
    focks = sample.get("fock_builds", sample.get("effective_potential_builds"))
    value = f"{sample['iterations']}/{focks}"
    if arm == "standard":
        guess = sample.get("initial_guess") or {}
        value += f"; guess {guess.get('preliminary_fock_builds', '?')}"
        incremental = sample.get("incremental_direct_jk") or {}
        if incremental.get("active"):
            fields = ("anchor_full_builds", "delta_builds", "post_scf_full_builds")
            value += "; A/D/C " + "/".join(str(incremental[k]) for k in fields)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also compare retained tables and figures byte for byte",
    )
    args = parser.parse_args()
    paths = [ROOT / f"{arm}.json.xz" for arm in ARMS]
    records = load_records(paths)
    expected = {
        (arm, method, mode, atoms)
        for arm in ARMS
        for method in METHODS
        for mode in ("direct", "df")
        for atoms in SIZES
    }
    assert len(records) == 128 and {key(r) for r in records} == expected
    assert all(r["status"] == "complete" and len(r["samples"]) == 4 for r in records)
    assert {r["source_revision"] for r in records} == {
        "068d24de3956f02626bd0c46b595a16d3f6f23ce"
    }
    provenance = read(ROOT / "provenance.json.xz")
    assert {
        r["native_build"]["library_sha256"] for r in records if r["arm"] == "standard"
    } == {provenance["libraries"]["libgenerativeqc.so"]}
    assert len({r["runner_sha256"] for r in records}) == 1
    previous = ROOT.parent / "df-dft48-work-20261025"
    old = {key(r): r for r in load_records([previous / f"{a}.json.xz" for a in ARMS])}
    for record in records:
        for field in INPUT_FIELDS:
            assert record.get(field) == old[key(record)].get(field), (
                key(record),
                field,
            )
    # Keep the audited oracle receipts and original producer/driver identities.
    for name in ("oracles.json.xz", "reference-reuse.json"):
        assert (ROOT / name).read_bytes() == (previous / name).read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.readme_defaults_report",
            "--receipts",
            *map(str, paths),
            "--oracles",
            str(ROOT / "oracles.json.xz"),
            "--reference-reuse",
            str(ROOT / "reference-reuse.json"),
            "--default-labels",
            "--include-unqualified",
            "--output",
            str(args.output),
        ],
        check=True,
    )
    rows = {key(r): reduce_record(r) for r in records}
    wins = []
    for mode in ("direct", "df"):
        for phase in ("cold", "warm"):
            pairs = [
                (rows[("standard", m, mode, n)], rows[("gpu4pyscf", m, mode, n)])
                for m in METHODS
                for n in SIZES
            ]
            wins.append(
                {
                    "mode": mode,
                    "phase": phase,
                    "pairs": 32,
                    "wins": sum(
                        a[phase]["median"] < b[phase]["median"] for a, b in pairs
                    ),
                }
            )
    changes = []
    for method, mode in (
        ("r2scan-3c", "direct"),
        ("r2scan-3c", "df"),
        ("r2scan", "df"),
    ):
        k = ("standard", method, mode, 96)
        changes.append(
            {
                "method": method,
                "mode": mode,
                "atoms": 96,
                "previous": reduce_record(old[k]),
                "current": rows[k],
                "gpu4pyscf": rows[("gpu4pyscf", method, mode, 96)],
            }
        )
    qualifications = read(args.output / "qualification.json")
    counts = {
        arm: {
            state: sum(q["arm"] == arm and q["status"] == state for q in qualifications)
            for state in ("qualified", "failed", "unverified")
        }
        for arm in ARMS
    }
    # These additional moved-owner checks are separate from figure timings.
    endpoints = read(ROOT / "endpoint-validation.json.xz")
    references = {
        (r["method"], r["mode"], r["atoms"], r["geometry"]): r["sample"]
        for r in endpoints
        if r["kind"] == "reference"
    }
    native = [r for r in endpoints if r["kind"] == "native"]
    assert len(native) == 24
    for r in native:
        reference = references[(r["method"], r["mode"], r["atoms"], r["geometry"])]
        actual = errors(r["energy"], r["forces"], reference, r["atoms"])
        assert actual == r["errors"] and actual["passed"]
    result = {
        "populations": len(records),
        "calls": 512,
        "wins": wins,
        "targets": changes,
        "qualification": counts,
        "endpoint_checks": len(native),
    }
    (args.output / "comparisons.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Executed SCF work",
        "",
        "Each entry is target iterations / Fock builds. GPU4PySCF reports effective-potential calls, a different operation count. Native preliminary Hückel Fock work is shown separately as `guess`. Active incremental solves also show full anchors / delta builds / post-SCF closure builds as A/D/C; periodic rebuilds are a subset of anchors, not extra work. All calls include one complete force evaluation. Full settings, residual histories and work metadata remain in the receipts.",
        "",
        "| Mode | Method | Atoms | Engine | Cold | Warm calls |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for r in sorted(
        records,
        key=lambda r: (
            r["mode"],
            list(METHODS).index(r["method"]),
            r["atoms"],
            r["arm"],
        ),
    ):
        phases = [
            "; ".join(work(s, r["arm"]) for s in r["samples"] if s["phase"] == p)
            for p in ("cold", "warm")
        ]
        lines.append(
            f"| {r['mode']} | {LABELS[r['method']]} | {r['atoms']} | {LABEL[r['arm']]} | {' | '.join(phases)} |"
        )
    (args.output / "work.md").write_text("\n".join(lines) + "\n")
    if args.verify:
        from tools.generativeqc_validation.publication import validate_publication

        manifest = read(ROOT / "publication.json")
        validate_publication(
            manifest,
            {
                entry["path"]: (ROOT / entry["path"]).read_bytes()
                for entry in manifest["files"]
            },
        )
        for name in ("qualification", "comparisons"):
            assert read(args.output / f"{name}.json") == read(ROOT / f"{name}.json.xz")
        for name in ("timings.md", "work.md"):
            assert (args.output / name).read_bytes() == (ROOT / name).read_bytes()
        assets = ROOT.parents[2] / "assets"
        for mode in ("direct", "df"):
            assert (args.output / f"{mode}.svg").read_bytes() == (
                assets / f"readme-{mode}-defaults.svg"
            ).read_bytes()
        assert hashlib.sha256(
            (assets / "readme-cbs-performance.svg").read_bytes()
        ).hexdigest() == (
            "5224236e07af9309cfe85bd77a25d797615a938dfc6517c13aef1a50e5478884"
        )
    print(json.dumps({k: v for k, v in result.items() if k != "targets"}, indent=2))


if __name__ == "__main__":
    main()
