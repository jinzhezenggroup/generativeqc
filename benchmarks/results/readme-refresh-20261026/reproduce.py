"""Replay the final default-only README cohort and independent static gates.

All figure samples use one frozen source and library. The earlier cohort is
consulted only to audit unchanged scientific inputs, never to fill timing gaps.
Verification binds the historical figures by hash; --check-readme additionally
checks that the current homepage still uses this publication.
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
from benchmarks.readme_defaults_report import LABELS, load_records, reduce_record

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "readme-refresh-20261025"
ARMS = ("standard", "gpu4pyscf")
INPUT_FIELDS = (
    "geometry_bohr",
    "orbital_basis",
    "basis_representation",
    "charge",
    "spin",
    "auxiliary_input_sha256",
)


def read(path: Path) -> dict | list:
    """Read lossless receipts while preserving their original producer identity."""
    data = path.read_bytes()
    return json.loads(lzma.decompress(data) if path.suffix == ".xz" else data)


def key(record: dict) -> tuple:
    """Identify a complete cold/three-warm population."""
    return tuple(record[name] for name in ("arm", "method", "mode", "atoms"))


def work(sample: dict, arm: str) -> str:
    """Keep guess and incremental work separate from target Fock totals."""
    focks = sample.get("fock_builds", sample.get("effective_potential_builds"))
    result = f"{sample['iterations']}/{focks}"
    if arm == "standard":
        guess = sample.get("initial_guess") or {}
        result += f"; guess {guess.get('preliminary_fock_builds', '?')}"
        incremental = sample.get("incremental_direct_jk") or {}
        if incremental.get("active"):
            fields = ("anchor_full_builds", "delta_builds", "post_scf_full_builds")
            result += "; A/D/C " + "/".join(str(incremental[k]) for k in fields)
    return result


def main() -> None:
    """Recompute every timing, independent gate, work table and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-readme", action="store_true")
    args = parser.parse_args()
    if args.check_readme and not args.verify:
        parser.error("--check-readme requires --verify")
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
    provenance = read(ROOT / "provenance.json.xz")
    if args.verify:
        import matplotlib

        if matplotlib.__version__ != provenance["render"]["matplotlib"]:
            raise ValueError(
                "Install render-requirements.txt for byte-identical replay"
            )
    # This campaign uses committed source and the previously qualified binary.
    # Its embedded source identity inherits a parent metadata object; authenticate
    # the actual library and incremental-build evidence rather than treating that
    # embedded string as the identity of the final source commit.
    assert provenance["source_archive"]["dirty"] is False
    for evidence in provenance["qualification_evidence"]:
        assert (
            hashlib.sha256((ROOT / evidence["path"]).read_bytes()).hexdigest()
            == (evidence["sha256"])
        )
    assert {r["source_revision"] for r in records} == {provenance["source_revision"]}
    assert {r["runner_sha256"] for r in records} == {provenance["runner_sha256"]}
    native = [r for r in records if r["arm"] == "standard"]
    assert {r["native_build"]["library_sha256"] for r in native} == {
        provenance["libraries"]["libgenerativeqc.so"]
    }
    assert {r["native_build"]["source_identity"] for r in native} == {
        provenance["source_identity"]
    }
    previous = {
        key(r): r for r in load_records([PREVIOUS / f"{a}.json.xz" for a in ARMS])
    }
    indexed = {key(r): r for r in records}
    jobs = read(ROOT / "jobs.json.xz")
    allocations = {r["SLURM_JOB_ID"]: r for r in jobs["allocations"]}
    assert len(jobs["commands"]) == 128
    assert all(r["returncode"] == 0 for r in jobs["commands"])
    assert all(r["returncode"] == 0 for r in jobs["worker_exits"])
    for record in records:
        for field in INPUT_FIELDS:
            assert record.get(field) == previous[key(record)].get(field), (
                key(record),
                field,
            )
        paired = indexed[
            ("gpu4pyscf", record["method"], record["mode"], record["atoms"])
        ]
        assert record["slurm_job_id"] == paired["slurm_job_id"]
        allocation = allocations[record["slurm_job_id"]]
        assert record["accelerator"]["pci_bus_id"] == allocation["pci"]
        assert (
            record["accelerator"]["nvidia_smi"]["uuid"]
            == (paired["accelerator"]["nvidia_smi"]["uuid"])
        )
        assert (
            record["environment"]["runtime"]["cuda_visible_devices"]
            == (allocation["CUDA_VISIBLE_DEVICES"])
        )
        assert record["accelerator"]["nvidia_smi"]["sampling_point"] == (
            "before benchmark measurements"
        )
    for name in ("oracles.json.xz", "reference-reuse.json", "auxiliary-inputs.json.xz"):
        assert (ROOT / name).read_bytes() == (PREVIOUS / name).read_bytes()
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
    wins, slower = [], []
    for mode in ("direct", "df"):
        for phase in ("cold", "warm"):
            count = 0
            for method in METHODS:
                for atoms in SIZES:
                    a, b = (
                        rows[(arm, method, mode, atoms)][phase]["median"]
                        for arm in ARMS
                    )
                    count += a < b
                    if a >= b:
                        slower.append(
                            {
                                "mode": mode,
                                "method": method,
                                "atoms": atoms,
                                "phase": phase,
                                "native_seconds": a,
                                "gpu4pyscf_seconds": b,
                                "native_over_gpu4pyscf": a / b,
                            }
                        )
            wins.append({"mode": mode, "phase": phase, "pairs": 32, "wins": count})
    qualifications = read(args.output / "qualification.json")
    result = {
        "populations": len(records),
        "calls": sum(len(r["samples"]) for r in records),
        "wins": wins,
        "slower": slower,
        "qualification": {
            arm: {
                state: sum(
                    q["arm"] == arm and q["status"] == state for q in qualifications
                )
                for state in ("qualified", "failed", "unverified")
            }
            for arm in ARMS
        },
        "wb97mv_direct96_warm_speedup": (
            rows[("gpu4pyscf", "wb97m-v", "direct", 96)]["warm"]["median"]
            / rows[("standard", "wb97m-v", "direct", 96)]["warm"]["median"]
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Executed SCF work",
        "",
        "Entries are target iterations / Fock builds; GPU4PySCF counts effective-potential calls. Native preliminary Hückel Focks are separate (`guess`). A/D/C denotes full anchors / delta builds / post-SCF closure; periodic rebuilds are a subset of anchors. Every sample includes complete analytic forces. Full settings, residual histories and work metadata remain in the receipts.",
        "",
        "| Mode | Method | Atoms | Engine | Cold | Warm calls |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for record in sorted(
        records,
        key=lambda r: (
            r["mode"],
            list(METHODS).index(r["method"]),
            r["atoms"],
            r["arm"],
        ),
    ):
        phases = [
            "; ".join(
                work(s, record["arm"]) for s in record["samples"] if s["phase"] == phase
            )
            for phase in ("cold", "warm")
        ]
        label = "GenerativeQC" if record["arm"] == "standard" else "GPU4PySCF"
        lines.append(
            f"| {record['mode']} | {LABELS[record['method']]} | {record['atoms']} | {label} | {' | '.join(phases)} |"
        )
    (args.output / "work.md").write_text("\n".join(lines) + "\n")
    if args.verify:
        from tools.generativeqc_validation.publication import validate_publication

        manifest = read(ROOT / "publication.json")
        validate_publication(
            manifest,
            {e["path"]: (ROOT / e["path"]).read_bytes() for e in manifest["files"]},
        )
        for name in ("qualification", "summary"):
            assert read(args.output / f"{name}.json") == read(ROOT / f"{name}.json.xz")
        for name in ("timings.md", "work.md"):
            assert (args.output / name).read_bytes() == (ROOT / name).read_bytes()
        figures = read(ROOT / "figures.json")
        for mode in ("direct", "df"):
            assert (
                hashlib.sha256((args.output / f"{mode}.svg").read_bytes()).hexdigest()
                == figures[mode]
            )
        if args.check_readme:
            assets = ROOT.parents[2] / "assets"
            for mode in ("direct", "df"):
                assert (args.output / f"{mode}.svg").read_bytes() == (
                    assets / f"readme-{mode}-defaults.svg"
                ).read_bytes()
            assert (
                hashlib.sha256(
                    (assets / "readme-cbs-performance.svg").read_bytes()
                ).hexdigest()
                == figures["cbs_unchanged"]
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
