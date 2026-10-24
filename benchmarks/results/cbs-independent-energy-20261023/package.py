"""Pack immutable reference receipts and the repository validation envelope.

Run from the repository with PYTHONPATH=.:python (Windows: .;python). Packing
preserves raw JSON/log text and never fills in missing reference energies.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

from generativeqc_compiler.common.evidence import (
    block_error,
    canonical_hash,
    new_evidence,
    outcome,
)

BASE = "4a783a27e104e9392a7227b7efca79abcfba60ae"
MEASURED = (
    "oracle.py",
    "inputs.json",
    "run-oracle.sh",
    "run-suite.sh",
    "environment.py",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    for carbon in (1, 3, 4, 5, 6, 7):
        path = args.raw / f"C{carbon}.json"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        reference = json.loads(raw)
        if reference["status"] != "completed":
            continue
        bundle = {
            "receipt_json": raw.decode(),
            "receipt_sha256": digest(raw),
            "logs": {
                str(c): (args.raw / f"C{carbon}-{c}.log").read_bytes().decode()
                for c in (2, 3)
            },
            "stdout": (args.raw / f"C{carbon}.stdout.log").read_bytes().decode(),
            "outcome_json": (args.raw / f"C{carbon}.outcome.json")
            .read_bytes()
            .decode(),
        }
        (receipts / f"C{carbon}.json.xz").write_bytes(
            lzma.compress(json.dumps(bundle).encode())
        )
    subprocess.run(
        [
            sys.executable,
            str(root / "replay.py"),
            "--receipts",
            str(receipts),
            "--output",
            str(root / "comparison.json"),
        ],
        check=True,
    )
    comparison = json.loads((root / "comparison.json").read_bytes())
    environment = json.loads((root / "environment.json").read_bytes())
    inputs = json.loads((root / "inputs.json").read_bytes())
    prefix = root.relative_to(Path.cwd()).as_posix()
    patch = ""
    for name in MEASURED:
        # The retry guard was added after measurement. Preserve the wrapper
        # actually used, rather than relabeling old receipts with revised code.
        text = (
            lzma.decompress((root / "run-oracle-measured.sh.xz").read_bytes()).decode()
            if name == "run-oracle.sh"
            else (root / name).read_text(encoding="utf-8")
        )
        path = prefix + "/" + name
        patch += f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
        patch += "".join(
            difflib.unified_diff(
                [],
                text.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="b/" + path,
            )
        )
    (root / "oracle-source.patch.xz").write_bytes(lzma.compress(patch.encode()))
    evidence = new_evidence(
        tier="cpu",
        subject="#2679 six QM7 full-DF CBS independent energies only",
        inputs_hash=canonical_hash(inputs["protocols"]),
    )
    evidence.update(
        revision=BASE,
        device={
            "kind": "CPU",
            "host": environment["host"],
            "resource": "CPU资源-2 4 CPU / 16 GiB",
            "cgroup": environment["cgroup"],
        },
        toolchain=environment,
        backend_selected="pyscf-cpu",
        hardware=outcome("pass", "Own CPU Notebook; no GPU requested or used"),
    )
    evidence["hashes"].update(
        equation=canonical_hash(
            {
                "hf_exponent": 1.63,
                "mp2_exponent": 3,
                "correction": "DZ DF-CCSD(T)-DF-MP2",
                "all_electron": True,
            }
        ),
        source=digest((root / "oracle.py").read_bytes()),
        schedule=canonical_hash({"threads": 4, "max_memory_mb": 8000}),
    )
    evidence["hash_reasons"]["ir"] = "External PySCF oracle has no GenerativeQC IR"
    evidence["settings"] = {
        "gate_eh": 5e-8,
        "scope": "independent energy only",
        "frozen_native": comparison["native_provenance"],
        "source_snapshot": "Pinned repository base plus oracle-source.patch.xz restores measured code, including run-oracle-measured.sh.xz; the current wrapper adds a post-measurement retry guard. Exact oracle.py bytes are bound in every receipt",
        "reference_seconds": {
            str(r["carbon"]): r.get("seconds") for r in comparison["molecules"]
        },
    }
    evidence["stages"]["representation"] = outcome(
        "pass",
        "Pinned QM7 geometries and exact PySCF basis-record checksums match frozen native inputs",
    )
    evidence["stages"]["source"] = outcome(
        "pass",
        "Frozen native source reconstructed; independent driver and numerical-library hashes retained",
    )
    measured = [r for r in comparison["molecules"] if "errors_eh" in r]
    for row in measured:
        for policy, native in row["native"].items():
            keys = sorted(row["reference"])
            evidence["block_errors"][f"C{row['carbon']}/{policy}"] = block_error(
                [native[k] for k in keys],
                [row["reference"][k] for k in keys],
                atol=5e-8,
                rtol=0,
            )
    status = comparison["status"]
    evidence["stages"]["numerical"] = outcome(
        "pass" if status == "PASS" else "fail" if status == "FAIL" else "not-run",
        "All six independent constituent/composite energy gates pass"
        if status == "PASS"
        else f"Energy cohort {status}; inspect per-molecule records",
    )
    evidence["stages"]["endpoint"] = outcome(
        "not-run",
        "No new native endpoint or force measurement in this energy-only slice",
    )
    evidence["compilation"] = {
        "seconds": None,
        "reason": "Read-only prebuilt PySCF numerical libraries; no compiler/JIT invocation",
    }
    evidence["memory"] = {
        "allocated_bytes": None,
        "peak_bytes": max((r["max_rss_kib"] * 1024 for r in measured), default=None),
        "reason": "Peak is per-reference-process ru_maxrss, not planner capacity or total instance peak; allocation payload not measured",
    }
    evidence["solver_trace_reason"] = (
        "Raw SCF/CC iteration histories are retained verbatim in compressed receipt logs"
    )
    evidence["performance"] = outcome(
        "not-run",
        "Reference runtimes are not speedup denominators; no new native timing",
    )
    dump(root / "evidence.json", evidence)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or relative == "publication.json"
            or relative == ".gitignore"
            or relative.startswith("raw/")
            or ".local" in relative
            or "__pycache__" in relative
            or path.name == "qm7.mat"
        ):
            continue
        role = (
            "evidence"
            if relative == "evidence.json"
            else "source-patch"
            if relative == "oracle-source.patch.xz"
            else "input"
            if relative in {"inputs.json", "inputs-measured.json.xz"}
            else "samples"
            if relative.startswith(("receipts/", "failures/"))
            or relative == "runner-retry-check.json.xz"
            else "reproduction"
            if path.suffix in {".py", ".sh"}
            or relative in {"oracle-attempt1.py.xz", "run-oracle-measured.sh.xz"}
            else "summary"
        )
        raw = path.read_bytes()
        files.append(
            {"path": relative, "role": role, "bytes": len(raw), "sha256": digest(raw)}
        )
    manifest = {
        "schema": "generativeqc.benchmark-publication.v1",
        "source": {"revision": BASE, "dirty": True},
        "decision": {
            "scope": "numerical",
            "status": "accepted"
            if status == "PASS"
            else "rejected"
            if status == "FAIL"
            else "inconclusive",
            "reason": f"Six-molecule independent energy slice {status}; force qualification and parent #2679 remain open",
        },
        "files": files,
        "archives": [],
        "reproduction": {
            "command": [
                "python",
                prefix + "/replay.py",
                "--receipts",
                prefix + "/receipts",
                "--output",
                "comparison-replay.json",
                *(["--require-pass"] if status == "PASS" else []),
            ]
        },
    }
    dump(root / "publication.json", manifest)
    from tools.generativeqc_validation.publication import validate_publication

    validate_publication(
        manifest, {e["path"]: (root / e["path"]).read_bytes() for e in files}
    )
    print("Publication validated:", status, len(files), "files")


if __name__ == "__main__":
    main()
