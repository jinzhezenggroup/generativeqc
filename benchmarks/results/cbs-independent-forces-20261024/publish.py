"""Build a numerical publication from retained references without changing gates."""

import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

from generativeqc_compiler.common.evidence import (
    block_error,
    canonical_hash,
    new_evidence,
    outcome,
)

from tools.generativeqc_validation.publication import validate_publication


def require_unoptimized_python() -> None:
    # Assertions are scientific gates; subprocesses also inherit this environment.
    if (
        not __debug__
        or sys.flags.optimize
        or os.environ.get("PYTHONOPTIMIZE", "") not in ("", "0")
    ):
        raise RuntimeError(
            "Scientific verification requires unoptimized Python; "
            "omit -O/-OO and unset PYTHONOPTIMIZE"
        )


require_unoptimized_python()

HERE = Path(__file__).parent.resolve()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compressed(path: Path, data: dict) -> None:
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(lzma.compress(raw, preset=9))


def main() -> None:
    require_unoptimized_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    comparison_path = args.raw / "publication-comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "replay.py"),
            "--raw",
            str(args.raw),
            "--output",
            str(comparison_path),
        ],
        check=True,
    )
    comparison = json.loads(comparison_path.read_bytes())
    spec = importlib.util.spec_from_file_location("force_replay", HERE / "replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execution = module.execution_records()
    environment = json.loads(execution["environment.json"])
    config = json.loads(execution["run-config.json"])
    freeze = json.loads(execution["source-freeze-final.json"])
    for name, entry in freeze["files"].items():
        relative = HERE.relative_to(Path.cwd()).as_posix() + "/" + name
        raw = subprocess.check_output(["git", "show", f"HEAD:{relative}"])
        assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], name
    inputs = json.loads(lzma.decompress((HERE / "inputs.json.xz").read_bytes()))
    status = comparison["status"]
    evidence = new_evidence(
        tier="cpu",
        subject="#2679 C4/C7 two-step independent directional forces",
        inputs_hash=canonical_hash(inputs),
    )
    evidence.update(
        revision=config["source_revision"],
        device={
            "kind": "CPU",
            "host": environment["host"],
            "resource": "CPU资源-2 4 CPU / 16 GiB",
            "cgroup": environment["cgroup"],
        },
        toolchain=environment,
        backend_selected="pyscf-cpu",
        hardware=outcome(
            "pass", "Received CPU Notebook; allocation and identity verified"
        ),
    )
    evidence["hashes"].update(
        equation=canonical_hash(
            {
                "hf_exponent": 1.63,
                "mp2_exponent": 3,
                "correction": "DZ DF-CCSD(T)-DF-MP2",
                "seed": 709,
                "steps_bohr": [1e-4, 3e-5],
            }
        ),
        source=config["driver_sha256"],
        schedule=canonical_hash(config),
    )
    evidence["hash_reasons"]["ir"] = "Standalone PySCF oracle has no GenerativeQC IR"
    evidence["settings"] = {
        "scope": "Frozen-native C4/C7 constituent/composite directional forces and translation only",
        "directional_gate_eh_per_bohr": 5e-6,
        "translation_gate_eh_per_bohr": 1e-6,
        "central_energy_gate_eh": 5e-8,
        "central_energy_proof": "Read-only delivered ENERGY publication at 334e7f489870dccc22ae8864c01f2353aa9689a4",
        "native_provenance": comparison["native_provenance"],
        "run_config": config,
        "no_claims": "No new native device endpoint, full-vector oracle, timing or true-CBS accuracy",
    }
    evidence["stages"]["representation"] = outcome(
        "pass",
        "Exact pinned QM7 centers, displaced geometries and basis identities verified",
    )
    evidence["stages"]["source"] = outcome(
        "pass" if comparison["terminal_postflight"]["status"] == "PASS" else "not-run",
        "Historical native source reconstructed; measured code/library identities require the terminal postflight proof",
    )
    for molecule in comparison["molecules"]:
        carbon = molecule["carbon"]
        for policy, values in molecule["policies"].items():
            translation = list(values["translation_eh_per_bohr"].values())
            evidence["block_errors"][f"C{carbon}/{policy}/translation"] = block_error(
                translation, [0] * len(translation), atol=1e-6, rtol=0
            )
            for step in values["steps"]:
                items = list(step["components"].values())
                evidence["block_errors"][f"C{carbon}/{policy}/h{step['step_bohr']}"] = (
                    block_error(
                        [x["native_force_projection"] for x in items],
                        [-x["energy_derivative"] for x in items],
                        atol=5e-6,
                        rtol=0,
                    )
                )
    evidence["stages"]["numerical"] = outcome(
        "pass" if status == "PASS" else "fail" if status == "FAIL" else "not-run",
        f"C4+C7 cohort {status}; both steps, both policies and all constituents required",
    )
    evidence["stages"]["endpoint"] = outcome(
        "not-run",
        "No new native endpoint; reference energies qualify frozen original forces",
    )
    evidence["performance"] = outcome(
        "not-run", "Oracle runtimes are provenance only, not speedup denominators"
    )
    evidence["compilation"] = {
        "seconds": None,
        "reason": "Prebuilt read-only PySCF libraries; no native build or JIT",
    }
    rss = [
        t["max_rss_kib"]
        for m in comparison["molecules"]
        for t in m["tasks"].values()
        if t.get("max_rss_kib") is not None
    ]
    evidence["memory"] = {
        "allocated_bytes": None,
        "peak_bytes": max(rss) * 1024 if rss else None,
        "reason": "Per-oracle process ru_maxrss, not whole-instance peak or measured allocation payload",
    }
    evidence["solver_trace_reason"] = (
        "Verbatim RHF/CCSD iteration logs retained in lossless raw records"
    )
    compressed(HERE / "comparison.json.xz", comparison)
    compressed(HERE / "evidence.json.xz", evidence)
    files = []
    for path in sorted(HERE.rglob("*")):
        relative = path.relative_to(HERE).as_posix()
        if not path.is_file() or relative in {
            "publication.json",
            ".gitignore",
            "comparison.json",
        }:
            continue
        if (
            relative.startswith("raw/")
            or "__pycache__" in relative
            or ".local." in relative
        ):
            continue
        role = (
            "evidence"
            if relative == "evidence.json.xz"
            else "input"
            if relative == "inputs.json.xz"
            else "samples"
            if relative == "receipts.json.xz" or relative.startswith("execution-")
            else "reproduction"
            if path.suffix in {".py", ".sh"} or relative.endswith(".sh.xz")
            else "summary"
        )
        raw = path.read_bytes()
        files.append(
            {
                "path": relative,
                "role": role,
                "bytes": len(raw),
                "sha256": sha(raw),
                "original_mode": "UNAVAILABLE",
                "git_mode": "100644",
            }
        )
    manifest = {
        "schema": "generativeqc.benchmark-publication.v1",
        "source": {"revision": config["source_revision"], "dirty": False},
        "decision": {
            "scope": "numerical",
            "status": "accepted"
            if status == "PASS"
            else "rejected"
            if status == "FAIL"
            else "inconclusive",
            "reason": f"Independent C4+C7 directional-force cohort {status}; parent #2679 remains OPEN for coordinating audit",
        },
        "files": files,
        "archives": [],
        "reproduction": {
            "command": [
                "python",
                HERE.relative_to(Path.cwd()).as_posix() + "/verify.py",
            ]
        },
    }
    validate_publication(
        manifest,
        {entry["path"]: (HERE / entry["path"]).read_bytes() for entry in files},
    )
    (HERE / "publication.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "status": status,
                "files": len(files),
                "stored_bytes": sum(e["bytes"] for e in files),
            }
        )
    )


if __name__ == "__main__":
    main()
