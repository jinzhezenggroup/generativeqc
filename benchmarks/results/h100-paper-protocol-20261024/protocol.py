"""Offline H100 protocol builder and fail-closed attempt reader; never launches work."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARCHIVE = "7522994e2e2e477e11c41cb2c2dbe74a2ce600e9"
HF = "cbd4d8074de35105717c8aac7cd1f86babbe90ec"
CBS = "976f756b391bc65edb79f76df1a8c3fe807518a7"
ENERGY = "334e7f489870dccc22ae8864c01f2353aa9689a4"
DIRECTION = "594343670cfdea574e5b1c6821b51165ce004f97"
PREFIX = "benchmarks/results/"
HF_DIR = PREFIX + "readme-repaired-20261021/"
CBS_DIR = PREFIX + "cbs-fast-memory-20261021/"
PREPARED = "df5c44a81c5d73210a8d2f7f09ba4930d7f67210"
PROFILE_PATH = "python/generativeqc_compiler/integral/production_profile.py"


def execution_contract(family: str, arm: str) -> dict:
    """Separate unchanged scientific identity from target-specific generation."""
    if arm == "gpu4pyscf":
        return {
            "cohort": "gpu4pyscf-h100-environment-port.v1",
            "target": "sm90",
            "native_profile": "not applicable",
            "hardware_only": False,
        }
    return {
        "cohort": "cbs-frozen-science-portable-cuda-sm90.v1"
        if family == "cbs"
        else "hf-dft-frozen-science-sm90.v1",
        "target": "sm90",
        "native_profile": "portable_cuda" if family == "cbs" else None,
        "profile_status": "EXPLICIT_ROOT_DECISION"
        if family == "cbs"
        else "UNAVAILABLE_UNTIL_PREFLIGHT",
        "profile_match": "portable" if family == "cbs" else None,
        "tuned": False if family == "cbs" else None,
        "hardware_only": False,
        "identical_tuned_schedule": False,
        "codegen": "target/profile-specific artifacts and new DSO hashes required",
        "science_changes": "forbidden; report to root for separately approved paired-version cohort",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def blob(revision: str, path: str) -> bytes:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "fixed Git OID required")
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def decoded(revision: str, path: str) -> object:
    raw = blob(revision, path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def binding(revision: str, path: str, selector: object = None) -> dict:
    raw = blob(revision, path)
    return {
        "revision": revision,
        "path": path,
        "sha256": digest(raw),
        "selector": selector,
    }


def build_manifest() -> dict:
    """Derive scientific contracts from immutable original receipts, not current code."""
    summary = decoded(ARCHIVE, CBS_DIR + "summary.json.xz")
    provenance = decoded(ARCHIVE, HF_DIR + "provenance.json.xz")
    records = decoded(ARCHIVE, HF_DIR + "standard.json.xz")
    oracles = decoded(ARCHIVE, HF_DIR + "oracles.json.xz")
    sources = {
        "hf_dft": {
            "revision": HF,
            "comparability": "authenticated Git base+patch; original complete measured-source manifest UNAVAILABLE; no full hardware-only claim",
            "reconstruction_base": provenance["source_binding"][
                "compiled_candidate_base"
            ],
            "patch_sha256": provenance["source_binding"][
                "compiled_candidate_patch_sha256"
            ],
            "patch": binding(
                ARCHIVE, HF_DIR + "provenance.json.xz", "native_source_patch"
            ),
            "old_sm120_library_sha256": provenance["source_binding"][
                "native_library_sha256"
            ],
            "driver": binding(HF, "benchmarks/readme_defaults.py"),
        },
        "cbs": {
            "revision": CBS,
            "comparability": "qualified Git base+patch reconstructed; H100 build/endpoint qualification pending",
            "patch_sha256": summary["qualification"]["source"]["candidate"][
                "patch_sha256"
            ],
            "patch": binding(ARCHIVE, CBS_DIR + "qualified-source.patch.xz"),
            "old_sm120_library_sha256": summary["qualification"]["source"]["candidate"][
                "library_sha256"
            ],
            "driver": binding(CBS, "benchmarks/readme_cbs.py"),
        },
    }
    references = {
        "hf_dft": binding(ARCHIVE, HF_DIR + "oracles.json.xz"),
        "alkane_energy": binding(
            ENERGY, PREFIX + "cbs-independent-energy-20261023/inputs.json"
        ),
        "c4_c7_direction": binding(
            DIRECTION, PREFIX + "cbs-independent-forces-20261024/inputs.json.xz"
        ),
        "c2_direction": binding(
            ARCHIVE, CBS_DIR + "summary.json.xz", "points[carbon=2].reference"
        ),
        "water_algebra": binding(
            ARCHIVE, CBS_DIR + "summary.json.xz", "qualification.water"
        ),
    }
    references["hf_auxiliary_tables"] = binding(
        ARCHIVE, HF_DIR + "auxiliary-inputs.json.xz"
    )
    for carbon in (4, 7):
        references[f"c{carbon}_energy_raw"] = binding(
            ENERGY,
            PREFIX + f"cbs-independent-energy-20261023/receipts/C{carbon}.json.xz",
        )
    references["c4_c7_direction_raw"] = binding(
        DIRECTION, PREFIX + "cbs-independent-forces-20261024/receipts.json.xz"
    )
    auxiliary = decoded(ARCHIVE, HF_DIR + "auxiliary-inputs.json.xz")
    cells = []
    for carbon in (2, 4, 7):
        record = next(p["native"] for p in summary["points"] if p["carbon"] == carbon)
        scientific = dict(record["protocol"])
        scientific.pop("precision")  # requested arm is separate from scientific input
        scientific.update(
            basis_representation="spherical", hf_exponent=1.63, mp2_exponent=3.0
        )
        refs = ["c2_direction"] if carbon == 2 else ["alkane_energy", "c4_c7_direction"]
        cells.append(
            {
                "id": f"cbs-c{carbon}",
                "family": "cbs",
                "source": "cbs",
                "input": scientific,
                "input_sha256": digest(canonical(scientific)),
                "arms": ["standard", "fast"],
                "references": refs,
                "gates": {
                    "independent_energy_eh": 5e-8,
                    "direction_force_eh_bohr": 5e-6,
                    "translation_eh_bohr": 1e-6,
                    "direction_seed": 709 if carbon != 2 else None,
                    "direction_steps_bohr": [1e-4, 3e-5],
                    "fast_standard": {
                        "energy_eh": 1e-5,
                        "force_max_eh_bohr": 1e-4,
                        "force_rms_eh_bohr": 3e-5,
                    },
                    "constituent_allowance": "composite allowance / expanded coefficient L1 norm",
                },
                "reference_scope": "constituent/composite energy + two-step direction + translation; no independent full vector; C2 displaced originals UNAVAILABLE (#2705)",
                "direction": summary["points"][1]["reference"]["direction"]
                if carbon == 2
                else "normalized NumPy default_rng(709) normal array with atoms x 3 shape",
                "calls_per_process": ["cold"],
                "attempts": [],
                "status": {
                    "preflight": "INCOMPLETE",
                    "science": "UNAVAILABLE",
                    "timing": "UNAVAILABLE",
                },
            }
        )
    for atoms in (48, 96):
        for method, mode in (
            ("pbe0", "direct"),
            ("pbe0", "df"),
            ("wb97m-v", "direct"),
            ("r2scan-3c", "df"),
        ):
            record = next(
                r
                for r in records
                if (r["atoms"], r["method"], r["mode"]) == (atoms, method, mode)
            )
            matches = [
                i
                for i, o in enumerate(oracles)
                if (o["atoms"], o["method"]) == (atoms, method)
            ]
            scientific = {
                k: record[k]
                for k in (
                    "atoms",
                    "method",
                    "mode",
                    "geometry_bohr",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                )
            }
            scientific["auxiliary_input_sha256"] = record.get("auxiliary_input_sha256")
            scientific["reference_auxiliary_basis_sha256"] = auxiliary[
                method + "-reference.json"
            ]["basis_sha256"]
            scientific["protocol"] = record["protocol"]
            cells.append(
                {
                    "id": f"water{atoms}-{method}-{mode}",
                    "family": "hf_dft",
                    "source": "hf_dft",
                    "input": scientific,
                    "input_sha256": digest(canonical(scientific)),
                    "arms": ["standard", "fast", "gpu4pyscf"],
                    "references": ["hf_dft"],
                    "oracle_indices": matches,
                    "gates": {
                        "energy_eh_per_atom": 1e-5,
                        "force_max_eh_bohr": 1e-4,
                        "force_rms_eh_bohr": 3e-5,
                        "grid_cpu_gpu_margin_fraction": 0.25,
                    },
                    "reference_scope": "independent direct target includes DF approximation; grid/CPU-GPU quarter margins; full moving-grid response; r2SCAN-3c includes independent D4+gCP",
                    "calls_per_process": ["cold", "warm1", "warm2", "warm3"],
                    "warm_seed": "own immutable cold density; same owner; never preceding warm density",
                    "attempts": [],
                    "status": {
                        "preflight": "INCOMPLETE",
                        "science": "UNAVAILABLE",
                        "timing": "UNAVAILABLE",
                    },
                }
            )
    schedule = []
    for cell in cells:
        arms = cell["arms"]
        for repeat in range(3):
            for arm in arms[repeat % len(arms) :] + arms[: repeat % len(arms)]:
                schedule.append({"cell": cell["id"], "arm": arm, "repeat": repeat})
    return {
        "schema": "generativeqc.h100-paper-protocol.v2",
        "prepared_v1": binding(
            PREPARED, PREFIX + "h100-paper-protocol-20261024/manifest.json"
        ),
        "profile_decision": {
            "authority": "root coordinator 01a10e8f-7b0a-7b52-a330-b2de34309210; explicit PR #2742 handoff 2026-10-24",
            "scope": "CBS task-owned build configuration only: portable_cuda/sm90; no source/parser/default/runtime edits",
            "reason": "frozen auto/sm90 rejects without tuned/compatible profile; explicit portable_cuda uses existing generic fallback",
            "resolver": binding(CBS, PROFILE_PATH),
            "preflight": "retain auto failure and explicit portable resolution receipt, source/patch, actual build config, generated inventory and new DSO hashes",
            "acceptance": "CPU build permission is separate from SDK, device/science acceptance and GPU launch authorization",
        },
        "execution_contracts": {
            family: {
                arm: execution_contract(family, arm)
                for arm in (
                    ("standard", "fast")
                    if family == "cbs"
                    else ("standard", "fast", "gpu4pyscf")
                )
            }
            for family in ("cbs", "hf_dft")
        },
        "issue": 2725,
        "archive_revision": ARCHIVE,
        "sources": sources,
        "references": references,
        "cells": cells,
        "schedule": schedule,
        "execution": {
            "authorized_here": False,
            "target": "sm90",
            "gpu_count": 1,
            "max_attempts_per_slot": 2,
            "automatic_retries": 0,
            "process_cap_seconds": 3600,
            "termination_grace_seconds": 30,
            "job_cap_seconds": 14400,
            "job_process_limit": 3,
            "pilot": ["cbs-c2", "water48-pbe0-direct"],
            "advance": "source/auth + environment + small-fixture science PASS, then one cold pilot per arm; recompute worksheet before repeats/larger cells",
            "stop": "stop affected cohort on source/input mismatch, nonfinite, science FAIL, timeout, OOM, work anomaly or cap/budget refusal; retain attempts; no automatic retry",
        },
        "optional": {
            "lambda": "only after core pilot identifies Lambda bottleneck; separately authenticated source family and explicit resource authorization",
            "vv10": "only if core WB97M-V indicates a reuse question; #2706 missing nine old originals remain UNAVAILABLE",
            "paired_version": "if sm90 requires science/runtime edits, report blocker to root and approve separate both-device paired cohort before changes",
            "full_population": "excluded; never silently expand to 192 populations",
        },
        "host_confounders": [
            "CPU model/affinity/cgroup and numerical threads",
            "host RAM vs numeric budget vs measured peak VRAM",
            "GPU UUID/driver/CUDA/power/clocks/temperature",
            "Python/native/AOT/compiler-cache identities and cache state",
            "original GPU4PySCF Python3.13.9/NumPy2.4.6 vs prepared qz Python3.11.16/NumPy2.2.6; record per-engine versions",
            "CBS sm120 historical tuned schedule vs explicit portable_cuda/sm90 target/profile/codegen delta",
            "clean E+F vs instrumented diagnostic timing",
            "precision/refinement/fallback and iteration/Fock/action counts",
        ],
    }


def validate_manifest(manifest: dict) -> None:
    """Authenticate v2 while retaining the immutable prepared v1 binding."""
    require(
        canonical(manifest) == canonical(build_manifest()),
        "manifest differs from frozen v2 scientific/execution contract",
    )


def authenticate_sources(manifest: dict) -> dict:
    """Reconstruct changed LF inputs temporarily; compare actual bytes, never compile."""
    report = {}
    for family, source in manifest["sources"].items():
        patch_ref = source["patch"]
        raw = blob(patch_ref["revision"], patch_ref["path"])
        require(digest(raw) == patch_ref["sha256"], "patch container hash mismatch")
        if family == "hf_dft":
            patch = json.loads(lzma.decompress(raw))["native_source_patch"].encode()
            base = source["reconstruction_base"]
        else:
            patch = lzma.decompress(raw)
            base = source["revision"]
        require(digest(patch) == source["patch_sha256"], "source patch hash mismatch")
        changed = [
            n.decode()
            for n in re.findall(rb"^diff --git a/(\S+) b/\S+$", patch, re.MULTILINE)
        ]
        require(bool(changed), "empty source patch")
        identities = {}
        if family == "cbs":
            identities = decoded(ARCHIVE, CBS_DIR + "summary.json.xz")["qualification"][
                "source"
            ]["candidate"]["file_sha256"]
        with tempfile.TemporaryDirectory(prefix="h100-protocol-source-") as directory:
            scratch = Path(directory)
            for name in set(changed) | set(identities):
                require(
                    not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "unsafe source path",
                )
                target = scratch / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob(base, name))
            for args in (["--check", "-"], ["-"]):
                subprocess.run(
                    ["git", "-c", "core.autocrlf=false", "apply", *args],
                    input=patch,
                    cwd=scratch,
                    check=True,
                    capture_output=True,
                )
            observed = {
                name: digest((scratch / name).read_bytes())
                for name in set(changed) | set(identities)
            }
            for name, expected in identities.items():
                require(
                    observed[name] == expected, f"reconstructed source mismatch: {name}"
                )
            if family == "hf_dft":
                for name in changed:
                    require(
                        observed[name] == digest(blob(HF, name)),
                        f"measured commit mismatch: {name}",
                    )
                delta = subprocess.check_output(
                    ["git", "diff", "--name-only", base, HF], cwd=ROOT, text=True
                ).splitlines()
                unexplained = [
                    n
                    for n in delta
                    if n not in changed and not n.startswith(".agents/notes/")
                ]
                require(not unexplained, "HF measured commit has unexplained changes")
        report[family] = {
            "status": "PASS",
            "base": base,
            "patch_sha256": digest(patch),
            "reconstructed_files": observed,
            "h100_build": "UNAVAILABLE",
            "h100_science": "UNAVAILABLE",
            "historical_full_source_manifest": "UNAVAILABLE"
            if family == "hf_dft"
            else "not applicable",
        }
    return report


def reference_inventory(manifest: dict) -> dict:
    """Authenticate original inputs and availability without reconverging any oracle."""
    for ref in manifest["references"].values():
        require(
            digest(blob(ref["revision"], ref["path"])) == ref["sha256"],
            "reference raw hash",
        )
    oracles = decoded(ARCHIVE, HF_DIR + "oracles.json.xz")
    require(len(oracles) == 40, "historical oracle inventory")
    for cell in manifest["cells"]:
        if cell["family"] == "hf_dft":
            selected = [oracles[i] for i in cell["oracle_indices"]]
            require(bool(selected), "missing selected oracle")
            for oracle in selected:
                require(
                    oracle["geometry_bohr"] == cell["input"]["geometry_bohr"],
                    "oracle geometry mismatch",
                )
                require(
                    oracle["method"] == cell["input"]["method"],
                    "oracle method mismatch",
                )
                for sample in oracle["samples"]:
                    require(
                        finite_number(sample["energy"])
                        and all(finite_number(x) for f in sample["forces"] for x in f),
                        "nonfinite oracle",
                    )
    summary = decoded(ARCHIVE, CBS_DIR + "summary.json.xz")
    water = summary["qualification"]["water"]
    driver = water["driver_source"]
    return {
        "raw_bindings": "PASS",
        "hf_dft_oracle_receipts": len(oracles),
        "water_driver_sha256": digest(driver.encode()),
        "water_groups": [
            {
                "reference": c["reference"],
                "geometry_id": c["geometry_id"],
                "native_samples": len(c["samples"]),
            }
            for c in water["cases"]
        ],
        "water_original_standard_gates": {
            "energy_abs": 1e-8,
            "force_max_abs": 3e-6,
            "force_rms": 3e-6,
        },
        "water_fast_composite_gates": {
            "energy_abs": 1e-5,
            "force_max_abs": 1e-4,
            "force_rms": 3e-5,
        },
        "h2_separate_gates": {"energy_abs": 1e-9, "force_max_abs": 2e-6},
        "h2_scope": "#2719 conventional/exact only; DF/RIJK/CUDA UNAVAILABLE; H2 cannot establish nonzero triples",
        "c2_displaced_originals": "UNAVAILABLE (#2705); stored projections are not displaced energies",
        "vv10_nine_old_originals": "UNAVAILABLE (#2706); only respective historical recalculation blocked",
        "direction_delivery": "#2721 merged at pinned OID; formal APPROVED absent at external merge, not transferred as approval",
        "science_replay": "not performed by inventory; use original offline verifiers; no H100 qualification",
    }


def finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else True)
    )


def attempt_template(manifest: dict, slot: dict, attempt_id: str) -> dict:
    cell = next(c for c in manifest["cells"] if c["id"] == slot["cell"])
    contract = manifest["execution_contracts"][cell["family"]][slot["arm"]]
    return {
        "schema": "generativeqc.h100-paper-attempt.v2",
        **slot,
        "attempt_id": attempt_id,
        "input_sha256": cell["input_sha256"],
        "source": manifest["sources"][cell["source"]],
        "cohort": contract["cohort"],
        "execution_contract_sha256": digest(canonical(contract)),
        "profile_decision_sha256": digest(canonical(manifest["profile_decision"])),
        "outcome": "INCOMPLETE",
        "exit_code": None,
        "preflight": "INCOMPLETE",
        "science": "UNAVAILABLE",
        "timing": "UNAVAILABLE",
        "runtime": None,
        "actual_policy": None,
        "work": None,
        "samples": [],
        "raw_artifacts": [],
        "failure_reason": "No observation registered",
    }


def validate_build_provenance(
    manifest: dict, cell: dict, arm: str, runtime: dict
) -> None:
    """Bind a preflight receipt to this source, profile, environment and binary.

    Hash consistency admits a receipt contract; independent raw replay still
    has to authenticate the referenced build/resolve observations.
    """
    preflight = runtime.get("build_preflight")
    require(isinstance(preflight, dict), "build preflight required")
    require(
        runtime.get("build_preflight_sha256") == digest(canonical(preflight)),
        "build preflight hash",
    )
    require(
        preflight.get("source_sha256")
        == digest(canonical(manifest["sources"][cell["source"]]))
        and preflight.get("library_sha256") == runtime["library_sha256"]
        and preflight.get("environment_sha256") == runtime["environment_sha256"]
        and preflight.get("target") == "sm90"
        and preflight.get("status") == "PASS",
        "preflight source/binary/environment/target binding",
    )
    if arm == "gpu4pyscf":
        require(preflight.get("native_profile") == "not applicable", "external engine")
        return
    for field in (
        "build_config_sha256",
        "generated_artifacts_sha256",
        "profile_resolution_receipt_sha256",
    ):
        value = preflight.get(field)
        require(
            isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)), field
        )
    require(
        preflight.get("profile_decision_sha256")
        == digest(canonical(manifest["profile_decision"])),
        "preflight root decision binding",
    )
    if cell["family"] == "cbs":
        require(
            preflight.get("native_profile") == "portable_cuda"
            and preflight.get("profile_match") == "portable"
            and preflight.get("tuned") is False
            and preflight.get("auto_sm90_outcome") == "ValueError",
            "explicit CBS portable schedule; retain auto/sm90 rejection",
        )
    else:
        require(
            isinstance(preflight.get("native_profile"), str)
            and bool(preflight["native_profile"])
            and preflight.get("profile_match") in {"exact", "compatible", "portable"}
            and type(preflight.get("tuned")) is bool,
            "HF/DFT resolved profile must be observed, not inferred from CBS",
        )


def validate_attempt(manifest: dict, attempt: dict) -> None:
    canonical(attempt)  # Reject NaN/Inf even in optional telemetry and failed receipts.
    require(
        attempt.get("schema") == "generativeqc.h100-paper-attempt.v2", "attempt schema"
    )
    slot = {k: attempt.get(k) for k in ("cell", "arm", "repeat")}
    require(
        type(slot["repeat"]) is int and slot in manifest["schedule"],
        "unknown process slot",
    )
    cell = next(c for c in manifest["cells"] if c["id"] == slot["cell"])
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", attempt.get("attempt_id", ""))),
        "attempt ID",
    )
    require(attempt.get("input_sha256") == cell["input_sha256"], "input hash")
    require(
        attempt.get("source") == manifest["sources"][cell["source"]], "source identity"
    )
    contract = manifest["execution_contracts"][cell["family"]][slot["arm"]]
    require(attempt.get("cohort") == contract["cohort"], "execution cohort")
    require(
        attempt.get("execution_contract_sha256") == digest(canonical(contract)),
        "execution contract hash",
    )
    require(
        attempt.get("profile_decision_sha256")
        == digest(canonical(manifest["profile_decision"])),
        "root profile decision hash",
    )
    outcome = attempt.get("outcome")
    require(
        outcome in {"PASS", "FAIL", "TIMEOUT", "OOM", "INCOMPLETE", "UNAVAILABLE"},
        "attempt outcome",
    )
    for field in ("preflight", "science", "timing"):
        require(
            attempt.get(field) in {"PASS", "FAIL", "INCOMPLETE", "UNAVAILABLE"}, field
        )
    samples = attempt.get("samples")
    require(isinstance(samples, list), "samples list")
    require(len({s["phase"] for s in samples}) == len(samples), "duplicate phase")
    for sample in samples:
        require(sample["phase"] in cell["calls_per_process"], "unknown phase")
        require(
            finite_number(sample.get("seconds"), positive=True),
            "nonfinite/nonpositive seconds",
        )
        require(finite_number(sample.get("energy")), "nonfinite energy")
        forces = sample.get("forces")
        atoms = len(
            cell["input"].get("atoms_bohr", cell["input"].get("geometry_bohr", []))
        )
        require(
            isinstance(forces, list)
            and len(forces) == atoms
            and all(
                isinstance(v, list) and len(v) == 3 and all(finite_number(x) for x in v)
                for v in forces
            ),
            "force shape/finiteness",
        )
    if outcome != "PASS":
        require(bool(attempt.get("failure_reason")), "retain failure reason")
        require(
            attempt["science"] != "PASS" or outcome in {"TIMEOUT", "FAIL"},
            "contradictory science",
        )
        return
    require(
        type(attempt.get("exit_code")) is int
        and attempt["exit_code"] == 0
        and all(attempt[k] == "PASS" for k in ("preflight", "science", "timing")),
        "PASS requires all gates",
    )
    require(
        [s["phase"] for s in samples] == cell["calls_per_process"],
        "missing/misordered samples",
    )
    runtime = attempt.get("runtime")
    require(isinstance(runtime, dict), "runtime identity required")
    for field in (
        "library_sha256",
        "driver_sha256",
        "environment_sha256",
        "raw_sha256",
    ):
        require(bool(re.fullmatch(r"[0-9a-f]{64}", runtime.get(field, ""))), field)
    require(runtime.get("cuda_architecture") == "sm90", "H100 target")
    validate_build_provenance(manifest, cell, slot["arm"], runtime)
    wall = runtime.get("process_wall_seconds")
    require(
        finite_number(wall, positive=True)
        and wall <= manifest["execution"]["process_cap_seconds"],
        "finite process wall cap",
    )
    require(
        sum(s["seconds"] for s in samples) <= wall,
        "endpoint intervals exceed process wall",
    )
    require(
        runtime.get("source_driver_sha256")
        == manifest["sources"][cell["source"]]["driver"]["sha256"],
        "measured source driver identity",
    )
    require(
        runtime["library_sha256"]
        != manifest["sources"][cell["source"]]["old_sm120_library_sha256"],
        "old binary is not H100 qualification",
    )
    for field in (
        "host",
        "threads",
        "gpu",
        "versions",
        "cache_state",
        "job_id",
        "process_id",
    ):
        require(bool(runtime.get(field)), f"missing {field}")
    require(
        bool(attempt.get("raw_artifacts")) and bool(attempt.get("actual_policy")),
        "raw/policy required",
    )
    require(
        isinstance(attempt.get("work"), dict),
        "work or explicit UNAVAILABLE fields required",
    )
    for field in ("iterations", "fock_builds", "actions"):
        entry = attempt["work"].get(field)
        require(
            isinstance(entry, dict)
            and (
                (entry.get("status") == "UNAVAILABLE" and entry.get("value") is None)
                or (
                    entry.get("status") == "OBSERVED"
                    and type(entry.get("value")) is int
                    and entry["value"] >= 0
                )
            ),
            "work count or explicit UNAVAILABLE required",
        )
    require(
        isinstance(attempt["actual_policy"], dict)
        and all(
            k in attempt["actual_policy"]
            for k in ("precision", "fallback", "refinement")
        ),
        "executed policy/fallback/refinement",
    )
    require(
        all(
            s.get("gate_replay") == "PASS"
            and bool(re.fullmatch(r"[0-9a-f]{64}", s.get("gate_receipt_sha256", "")))
            for s in samples
        ),
        "independent gate receipts required",
    )
    if cell["family"] == "hf_dft":
        seeds = [s.get("cold_density_sha256") for s in samples]
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", seeds[0] or ""))
            and len(set(seeds)) == 1,
            "warm density must match own cold",
        )


def validate_attempts(
    manifest: dict, attempts: list[dict], *, require_complete: bool = False
) -> dict:
    ids = set()
    processes = set()
    by_slot = {}
    for attempt in attempts:
        validate_attempt(manifest, attempt)
        require(attempt["attempt_id"] not in ids, "duplicate attempt")
        ids.add(attempt["attempt_id"])
        if attempt["outcome"] == "PASS":
            identity = (attempt["runtime"]["job_id"], attempt["runtime"]["process_id"])
            require(
                identity not in processes,
                "fresh processes must have independent identities",
            )
            processes.add(identity)
        slot = tuple(attempt[k] for k in ("cell", "arm", "repeat"))
        by_slot.setdefault(slot, []).append(attempt)
        require(
            len(by_slot[slot]) <= manifest["execution"]["max_attempts_per_slot"],
            "attempt cap",
        )
        require(
            sum(a["outcome"] == "PASS" for a in by_slot[slot]) <= 1,
            "duplicate successful process",
        )
    complete = len(by_slot) == len(manifest["schedule"]) and all(
        any(a["outcome"] == "PASS" for a in group) for group in by_slot.values()
    )
    require(not require_complete or complete, "missing independent processes")
    return {
        "status": "COMPLETE_RECEIPT_CONTRACT" if complete else "INCOMPLETE",
        "attempts": len(attempts),
        "observed_slots": len(by_slot),
        "required_slots": len(manifest["schedule"]),
        "scientific_acceptance": "REQUIRES_INDEPENDENT_RAW_REPLAY; declared gate statuses are not independently certified by this reader",
    }


def timing_statistics(manifest: dict, attempts: list[dict]) -> list[dict]:
    """Three process observations; warm medians are nested within each process."""
    validate_attempts(manifest, attempts, require_complete=True)
    rows = []
    for cell in manifest["cells"]:
        for arm in cell["arms"]:
            selected = sorted(
                (
                    a
                    for a in attempts
                    if a["cell"] == cell["id"]
                    and a["arm"] == arm
                    and a["outcome"] == "PASS"
                ),
                key=lambda a: a["repeat"],
            )
            cold = [a["samples"][0]["seconds"] for a in selected]
            warm = [[s["seconds"] for s in a["samples"][1:]] for a in selected]
            row = {
                "cell": cell["id"],
                "arm": arm,
                "cold_process_samples": cold,
                "cold_median": statistics.median(cold),
                "cold_range": [min(cold), max(cold)],
            }
            if cell["family"] == "hf_dft":
                medians = [statistics.median(v) for v in warm]
                row.update(
                    warm_within_process_samples=warm,
                    warm_process_medians=medians,
                    warm_median=statistics.median(medians),
                    warm_range=[min(medians), max(medians)],
                )
            rows.append(row)
    return rows


def cost_worksheet(manifest: dict, pilot: dict) -> dict:
    """Use measured pilot wall times only; an empty pilot cannot authorize expansion."""
    require(
        bool(pilot) and set(pilot) <= {c["id"] for c in manifest["cells"]},
        "nonempty known pilot cells required",
    )
    rows = []
    missing = []
    for cell in manifest["cells"]:
        observations = pilot.get(cell["id"], {})
        require(set(observations) <= set(cell["arms"]), "unknown pilot arm")
        missing.extend(
            {"cell": cell["id"], "arm": arm, "status": "UNAVAILABLE"}
            for arm in cell["arms"]
            if arm not in observations
        )
        for arm, observation in observations.items():
            wall, setup = (
                observation["process_wall_seconds"],
                observation["setup_seconds"],
            )
            rate = observation["price_per_gpu_hour"]
            require(
                all(finite_number(x, positive=True) for x in (wall, rate))
                and finite_number(setup)
                and setup >= 0,
                "pilot cost values",
            )
            cap = math.ceil(2 * wall + setup + 30)
            require(
                cap <= manifest["execution"]["process_cap_seconds"],
                "pilot exceeds finite process cap; stop and report",
            )
            require(
                observation.get("status") == "PASS"
                and bool(
                    re.fullmatch(r"[0-9a-f]{64}", observation.get("receipt_sha256", ""))
                ),
                "pilot gate receipt",
            )
            max_processes = 3 * manifest["execution"]["max_attempts_per_slot"]
            grace = manifest["execution"]["termination_grace_seconds"]
            batch = min(
                manifest["execution"]["job_process_limit"],
                math.floor(
                    (manifest["execution"]["job_cap_seconds"] - setup) / (cap + grace)
                ),
            )
            require(batch > 0, "pilot exceeds finite job cap")
            jobs = math.ceil(max_processes / batch)
            hours = (wall + max_processes * (cap + grace) + jobs * setup) / 3600
            rows.append(
                {
                    "cell": cell["id"],
                    "arm": arm,
                    "process_cap_seconds": cap,
                    "max_processes_per_job": batch,
                    "max_formal_jobs": jobs,
                    "job_wall_ceiling_seconds": setup + batch * (cap + grace),
                    "pilot_plus_max_attempt_gpu_hour_ceiling": hours,
                    "cost_ceiling": hours * rate,
                    "pilot_seconds": wall,
                    "pilot_receipt_sha256": observation["receipt_sha256"],
                }
            )
    total = sum(r["cost_ceiling"] for r in rows)
    require(bool(rows), "no measured pilot arms")
    return {
        "status": "ESTIMATE_REQUIRES_ROOT_RESOURCE_APPROVAL",
        "rows": rows,
        "cost_ceiling": total,
        "scope": "measured pilot arms only",
        "full_matrix_cost_ceiling": None if missing else total,
        "not_estimated": missing,
        "excludes": "retention/queue/CPU/build costs must be added from actual allocation; not a reservation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "attempts", "cost"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_manifest(), indent=2, allow_nan=False))
        return
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if args.command == "verify":
        report = {
            "protocol": "PASS",
            "source_authentication": authenticate_sources(manifest),
            "references": reference_inventory(manifest),
            "attempts": validate_attempts(manifest, []),
            "device_science": "UNAVAILABLE",
        }
    else:
        require(args.input is not None, "--input required")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        report = (
            validate_attempts(manifest, data, require_complete=args.require_complete)
            if args.command == "attempts"
            else cost_worksheet(manifest, data)
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
