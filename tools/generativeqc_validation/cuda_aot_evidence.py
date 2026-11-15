"""Bind retained audit-only CUDA compilations without promoting runtime policy."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import re
from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.evidence import file_hash
from generativeqc_compiler.integral.production_emission import emit_profile_shard
from generativeqc_compiler.integral.production_profile import resolve_production_profile

from tools.generativeqc_validation.cuda_aot_policy import (
    ENERGY_LIMIT,
    FORCE_LIMIT,
    PHASES,
)
from tools.validate_direct_jk_aot_resources import cuobjdump_resources

if TYPE_CHECKING:
    from pathlib import Path


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _retained_bundle(root: Path, declaration: dict) -> dict:
    path = (root / declaration["path"]).resolve()
    _require(path.is_relative_to(root.resolve()), "compile evidence escapes repository")
    _require(file_hash(path) == declaration["sha256"], "compile bundle hash mismatch")
    return json.loads(lzma.decompress(path.read_bytes()))


def apply_compile_evidence(
    report: dict,
    root: Path,
    declaration: dict,
    source_identity: str,
    *,
    verify_inputs: bool = False,
) -> None:
    """Annotate only source-matched primary/mixed compilations in an audit.

    Resource observations are not resource admission, numerical acceptance,
    device execution or endpoint qualification. Historical bundles remain
    readable after source changes but never qualify the new source. Hashes
    protect retained integrity; they do not authenticate the acquisition.
    """
    _require(declaration["schema_version"] == 1, "unsupported compile evidence")
    bundle = _retained_bundle(root, declaration)
    _require(
        bundle["schema"] == "generativeqc.cuda-aot-retained-compile/v1",
        "unsupported retained compile bundle",
    )
    index = bundle["index"]
    _require(
        index["schema"] == "generativeqc.cuda-aot-audit-compile/v1",
        "unsupported acquisition compile index",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", index["source_commit"]))
        and bool(re.fullmatch(r"[0-9a-f]{64}", index["native_source_identity"])),
        "missing compile source identity",
    )
    _require(
        all(
            index["toolchain"].get(name)
            for name in ("nvcc", "ptxas", "cuobjdump", "host_compiler")
        )
        and bool(index["cache_launcher"]),
        "missing compile toolchain/cache provenance",
    )
    scheduler = json.loads(bundle["files"]["scheduler.json"])
    _require(
        scheduler.get("SLURM_JOB_ID") and scheduler.get("CUDA_VISIBLE_DEVICES"),
        "missing compile scheduler allocation",
    )
    current = index["native_source_identity"] == source_identity
    _require(
        not verify_inputs or current, "input reproduction requires the original source"
    )
    profile = resolve_production_profile(root / report["rows"][0]["source"], "sm_120")
    selections = {item.spec.name: item for item in profile.selections}
    observed = {}
    for row in index["rows"]:
        key = row["architecture"], row["shell_class"]
        _require(key not in observed, "duplicate compile target/class")
        _require(row["status"] == "passed", "compile did not pass")
        compilation = row["compile"]
        _require(
            compilation["returncode"] == 0
            and compilation["timed_out"] is False
            and math.isfinite(compilation["duration_seconds"])
            and compilation["duration_seconds"] >= 0,
            "invalid compile outcome",
        )
        _require(
            all(
                re.fullmatch(r"[0-9a-f]{64}", row[name])
                for name in ("source_sha256", "object_sha256")
            ),
            "invalid compile input/object identity",
        )
        basename = row["architecture"] + "-" + row["shell_class"]
        _require(
            _sha256(bundle["files"][basename + ".log"]) == row["compile_log_sha256"],
            "compile log hash mismatch",
        )
        resource = bundle["files"][row["resource_file"]]
        _require(_sha256(resource) == row["resource_sha256"], "resource hash mismatch")
        _require(
            re.findall(r"^arch = (sm_\d+)$", resource, re.MULTILINE)
            and set(re.findall(r"^arch = (sm_\d+)$", resource, re.MULTILINE))
            == {row["architecture"]},
            "resource architecture mismatch",
        )
        kernels = cuobjdump_resources(resource)
        if current:
            selection = selections[row["shell_class"]]
            if verify_inputs:
                emitted = emit_profile_shard(
                    replace(profile, target=cuda_target_info(row["architecture"])),
                    (
                        replace(
                            selection, architecture=row["architecture"], tuned=False
                        ),
                    ),
                )
                _require(
                    _sha256(emitted) == row["source_sha256"],
                    "compiled input reproduction mismatch",
                )
            _require(
                sorted(row["consumers"])
                == sorted(item.value for item in selection.consumers)
                and row["mixed_fock"] == selection.has_capability("mixed_fock"),
                "compile consumer/capability mismatch",
            )
        observed[key] = {"row": row, "kernels": kernels}
    report["compile_evidence"] = {
        "path": declaration["path"],
        "sha256": declaration["sha256"],
        "source_commit": index["source_commit"],
        "native_source_identity": index["native_source_identity"],
        "status": "current-source-compile-only"
        if current
        else "historical-source-only",
        "translation_units": len(observed),
        "scope": index["scope"],
        "qualification_boundary": "Observed compiler resources are not admission or device/endpoint qualification; exact derivative packages and optional competitors are not covered.",
    }
    if not current:
        return
    for feature in report["rows"]:
        category, shell_class = feature["feature"].split(":", 1)
        if category == "exact-derivative":
            continue
        for architecture, target in feature["targets"].items():
            item = observed.get((architecture, shell_class))
            if item is None:
                continue
            row = item["row"]
            _require(
                feature["consumer"] in row["consumers"], "uncompiled feature consumer"
            )
            _require(
                category != "mixed-fock" or row["mixed_fock"],
                "uncompiled mixed feature",
            )
            target["nvcc_compile_and_resources"] = {
                "status": "compiled-resources-recorded-not-admitted",
                "bundle_sha256": declaration["sha256"],
                "source_sha256": row["source_sha256"],
                "object_sha256": row["object_sha256"],
                "resource_sha256": row["resource_sha256"],
                "kernel_count": len(item["kernels"]),
                "resource_maxima": {
                    field: max(kernel[field] for kernel in item["kernels"].values())
                    for field in ("REG", "STACK", "SHARED", "LOCAL")
                },
            }


def apply_endpoint_evidence(
    report: dict, root: Path, declaration: dict, source_identity: str
) -> None:
    """Replay diagnostic endpoints, retaining failed scientific gates honestly.

    A route-level water smoke does not establish per-shell coverage. Missing
    quartet counters and resource admission also prohibit global promotion.
    Diagnostic convergence overrides must never replace stock acquisitions.
    Explicit compensated totals avoid version-specific builtin sum semantics.
    """
    bundle = _retained_bundle(root, declaration)
    _require(
        bundle["schema"] == "generativeqc.cuda-aot-retained-endpoint-smoke/v1",
        "unsupported endpoint smoke bundle",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", bundle["source_commit"]))
        and bool(re.fullmatch(r"[0-9a-f]{64}", bundle["native_source_identity"])),
        "missing endpoint source identity",
    )
    device_uuids, process_ids, artifact_hashes = set(), set(), set()
    device_names, compute_capabilities = set(), set()
    libraries = {}
    results = {}
    controller = bundle["control"]["scontrol.txt"]
    _require(
        f"JobId={bundle['slurm_job_id']} " in controller
        and re.search(r"\bTimeLimit=(?:\d+-)?\d+:\d+:\d+\b", controller),
        "missing finite endpoint scheduler receipt",
    )
    for basis, files in bundle["campaigns"].items():
        index = json.loads(files["index.json"])
        _require(
            index["source_commit"] == bundle["source_commit"],
            "endpoint source commit mismatch",
        )
        reference = json.loads(files["reference.json"])
        _require(
            _sha256(files["reference.json"]) == index["reference_sha256"],
            "endpoint reference hash mismatch",
        )
        _require(
            reference["engine"] == "pyscf"
            and bool(reference["version"])
            and reference["basis"] == basis
            and reference["unit"] == "Bohr"
            and reference["representation"] == "cartesian"
            and reference["atoms"] != reference["moved_atoms"],
            "endpoint scientific contract mismatch",
        )
        arms = {"auto": [], "portable": []}
        for row in index["rows"]:
            _require(row["returncode"] == 0, "endpoint process did not finish")
            digest = row["artifact_sha256"]
            _require(digest not in artifact_hashes, "endpoint process artifact reused")
            _require(
                _sha256(files[row["file"]]) == digest, "endpoint artifact hash mismatch"
            )
            artifact_hashes.add(digest)
            data = json.loads(files[row["file"]])
            arm = row["arm"]
            _require(arm in arms and data["arm"] == arm, "endpoint arm mismatch")
            _require(data["basis"] == basis, "endpoint basis mismatch")
            _require(
                not data.get("nondefault_controls"), "nondefault endpoint controls"
            )
            _require(data["pid"] not in process_ids, "endpoint process identity reused")
            process_ids.add(data["pid"])
            device_uuids.add(data["device_uuid"])
            probe = data["probe"]
            device = probe["device"]
            device_names.add(device["name"])
            compute_capabilities.add((device["major"], device["minor"]))
            _require(
                probe["source_identity"]
                == index["native_source_identity"]
                == bundle["native_source_identity"]
                and device["official_profile"]
                == ("sm_120" if arm == "auto" else "portable_cuda")
                and device["registry_status"] == "matched"
                and device["release_build"] == 1
                and device["fast_compile"] == 0,
                "endpoint source/build/profile mismatch",
            )
            _require(
                data["slurm"]["SLURM_JOB_ID"] == bundle["slurm_job_id"]
                and bool(data["slurm"]["CUDA_VISIBLE_DEVICES"]),
                "endpoint scheduler mismatch",
            )
            library = row["library_sha256"]
            _require(
                libraries.setdefault(arm, library) == library,
                "endpoint library changed",
            )
            _require(
                set(data["phases"]) == set(PHASES),
                "partial endpoint phases",
            )
            outcomes = {}
            for phase, sample in data["phases"].items():
                expected = reference["records"][
                    "moved" if phase.startswith("moved") else "initial"
                ]
                _require(
                    sample["converged"] and expected["converged"],
                    "endpoint did not converge",
                )
                _require(
                    sample["executed_backend"] == "cuda", "endpoint fell back to CPU"
                )
                _require(
                    math.isfinite(sample["seconds"])
                    and sample["seconds"] > 0
                    and sample["iterations"] > 0
                    and sample["fock_builds"] > 0,
                    "endpoint timing/work incomplete",
                )
                forces = sample["forces"]
                _require(
                    len(forces) == len(expected["forces"]) == len(reference["atoms"]),
                    "partial endpoint forces",
                )
                _require(
                    all(
                        math.isfinite(value)
                        for collection in (forces, expected["forces"])
                        for force in collection
                        for value in force
                    ),
                    "nonfinite endpoint forces",
                )
                _require(
                    all(
                        len(force) == len(want) == 3
                        for force, want in zip(forces, expected["forces"], strict=True)
                    ),
                    "partial force components",
                )
                energy_error = abs(sample["energy"] - expected["energy"])
                force_error = max(
                    abs(value - want)
                    for force, target in zip(forces, expected["forces"], strict=True)
                    for value, want in zip(force, target, strict=True)
                )
                translation = max(
                    abs(math.fsum(force[axis] for force in forces)) for axis in range(3)
                )
                _require(
                    all(
                        math.isfinite(value)
                        for value in (energy_error, force_error, translation)
                    ),
                    "nonfinite endpoint values",
                )
                passed = (
                    energy_error <= ENERGY_LIMIT
                    and force_error <= FORCE_LIMIT
                    and translation <= FORCE_LIMIT
                )
                _require(
                    row["validation"][phase]["passed"] is passed,
                    "endpoint gate outcome was relabeled",
                )
                outcomes[phase] = {
                    "energy_error": energy_error,
                    "force_error": force_error,
                    "translation_error": translation,
                    "passed": passed,
                }
            arms[arm].append(outcomes)
        _require(
            all(len(rows) == 4 for rows in arms.values()),
            "endpoint needs four fresh processes per arm",
        )
        results[basis] = {
            arm: {
                "processes": len(rows),
                "phases": {
                    phase: {
                        "status": "passed"
                        if all(row[phase]["passed"] for row in rows)
                        else "failed-scientific-gate",
                        "maximum_energy_error": max(
                            row[phase]["energy_error"] for row in rows
                        ),
                        "maximum_force_error": max(
                            row[phase]["force_error"] for row in rows
                        ),
                        "maximum_translation_error": max(
                            row[phase]["translation_error"] for row in rows
                        ),
                    }
                    for phase in PHASES
                },
            }
            for arm, rows in arms.items()
        }
    _require(
        len(device_uuids) == 1 and bool(next(iter(device_uuids))),
        "endpoint crossed physical GPUs",
    )
    _require(
        len(device_names) == 1
        and bool(next(iter(device_names)))
        and compute_capabilities == {(12, 0)},
        "endpoint device description mismatch",
    )
    for arm, digest in libraries.items():
        _require(
            any(
                line.split()[0] == digest and line.endswith(f"libgenerativeqc-{arm}.so")
                for line in bundle["control"]["library-sha256.txt"].splitlines()
            ),
            "endpoint library differs from acquired binary receipt",
        )
    report["endpoint_smoke_evidence"] = {
        "path": declaration["path"],
        "sha256": declaration["sha256"],
        "source_commit": bundle["source_commit"],
        "native_source_identity": bundle["native_source_identity"],
        "source_status": "current-source"
        if bundle["native_source_identity"] == source_identity
        else "historical-source-only",
        "device_uuid": next(iter(device_uuids)),
        "device_name": next(iter(device_names)),
        "compute_capability": list(next(iter(compute_capabilities))),
        "library_sha256": libraries,
        "results": results,
        "qualification_boundary": "Diagnostic water/RHF route smoke only; no per-shell numerical coverage, no valid quartet counters, no default promotion or performance ranking. Failed stock gates remain failed.",
    }


def apply_card_comparison_evidence(
    report: dict, root: Path, declaration: dict, source_identity: str
) -> None:
    """Bind a second card only when both acquisitions load identical binaries.

    Matching compute capabilities do not identify a card or establish equal
    performance. Replay each independent oracle campaign, then enforce the
    source/image controls needed for a same-binary, cross-card diagnostic.
    """
    secondary_report = {}
    apply_endpoint_evidence(secondary_report, root, declaration, source_identity)
    primary = report["endpoint_smoke_evidence"]
    secondary = secondary_report["endpoint_smoke_evidence"]
    secondary_bundle = _retained_bundle(root, declaration)
    shared = secondary_bundle.get("build_receipts", {})
    if "retained_bundle" in shared:
        _require(
            shared["retained_bundle"]
            == {name: primary[name] for name in ("path", "sha256")}
            and shared["section"] == "build_receipts",
            "card comparison shared build origin mismatch",
        )
        build_receipts = _retained_bundle(root, shared["retained_bundle"])[
            "build_receipts"
        ]
        _require(
            bool(shared["file_sha256"])
            and all(
                _sha256(build_receipts[name]) == digest
                for name, digest in shared["file_sha256"].items()
            ),
            "card comparison shared build receipt mismatch",
        )
    _require(
        primary["native_source_identity"] == secondary["native_source_identity"]
        and primary["library_sha256"] == secondary["library_sha256"],
        "card comparison requires identical source and binaries",
    )
    _require(
        primary["compute_capability"] == secondary["compute_capability"]
        and primary["device_uuid"] != secondary["device_uuid"],
        "card comparison requires distinct devices with the same SM",
    )
    report["same_binary_card_comparison"] = {
        "primary_device_uuid": primary["device_uuid"],
        "secondary": secondary,
        "qualification_boundary": "Same-binary, same-SM diagnostic only. Different host/runtime environments preclude isolating GPU-only performance; neither card supplies SM80/SM90 execution or global-default acceptance.",
    }
