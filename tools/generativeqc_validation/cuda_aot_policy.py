"""Offline CUDA AOT audits and fail-closed global-default qualification.

This owner never tunes a runtime or changes scientific settings. Local warm
autotuning has a deliberately narrower contract and does not call this gate.
"""

from __future__ import annotations

import json
import lzma
import math
import re
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.evidence import canonical_hash, file_hash
from generativeqc_compiler.integral.capabilities import CAPABILITY_MIXED_FOCK
from generativeqc_compiler.integral.fused_schedule import build_fused_shell_plan
from generativeqc_compiler.integral.ir import KernelConsumer, build_integral_ir
from generativeqc_compiler.integral.production_cost import shell_class_index
from generativeqc_compiler.integral.production_policy import resolved_profile_inventory
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.shell_spec import FUSED_SHELL_SPECS
from generativeqc_compiler.integral.specialize import specialize_integral_ir
from generativeqc_compiler.integral.tuning.policy import schedule_payload

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("python/generativeqc_compiler/integral/production_shell_classes.json")
REPRESENTATIVES = {"sm_80": "A100", "sm_90": "H100", "sm_120": "RTX 5090"}
PHASES = ("cold", "warm", "moved", "moved_warm")
WORK_FIELDS = ("scf_iterations", "fock_builds", "force_evaluations", "quartet_tasks")
ENERGY_LIMIT = 1e-9
FORCE_LIMIT = 1e-7


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}",
    )
    return float(value)


def _digest(value: Any, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {name} digest",
    )
    return value


def _finite_slurm_time(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", value)
    return match is not None and any(int(part or 0) > 0 for part in match.groups())


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(
        not Path(relative).is_absolute() and path.is_relative_to(root.resolve()),
        "evidence path escapes root",
    )
    _require(path.is_file(), f"missing evidence: {relative}")
    return path


def default_policies(manifest: Path) -> dict[str, dict]:
    """Compare semantic stock defaults, never generated-source spelling.

    Named target and explicit compatibility additions also need a decision.
    Per-target data includes full schedules/recurrences, not just class counts.
    """
    payload = json.loads(manifest.read_text())
    architectures = {"sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"}
    for name, profile in payload["architectures"].items():
        if re.fullmatch(r"sm_\d+", name):
            architectures.add(name)
        architectures.update(profile.get("compatible_architectures", ()))
        for capability in profile.get("compatible_compute_capabilities", ()):
            if not isinstance(capability, str):
                continue
            numeric = re.fullmatch(r"(\d+)\.([0-9]|x)", capability)
            wildcard = re.fullmatch(r"sm_(\d+)x", capability)
            if numeric:
                architectures.add(
                    f"sm_{numeric[1]}{numeric[2] if numeric[2] != 'x' else '0'}"
                )
            elif wildcard:
                architectures.add(f"sm_{wildcard[1]}0")
            elif re.fullmatch(r"sm_\d+", capability):
                architectures.add(capability)
    result = {}
    for architecture in sorted(architectures):
        profile = resolve_production_profile(manifest, architecture)
        inventory = resolved_profile_inventory(profile)
        result[architecture] = {
            "target": asdict(profile.target),
            "compiled_masks": inventory["compiled_masks"],
            "default_preferences": inventory["default_preferences"],
            "packaging": inventory["packaging"],
            "selections": [
                {
                    "shell_class": item.spec.name,
                    "consumers": sorted(consumer.value for consumer in item.consumers),
                    "recurrence": item.recurrence,
                    "resident_force_recurrence": item.resident_force_recurrence,
                    "fock_recurrence": item.fock_recurrence,
                    "schedule": schedule_payload(item.schedule),
                    "fock_schedule": schedule_payload(item.fock_schedule)
                    if item.fock_schedule
                    else None,
                    "fock_route": item.fock_route,
                    "capabilities": sorted(item.capabilities),
                }
                for item in sorted(profile.selections, key=lambda item: item.spec.name)
            ],
        }
    return result


def capability_audit(manifest: Path) -> dict:
    """Enumerate every tuned-only/portable-disabled consumer conservatively.

    IR/target-plan admission is executable. It is deliberately not labeled an
    NVCC compile, real-device oracle run, or full-endpoint performance result.
    Retargeted plans are audit-only and never enter production selection.
    """
    tuned = resolve_production_profile(manifest, "sm_120")
    portable = resolve_production_profile(manifest, "sm_120", "portable")
    portable_consumers = {
        (item.spec.name, consumer)
        for item in portable.selections
        for consumer in item.consumers
    }
    features = []
    portable_by_name = {item.spec.name: item for item in portable.selections}
    for item in tuned.selections:
        for consumer in item.consumers:
            if (item.spec.name, consumer) not in portable_consumers:
                features.append(
                    (f"extra-{consumer.value}:{item.spec.name}", item, consumer, "fp64")
                )
        counterpart = portable_by_name.get(item.spec.name)
        if (
            counterpart is not None
            and KernelConsumer.FOCK in item.consumers
            and (
                item.fock_recurrence != counterpart.fock_recurrence
                or item.fock_schedule != counterpart.fock_schedule
                or item.fock_route != counterpart.fock_route
                or (
                    item.recurrence != counterpart.recurrence
                    and item.fock_recurrence is None
                )
            )
        ):
            features.append(
                (f"fock-mapping:{item.spec.name}", item, KernelConsumer.FOCK, "fp64")
            )
        if item.has_capability(CAPABILITY_MIXED_FOCK):
            features.append(
                (
                    f"mixed-fock:{item.spec.name}",
                    item,
                    KernelConsumer.FOCK,
                    "mixed-fock-only",
                )
            )
        if KernelConsumer.FORCE in item.consumers:
            features.append(
                (
                    f"exact-derivative:{item.spec.name}",
                    item,
                    KernelConsumer.FORCE,
                    "fp64-full-sr-lr",
                )
            )
    rows = []
    for feature, selection, consumer, precision in features:
        targets = {}
        for architecture, device in REPRESENTATIVES.items():
            try:
                retargeted = replace(selection, architecture=architecture, tuned=False)
                integral = retargeted.integral or build_integral_ir(
                    selection.spec, selection.consumers, recurrence=selection.recurrence
                )
                if consumer == KernelConsumer.FOCK:
                    integral = specialize_integral_ir(
                        integral,
                        consumers=(consumer,),
                        recurrence=selection.fock_recurrence
                        or (
                            "subset_wick"
                            if integral.derivative is not None
                            else selection.recurrence
                        ),
                    )
                plan = build_fused_shell_plan(
                    selection.spec,
                    integral=integral,
                    target=cuda_target_info(architecture),
                    schedule=selection.fock_schedule
                    if consumer == KernelConsumer.FOCK and selection.fock_schedule
                    else selection.schedule,
                )
                legality = {
                    "status": "target-plan-accepted",
                    "threads": plan.block_threads,
                }
            except (ValueError, TypeError) as error:
                legality = {"status": "rejected", "reason": str(error)}
            targets[architecture] = {
                "representative_device": device,
                "ir_and_static_resource_admission": legality,
                "nvcc_compile_and_resources": "not-run-for-this-audit",
                "independent_device_numerics": "not-run-for-this-audit",
                "full_endpoint": {phase: "not-run-for-this-audit" for phase in PHASES},
                "retained_stock_selection": architecture == "sm_120",
                "decision": "explicit-experimental"
                if legality["status"] == "target-plan-accepted"
                else "fallback-only",
            }
        rows.append(
            {
                "feature": feature,
                "source": MANIFEST.as_posix(),
                "consumer": consumer.value,
                "precision_contract": precision,
                "precision_admission": "existing guarded consumer only; target-plan acceptance does not qualify mixed precision or range derivatives",
                "targets": targets,
            }
        )
    routes = {}
    for architecture in REPRESENTATIVES:
        stock = resolve_production_profile(manifest, architecture)
        explicit = replace(
            stock,
            optional_fock_variants=True,
            prefer_rys_task_fock=False,
            compile_derivative_shells=True,
        )
        routes[architecture] = {
            "stock": resolved_profile_inventory(stock),
            "explicit_compile_only": resolved_profile_inventory(explicit),
            "decision": "preserve-stock-packaging-and-preferences",
            "qualification": "optional-compilation-is-not-global-default-qualification",
        }
    route_features = []
    for route, source in (
        ("rys_value_fock", "production_rys_values.py"),
        ("k_block_fock", "production_k_block.py"),
        ("rys_task_fock", "production_rys_tasks.py"),
        ("shared_rys_xy_fock", "production_shared_rys_xy.py"),
        ("shared_rys_weighted_z_fock", "production_shared_rys_xy.py"),
        ("shared_rys_xy_streaming_fock", "production_shared_rys_xy.py"),
    ):
        targets = {}
        for architecture, inventory in routes.items():
            stock_mask = inventory["stock"]["compiled_masks"][route]
            explicit_mask = inventory["explicit_compile_only"]["compiled_masks"][route]
            targets[architecture] = {
                "stock_mask": stock_mask,
                "explicit_compile_only_mask": explicit_mask,
                "shell_classes": [
                    spec.name
                    for spec in FUSED_SHELL_SPECS
                    if explicit_mask & (1 << shell_class_index(spec))
                ],
                "ir_admission": "profile-capability-and-target-plan-only"
                if explicit_mask
                else "no-admitted-companion",
                "nvcc_compile_and_resources": "not-run-for-this-audit",
                "independent_device_numerics": "not-run-for-this-audit",
                "full_endpoint": {phase: "not-run-for-this-audit" for phase in PHASES},
                "decision": "explicit-experimental"
                if explicit_mask
                else "fallback-only",
                "preserve_existing_runtime_guards": True,
            }
        route_features.append(
            {
                "feature": route,
                "source": f"python/generativeqc_compiler/integral/{source}",
                "targets": targets,
            }
        )
    return {
        "schema": "generativeqc.cuda-aot-capability-audit",
        "schema_version": 1,
        "manifest_sha256": file_hash(manifest),
        "qualification_boundary": "Static legality is not compile, numerical or endpoint qualification; historical evidence must retain its own source/device identity.",
        "rows": sorted(rows, key=lambda row: row["feature"]),
        "route_features": route_features,
        "routes": routes,
        "shared_primary_force_contract": {
            item.spec.name: {
                "same_primary_recurrence_and_schedule": item.recurrence
                == portable_by_name[item.spec.name].recurrence
                and item.schedule == portable_by_name[item.spec.name].schedule,
                "decision": "generic-mathematical-capability",
                "qualification_boundary": "shared force declarations do not imply identical Fock mappings, binaries or runtime endpoints",
            }
            for item in tuned.selections
            if (item.spec.name, KernelConsumer.FORCE) in portable_consumers
            and KernelConsumer.FORCE in item.consumers
        },
        "historical_context": [
            {
                "path": path,
                "authority": "historical-source-bound-only-not-current-qualification",
            }
            for path in (
                "benchmarks/results/f-shell-135/index.json",
                "benchmarks/results/readme-a100-20261108/README.md",
                "benchmarks/results/h100-source-pair-20261030/README.md",
                ".agents/notes/implemented/performance/2026-11-03-sm80-direct-force-class-grouping.md",
                "docs/maintainer/f_shell_validation.md",
            )
        ],
        "unmatched_registry": {
            "physical_target": "sm_89",
            "compiled_targets": ["sm_120"],
            "expected_status": "no_matching_kernel_set",
            "decision": "bounded-native-fallback",
        },
    }


def _observables(sample: dict, oracle: dict, name: str) -> None:
    _require(
        oracle["converged"] is True, f"{name}: independent reference is unconverged"
    )
    _require(
        sample["converged"] is True and sample["backend"] == "cuda",
        f"{name}: incomplete or non-CUDA endpoint",
    )
    _require(
        sample["geometry_sha256"] == oracle["geometry_sha256"],
        f"{name}: wrong geometry",
    )
    energies, forces = sample["energies"], sample["forces"]
    _require(
        isinstance(energies, list)
        and energies
        and len(energies) == len(oracle["energies"]),
        f"{name}: incomplete energies",
    )
    _require(
        isinstance(forces, list) and forces and len(forces) == len(oracle["forces"]),
        f"{name}: incomplete full forces",
    )
    for actual, expected in zip(energies, oracle["energies"], strict=True):
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and math.isfinite(expected)
            and abs(actual - expected) <= ENERGY_LIMIT,
            f"{name}: independent energy gate",
        )
    for actual, expected in zip(forces, oracle["forces"], strict=True):
        _require(
            len(actual) == len(expected) == 3,
            f"{name}: forces must have every Cartesian component",
        )
        for component, reference in zip(actual, expected, strict=True):
            _require(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(component)
                and isinstance(reference, (int, float))
                and not isinstance(reference, bool)
                and math.isfinite(reference)
                and abs(component - reference) <= FORCE_LIMIT,
                f"{name}: independent full-force gate",
            )
    _require(
        all(
            abs(sum(force[axis] for force in forces)) <= FORCE_LIMIT
            for axis in range(3)
        ),
        f"{name}: force translation gate",
    )


def _bootstrap_lower(baseline: list[float], candidate: list[float]) -> float:
    import numpy as np

    indices = np.random.default_rng(136).integers(
        len(baseline), size=(4096, len(baseline))
    )
    ratios = np.median(np.asarray(baseline)[indices], axis=1) / np.median(
        np.asarray(candidate)[indices], axis=1
    )
    return float(np.quantile(ratios, 0.05))


def promotion_gate(
    campaign: dict,
    *,
    source_identity: str,
    architecture: str,
    baseline_policy: str,
    candidate_policy: str,
) -> dict:
    """Require fresh, paired cold/warm/moved complete endpoints and raw counts.

    Evidence checks identify measurements; they are not a signature proving an
    honest acquisition. Reviewers must inspect the source-bound raw receipts.
    Any missing/invalid field rejects admission rather than inventing a pass.
    """
    _require(
        campaign["schema"] == "generativeqc.cuda-aot-global-promotion"
        and campaign["schema_version"] == 1,
        "unsupported global-promotion schema",
    )
    _require(
        campaign["source_identity"] == _digest(source_identity, "current source"),
        "stale campaign source",
    )
    _require(campaign["architecture"] == architecture, "wrong campaign architecture")
    _require(
        campaign["baseline_policy_sha256"] == baseline_policy
        and campaign["candidate_policy_sha256"] == candidate_policy,
        "campaign policy mismatch",
    )
    science = _digest(campaign["scientific_contract_sha256"], "scientific contract")
    oracle = campaign["oracle"]
    _require(
        isinstance(oracle["engine"], str) and isinstance(oracle["version"], str),
        "versioned independent oracle required",
    )
    _require(
        "generativeqc" not in oracle["engine"].casefold()
        and oracle["engine"].casefold() not in ("native", "candidate", "baseline")
        and bool(oracle["engine"])
        and bool(oracle["version"]),
        "independent versioned oracle required",
    )
    _digest(oracle["input_sha256"], "oracle input")
    _require(
        oracle["scientific_contract_sha256"] == science,
        "oracle scientific task changed",
    )
    _require(
        set(oracle["geometries"]) == {"initial", "moved"},
        "both oracle geometries required",
    )
    _require(
        oracle["geometries"]["initial"]["geometry_sha256"]
        != oracle["geometries"]["moved"]["geometry_sha256"],
        "moved geometry is unchanged",
    )
    pairs = campaign["pairs"]
    _require(
        isinstance(pairs, list) and len(pairs) >= 4,
        "four independent paired processes required",
    )
    identifiers: set[str] = set()
    device_identity = None
    libraries: dict[str, str] = {}
    timings = {
        phase: {arm: [] for arm in ("baseline", "candidate")} for phase in PHASES
    }
    first_use = {arm: [] for arm in ("baseline", "candidate")}
    for pair in pairs:
        for arm, policy in (
            ("baseline", baseline_policy),
            ("candidate", candidate_policy),
        ):
            run = pair[arm]
            identifier = run["process_id"]
            _require(
                isinstance(identifier, str)
                and bool(identifier)
                and identifier not in identifiers,
                "processes must be fresh and distinct",
            )
            identifiers.add(identifier)
            _require(
                run["source_identity"] == source_identity
                and run["policy_sha256"] == policy,
                "run source or policy mismatch",
            )
            _require(
                run["scientific_contract_sha256"] == science,
                "scientific settings changed",
            )
            _require(
                run["production_oracle_calls"] == 0
                and run["nondefault_controls"] == {},
                "reference work or nondefault controls entered production endpoint",
            )
            scheduler = run["scheduler"]
            _require(
                bool(scheduler["job_id"])
                and bool(scheduler["node"])
                and bool(scheduler["cuda_visible_devices"])
                and _finite_slurm_time(scheduler["time_limit"]),
                "finite scheduler provenance required",
            )
            device = run["device"]
            _require(
                device["architecture"] == architecture
                and bool(device["name"])
                and bool(device["uuid"])
                and bool(device["driver"]),
                "device provenance incomplete",
            )
            identity = (
                device["uuid"],
                device["name"],
                device["architecture"],
                device["driver"],
            )
            device_identity = device_identity or identity
            _require(identity == device_identity, "A/B device or driver changed")
            build = run["build"]
            _require(
                build["release"] is True
                and build["fast_compile"] is False
                and bool(build["compiler"])
                and bool(build["cuda_images"]),
                "release compiler/image provenance required",
            )
            library = _digest(build["library_sha256"], "library")
            libraries.setdefault(arm, library)
            _require(libraries[arm] == library, "arm binary changed between processes")
            _require(
                build["resource_status"] == "passed"
                and bool(build["resource_receipt_sha256"]),
                "compile/resource receipt required",
            )
            _digest(build["resource_receipt_sha256"], "resource receipt")
            usage = build["resource_usage"]
            _require(
                set(usage)
                == {
                    "registers_per_thread",
                    "stack_bytes",
                    "local_bytes",
                    "shared_bytes",
                },
                "resource usage incomplete",
            )
            for name, value in usage.items():
                _number(value, name)
            _require(
                set(run["phases"]) == set(PHASES),
                "cold/warm/moved/moved-warm endpoints required",
            )
            first_use[arm].append(
                _number(
                    run["module_first_use_seconds"], "module first-use", positive=True
                )
            )
            _number(run["package_bytes"], "package bytes", positive=True)
            _number(run["peak_device_bytes"], "peak device bytes", positive=True)
            for phase in PHASES:
                sample = run["phases"][phase]
                reference = oracle["geometries"][
                    "moved" if phase.startswith("moved") else "initial"
                ]
                _observables(sample, reference, f"{arm}/{phase}")
                _require(
                    sample["scope"] == "complete-energy-and-analytic-forces",
                    "partial endpoint timing",
                )
                timings[phase][arm].append(
                    _number(sample["seconds"], f"{phase} endpoint", positive=True)
                )
                work = sample["semantic_work"]
                _require(
                    set(WORK_FIELDS) <= set(work), "semantic work counters incomplete"
                )
                for name, value in work.items():
                    _require(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0,
                        f"invalid semantic count {name}",
                    )
                _require(
                    work["scf_iterations"] > 0
                    and work["fock_builds"] > 0
                    and work["force_evaluations"] > 0,
                    "endpoint did not execute required work",
                )
                _require(
                    isinstance(sample["iterations"], list)
                    and len(sample["iterations"]) == len(sample["energies"])
                    and all(
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and count > 0
                        for count in sample["iterations"]
                    ),
                    "SCF iteration branch data incomplete",
                )
                _require(
                    sum(sample["iterations"]) == work["scf_iterations"],
                    "iteration count disagrees with semantic work",
                )
        left, right = pair["baseline"], pair["candidate"]
        _require(
            right["package_bytes"] <= left["package_bytes"]
            and right["peak_device_bytes"] <= left["peak_device_bytes"],
            "package or memory regression",
        )
        _require(
            right["build"]["compiler"] == left["build"]["compiler"]
            and right["build"]["cuda_images"] == left["build"]["cuda_images"],
            "A/B compiler or images changed",
        )
        _require(
            all(
                right["build"]["resource_usage"][name] <= value
                for name, value in left["build"]["resource_usage"].items()
            ),
            "compiled resource regression",
        )
        for phase in PHASES:
            baseline, candidate = left["phases"][phase], right["phases"][phase]
            _require(
                candidate["iterations"] == baseline["iterations"]
                and bool(candidate["iterations"]),
                "SCF iteration branches changed",
            )
            _require(
                set(candidate["semantic_work"]) == set(baseline["semantic_work"]),
                "semantic work axes changed",
            )
            _require(
                all(
                    candidate["semantic_work"][name] <= count
                    for name, count in baseline["semantic_work"].items()
                ),
                "semantic work amplification",
            )
    reports = {}
    improved = False
    for phase, arms in {**timings, "module_first_use": first_use}.items():
        baseline, candidate = arms["baseline"], arms["candidate"]
        speedup = statistics.median(baseline) / statistics.median(candidate)
        lower = _bootstrap_lower(baseline, candidate)
        _require(
            speedup >= 1.0
            and statistics.mean(candidate) <= statistics.mean(baseline)
            and lower >= 1.0,
            f"{phase}: nonregression gate failed",
        )
        improved |= phase in PHASES and speedup >= 1.02 and lower > 1.0
        reports[phase] = {"median_speedup": speedup, "bootstrap_lower_95": lower}
    _require(improved, "no complete endpoint demonstrates a stable improvement")
    return {
        "passed": True,
        "scope": "global-default",
        "architecture": architecture,
        "source_identity": source_identity,
        "device": device_identity,
        "phases": reports,
        "process_pairs": len(pairs),
    }


def check_defaults(root: Path, inventory: dict, *, source_identity: str) -> dict:
    """Retain an identified baseline, or require source-matched promotions.

    Merely updating an audit table or importing a local warm winner never
    supplies a global promotion. A campaign is read and evaluated, not trusted
    because it contains a precomputed ``passed`` flag.
    """
    _require(
        inventory["schema"] == "generativeqc.cuda-aot-default-decisions"
        and inventory["schema_version"] == 1,
        "unsupported default-decision schema",
    )
    retained = inventory["retained_baseline"]
    reference = _path(root, retained["manifest"])
    _require(
        file_hash(reference) == retained["sha256"],
        "retained baseline integrity mismatch",
    )
    _require(len(retained["source_commit"]) == 40, "baseline source commit required")
    snapshot = _path(root, retained["effective_policies"])
    _require(
        file_hash(snapshot) == retained["effective_policies_sha256"],
        "retained effective-policy integrity mismatch",
    )
    # Re-resolving the old manifest with today's compiler would apply today's
    # implicit choices to both sides, hiding an implementation-driven promotion.
    before = json.loads(lzma.decompress(snapshot.read_bytes()))
    after = default_policies(root / MANIFEST)
    for target in set(after) - set(before):
        before[target] = {
            "target": after[target]["target"],
            "compiled_masks": dict.fromkeys(after[target]["compiled_masks"], 0),
            "default_preferences": dict.fromkeys(
                after[target]["default_preferences"], 0
            ),
            "packaging": {
                "optional_fock_variants": False,
                "compile_derivative_shells": False,
            },
            "selections": [],
        }
    _require(
        set(before) <= set(after),
        "removing a target requires an explicit retained fallback decision",
    )
    changed = sorted(
        target
        for target in set(before) | set(after)
        if canonical_hash(before.get(target)) != canonical_hash(after.get(target))
    )
    reports = {}
    for target in changed:
        matches = [
            entry
            for entry in inventory["promotions"]
            if entry["architecture"] == target
            and entry["candidate_policy_sha256"] == canonical_hash(after[target])
        ]
        _require(
            len(matches) == 1,
            f"{target}: fresh global-default promotion campaign required",
        )
        entry = matches[0]
        path = _path(root, entry["campaign"])
        _require(
            path.stat().st_size <= 8 * 1024 * 1024
            and file_hash(path) == entry["sha256"],
            "campaign integrity or size mismatch",
        )
        artifact_paths = {}
        for artifact in entry["artifacts"]:
            artifact_path = _path(root, artifact["path"])
            _require(
                artifact_path.stat().st_size <= 64 * 1024 * 1024
                and file_hash(artifact_path) == artifact["sha256"],
                "raw receipt integrity mismatch",
            )
            artifact_paths[artifact["sha256"]] = artifact_path
        _require(bool(entry["artifacts"]), "raw build/acquisition receipts required")
        campaign = json.loads(path.read_text())
        oracle = campaign["oracle"]
        _require(
            oracle["input_sha256"] in artifact_paths
            and oracle["reference_receipt_sha256"] in artifact_paths,
            "independent reference inputs and receipt must be retained",
        )
        reference_receipt = json.loads(
            artifact_paths[oracle["reference_receipt_sha256"]].read_text()
        )
        _require(
            reference_receipt
            == {
                key: value
                for key, value in oracle.items()
                if key != "reference_receipt_sha256"
            },
            "independent reference receipt mismatch",
        )
        for pair in campaign["pairs"]:
            for arm, run in pair.items():
                _require(
                    run["build"]["resource_receipt_sha256"] in artifact_paths
                    and run["acquisition_receipt_sha256"] in artifact_paths,
                    "run references an unretained raw receipt",
                )
                _require(
                    run["compiled_masks"]
                    == (before if arm == "baseline" else after)[target][
                        "compiled_masks"
                    ],
                    "executed compiled inventory differs from declared policy",
                )
                _check_raw_receipts(run, artifact_paths)
        reports[target] = promotion_gate(
            campaign,
            source_identity=source_identity,
            architecture=target,
            baseline_policy=canonical_hash(before[target]),
            candidate_policy=canonical_hash(after[target]),
        )
    return {
        "changed_targets": changed,
        "retained_defaults": not changed,
        "promotions": reports,
    }


def _check_raw_receipts(run: dict, artifacts: dict[str, Path]) -> None:
    """Recompute raw-resource aggregates and bind complete acquisition records.

    A resource ``passed`` string alone is insufficient. The existing cubin
    parser owns the measured register/stack/shared/local resource contract.
    """
    from tools.validate_direct_jk_aot_resources import cuobjdump_resources

    build = run["build"]
    resource = json.loads(artifacts[build["resource_receipt_sha256"]].read_text())
    _require(
        resource["schema"] == "generativeqc.cuda-aot-resource-receipt/v1",
        "unsupported resource receipt",
    )
    _require(
        resource["source_identity"] == run["source_identity"]
        and resource["library_sha256"] == build["library_sha256"]
        and resource["architecture"] == run["device"]["architecture"],
        "resource receipt source/binary mismatch",
    )
    _require(
        resource["compiler"] == build["compiler"]
        and resource["cuda_images"] == build["cuda_images"]
        and resource["exit_code"] == 0
        and bool(resource["commands"]),
        "resource compile context mismatch",
    )
    log = resource["cuobjdump_sha256"]
    _require(log in artifacts, "raw cubin-resource log is not retained")
    kernels = cuobjdump_resources(artifacts[log].read_text())
    usage = {
        name: max(row[field] for row in kernels.values())
        for name, field in (
            ("registers_per_thread", "REG"),
            ("stack_bytes", "STACK"),
            ("shared_bytes", "SHARED"),
            ("local_bytes", "LOCAL"),
        )
    }
    _require(
        usage == build["resource_usage"],
        "resource summary disagrees with raw cubin measurements",
    )
    acquisition = json.loads(artifacts[run["acquisition_receipt_sha256"]].read_text())
    _require(
        acquisition["schema"] == "generativeqc.cuda-aot-acquisition-receipt/v1",
        "unsupported acquisition receipt",
    )
    measured = {
        key: value for key, value in run.items() if key != "acquisition_receipt_sha256"
    }
    _require(
        acquisition["run_sha256"] == canonical_hash(measured),
        "acquisition receipt does not bind the measured run",
    )
