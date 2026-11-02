"""Fail-closed audit for #1598 default-promotion controls."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "manifests/maintenance/default_promotion_inventory.json"
SCHEMA = "generativeqc.default-promotion-inventory"
SCHEMA_VERSION = 1
CLASSIFICATIONS = frozenset(
    {
        "diagnostic-test-only",
        "negative-evidence",
        # A completed decision to require explicit selection, without asserting
        # that every eligible workload is slower or still awaiting a default.
        "retained-opt-in",
        "needs-qualification",
        "scientific-choice",
        "guarded-promotion-candidate",
        "already-default",
        "retire",
    }
)
AUDITED_PREFIXES = (
    "scf-option:",
    "public-policy:",
    "hf-runtime:",
    "ks-runtime:",
    "tensor-schedule:",
    "tensor-execution:",
    "tensor-ad:",
    "dft-policy:",
    "public-model:",
    "initial-guess:",
    "cc-option:",
    "cc-execution:",
    "response-option:",
)


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read audited source {path}: {exc}") from exc


def _assignment_map(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = node.value
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def _string_tuple(assignments: dict[str, ast.AST], name: str) -> tuple[str, ...]:
    visiting: set[str] = set()

    def resolve(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (ast.Tuple, ast.List)):
            values: list[str] = []
            for item in node.elts:
                values.extend(
                    resolve(item.value if isinstance(item, ast.Starred) else item)
                )
            return values
        if isinstance(node, ast.Name):
            if node.id in visiting:
                raise ValueError(f"cyclic audited tuple reference: {node.id}")
            if node.id not in assignments:
                raise ValueError(f"unknown audited tuple reference: {node.id}")
            visiting.add(node.id)
            try:
                return resolve(assignments[node.id])
            finally:
                visiting.remove(node.id)
        raise ValueError(f"audited tuple {name} contains a non-string expression")

    if name not in assignments:
        raise ValueError(f"missing audited tuple {name}")
    values = tuple(resolve(assignments[name]))
    if len(values) != len(set(values)):
        raise ValueError(f"audited tuple {name} contains duplicate controls")
    return values


def _discover_scf_options(root: Path) -> dict[str, str]:
    relative = Path("src/scf/types.hpp")
    source = _read(root / relative)
    # Audit the field name, not its initializer: {}, {false}, = false and
    # non-literal defaults must all require the same ownership registration.
    source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", source, flags=re.DOTALL)
    names = re.findall(
        r"\bbool\s+((?:experimental_|incremental_)[A-Za-z0-9_]*)\b",
        source,
    )
    return {f"scf-option:{name}": relative.as_posix() for name in names}


def _discover_ks_tile_policy(root: Path) -> dict[str, str]:
    """Register the public tile choice without pinning its implementation."""
    relative = Path("python/generativeqc/ks.py")
    tree = ast.parse(_read(root / relative), filename=str(relative))
    controls = {}
    names = {
        "tile_points": "dft-policy:ks-scf-tile-points",
        "force_tile_points": "dft-policy:ks-force-tile-points",
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KsOptions":
            for field in node.body:
                if (
                    isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                    and field.target.id in names
                ):
                    controls[names[field.target.id]] = relative.as_posix()
    if set(controls) != set(names.values()):
        raise ValueError("public KS tile policies are missing")
    return controls


def _discover_runtime_controls(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc/resources_hf.py")
    tree = ast.parse(_read(root / relative), filename=str(relative))
    names = _string_tuple(_assignment_map(tree), "_CUDA_SCHEDULE_VARIABLES")
    return {f"hf-runtime:{name}": relative.as_posix() for name in names}


def _literal_default(node: ast.AST | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
    ):
        for keyword in node.keywords:
            if keyword.arg == "default":
                return _literal_default(keyword.value)
    return None


def _discover_tensor_schedule(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc_compiler/tensor/cuda_plan.py")
    tree = ast.parse(_read(root / relative), filename=str(relative))
    schedule = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TensorSchedule"
        ),
        None,
    )
    if schedule is None:
        raise ValueError("missing audited TensorSchedule")
    result: dict[str, str] = {}
    for node in schedule.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        annotation = node.annotation
        if isinstance(annotation, ast.Name) and annotation.id == "bool":
            default = _literal_default(node.value)
            if not isinstance(default, bool):
                raise ValueError(
                    f"TensorSchedule.{name} has a non-literal boolean default"
                )
            result[f"tensor-schedule:{name}"] = relative.as_posix()
    return result


def _discover_tensor_execution(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc_compiler/tensor/cuda_execute.py")
    source = _read(root / relative)
    if not re.search(r'execution_mode:\s*str\s*=\s*"ordinary"', source):
        raise ValueError("PreparedCuda execution_mode default drifted from ordinary")
    if '"cuda-graph"' not in source:
        raise ValueError("missing audited cuda-graph execution mode")
    return {"tensor-execution:cuda-graph": relative.as_posix()}


def _discover_tensor_ad(root: Path) -> dict[str, str]:
    """Register explicit Boolean API controls, without pinning derivative bodies."""
    relative = Path("python/generativeqc_compiler/tensor/ad_program.py")
    tree = ast.parse(_read(root / relative), filename=str(relative))
    result = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults, strict=True
        ):
            if (
                isinstance(argument.annotation, ast.Name)
                and argument.annotation.id == "bool"
            ):
                if (
                    not isinstance(default, ast.Constant)
                    or type(default.value) is not bool
                ):
                    raise ValueError(
                        "Tensor AD Boolean control requires an explicit Boolean default"
                    )
                result[f"tensor-ad:{function.name}.{argument.arg}"] = (
                    relative.as_posix()
                )
    return result


def _discover_public_precision(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc/calculator.py")
    source = _read(root / relative)
    if not re.search(r'precision:\s*str\s*=\s*"fp64"', source):
        raise ValueError("public precision default drifted from fp64")
    if '"auto"' not in source:
        raise ValueError("public precision auto mode is missing")
    return {"public-policy:precision-auto": relative.as_posix()}


def _discover_force_active_ao(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc/_force_active_ao.py")
    tree = ast.parse(_read(root / relative), filename=str(relative))
    default = _assignment_map(tree).get("DEFAULT_FORCE_ACTIVE_AO_POLICY")
    if not isinstance(default, ast.Constant) or default.value != "auto":
        raise ValueError("force active-AO policy default drifted from auto")
    return {"dft-policy:force-active-ao-auto": relative.as_posix()}


def _discover_xc_point_batching(root: Path) -> dict[str, str]:
    """Inventory rollback controls; compiled resource tests qualify the policy."""
    relative = Path("src/dft/cuda_ks.cpp")
    source = _read(root / relative)
    for control in (
        "GENERATIVEQC_CUDA_XC_BATCH_TILES",
        "GENERATIVEQC_CUDA_XC_BATCH_BYTES",
        "GENERATIVEQC_CUDA_XC_COMPACT_BATCH",
        "GENERATIVEQC_CUDA_XC_RAGGED_BATCH",
    ):
        if f'"{control}"' not in source:
            raise ValueError(f"XC point-batch rollback selector missing: {control}")
    return {
        "dft-policy:xc-point-batch-auto": relative.as_posix(),
        "dft-policy:GENERATIVEQC_CUDA_XC_COMPACT_BATCH": relative.as_posix(),
        "dft-policy:GENERATIVEQC_CUDA_XC_RAGGED_BATCH": relative.as_posix(),
    }


def _discover_ks_final_validation_default(root: Path) -> dict[str, str]:
    """Audit model/resource admission and opt-out; executable tests own arithmetic.

    Final FP64 arithmetic may follow mixed proposals. Do not pin the spelling of
    that state decision to the original requested SCF precision.
    """
    relative = Path("src/dft/cuda_ks.cpp")
    source = re.sub(r"\s+", " ", _read(root / relative))
    policy = _read(root / "src/dft/cuda_ks_final_validation_policy.hpp")
    if not re.search(
        r"return\s+spins\s*==\s*1\s*&&\s*full_precision\s*&&\s*"
        r"\(\(aos\s*>=\s*384\s*&&\s*\w+\)\s*\|\|\s*"
        r"\(aos\s*>=\s*544\s*&&\s*full_range_coulomb_only\)\);",
        policy,
    ) or any(
        fragment not in policy
        for fragment in (
            "std::isfinite(coefficient)",
            "coefficient < 0.0",
            "coefficient >= -0.5",
            "if (!setting) return default_eligible;",
            'std::strcmp(setting, "0") == 0) return false;',
            'std::strcmp(setting, "1") == 0) return true;',
            "throw std::invalid_argument",
        )
    ):
        raise ValueError("KS final-validation default scope or opt-out drifted")
    if any(
        fragment not in source
        for fragment in (
            "restricted_global_hybrid_exchange_coefficient(exchange_coefficient)",
            "!has_range_correction && !nonlocal_correlation",
            "!nonlocal_correlation &&",
            "provider.system().ecp_terms.empty()",
            "atom.ecp_core != 0",
            "final_spec.coulomb.approximation == scf::FockApproximation::Exact",
            "final_spec.coulomb.op == scf::FockOperator::FullRange",
            "final_spec.exchange.approximation == final_spec.coulomb.approximation",
            "final_spec.exchange.op == scf::FockOperator::FullRange",
            "!has_exchange && !fitted_exchange && !has_range_correction && !nonlocal_correlation",
            "!final_spec.exchange.present",
            "final_spec.coulomb.approximation == scf::FockApproximation::DensityFitted",
            "device_final_validation_default_eligible(",
            'std::getenv("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"),',
        )
    ):
        raise ValueError("KS final-validation default owner or resource guard drifted")
    # Resource guards must belong to this consumer's admission, not merely
    # occur elsewhere (for example in the independent early-final SCF probe).
    admission = re.search(r"\bconst bool device_validation\s*=\s*([^;]+);", source)
    if admission is None or any(
        fragment not in admission.group(1)
        for fragment in (
            "device_final_validation_requested(",
            'std::getenv("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"),',
            "default_device_validation)",
            "final_validation_partial && matrix_products.library_enabled()",
            "!final_stationary_weights_ready",
        )
    ):
        raise ValueError("KS final-validation default owner or resource guard drifted")
    return {"dft-policy:ks-device-final-validation-auto": relative.as_posix()}


def _discover_md_j_default(root: Path) -> dict[str, str]:
    """Keep diagnostic controls registered without pinning admission syntax.

    Domain, opt-out behavior, capacity and rollback are enforced by executable
    probes in test_md_j_resources and test_device_ledger_requested_bytes. A
    source spelling is not proof that mandatory future owners remain protected.
    """
    relative = Path("src/scf/cuda/direct_jk.cpp")
    source = _read(root / relative)
    if (
        'std::getenv("GENERATIVEQC_DISABLE_MD_J")' not in source
        or 'std::getenv("GENERATIVEQC_MD_J_ENGINE")' not in source
    ):
        raise ValueError("MD-J default diagnostic control registration drifted")
    return {
        "dft-policy:md-j-default": relative.as_posix(),
        "dft-policy:GENERATIVEQC_DISABLE_MD_J": relative.as_posix(),
        "dft-policy:GENERATIVEQC_MD_J_ENGINE": relative.as_posix(),
    }


def _discover_spatial_force_default(root: Path) -> dict[str, str]:
    """Audit exact Direct capabilities; executable force tests own arithmetic.

    The force invocation supplies precision independently of the SCF descriptor.
    Keep model/resource boundaries here without duplicating its implementation.
    """
    relative = Path("src/dft/cuda_spd_force_policy.hpp")
    policy = _read(root / relative)
    schedule = re.sub(
        r"\s+", " ", _read(root / "src/scf/cuda/direct_spd_force_schedule.hpp")
    )
    if not re.search(
        r"return\s+aos\s*>=\s*384\s*&&\s*spins\s*==\s*1\s*&&\s*direct_full_range\s*&&\s*full_precision;",
        policy,
    ) or any(
        fragment not in schedule
        for fragment in (
            "if (selector.empty()) return DirectSpdForceSchedule::Automatic;",
            'selector == "1" || selector == "spatial-reuse"',
            "return DirectSpdForceSchedule::Incumbent;",
            "if (captured == DirectSpdForceSchedule::Automatic)",
            "return qualified_default ? DirectSpdForceSchedule::DensityReuse : DirectSpdForceSchedule::Incumbent;",
        )
    ):
        raise ValueError("Spatial force default scope or opt-out drifted")
    owner = re.sub(r"\s+", " ", _read(root / "src/methods/dft_method.cpp"))
    if any(
        fragment not in owner
        for fragment in (
            "dft::restricted_global_hybrid_exchange_coefficient(spec.exchange.coefficient)",
            "spec.coulomb.approximation == scf::FockApproximation::Exact",
            "spec.exchange.approximation == scf::FockApproximation::Exact",
            "spec.coulomb.op == scf::FockOperator::FullRange",
            "spec.exchange.op == scf::FockOperator::FullRange",
            "!execution_plan_.range_exchange && !range_strategy_",
            "!execution_plan_.nonlocal_correlation",
            "spec.exchange.op == scf::FockOperator::FullRange && !execution_plan_.d4_correction",
            "options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE",
            "system_.ecp_terms.empty()",
            "dft::spatial_spd_force_default_eligible(",
        )
    ):
        raise ValueError("Spatial force default method authentication drifted")
    return {"dft-policy:spatial-spd-force-auto": relative.as_posix()}


def _discover_rys_task_default(root: Path) -> dict[str, str]:
    """Guard shared capability defaults and exact-tuned profile overrides."""
    preference_relative = Path(
        "python/generativeqc_compiler/integral/production_rys_tasks.py"
    )
    preference = _read(root / preference_relative)
    manifest_relative = Path(
        "python/generativeqc_compiler/integral/production_shell_classes.json"
    )
    profile_relative = Path(
        "python/generativeqc_compiler/integral/production_profile.py"
    )
    manifest = json.loads(_read(root / manifest_relative))
    profiles = manifest.get("architectures")
    qualified: dict[str, set[str]] = {}
    if not isinstance(profiles, dict):
        raise TypeError("Rys-task default target/class admission drifted")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise TypeError("Rys-task default target/class admission drifted")
        classes = profile.get("preferred_rys_task_fock_shell_classes", [])
        if (
            not isinstance(classes, list)
            or any(not isinstance(item, str) or not item for item in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError("Rys-task default target/class admission drifted")
        if "preferred_rys_task_fock_shell_classes" in profile:
            qualified[name] = set(classes)
    resolution = _read(root / profile_relative)
    preferred = re.search(r"preferred = (\{[^\n]+\})", preference)
    if (
        preferred is None
        or ast.literal_eval(preferred.group(1))
        != {"psps", "ppps", "dsss", "dpss", "dsps", "ddss", "dsds", "dpps", "dspp"}
        or qualified
        != {
            "sm_120": {
                "psps",
                "ppps",
                "dsss",
                "dpss",
                "dsps",
                "ddss",
                "dsds",
                "dpps",
                "dspp",
            }
        }
        or "if profile.portable:" not in preference
        or "schedule_candidates(integral, profile.target)" not in preference
        or "if schedule is None:" not in preference
        or 'profile.target.architecture != "sm_120"' in preference
        or "override = profile.preferred_rys_task_fock_shell_classes if profile.tuned else None"
        not in preference
        or "if override is not None:" not in preference
        or "preferred = set(override)" not in preference
        or "missing = preferred - available" not in preference
        or "match == ProfileMatch.EXACT" not in resolution
        or '_profile_kind(profile_name, profile_payload) == "tuned"' not in resolution
        or "return tuple(raw) if tuned else None" not in resolution
        or 'if "preferred_rys_task_fock_shell_classes" not in profile:\n        return None'
        not in resolution
    ):
        raise ValueError("Rys-task default target/class admission drifted")
    lowering_relative = Path("src/scf/cuda/direct_fock_lowering.hpp")
    lowering = _read(root / lowering_relative)
    registry_relative = Path(
        "python/generativeqc_compiler/integral/production_registry.py"
    )
    registry = _read(root / registry_relative)
    if (
        '"GENERATIVEQC_DIRECT_K_FOCK_LOWERING"' not in lowering
        or "return exchange ? DirectFockLowering::Default : DirectFockLowering::Incumbent;"
        not in lowering
        or "case DirectFockLowering::Default:\n"
        "      selection.rys_task_fock_mask = generated::preferred_rys_task_fock_shell_class_mask();"
        not in lowering
        or "selection.rys_task_fock_mask &= class_mask;" not in lowering
        or 'std::strcmp(value, "rys-task") == 0' not in lowering
        or '"GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES"' not in registry
        or "kernels->preferred_rys_task_fock_mask\n"
        "      & enabled_rys_task_fock_shell_class_mask();"
        not in registry
    ):
        raise ValueError("Rys-task default selection/filter guard drifted")
    return {
        "dft-policy:rys-task-k-default": preference_relative.as_posix(),
        "dft-policy:GENERATIVEQC_DIRECT_K_FOCK_LOWERING": lowering_relative.as_posix(),
        "dft-policy:GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES": registry_relative.as_posix(),
    }


def _discover_direct_k_work_default(root: Path) -> dict[str, str]:
    """Keep work-default promotion separate from qualified Rys-task preference."""
    relative = Path("src/scf/cuda/direct_fock_lowering.hpp")
    source = _read(root / relative)
    if (
        'std::getenv("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE")' not in source
        or not re.search(
            r'if \(value == nullptr \|\| \*value == \'\\0\' \|\| std::strcmp\(value, "work"\) == 0\)\s*'
            r"return detail::GeneratedExchangeTaskSchedule::Work;",
            source,
        )
        or "selection.task_schedule = prepare_direct_exchange_task_schedule();"
        not in source
    ):
        raise ValueError("Direct K work schedule default/selection guard drifted")
    return {"dft-policy:GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": relative.as_posix()}


def _discover_explicit_model_and_guess_choices(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}

    calculator_relative = Path("python/generativeqc/calculator.py")
    calculator = _read(root / calculator_relative)
    if not re.search(r'density_fitting:\s*str\s*\|\s*bool\s*=\s*"none"', calculator):
        raise ValueError("public density-fitting default drifted from none")
    result["public-model:density-fitting"] = calculator_relative.as_posix()

    if not re.search(
        r'initial_guess:\s*InitialGuessSpec\s*\|\s*typing\.Literal\["auto"\]\s*\|\s*None\s*=\s*"auto"',
        calculator,
    ):
        raise ValueError("initial-guess default drifted from guarded automatic Huckel")
    result["initial-guess:preliminary-scf"] = calculator_relative.as_posix()
    result["initial-guess:minao-auto"] = calculator_relative.as_posix()
    result["initial-guess:huckel-auto"] = calculator_relative.as_posix()

    progressive_relative = Path("python/generativeqc/progressive.py")
    progressive = _read(root / progressive_relative)
    if not re.search(r"def\s+projected_singlepoint\s*\(", progressive):
        raise ValueError(
            "missing explicit cross-basis projected_singlepoint entry point"
        )
    result["initial-guess:basis-projection"] = progressive_relative.as_posix()

    fock_relative = Path("src/scf/fock_build.hpp")
    fock = _read(root / fock_relative)
    fock = re.sub(r"//[^\n]*|/\*.*?\*/", " ", fock, flags=re.DOTALL)
    if "SeminumericalCosx" not in fock:
        raise ValueError("missing audited seminumerical COSX approximation")
    term = re.search(
        r"struct\s+FockTermSpec\s*\{(?P<body>.*?)\n\};", fock, flags=re.DOTALL
    )
    if term is None:
        raise ValueError("missing audited FockTermSpec")
    if not re.search(
        r"FockApproximation\s+approximation\s*"
        r"(?:\{\s*FockApproximation::Exact\s*\}|=\s*FockApproximation::Exact)\s*;",
        term.group("body"),
    ):
        raise ValueError("Fock approximation default drifted from Exact")
    result["public-model:cosx-exchange"] = fock_relative.as_posix()
    return result


def _discover_cc_options(root: Path) -> dict[str, str]:
    """Discover default-off primal and Lambda controls in distinct namespaces."""
    result = {}
    for relative, structure, prefix in (
        (Path("src/cc/solver.hpp"), "SolverOptions", "cc-option:"),
        (Path("src/cc/lambda_response.hpp"), "LambdaOptions", "cc-option:lambda."),
    ):
        source = _read(root / relative)
        match = re.search(
            rf"struct\s+{structure}\s*\{{(?P<body>.*?)\n\}};",
            source,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"missing audited cc::{structure}")
        body = re.sub(r"//[^\n]*|/\*.*?\*/", " ", match.group("body"), flags=re.DOTALL)
        names = re.findall(
            r"\bbool\s+([A-Za-z0-9_]+)\s*(?:\{\s*(?:false)?\s*\}|=\s*false)\s*;",
            body,
        )
        result.update({f"{prefix}{name}": relative.as_posix() for name in names})
    return result


def _discover_cc_execution(root: Path) -> dict[str, str]:
    """Audit bounded physical execution defaults without inventing user flags."""
    relative = Path("src/cc/df_triples_cuda.cu")
    source = _read(root / relative)
    energy_start = source.find("DFCudaResult evaluate_df_cuda(")
    response_start = source.find("static DFCudaResponseResult pullback_df_cuda_impl(")
    joint_start = source.find("DFCudaCombinedResponseResult pullback_and_fock_df_cuda(")
    if (
        energy_start < 0
        or response_start <= energy_start
        or joint_start <= response_start
    ):
        raise ValueError("missing audited DF triples energy/response boundary")
    body = re.sub(
        r"//[^\n]*|/\*.*?\*/", " ", source[energy_start:response_start], flags=re.DOTALL
    )
    body = re.sub(r"\s+", "", body)
    required = (
        "constbooldistinct_moments=result.w_contraction_storage_bits==64&&result.w_contraction_compute_bits==64&&result.w_contraction_accumulation_bits==64;",
        "constboolalias_tile=distinct_moments&&(i==j||j==k);",
        "constautosources=alias_tile?generated_df::occupied_moment_sources(i,j,k):generated_df::MomentSourceMap{};",
        "if(sources.index[permutation]!=permutation)continue;",
        "if(alias_tile)generated_df::energy_distinct_tile(o,v,i,j,k,degeneracy,threshold,in,moments,sources,p.blocks,partials,context.error,context.stream);elsegenerated_df::energy_tile(o,v,i,j,k,degeneracy,threshold,in,moments,p.blocks,partials,context.error,context.stream);",
    )
    if any(body.count(fragment) != 1 for fragment in required) or any(
        fragment in source[response_start:joint_start]
        for fragment in ("MomentSourceMap", "energy_distinct_tile")
    ):
        raise ValueError("DF triples distinct-moment FP64 energy admission drifted")
    joint = re.sub(r"//[^\n]*|/\*.*?\*/", " ", source[joint_start:], flags=re.DOTALL)
    joint = re.sub(r"\s+", "", joint)
    forward_required = (
        "constbooldistinct_moments=d.w_contraction_storage_bits==64&&d.w_contraction_compute_bits==64&&d.w_contraction_accumulation_bits==64;",
        "constautosources=distinct_moments?generated_df::occupied_moment_sources(i,j,k):generated_df::MomentSourceMap{};",
        "constautosource=sources.index[perm];",
        "if(source!=perm){",
        "cudaMemcpyAsync(moments+perm*p.v3,moments+source*p.v3,bytes(p.v3),cudaMemcpyDeviceToDevice,context.stream)",
        "++fock.w_copy_calls;",
        "fock.w_copy_bytes=checked_add(fock.w_copy_bytes,bytes(p.v3));",
    )
    reverse = joint.split("constautorun_pullback=", 1)[-1].split(
        "constbooldistinct_moments=", 1
    )[0]
    if (
        any(joint.count(fragment) != 1 for fragment in forward_required)
        or "MomentSourceMap" in reverse
        or "sources.index" in reverse
        or "energy_distinct_tile" in joint
        or "bar_w+perm*p.v3,bar_panel,out,reverse_gemm" not in reverse
        or "for(std::size_tperm=0;perm<6;++perm)" not in reverse
    ):
        raise ValueError(
            "DF triples distinct-moment FP64 forward-only force admission drifted"
        )
    traversal = (
        "detail::visit_occupied_tiles(o,distinct_moments&&p.panel_capacity==3,visit_tile);",
        "reduce<<<1,256,0,context.stream>>>(partials,p.blocks,energies+canonical_tile,context.error);",
    )
    if (
        any(body.count(fragment) != 1 for fragment in traversal)
        or "visit_occupied_tiles" in source[response_start:]
    ):
        raise ValueError("DF triples bounded panel traversal admission drifted")
    independent_pair = (
        "constautounpaired_layout=p;",
        "if(p.panel_capacity==3&&execution_plan.pair_pointer_bytes())",
        "if(paired.total<=max_bytes)p=paired;",
        "std::array<std::size_t,6>seed_permutations{};",
        "if(seed_count==2&&p.paired_pointer_bytes)",
        "moments+seed_permutations[0]*p.v3,moments+seed_permutations[1]*p.v3,context.arena+p.paired_pointers)",
    )
    generator = re.sub(
        r"\s+", "", _read(root / "tools/generate_df_occupied_triples.py")
    )
    if (
        any(body.count(fragment) != 1 for fragment in independent_pair)
        or "build_w_pair" in source[response_start:]
        or "precision.arithmetic.is_strict_fp64()&&algorithm()==tensor::ContractionAlgorithm::PedanticBlas"
        not in generator
        or "second_output!=first_output+v3" not in generator
    ):
        raise ValueError("DF triples independent-pair admission drifted")
    return {
        "cc-execution:df-triples-distinct-moments": relative.as_posix(),
        "cc-execution:df-triples-panel-traversal": relative.as_posix(),
        "cc-execution:df-triples-independent-pair": relative.as_posix(),
    }


def _discover_response_options(root: Path) -> dict[str, str]:
    relative = Path("src/hf/rhf_frame_response.hpp")
    source = _read(root / relative)
    match = re.search(
        r"struct\s+RHFFrameResponseOptions\s*\{(?P<body>.*?)\n\};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("missing audited RHFFrameResponseOptions")
    body = re.sub(r"//[^\n]*|/\*.*?\*/", " ", match.group("body"), flags=re.DOTALL)
    names = re.findall(
        r"\bbool\s+([A-Za-z0-9_]+)\s*"
        r"(?:\{\s*(?:false|true)?\s*\}|=\s*(?:false|true))\s*;",
        body,
    )
    result = {f"response-option:{name}": relative.as_posix() for name in names}

    if not re.search(
        r"\bdouble\s+orbital_screening_tolerance\s*"
        r"(?:\{\s*(?:0(?:\.0)?)?\s*\}|=\s*0(?:\.0)?)\s*;",
        body,
    ):
        raise ValueError("RHF response orbital screening default drifted from zero")
    result["response-option:orbital_screening_tolerance"] = relative.as_posix()

    if not re.search(r"RHFFrameResponseRecycle\s*\*\s*recycling\s*\{\s*\}\s*;", body):
        raise ValueError("RHF response recycling default drifted from disabled")
    result["response-option:recycling"] = relative.as_posix()
    return result


def _discover_df_rhf_preconvergence(root: Path) -> dict[str, str]:
    """Discover the rollback selector; compiled policy tests qualify admission."""
    relative = Path("src/methods/df_hf_guess.cpp")
    source = _read(root / relative)
    required = (
        'std::getenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS")',
        'value != "direct" && value != "auto"',
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("DF-RHF preconvergence rollback selector drifted")
    return {"initial-guess:df-rhf-preconvergence-auto": relative.as_posix()}


def _discover_rhf_phase_values_default(root: Path) -> dict[str, str]:
    """Discover the control; compiled source-policy tests own AO thresholds."""
    policy_relative = Path("src/scf/cuda/direct_jk_source_policy.hpp")
    policy = _read(root / policy_relative)
    required = (
        '!value || std::string_view(value) == "auto"',
        "if (!cold_phase)",
        "maximum_angular != 3 || !generic_fock",
        "maximum_iterations < 8",
    )
    owner_relative = Path("src/scf/cuda/rhf_resident_values.cpp")
    owner = _read(root / owner_relative)
    source = " ".join(_read(root / "src/scf/cuda/direct_jk_source_lease.cpp").split())
    header = _read(root / "src/scf/cuda/direct_jk_source_lease.hpp")
    facade = _read(root / "src/scf/cuda/reference_eri_policy.hpp")
    if (
        any(fragment not in policy for fragment in required)
        or 'std::getenv("GENERATIVEQC_RHF_RESIDENT_VALUES")' not in owner
        or "return direct_jk_source_selection(value)" not in facade
        or "return direct_jk_source_auto_refusal(" not in facade
        or "value_ceiling = 8ULL << 30" not in header
        or any(
            fragment not in source
            for fragment in (
                "budget, value_ceiling",
                "device_overhead, 256ULL << 20",
            )
        )
    ):
        raise ValueError("RHF phase-value auto default or admission domain drifted")
    return {"hf-runtime:GENERATIVEQC_RHF_RESIDENT_VALUES": owner_relative.as_posix()}


def _discover_ks_phase_values(root: Path) -> dict[str, str]:
    """Register stable KS controls; executable probes qualify their admission."""
    relative = Path("src/dft/cuda_ks.cpp")
    source = _read(root / relative)
    # The environment name may be chosen by a conditional or read through a
    # policy helper. Inventory ownership follows each live control literal,
    # independently of that harmless call-site spelling. Admission semantics
    # are covered by the executable source-policy and endpoint tests.
    source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", source, flags=re.DOTALL)
    names = re.findall(
        r'"(GENERATIVEQC_KS_(?:RANGE_)?RESIDENT_(?:VALUES|REPLAY))"', source
    )
    names.extend(re.findall(r'"(GENERATIVEQC_CUDA_KS_WARM_FACTOR(?:_AUDIT)?)"', source))
    names.extend(re.findall(r'"(GENERATIVEQC_CUDA_KS_COLD_FACTOR)"', source))
    names.extend(re.findall(r'"(GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY)"', source))
    names.extend(re.findall(r'"(GENERATIVEQC_CUDA_KS_OCCUPIED_XC)"', source))
    names.extend(re.findall(r'"(GENERATIVEQC_KS_FAST_ANCHOR_DELTA)"', source))
    names.extend(
        re.findall(r'"(GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING)"', source)
    )
    names.extend(
        re.findall(
            r'"(GENERATIVEQC_KS_INCREMENTAL_(?:PROPOSALS|COARSE_SCREENING))"', source
        )
    )
    names.extend(re.findall(r'"(GENERATIVEQC_KS_EARLY_FINAL_PROBE)"', source))
    names.extend(
        re.findall(
            r'"(GENERATIVEQC_KS_(?:RANGE_)?(?:HOST|DISK)_(?:MAX_BYTES|TILE_BYTES|BUCKET_MASK))"',
            source,
        )
    )
    names.extend(re.findall(r'"(GENERATIVEQC_CUDA_XC_SPATIAL_ORDER)"', source))
    names.extend(re.findall(r'"(GENERATIVEQC_ERI_TMPDIR)"', source))
    controls = {f"ks-runtime:{name}": relative.as_posix() for name in names}
    method = Path("src/methods/dft_method.cpp")
    if (root / method).is_file():
        owner = re.sub(
            r"//[^\n]*|/\*.*?\*/", " ", _read(root / method), flags=re.DOTALL
        )
        for name in re.findall(r'"(GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE)"', owner):
            controls[f"ks-runtime:{name}"] = method.as_posix()
    geometry = Path("python/generativeqc/_stationary_composite_cuda.py")
    if (root / geometry).is_file():
        for name in re.findall(
            r'"(GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE)"', _read(root / geometry)
        ):
            controls[f"ks-runtime:{name}"] = geometry.as_posix()
    return controls


def _discover_partial_phase_values(root: Path) -> dict[str, str]:
    """Experimental masks must not silently replace the full-resident default."""
    policy = _read(root / "src/scf/cuda/direct_jk_source_policy.hpp")
    if any(
        fragment not in policy
        for fragment in (
            "std::uint32_t mask{(1U << 28) - 1}",
            "policy.ceiling = default_ceiling;",
            "if (!text) return true;",
            "policy.mask < (1U << 28)",
        )
    ):
        raise ValueError("experimental partial source default or admission drifted")
    result = {}
    for method, relative in (
        ("RHF", "src/scf/cuda/reference_eri_policy.hpp"),
        ("KS", "src/dft/cuda_ks.cpp"),
    ):
        source = _read(root / relative)
        if "direct_jk_resident_bucket_policy(" not in source:
            raise ValueError("experimental partial source parser drifted")
        prefix = "hf-runtime" if method == "RHF" else "ks-runtime"
        for suffix in ("BUCKET_MASK", "MAX_BYTES"):
            control = f"GENERATIVEQC_{method}_RESIDENT_{suffix}"
            if f'std::getenv("{control}")' not in source:
                raise ValueError("experimental partial source controls drifted")
            result[f"{prefix}:{control}"] = relative
    return result


def _discover_prepared_df_controls(root: Path) -> dict[str, str]:
    """Inventory optional prepared value owners without pinning their admission syntax."""
    relative = Path("src/scf/fock_prepared.cpp")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", _read(root / relative), flags=re.DOTALL)
    return {
        f"hf-runtime:{name}": relative.as_posix()
        for name in re.findall(r'"(GENERATIVEQC_DF_(?:MD|RAW)_COULOMB)"', source)
    }


def _discover_nonlocal_force_reuse(root: Path) -> dict[str, str]:
    """Register native force reuse; executed numerical/work gates own admission."""
    relative = Path("src/dft/cuda_stationary_controller.cpp")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", _read(root / relative), flags=re.DOTALL)
    return {
        f"dft-policy:{name}": relative.as_posix()
        for name in re.findall(r'"(GENERATIVEQC_VV10_COMPONENT_REUSE)"', source)
    }


def _discover_response_host_controls(root: Path) -> dict[str, str]:
    """Track resource controls without prescribing admission implementation."""
    relative = Path("src/hf/rhf_frame_response.cu")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", _read(root / relative), flags=re.DOTALL)
    return {
        f"response-option:{name}": relative.as_posix()
        for name in re.findall(r'"(GENERATIVEQC_RHF_RESPONSE_HOST_[A-Z_]+)"', source)
    }


def discover_controls(root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for discovered in (
        _discover_scf_options(root),
        _discover_ks_tile_policy(root),
        _discover_public_precision(root),
        _discover_runtime_controls(root),
        _discover_prepared_df_controls(root),
        _discover_nonlocal_force_reuse(root),
        _discover_ks_phase_values(root),
        _discover_partial_phase_values(root),
        _discover_tensor_schedule(root),
        _discover_tensor_execution(root),
        _discover_tensor_ad(root),
        _discover_force_active_ao(root),
        _discover_xc_point_batching(root),
        _discover_ks_final_validation_default(root),
        _discover_spatial_force_default(root),
        _discover_md_j_default(root),
        _discover_rys_task_default(root),
        _discover_direct_k_work_default(root),
        _discover_explicit_model_and_guess_choices(root),
        _discover_cc_options(root),
        _discover_cc_execution(root),
        _discover_response_options(root),
        _discover_response_host_controls(root),
        _discover_df_rhf_preconvergence(root),
        _discover_rhf_phase_values_default(root),
    ):
        overlap = set(result) & set(discovered)
        if overlap:
            raise ValueError(
                f"control discovered by multiple audits: {sorted(overlap)}"
            )
        result.update(discovered)
    return result


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_inventory(
    payload: dict[str, Any],
    *,
    root: Path = ROOT,
    check_sources: bool = True,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    classifications = payload.get("classifications")
    if (
        not isinstance(classifications, list)
        or any(not isinstance(value, str) for value in classifications)
        or (
            len(classifications) != len(set(classifications))
            or set(classifications) != CLASSIFICATIONS
        )
    ):
        errors.append(
            "classifications must declare exactly the supported policy taxonomy"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return errors + ["entries must be a non-empty list"]

    ids: set[str] = set()
    registered: dict[str, str] = {}
    entry_sources: dict[str, set[str]] = {}
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        entry_id = entry.get("id")
        if not _nonempty_string(entry_id):
            errors.append(f"{label}.id must be a non-empty string")
            continue
        if entry_id in ids:
            errors.append(f"duplicate entry id: {entry_id}")
        ids.add(entry_id)
        classification = entry.get("classification")
        if classification not in CLASSIFICATIONS:
            errors.append(f"{entry_id}: invalid classification {classification!r}")
        owners = entry.get("owner_issues")
        if (
            not isinstance(owners, list)
            or not owners
            or any(type(issue) is not int or issue <= 0 for issue in owners)
        ):
            errors.append(
                f"{entry_id}: owner_issues must contain positive issue numbers"
            )
        elif classification == "needs-qualification" and set(owners) == {1598}:
            errors.append(
                f"{entry_id}: needs-qualification requires a narrower owner than #1598"
            )
        for field in ("rationale", "revisit_condition"):
            if not _nonempty_string(entry.get(field)):
                errors.append(f"{entry_id}: {field} must be a non-empty string")
        sources = entry.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not _nonempty_string(source) for source in sources)
        ):
            errors.append(
                f"{entry_id}: sources must be non-empty repository-relative paths"
            )
            source_set: set[str] = set()
        else:
            source_set = set(sources)
            if len(source_set) != len(sources):
                errors.append(f"{entry_id}: sources contain duplicates")
            if check_sources:
                for source in source_set:
                    path = Path(source)
                    if path.is_absolute() or ".." in path.parts:
                        errors.append(f"{entry_id}: unsafe source path {source!r}")
                    elif not (root / path).is_file():
                        errors.append(f"{entry_id}: source does not exist: {source}")
        entry_sources[entry_id] = source_set
        controls = entry.get("controls")
        if (
            not isinstance(controls, list)
            or not controls
            or any(not _nonempty_string(control) for control in controls)
        ):
            errors.append(f"{entry_id}: controls must be a non-empty list")
            continue
        for control in controls:
            if not control.startswith(AUDITED_PREFIXES):
                errors.append(f"{entry_id}: unsupported control key {control!r}")
                continue
            if control in registered:
                errors.append(
                    f"control {control!r} is registered by both "
                    f"{registered[control]!r} and {entry_id!r}"
                )
            registered[control] = entry_id

    if errors and not check_sources:
        return errors

    try:
        discovered = discover_controls(root)
    except (SyntaxError, TypeError, ValueError) as exc:
        return errors + [f"control discovery failed: {exc}"]

    missing = sorted(set(discovered) - set(registered))
    stale = sorted(set(registered) - set(discovered))
    if missing:
        errors.append("unregistered audited controls: " + ", ".join(missing))
    if stale:
        errors.append("inventory controls no longer discovered: " + ", ".join(stale))
    for control in sorted(set(discovered) & set(registered)):
        entry_id = registered[control]
        source = discovered[control]
        if source not in entry_sources.get(entry_id, set()):
            errors.append(
                f"{entry_id}: discovered source {source!r} is missing for control {control!r}"
            )
    return errors


def expand_inventory_shards(
    payload: dict[str, Any], *, root: Path = ROOT
) -> tuple[dict[str, Any], list[str]]:
    """Load version-two policy fragments into the original validated model.

    The stable root declaration is never rewritten when a policy is added.
    Existing callers of validate_inventory still see the same v1 entries list.
    """
    if payload.get("schema_version") != 2:
        return payload, []
    if "entries" in payload:
        return payload, ["sharded inventory must not embed mutable entries"]
    directory = payload.get("entries_dir")
    if (
        not isinstance(directory, str)
        or not directory.startswith("manifests/maintenance/")
        or Path(directory).is_absolute()
        or ".." in Path(directory).parts
    ):
        return payload, ["invalid default-promotion entry directory"]
    entry_dir = root / directory
    if not entry_dir.is_dir() or entry_dir.is_symlink():
        return payload, [f"missing or unsafe default-promotion directory: {directory}"]
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(entry_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            errors.append(f"unexpected default-promotion shard: {path.name}")
            continue
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError, UnicodeError) as error:
            errors.append(f"invalid default-promotion shard {path.name}: {error}")
            continue
        if not isinstance(entry, dict) or entry.get("id") != path.stem:
            errors.append(f"default-promotion shard id differs from {path.name}")
            continue
        entries.append(entry)
    if not entries:
        errors.append("default-promotion entry directory is empty")
    assembled = {key: value for key, value in payload.items() if key != "entries_dir"}
    assembled["schema_version"] = SCHEMA_VERSION
    assembled["entries"] = entries
    return assembled, errors


def load_and_validate(
    inventory: Path = DEFAULT_INVENTORY,
    *,
    root: Path = ROOT,
) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(inventory.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"cannot load {inventory}: {exc}"]
    if not isinstance(payload, dict):
        return {}, ["inventory root must be an object"]
    payload, errors = expand_inventory_shards(payload, root=root)
    return payload, [*errors, *validate_inventory(payload, root=root)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    payload, errors = load_and_validate(args.inventory, root=args.root)
    if errors:
        for error in errors:
            print(f"default-promotion inventory: {error}")
        return 1
    entries = payload["entries"]
    controls = sum(len(entry["controls"]) for entry in entries)
    print(
        f"default-promotion inventory: {len(entries)} entries, "
        f"{controls} audited controls, all registered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
