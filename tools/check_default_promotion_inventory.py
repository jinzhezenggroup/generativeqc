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
    "tensor-schedule:",
    "tensor-execution:",
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
    reduction_provider_seen = False
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
        elif name == "reduction_provider":
            if _literal_default(node.value) != "generated":
                raise ValueError(
                    "TensorSchedule.reduction_provider default drifted from generated"
                )
            reduction_provider_seen = True
            result["tensor-schedule:reduction_provider"] = relative.as_posix()
    if not reduction_provider_seen:
        raise ValueError("missing audited TensorSchedule.reduction_provider")
    return result


def _discover_tensor_execution(root: Path) -> dict[str, str]:
    relative = Path("python/generativeqc_compiler/tensor/cuda_execute.py")
    source = _read(root / relative)
    if not re.search(r'execution_mode:\s*str\s*=\s*"ordinary"', source):
        raise ValueError("PreparedCuda execution_mode default drifted from ordinary")
    if '"cuda-graph"' not in source:
        raise ValueError("missing audited cuda-graph execution mode")
    return {"tensor-execution:cuda-graph": relative.as_posix()}


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
    """Audit both limits of the promoted, resource-guarded native KS policy."""
    relative = Path("src/dft/cuda_ks.cpp")
    source = _read(root / relative)
    if not re.search(
        r'point_batch_size\("GENERATIVEQC_CUDA_XC_BATCH_TILES",\s*32\)', source
    ) or not re.search(
        r'point_batch_size\("GENERATIVEQC_CUDA_XC_BATCH_BYTES",\s*32\s*\*\s*1024\s*\*\s*1024\)',
        source,
    ):
        raise ValueError("XC point-batch default drifted from 32 tiles / 32 MiB")
    if not re.search(
        r'point_batch_size\("GENERATIVEQC_CUDA_XC_COMPACT_BATCH",\s*1\)', source
    ):
        raise ValueError(
            "XC compact contraction default drifted from resource-guarded enablement"
        )
    return {
        "dft-policy:xc-point-batch-auto": relative.as_posix(),
        "dft-policy:GENERATIVEQC_CUDA_XC_COMPACT_BATCH": relative.as_posix(),
    }


def _discover_ks_final_validation_default(root: Path) -> dict[str, str]:
    """Keep default selection separate from explicit experimental admission."""
    relative = Path("src/dft/cuda_ks.cpp")
    source = re.sub(r"\s+", " ", _read(root / relative))
    policy = _read(root / "src/dft/cuda_ks_final_validation_policy.hpp")
    if not re.search(
        r"return\s+aos\s*>=\s*384\s*&&\s*spins\s*==\s*1\s*&&\s*direct_pbe0\s*&&\s*full_precision;",
        policy,
    ) or any(
        fragment not in policy
        for fragment in (
            "fock_exchange_coefficient == -0.125",
            "semilocal_exchange_scale == 0.75",
            "semilocal_correlation_scale == 1.0",
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
            "pbe0_rks_final_validation_composition(",
            "is_semilocal_family(functional, SemilocalFamily::Pbe)",
            "options.semilocal_exchange_scale, options.semilocal_correlation_scale",
            "has_exchange && !fitted_coulomb && !fitted_exchange && !has_range_correction",
            "!nonlocal_correlation;",
            "n, spins, direct_pbe0, !precision_schedule.any_lower_precision()",
            'std::getenv("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"), default_device_validation',
            "final_validation_partial && matrix_products.library_enabled()",
            "!final_stationary_weights_ready;",
        )
    ):
        raise ValueError("KS final-validation default owner or resource guard drifted")
    return {"dft-policy:ks-device-final-validation-auto": relative.as_posix()}


def _discover_md_j_default(root: Path) -> dict[str, str]:
    """Keep the admitted domain, optional cap and diagnostic opt-out reviewable."""
    relative = Path("src/scf/cuda/direct_jk.cpp")
    source = _read(root / relative)
    layout = _read(root / "src/scf/cuda/direct_md_j.hpp")
    if not re.search(r"kMdJResidentCap\s*=\s*128U\s*<<\s*20", layout):
        raise ValueError("MD-J default resident cap drifted from 128 MiB")
    if (
        'std::getenv("GENERATIVEQC_DISABLE_MD_J")' not in source
        or 'std::strcmp(disabled, "1") == 0' not in source
        or "host.nbf < 8" not in source
        or "return angular > 2;" not in source
        or "md_ready = !runtime::active_device_resource_ledger" not in source
    ):
        raise ValueError("MD-J default admission/opt-out guard drifted")
    return {
        "dft-policy:md-j-default": relative.as_posix(),
        "dft-policy:GENERATIVEQC_DISABLE_MD_J": relative.as_posix(),
    }


def _discover_rys_task_default(root: Path) -> dict[str, str]:
    """Guard shared Rys classes, target capabilities and portable fallback."""
    preference_relative = Path(
        "python/generativeqc_compiler/integral/production_rys_tasks.py"
    )
    preference = _read(root / preference_relative)
    preferred = re.search(r"preferred\s*=\s*(\{[^}]+\})", preference)
    if (
        preferred is None
        or ast.literal_eval(preferred.group(1))
        != {"psps", "ppps", "dsss", "dpss", "dsps", "ddss", "dsds", "dpps", "dspp"}
        or "if profile.portable:" not in preference
        or "schedule_candidates(integral, profile.target)" not in preference
        or "if schedule is None:" not in preference
        or 'profile.target.architecture != "sm_120"' in preference
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
        raise ValueError("initial-guess default drifted from guarded automatic MINAO")
    result["initial-guess:preliminary-scf"] = calculator_relative.as_posix()
    result["initial-guess:minao-auto"] = calculator_relative.as_posix()

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
    relative = Path("src/cc/solver.hpp")
    source = _read(root / relative)
    match = re.search(
        r"struct\s+SolverOptions\s*\{(?P<body>.*?)\n\};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("missing audited cc::SolverOptions")
    body = re.sub(r"//[^\n]*|/\*.*?\*/", " ", match.group("body"), flags=re.DOTALL)
    names = re.findall(
        r"\bbool\s+([A-Za-z0-9_]+)\s*(?:\{\s*(?:false)?\s*\}|=\s*false)\s*;",
        body,
    )
    return {f"cc-option:{name}": relative.as_posix() for name in names}


def _discover_cc_execution(root: Path) -> dict[str, str]:
    """Audit the energy-only source alias default without inventing a user flag."""
    relative = Path("src/cc/df_triples_cuda.cu")
    source = _read(root / relative)
    energy_start = source.find("DFCudaResult evaluate_df_cuda(")
    response_start = source.find("static DFCudaResponseResult pullback_df_cuda_impl(")
    if energy_start < 0 or response_start <= energy_start:
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
        fragment in source[response_start:]
        for fragment in ("MomentSourceMap", "energy_distinct_tile")
    ):
        raise ValueError("DF triples distinct-moment FP64 energy admission drifted")
    return {"cc-execution:df-triples-distinct-moments": relative.as_posix()}


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
        r"\bbool\s+([A-Za-z0-9_]+)\s*(?:\{\s*(?:false)?\s*\}|=\s*false)\s*;",
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
    """Keep the density-only default's qualified domain and finite work cap explicit."""
    relative = Path("src/methods/df_hf_guess.cpp")
    source = _read(root / relative)
    required = (
        'std::getenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS")',
        'value != "direct" && value != "auto"',
        "reference_quartet_direct(functions, maximum_angular->angular_momentum)",
        "guess.work_amortization_ratio < 1.0",
        "cartesian * cartesian",
        "guess.preparation_peak_bytes > guess.value_budget_bytes",
        "scf::plan_requested_density_fitting_tiles(",
        "!plan.stores_full_three_center",
        "system.charge != 0",
        "preliminary_iterations = 32",
        "options.max_iterations = preliminary_iterations",
        "options.energy_tolerance = 1e-4",
        "options.density_tolerance = 1e-4",
        "options.export_physical_reference = false",
        "512ULL << 20",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("DF-RHF preconvergence default or admission domain drifted")
    return {"initial-guess:df-rhf-preconvergence-auto": relative.as_posix()}


def _discover_rhf_phase_values_default(root: Path) -> dict[str, str]:
    """Keep automatic source construction tied to its qualified bounded domain."""
    policy_relative = Path("src/scf/cuda/reference_eri_policy.hpp")
    policy = _read(root / policy_relative)
    required = (
        '!value || std::string_view(value) == "auto"',
        "if (!cold_reference)",
        "maximum_angular != 3 || !generic_fock",
        "nbf < 64 || direct_nbf < 128",
        "maximum_iterations < 8",
    )
    owner_relative = Path("src/scf/cuda/rhf_resident_values.cpp")
    owner = _read(root / owner_relative)
    if any(fragment not in policy for fragment in required) or any(
        fragment not in owner
        for fragment in (
            'std::getenv("GENERATIVEQC_RHF_RESIDENT_VALUES")',
            "budget, 8ULL << 30",
            "device_overhead, 256ULL << 20",
        )
    ):
        raise ValueError("RHF phase-value auto default or admission domain drifted")
    return {"hf-runtime:GENERATIVEQC_RHF_RESIDENT_VALUES": owner_relative.as_posix()}


def discover_controls(root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for discovered in (
        _discover_scf_options(root),
        _discover_public_precision(root),
        _discover_runtime_controls(root),
        _discover_tensor_schedule(root),
        _discover_tensor_execution(root),
        _discover_force_active_ao(root),
        _discover_xc_point_batching(root),
        _discover_ks_final_validation_default(root),
        _discover_md_j_default(root),
        _discover_rys_task_default(root),
        _discover_direct_k_work_default(root),
        _discover_explicit_model_and_guess_choices(root),
        _discover_cc_options(root),
        _discover_cc_execution(root),
        _discover_response_options(root),
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
    except (SyntaxError, ValueError) as exc:
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
    return payload, validate_inventory(payload, root=root)


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
