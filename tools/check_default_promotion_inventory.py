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


def discover_controls(root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for discovered in (
        _discover_scf_options(root),
        _discover_public_precision(root),
        _discover_runtime_controls(root),
        _discover_tensor_schedule(root),
        _discover_tensor_execution(root),
        _discover_force_active_ao(root),
        _discover_explicit_model_and_guess_choices(root),
        _discover_cc_options(root),
        _discover_response_options(root),
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
