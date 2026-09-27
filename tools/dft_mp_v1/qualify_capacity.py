"""Report frozen DFT-MP-v1 stationary CUDA capacity without running science.

The report resolves the committed inputs, bundled basis expansion, frozen grid,
current stationary plan and current fail-closed limits.  It performs no native
library load, CUDA initialization, SCF calculation or scientific compilation.
Passing this static report is therefore only a preflight result, never a
scientific qualification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SOURCE_REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = SOURCE_REPOSITORY / "python"

source_python = str(SOURCE_PYTHON)
if source_python in sys.path:
    sys.path.remove(source_python)
sys.path.insert(0, source_python)

import numpy as np
from vibeqc import Atom
from vibeqc import _generated_methods as generated_methods
from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
from vibeqc.basis_capabilities import resolved_basis_metadata
from vibeqc.calculator import _named_basis_record, _named_basis_shells
from vibeqc_compiler.dft.grid import GridSpec, MolecularGrid
from vibeqc_compiler.dft.plan import plan_tiles
from vibeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_COMPONENTS,
    STATIONARY_RUNTIME_SOURCE_NAMES,
    _qualified_aot_plan,
    _stationary_aot_name,
    load_stationary_aot_artifact,
    stationary_aot_contract_identity,
    stationary_runtime_sources,
)

SCHEMA = "vibeqc.dft-mp-v1.stationary-capacity.v1"
SEMILOCAL_FUNCTIONALS = {"lda": 0, "pbe": 1, "r2scan": 2}
SPARSE_SPHERICAL_COMPONENT_TERMS = {0: 1, 1: 3, 2: 8}
PRIMITIVE_SUM_DEFINITION = (
    "sum((int(row[2]) * len(expansion) for row, expansion in "
    "zip(aos, expansions, strict=True)))"
)
PRIMITIVE_RECORDS_DEFINITION = (
    "(1 + int(has_exchange)) * primitive_sum ** 4 + "
    "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
)
STATIONARY_OWNER = {
    "file": "python/vibeqc/_stationary_cuda.py",
    "function": "_complete_rks_cuda_gradient_diagnostic",
}


def _assert_local_imports() -> None:
    """Reject helpers already imported from an installed or foreign checkout."""

    helpers = {
        "Atom": Atom,
        "generated_methods": generated_methods,
        "complete_rks_cuda_gradient_diagnostic": complete_rks_cuda_gradient_diagnostic,
        "resolved_basis_metadata": resolved_basis_metadata,
        "_named_basis_record": _named_basis_record,
        "_named_basis_shells": _named_basis_shells,
        "GridSpec": GridSpec,
        "MolecularGrid": MolecularGrid,
        "plan_tiles": plan_tiles,
        "_qualified_aot_plan": _qualified_aot_plan,
        "load_stationary_aot_artifact": load_stationary_aot_artifact,
        "stationary_aot_contract_identity": stationary_aot_contract_identity,
        "stationary_runtime_sources": stationary_runtime_sources,
    }
    foreign = []
    for name, helper in helpers.items():
        module = helper if inspect.ismodule(helper) else inspect.getmodule(helper)
        source = None if module is None else getattr(module, "__file__", None)
        if source is None or not Path(source).resolve().is_relative_to(SOURCE_PYTHON):
            foreign.append(name)
    if foreign:
        raise RuntimeError(
            "capacity helper imported outside the tool checkout: "
            + ", ".join(sorted(foreign))
        )


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_limits(repository: Path) -> dict[str, Any]:
    """Read the current owner's literal shape caps and public work defaults."""

    source_path = repository / STATIONARY_OWNER["file"]
    source = source_path.read_text(encoding="utf-8")
    small = re.search(
        r"if not 1 <= na <= (?P<atoms>\d+) or not 1 <= n <= (?P<aos>\d+):\s+"
        r'raise ValueError\("CUDA diagnostic small-domain atom/AO cap exceeded"\)',
        source,
    )
    primitives = re.search(
        r"if not 1 <= basis\.nprimitive <= (?P<primitives>\d+):\s+"
        r'raise ValueError\("CUDA diagnostic primitive-topology cap exceeded"\)',
        source,
    )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == STATIONARY_OWNER["function"]
    ]
    if len(functions) != 1:
        raise RuntimeError("stationary CUDA admission owner is missing or ambiguous")
    owner = functions[0]
    definitions = {
        target.id: ast.unparse(node.value)
        for node in owner.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id in ("primitive_sum", "records")
    }
    if definitions.get("primitive_sum") != PRIMITIVE_SUM_DEFINITION:
        raise RuntimeError("stationary CUDA primitive-sum definition changed")
    if definitions.get("records") != PRIMITIVE_RECORDS_DEFINITION:
        raise RuntimeError("stationary CUDA primitive-record definition changed")
    whole_force_primitive_budget = next(
        (
            node
            for node in owner.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "records > max_primitive_records"
        ),
        None,
    )
    if small is None or primitives is None:
        raise RuntimeError(
            "stationary CUDA admission source no longer matches the audited gates"
        )
    if whole_force_primitive_budget is None:
        raise RuntimeError(
            "capacity qualifier requires a source-verified whole-force cumulative "
            "primitive-record admission gate"
        )

    signature = inspect.signature(complete_rks_cuda_gradient_diagnostic)

    def default(name: str) -> int:
        value = signature.parameters[name].default
        if type(value) is not int:
            raise RuntimeError(f"stationary CUDA {name} default is not an integer")
        return value

    messages = (
        "primitive work budget exceeded",
        "grid point work budget exceeded",
        "grid work budget exceeded",
        "stationary additional-host byte budget exceeded",
        "stationary additional-device budget exceeded",
    )
    positions = [source.find(message) for message in messages]
    if any(position < 0 for position in positions):
        raise RuntimeError("stationary CUDA admission messages are incomplete")
    if positions[:3] != sorted(positions[:3]):
        raise RuntimeError("stationary CUDA scalar work-gate order changed")

    return {
        "owner": STATIONARY_OWNER,
        "small_domain": {
            "atom_count": int(small.group("atoms")),
            "ao_count": int(small.group("aos")),
        },
        "basis_primitive_count": int(primitives.group("primitives")),
        "primitive_records": default("max_primitive_records"),
        "primitive_records_scope": "whole_force_cumulative",
        "primitive_sum_definition": PRIMITIVE_SUM_DEFINITION,
        "primitive_records_definition": PRIMITIVE_RECORDS_DEFINITION,
        "grid_points": default("max_grid_points"),
        "grid_pair_visits": default("max_grid_pair_visits"),
        "additional_device_bytes": default("max_device_bytes"),
        "additional_host_bytes": default("max_host_bytes"),
        "gate_order": [
            "small_domain_atom_ao_cap",
            "primitive_topology_cap",
            "primitive_work_budget",
            "grid_point_work_budget",
            "grid_pair_work_budget",
            "additional_device_budget",
            "additional_host_budget",
        ],
    }


def _grid_spec(payload: dict[str, Any]) -> GridSpec:
    values = dict(payload)
    values["element_radii"] = tuple(tuple(item) for item in values["element_radii"])
    return GridSpec(**values)


def _basis_shape(
    atoms: tuple[Atom, ...], charge: int, multiplicity: int
) -> tuple[dict[str, Any], Any]:
    shells = _named_basis_shells("def2-svp", atoms)
    if any(
        shell.angular_momentum not in SPARSE_SPHERICAL_COMPONENT_TERMS
        for shell in shells
    ):
        raise NotImplementedError(
            "DFT-MP-v1 capacity audit only covers the frozen s/p/d def2-SVP domain"
        )
    ao_count = sum(2 * shell.angular_momentum + 1 for shell in shells)
    primitive_count = sum(len(shell.primitives) for shell in shells)
    component_primitive_sum = sum(
        len(shell.primitives) * SPARSE_SPHERICAL_COMPONENT_TERMS[shell.angular_momentum]
        for shell in shells
    )
    packed_elements = 3 * len(atoms) + 2 * primitive_count + 16 * ao_count
    numeric_bytes = (
        2 * 8 * packed_elements
        + 32 * len(atoms)
        + 32 * len(shells)
        + 16 * primitive_count
    )
    synthetic = SimpleNamespace(
        nao=ao_count,
        natom=len(atoms),
        nprimitive=primitive_count,
        numeric_bytes=numeric_bytes,
        packed=np.empty(packed_elements),
    )
    record = _named_basis_record("def2-svp", "spherical")
    metadata = resolved_basis_metadata(
        record,
        shells,
        atoms,
        representation="spherical",
        charge=charge,
        multiplicity=multiplicity,
    )
    return (
        {
            "atom_count": len(atoms),
            "ao_count_spherical": ao_count,
            "shell_count": len(shells),
            "basis_primitive_count": primitive_count,
            "component_primitive_sum": component_primitive_sum,
        },
        (synthetic, metadata),
    )


def _method_resources(
    basis: Any,
    *,
    atom_count: int,
    functional: int,
    spin: str,
    limits: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    plan = _qualified_aot_plan(functional, spin)
    source_names = stationary_runtime_sources(plan)
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=1 if functional == 0 else 2,
        tile_points=256,
        active_ao_capacity=basis.nao,
        # Estimate even a losing case so the report cannot hide it by raising
        # from the live budget gate before recording the required bytes.
        budget_bytes=(1 << 63) - 1,
    )
    source_bytes = (
        8
        * (
            22 * 4096
            + 2 * basis.nprimitive
            + 4 * basis.nao
            + (579 + 3 * len(source_names)) * atom_count
            + 3 * 256
            + 2 * plan.spin_blocks * basis.nao * basis.nao
        )
        + 256
    )
    device_bound = grid_plan.peak_bytes + source_bytes
    host_bound = grid_plan.host_bytes + 8 * (
        34 * 4096
        + 4 * plan.spin_blocks * basis.nao * basis.nao
        + 120 * atom_count
        + 12 * (len(source_names) - len(STATIONARY_RUNTIME_SOURCE_NAMES)) * atom_count
        + 26 * 32
        + 3 * 256
        + 2 * basis.nprimitive
        + 4 * basis.nao
        + 80
    )
    return (
        {
            "grid_tile_peak_bytes": grid_plan.peak_bytes,
            "stationary_source_bytes": source_bytes,
            "additional_device_peak_bound": device_bound,
            "additional_device_budget": limits["additional_device_bytes"],
            "additional_host_numeric_bound": host_bound,
            "additional_host_budget": limits["additional_host_bytes"],
        },
        plan,
    )


def _failure(
    gate: str,
    message: str,
    *,
    required: Any,
    cap: Any,
    exceeded: list[str] | None = None,
) -> dict[str, Any]:
    result = {
        "gate": gate,
        "owner": STATIONARY_OWNER,
        "message": message,
        "required": required,
        "cap": cap,
    }
    if exceeded is not None:
        result["exceeded"] = exceeded
    return result


def _case_failures(
    shape: dict[str, Any],
    requirements: dict[str, Any],
    memory: dict[str, Any],
    limits: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    exceeded = [
        name
        for name, actual, cap in (
            (
                "atom_count",
                shape["atom_count"],
                limits["small_domain"]["atom_count"],
            ),
            (
                "ao_count",
                shape["ao_count_spherical"],
                limits["small_domain"]["ao_count"],
            ),
        )
        if actual > cap
    ]
    if exceeded:
        failures.append(
            _failure(
                "small_domain_atom_ao_cap",
                "CUDA diagnostic small-domain atom/AO cap exceeded",
                required={
                    "atom_count": shape["atom_count"],
                    "ao_count": shape["ao_count_spherical"],
                },
                cap=limits["small_domain"],
                exceeded=exceeded,
            )
        )
    if shape["basis_primitive_count"] > limits["basis_primitive_count"]:
        failures.append(
            _failure(
                "primitive_topology_cap",
                "CUDA diagnostic primitive-topology cap exceeded",
                required=shape["basis_primitive_count"],
                cap=limits["basis_primitive_count"],
            )
        )
    for key, gate, message in (
        (
            "primitive_records",
            "primitive_work_budget",
            "primitive work budget exceeded",
        ),
        ("grid_points", "grid_point_work_budget", "grid point work budget exceeded"),
        ("grid_pair_visits", "grid_pair_work_budget", "grid work budget exceeded"),
    ):
        if requirements[key] > limits[key]:
            failures.append(
                _failure(
                    gate,
                    message,
                    required=requirements[key],
                    cap=limits[key],
                )
            )
    if memory["additional_device_peak_bound"] > limits["additional_device_bytes"]:
        failures.append(
            _failure(
                "additional_device_budget",
                "stationary additional-device budget exceeded",
                required=memory["additional_device_peak_bound"],
                cap=limits["additional_device_bytes"],
            )
        )
    if memory["additional_host_numeric_bound"] > limits["additional_host_bytes"]:
        failures.append(
            _failure(
                "additional_host_budget",
                "stationary additional-host byte budget exceeded",
                required=memory["additional_host_numeric_bound"],
                cap=limits["additional_host_bytes"],
            )
        )
    return failures


def _source_package_inventory(repository: Path) -> None:
    cmake = (repository / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    names = ("lda_rks", "lda_uks", "pbe_rks", "pbe_uks", "r2scan_rks", "r2scan_uks")
    required = (
        "vibeqc_stationary_spd_primitives",
        'OUTPUT_NAME "vibeqc_stationary_${_vibeqc_stationary_name}_spd"',
        *names,
    )
    missing = [token for token in required if token not in cmake]
    if missing:
        raise RuntimeError(
            "stationary s/p/d package declaration is incomplete: " + ", ".join(missing)
        )


def _source_public_route(repository: Path) -> None:
    """Fail closed if the source predicates supporting the reported route move."""

    calculator = (repository / "python/vibeqc/calculator.py").read_text(
        encoding="utf-8"
    )
    batch = (repository / "python/vibeqc/batch.py").read_text(encoding="utf-8")
    required = {
        "python/vibeqc/calculator.py": (
            "semilocal_force = (",
            "supported_properties=self._capabilities.supported_properties",
            '| {"forces"}',
        ),
        "python/vibeqc/batch.py": (
            "packaged = (",
            "not state._source.method_ir.full_range_exact_exchange",
            "None if packaged else self._stationary_cuda_compiler()",
            '"aot_directory": native_library.parent if packaged else None',
        ),
    }
    missing = [
        f"{path}: {token}"
        for path, tokens in required.items()
        for token in tokens
        if token not in (calculator if path.endswith("calculator.py") else batch)
    ]
    if missing:
        raise RuntimeError(
            "public stationary CUDA route no longer matches the audited predicates: "
            + "; ".join(missing)
        )


def _artifact_verification(
    directory: Path | None,
    *,
    functional: int,
    spin: str,
    plan: Any,
) -> dict[str, Any]:
    if directory is None:
        return {"status": "not_checked", "detail": "no AOT directory supplied"}
    try:
        artifact = load_stationary_aot_artifact(
            directory,
            functional=functional,
            spin=spin,
            plan=plan,
            architecture="sm_120",
            component_domain=QUALIFIED_SPD_COMPONENTS,
        )
    except (FileNotFoundError, NotImplementedError, TypeError, ValueError) as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except KeyError as error:
        field = error.args[0] if error.args else "unknown"
        return {
            "status": "missing_or_invalid",
            "detail": f"missing AOT manifest field: {field}",
        }
    except AttributeError as error:
        return {
            "status": "missing_or_invalid",
            "detail": f"invalid AOT manifest schema: {error}",
        }
    return {
        "status": "verified",
        "detail": "packaged s/p/d AOT contract and binary identity verified",
        "library": str(Path(artifact.library).resolve()),
        "binary_sha256": artifact.metadata["binary_sha256"],
        "artifact_key": artifact.metadata["key"],
        "code_kinds": list(artifact.metadata["identity"]["target"]["code_kinds"]),
        "driver_ptx_jit_required": artifact.metadata["driver_ptx_jit_required"],
    }


def build_report(
    repository: Path,
    *,
    source_sha: str,
    aot_directory: Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository).resolve()
    _assert_local_imports()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be a full lowercase Git commit SHA")
    root = repository / "tools/dft_mp_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frozen_hash = manifest.get("contract_sha256")
    unhashed = dict(manifest)
    unhashed.pop("contract_sha256", None)
    if frozen_hash != _canonical_sha256(unhashed):
        raise ValueError("DFT-MP-v1 frozen contract hash mismatch")

    basis_pack = repository / "python/vibeqc/data/basis_pack.json"
    basis_pack_sha = _lf_sha256(basis_pack.read_bytes())
    basis_pack_matches = basis_pack_sha == manifest["basis"]["basis_pack_sha256"]
    if not basis_pack_matches:
        raise ValueError(
            "current basis pack differs from the frozen DFT-MP-v1 contract"
        )

    limits = _source_limits(repository)
    _source_package_inventory(repository)
    _source_public_route(repository)
    grid_spec = _grid_spec(manifest["model"]["grid_spec"])

    required_rows = [
        row
        for row in manifest["rows"]
        if row["required"]
        and row["method"] in SEMILOCAL_FUNCTIONALS
        and row["level"] == "fp64_energy_forces"
    ]
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in required_rows:
        rows_by_case.setdefault(row["case"], []).append(row)

    cases = []
    case_work: dict[str, dict[str, Any]] = {}
    ao_counts_match = True
    for case_name, frozen in sorted(manifest["cases"].items()):
        input_path = root / frozen["input"]
        raw = input_path.read_bytes()
        if _lf_sha256(raw) != frozen["input_sha256"]:
            raise ValueError(f"frozen input hash mismatch for {case_name}")
        value = json.loads(raw)
        atoms = tuple(Atom.from_value(atom) for atom in value["atoms"])
        changed_path = root / frozen["changed_input"]
        changed_raw = changed_path.read_bytes()
        if _lf_sha256(changed_raw) != frozen["changed_input_sha256"]:
            raise ValueError(f"frozen changed input hash mismatch for {case_name}")
        changed_value = json.loads(changed_raw)
        changed_atoms = tuple(Atom.from_value(atom) for atom in changed_value["atoms"])
        shape, (basis, basis_metadata) = _basis_shape(
            atoms, value["charge"], value["multiplicity"]
        )
        ao_match = shape["ao_count_spherical"] == frozen["ao_count_spherical"]
        ao_counts_match = ao_counts_match and ao_match
        if not ao_match:
            raise ValueError(
                f"actual spherical AO count differs from manifest for {case_name}"
            )
        grid = MolecularGrid(
            atoms,
            grid_spec,
            charge=value["charge"],
            multiplicity=value["multiplicity"],
        )
        if grid.identity != frozen["grid_identity"]:
            raise ValueError(
                f"current grid identity differs from manifest for {case_name}"
            )
        changed_grid = MolecularGrid(
            changed_atoms,
            grid_spec,
            charge=changed_value["charge"],
            multiplicity=changed_value["multiplicity"],
        )
        if changed_grid.identity != frozen["changed_grid_identity"]:
            raise ValueError(
                f"current changed grid identity differs from manifest for {case_name}"
            )

        atom_pairs = shape["atom_count"] * (shape["atom_count"] - 1) // 2
        primitive_sum = shape["component_primitive_sum"]
        requirements = {
            "primitive_records": primitive_sum**4
            + (shape["atom_count"] + 2) * primitive_sum**2
            + atom_pairs,
            "grid_points": grid.npoint,
            "grid_pair_visits": (1 + 2 * grid.npoint) * atom_pairs,
        }
        method_memory = {}
        method_plans = {}
        for row in rows_by_case.get(case_name, ()):
            spin = "unpolarized" if row["spin"] == "rks" else "polarized"
            key = f"{row['method']}/{row['spin']}"
            if key not in method_memory:
                method_memory[key], method_plans[key] = _method_resources(
                    basis,
                    atom_count=shape["atom_count"],
                    functional=SEMILOCAL_FUNCTIONALS[row["method"]],
                    spin=spin,
                    limits=limits,
                )
        maximum_memory = {
            "additional_device_peak_bound": max(
                item["additional_device_peak_bound"] for item in method_memory.values()
            ),
            "additional_host_numeric_bound": max(
                item["additional_host_numeric_bound"] for item in method_memory.values()
            ),
        }
        failures = _case_failures(shape, requirements, maximum_memory, limits)
        record = {
            "id": case_name,
            "classification": frozen["classification"],
            "charge": value["charge"],
            "multiplicity": value["multiplicity"],
            "shape": {**shape, "grid_points": grid.npoint},
            "identities": {
                "input_sha256": frozen["input_sha256"],
                "changed_input_sha256": frozen["changed_input_sha256"],
                "basis": basis_metadata["mathematical_identity"],
                "basis_pack_sha256": basis_pack_sha,
                "ao_representation": "real_spherical/libcint-PySCF-order",
                "grid": grid.identity,
                "changed_grid": changed_grid.identity,
            },
            "requirements": {
                **requirements,
                "memory_by_method_spin": method_memory,
            },
            "requested_rows": sorted(
                row["id"] for row in rows_by_case.get(case_name, ())
            ),
            "admission": {
                "outcome": "blocked" if failures else "passes_static_stationary_caps",
                "first_blocker": failures[0] if failures else None,
                "failures": failures,
                "scope": (
                    "static source/resource preflight only; no SCF, CUDA execution, "
                    "AOT binary, numerical, or performance qualification"
                ),
            },
        }
        cases.append(record)
        case_work[case_name] = {
            "basis": basis,
            "plans": method_plans,
            "memory": method_memory,
            "record": record,
        }

    artifact_cache: dict[tuple[int, str], dict[str, Any]] = {}
    rows = []
    for frozen_row in sorted(required_rows, key=lambda item: item["id"]):
        method = frozen_row["method"]
        functional = SEMILOCAL_FUNCTIONALS[method]
        spin = "unpolarized" if frozen_row["spin"] == "rks" else "polarized"
        method_key = f"{method}/{frozen_row['spin']}"
        plan = case_work[frozen_row["case"]]["plans"][method_key]
        aot_key = (functional, spin)
        if aot_key not in artifact_cache:
            artifact_cache[aot_key] = _artifact_verification(
                aot_directory,
                functional=functional,
                spin=spin,
                plan=plan,
            )
        selector = f"{method}-{'rks' if spin == 'unpolarized' else 'uks'}"
        native_properties = list(
            generated_methods.METHOD_METADATA[selector]["properties"]
        )
        rows.append(
            {
                **frozen_row,
                "stationary_plan": {
                    "identity": plan.identity,
                    "source_names": list(stationary_runtime_sources(plan)),
                    "component_domain": list(QUALIFIED_SPD_COMPONENTS),
                },
                "resource_requirements": {
                    "atom_count": case_work[frozen_row["case"]]["record"]["shape"][
                        "atom_count"
                    ],
                    "ao_count_spherical": case_work[frozen_row["case"]]["record"][
                        "shape"
                    ]["ao_count_spherical"],
                    **{
                        key: case_work[frozen_row["case"]]["record"]["requirements"][
                            key
                        ]
                        for key in (
                            "primitive_records",
                            "grid_points",
                            "grid_pair_visits",
                        )
                    },
                    **case_work[frozen_row["case"]]["memory"][method_key],
                },
                "public_capability": {
                    "forces": True,
                    "native_registry_properties": native_properties,
                    "promotion": "python semilocal direct-CUDA stationary-force predicate",
                    "owner": "python/vibeqc/calculator.py::Calculator.__init__",
                    "source_audited": True,
                },
                "public_route": {
                    "scientific_runtime_compilation_required": False,
                    "selection": "all-electron semilocal packaged stationary CUDA",
                    "owner": "python/vibeqc/batch.py::_public_dft_cuda_force",
                    "missing_aot_behavior": "fail closed; no NVCC fallback",
                    "source_audited": True,
                },
                "packaged_aot": {
                    "name": _stationary_aot_name(
                        functional,
                        spin,
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                    ),
                    "architecture": "sm_120",
                    "source_package_declared": True,
                    "source_owner": "cmake/VibeQCCuda.cmake",
                    "contract_identity": stationary_aot_contract_identity(
                        functional,
                        spin=spin,
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                    ),
                    "binary_verification": artifact_cache[aot_key],
                },
                "admission": case_work[frozen_row["case"]]["record"]["admission"],
            }
        )

    owner_files = (
        "python/vibeqc/_stationary_cuda.py",
        "python/vibeqc/calculator.py",
        "python/vibeqc/batch.py",
        "python/vibeqc_compiler/method/stationary_cuda.py",
        "cmake/VibeQCCuda.cmake",
    )
    blocked = sum(row["admission"]["outcome"] == "blocked" for row in rows)
    return {
        "schema": SCHEMA,
        "source": {
            "sha": source_sha,
            "runtime_owner_sha256": {
                path: _lf_sha256((repository / path).read_bytes())
                for path in owner_files
            },
        },
        "contract": {
            "id": manifest["contract_id"],
            "version": manifest["version"],
            "sha256": frozen_hash,
        },
        "basis": {
            "name": manifest["basis"]["name"],
            "representation": manifest["basis"]["representation"],
            "basis_pack_sha256": basis_pack_sha,
            "basis_pack_sha256_match": basis_pack_matches,
            "manifest_ao_counts_match": ao_counts_match,
            "component_primitive_sum_derivation": (
                "sum(shell primitive count * sparse public-AO Cartesian term count); "
                "s=1, p=3, spherical d=8 from src/molecule/basis.cpp"
            ),
        },
        "admission_limits": limits,
        "aot_binary_directory": None
        if aot_directory is None
        else str(Path(aot_directory).resolve()),
        "cases": cases,
        "rows": rows,
        "summary": {
            "required_semilocal_fp64_force_rows": len(rows),
            "statically_blocked_rows": blocked,
            "rows_passing_static_stationary_caps": len(rows) - blocked,
            "scientific_qualification": "NOT_RUN",
        },
    }


def _clean_git_sha(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("capacity report requires a clean Git worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("Git did not return a full source SHA")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=SOURCE_REPOSITORY,
        help="checkout containing this tool; other repositories are rejected",
    )
    parser.add_argument(
        "--aot-directory",
        type=Path,
        help="optionally verify the actual sm_120 packaged s/p/d binaries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    payload = build_report(
        repository,
        source_sha=_clean_git_sha(repository),
        aot_directory=args.aot_directory,
    )
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
