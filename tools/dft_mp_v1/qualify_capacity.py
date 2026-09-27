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
SPD_EXPANSION_CONTRACT_SHA256 = (
    "f0d9be746f30067f6dba8293bcc35a9dc06db74037a6c76d322d21051bc61334"
)
AO_PACKER_CONTRACT_SHA256 = (
    "07858ba7f9a78fe6348bbcb9430eb4f8321db8774ea3ce1ecef495629abe2a1c"
)
AO_PACK_BRIDGE_CONTRACT_SHA256 = (
    "c5c8a0181075e7d171e1d189c875d5cc9e69467cb069b13267f91e73b1e1dd7e"
)
NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256 = (
    "c08f40375765a126782325dd4d03ded0ea9bf3caa25f23953a8a5cdc5c75c01b"
)
STATIONARY_LAYOUT_CONTRACT_SHA256 = (
    "2f1bb49d43cbfd93e65f69c769ec26c9d04b84bfe5e4be2d705b1262a386b030"
)
NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256 = (
    "23785e9e006f9a100b4fecc690e6936a348581beba073507c154b185564832c6"
)
PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256 = (
    "069f7414cb61d55c543d5829b5d793aed9b882318fbdeb157f4103c486ea0e71"
)
PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256 = (
    "3ea6ef6ce2c0d8ea5849161ef4ccd706f987261e2ceacdd13c7cfb185525d2d8"
)
PUBLIC_CUDA_FORCE_METHOD_CONTRACT_SHA256 = (
    "1d0df874a38441e94168f329e8055f9b9d27e7f26649e15e106db9ab7694c79e"
)
PYTHON_GRID_CONTRACT_SHA256 = (
    "03a43444cd793167823c0c30c0b66b51c2a464d8f65946118dd781813dc7f0a4"
)
QUADRATURE_LAYOUT_CONTRACT_SHA256 = (
    "7ad4c84286cce70329233f7aa2dcaf2b934e2e7cf46137cc3ed32cc6076754c3"
)
NATIVE_CUDA_GRID_CONTRACT_SHA256 = (
    "eed5f5bff7c67622c75fd0d21448b66502b581c102637036a748b459a288a41b"
)
NATIVE_GRID_ROUTE_CONTRACT_SHA256 = (
    "cc639c77e261810ff35a30f3bf4967a398b6408e72f86446f94a4d5e760e1a42"
)
NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256 = (
    "92cd50078b7a96f371ed8d4fcdb77930b8c472134bd1e97bba803ac445d85867"
)
PRIMITIVE_SUM_DEFINITION = (
    "sum((int(row[2]) * len(expansion) for row, expansion in "
    "zip(aos, expansions, strict=True)))"
)
PRIMITIVE_RECORDS_DEFINITION = (
    "(1 + int(has_exchange)) * primitive_sum ** 4 + "
    "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
)
GRID_PAIR_VISITS_DEFINITION = "(1 + 2 * len(state.grid.points)) * na * (na - 1) // 2"
SOURCE_BYTES_DEFINITION = (
    "8 * (22 * primitive_tile + 2 * basis.nprimitive + 4 * n + "
    "(579 + 3 * len(source_names)) * na + 3 * tile_points + "
    "2 * plan.spin_blocks * n * n) + 256"
)
HOST_BOUND_DEFINITION = (
    "grid_plan.host_bytes + 8 * (34 * primitive_tile + "
    "4 * plan.spin_blocks * n * n + 120 * na + "
    "12 * (len(source_names) - len(_SOURCE_NAMES)) * na + "
    "26 * integral_terms + 3 * tile_points + 2 * basis.nprimitive + "
    "4 * n + 80) + max((tp.host_bytes for tp in tensor_plans.values()), default=0)"
)
AVAILABLE_DEVICE_BYTES_DEFINITION = (
    "max_device_bytes - grid_plan.peak_bytes - source_bytes"
)
GATE_PREDICATES = {
    "primitive_records": "records > max_primitive_records",
    "grid_points": "len(state.grid.points) > max_grid_points",
    "grid_pair_visits": "pair_visits > max_grid_pair_visits",
    "additional_device": "available <= 0",
    "additional_host": "host_bound > max_host_bytes",
}
BASIS_PACKED_CAPACITY_DEFINITION = (
    "np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)"
)
BASIS_NUMERIC_CAPACITY_DEFINITION = (
    "2 * self.packed.nbytes + 32 * self.natom + "
    "32 * len(self.shells) + 16 * self.nprimitive"
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


def _source_node_sha256(source: str, node: ast.AST) -> str:
    """Hash the selected source span without Python-version AST formatting."""

    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("source contract segment is unavailable")
    return _lf_sha256(segment.encode())


def _source_span_sha256(
    source: str,
    *,
    begin: str,
    end: str,
    label: str,
) -> str:
    try:
        start = source.index(begin)
        stop = source.index(end, start)
    except ValueError as error:
        raise RuntimeError(f"{label} source contract is missing") from error
    return _lf_sha256(source[start:stop].encode())


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
    definition_nodes = {
        name: [
            node
            for node in owner.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        for name in (
            "primitive_sum",
            "records",
            "pair_visits",
            "source_bytes",
            "available",
            "host_bound",
        )
    }
    if any(len(nodes) != 1 for nodes in definition_nodes.values()):
        raise RuntimeError(
            "stationary CUDA capacity definitions are missing or ambiguous"
        )
    definitions = {
        name: ast.unparse(nodes[0].value) for name, nodes in definition_nodes.items()
    }
    expected_definitions = {
        "primitive_sum": PRIMITIVE_SUM_DEFINITION,
        "records": PRIMITIVE_RECORDS_DEFINITION,
        "pair_visits": GRID_PAIR_VISITS_DEFINITION,
        "source_bytes": SOURCE_BYTES_DEFINITION,
        "available": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "host_bound": HOST_BOUND_DEFINITION,
    }
    definition_labels = {
        "primitive_sum": "primitive-sum",
        "records": "primitive-record",
        "pair_visits": "grid-pair-visits",
        "source_bytes": "source-bytes",
        "available": "available-device-bytes",
        "host_bound": "host-bound",
    }
    for name, expected in expected_definitions.items():
        if definitions[name] != expected:
            raise RuntimeError(
                f"stationary CUDA {definition_labels[name]} definition changed"
            )
    direct_if_tests = [
        ast.unparse(node.test) for node in owner.body if isinstance(node, ast.If)
    ]
    gate_labels = {
        "primitive_records": "whole-force cumulative primitive-record",
        "grid_points": "grid-point",
        "grid_pair_visits": "grid-pair-visits",
        "additional_device": "positive additional-device remainder",
        "additional_host": "additional-host",
    }
    for name, predicate in GATE_PREDICATES.items():
        if direct_if_tests.count(predicate) != 1:
            raise RuntimeError(f"stationary CUDA {gate_labels[name]} predicate changed")
    if small is None or primitives is None:
        raise RuntimeError(
            "stationary CUDA admission source no longer matches the audited gates"
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
        "grid_pair_visits_definition": GRID_PAIR_VISITS_DEFINITION,
        "source_bytes_definition": SOURCE_BYTES_DEFINITION,
        "host_bound_definition": HOST_BOUND_DEFINITION,
        "available_device_bytes_definition": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "additional_device_admission": (
            "additional_device_peak_bound < additional_device_budget"
        ),
        "gate_predicates": dict(GATE_PREDICATES),
        "tile_points": default("tile_points"),
        "primitive_tile": default("primitive_tile"),
        "integral_terms": default("integral_terms"),
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


def _basis_layout_contract(repository: Path) -> dict[str, str]:
    source = (repository / "python/vibeqc_compiler/dft/ao.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeAO"
    ]
    if len(classes) != 1:
        raise RuntimeError("NativeAO capacity owner is missing or ambiguous")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("NativeAO capacity constructor is missing or ambiguous")
    constructor_digest = _source_node_sha256(source, constructors[0])
    packed = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "packed"
    ]
    numeric = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and ast.unparse(node.targets[0]) == "self.numeric_bytes"
    ]
    if packed != [BASIS_PACKED_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO packed capacity definition changed")
    if numeric != [BASIS_NUMERIC_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO numeric capacity definition changed")
    if constructor_digest != NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256:
        raise RuntimeError("NativeAO constructor contract changed")
    return {
        "packed_capacity_definition": packed[0],
        "numeric_capacity_definition": numeric[0],
        "native_ao_constructor_contract_sha256": constructor_digest,
    }


def _spd_expansion_contract(repository: Path) -> dict[str, Any]:
    """Bind frozen s/p/d term counts to the native public-AO expansion table."""

    source = (repository / "src/molecule/basis.cpp").read_text(encoding="utf-8")
    try:
        begin = source.index("std::vector<CartesianComponent> cartesian_components")
        end = source.index("  if (l == 3)", begin)
    except ValueError as error:
        raise RuntimeError("native s/p/d expansion owner is missing") from error
    digest = _lf_sha256(source[begin:end].encode())
    if digest != SPD_EXPANSION_CONTRACT_SHA256:
        raise RuntimeError("native s/p/d expansion contract changed")

    packer_source = (repository / "src/dft/ao_grid.cpp").read_text(encoding="utf-8")
    try:
        packer_begin = packer_source.index("AoBasis::AoBasis(")
        packer_end = packer_source.index("void AoBasis::evaluate(", packer_begin)
    except ValueError as error:
        raise RuntimeError("native packed-AO owner is missing") from error
    packer_digest = _lf_sha256(packer_source[packer_begin:packer_end].encode())
    if packer_digest != AO_PACKER_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    bridge_source = (repository / "src/dft/bridge.cpp").read_text(encoding="utf-8")
    try:
        bridge_begin = bridge_source.index("VIBEQC_API int vibeqc_grid_basis_create_v1")
        bridge_end = bridge_source.index(
            "VIBEQC_API int vibeqc_grid_ao_v1", bridge_begin
        )
    except ValueError as error:
        raise RuntimeError("native AO pack bridge is missing") from error
    bridge_digest = _lf_sha256(bridge_source[bridge_begin:bridge_end].encode())
    if bridge_digest != AO_PACK_BRIDGE_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    ao_count_digest = _source_span_sha256(
        source,
        begin="std::size_t ao_count(",
        end="std::size_t cartesian_ao_count(",
        label="native spherical AO count",
    )
    if ao_count_digest != NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256:
        raise RuntimeError("native spherical AO count contract changed")

    stationary_source = (repository / "python/vibeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_tree = ast.parse(stationary_source)
    layouts = [
        node
        for node in stationary_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_layout"
    ]
    if len(layouts) != 1:
        raise RuntimeError("stationary layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(stationary_source, layouts[0])
    if layout_digest != STATIONARY_LAYOUT_CONTRACT_SHA256:
        raise RuntimeError("stationary layout contract changed")

    return {
        "spd_expansion_contract_sha256": digest,
        "ao_packer_contract_sha256": packer_digest,
        "ao_pack_bridge_contract_sha256": bridge_digest,
        "native_spherical_ao_count_contract_sha256": ao_count_digest,
        "stationary_layout_contract_sha256": layout_digest,
        "sparse_spherical_component_terms": {
            "s": SPARSE_SPHERICAL_COMPONENT_TERMS[0],
            "p": SPARSE_SPHERICAL_COMPONENT_TERMS[1],
            "d": SPARSE_SPHERICAL_COMPONENT_TERMS[2],
        },
        "spd_expansion_owner": (
            "basis.cpp::cartesian_components+ao_expansions -> "
            "ao_grid.cpp::AoBasis -> bridge.cpp::vibeqc_grid_basis_pack_v1 -> "
            "_stationary_cuda.py::_layout"
        ),
    }


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
    packed = np.empty(3 * len(atoms) + 2 * primitive_count + 16 * ao_count)
    numeric_bytes = (
        2 * packed.nbytes + 32 * len(atoms) + 32 * len(shells) + 16 * primitive_count
    )
    synthetic = SimpleNamespace(
        nao=ao_count,
        natom=len(atoms),
        nprimitive=primitive_count,
        numeric_bytes=numeric_bytes,
        packed=packed,
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
    tile_points = limits["tile_points"]
    primitive_tile = limits["primitive_tile"]
    integral_terms = limits["integral_terms"]
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=1 if functional == 0 else 2,
        tile_points=tile_points,
        active_ao_capacity=basis.nao,
        # Estimate even a losing case so the report cannot hide it by raising
        # from the live budget gate before recording the required bytes.
        budget_bytes=(1 << 63) - 1,
    )
    source_bytes = (
        8
        * (
            22 * primitive_tile
            + 2 * basis.nprimitive
            + 4 * basis.nao
            + (579 + 3 * len(source_names)) * atom_count
            + 3 * tile_points
            + 2 * plan.spin_blocks * basis.nao * basis.nao
        )
        + 256
    )
    device_bound = grid_plan.peak_bytes + source_bytes
    host_bound = grid_plan.host_bytes + 8 * (
        34 * primitive_tile
        + 4 * plan.spin_blocks * basis.nao * basis.nao
        + 120 * atom_count
        + 12 * (len(source_names) - len(STATIONARY_RUNTIME_SOURCE_NAMES)) * atom_count
        + 26 * integral_terms
        + 3 * tile_points
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
    if memory["additional_device_peak_bound"] >= limits["additional_device_bytes"]:
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


def _source_public_route(repository: Path) -> dict[str, str]:
    """Fail closed if the source predicates supporting the reported route move."""

    calculator = (repository / "python/vibeqc/calculator.py").read_text(
        encoding="utf-8"
    )
    batch = (repository / "python/vibeqc/batch.py").read_text(encoding="utf-8")
    calculator_tree = ast.parse(calculator)
    calculator_classes = [
        node
        for node in calculator_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    ]
    if len(calculator_classes) != 1:
        raise RuntimeError("public Calculator owner is missing or ambiguous")
    constructors = [
        node
        for node in calculator_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("public Calculator constructor is missing or ambiguous")
    semilocal_assignments = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "semilocal_force"
            for target in node.targets
        )
    ]
    force_promotions = [
        node
        for node in constructors[0].body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Name) and item.id == "semilocal_force"
            for item in ast.walk(node.test)
        )
    ]
    if len(semilocal_assignments) != 1 or len(force_promotions) != 1:
        raise RuntimeError("public semilocal force capability owner is ambiguous")
    semilocal_digest = _source_node_sha256(calculator, semilocal_assignments[0])
    promotion_digest = _source_node_sha256(calculator, force_promotions[0])
    if semilocal_digest != PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256:
        raise RuntimeError("public semilocal force predicate changed")
    if promotion_digest != PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256:
        raise RuntimeError("public force capability promotion changed")

    batch_tree = ast.parse(batch)
    batch_classes = [
        node
        for node in batch_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedBatch"
    ]
    if len(batch_classes) != 1:
        raise RuntimeError("public PreparedBatch owner is missing or ambiguous")
    force_methods = [
        node
        for node in batch_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_public_dft_cuda_force"
    ]
    if len(force_methods) != 1:
        raise RuntimeError("public CUDA force route is missing or ambiguous")
    batch_digest = _source_node_sha256(batch, force_methods[0])
    if batch_digest != PUBLIC_CUDA_FORCE_METHOD_CONTRACT_SHA256:
        raise RuntimeError("public CUDA force route changed")
    return {
        "semilocal_force_predicate_sha256": semilocal_digest,
        "force_capability_promotion_sha256": promotion_digest,
        "cuda_force_method_sha256": batch_digest,
    }


def _grid_count_contract(repository: Path) -> dict[str, str]:
    """Bind the source-only point count to the native CUDA grid owner."""

    python_grid = (repository / "python/vibeqc_compiler/dft/grid.py").read_text(
        encoding="utf-8"
    )
    python_tree = ast.parse(python_grid)
    grid_classes = [
        node
        for node in python_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MolecularGrid"
    ]
    if len(grid_classes) != 1:
        raise RuntimeError("source-only MolecularGrid owner is missing or ambiguous")
    post_init = [
        node
        for node in grid_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    ]
    if len(post_init) != 1:
        raise RuntimeError(
            "source-only MolecularGrid constructor is missing or ambiguous"
        )
    python_digest = _source_node_sha256(python_grid, post_init[0])

    quadrature_source = (
        repository / "python/vibeqc_compiler/xc/quadrature_cuda.py"
    ).read_text(encoding="utf-8")
    quadrature_tree = ast.parse(quadrature_source)
    layouts = [
        node
        for node in quadrature_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_LAYOUT"
            for target in node.targets
        )
    ]
    if len(layouts) != 1:
        raise RuntimeError("generated quadrature layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(quadrature_source, layouts[0])

    cuda_source = (repository / "src/dft/cuda_quadrature.cu").read_text(
        encoding="utf-8"
    )
    cuda_digest = _source_span_sha256(
        cuda_source,
        begin="MolecularGrid MolecularGrid::from_cuda(",
        end="}  // namespace vibeqc::dft",
        label="native CUDA grid",
    )
    route_source = (repository / "src/methods/dft_method.cpp").read_text(
        encoding="utf-8"
    )
    route_digest = _source_span_sha256(
        route_source,
        begin="dft::MolecularGrid ks_molecular_grid(",
        end="class KsPreparedCalculation",
        label="native CUDA grid route",
    )
    header_source = (repository / "src/dft/grid.hpp").read_text(encoding="utf-8")
    point_count_digest = _source_span_sha256(
        header_source,
        begin="  std::size_t point_count()",
        end="  const std::vector<double>& points()",
        label="native grid point-count publication",
    )
    contracts = {
        "source_only_molecular_grid_sha256": python_digest,
        "generated_quadrature_layout_sha256": layout_digest,
        "native_cuda_grid_sha256": cuda_digest,
        "native_cuda_grid_route_sha256": route_digest,
        "native_grid_point_count_sha256": point_count_digest,
    }
    expected = {
        "source_only_molecular_grid_sha256": PYTHON_GRID_CONTRACT_SHA256,
        "generated_quadrature_layout_sha256": QUADRATURE_LAYOUT_CONTRACT_SHA256,
        "native_cuda_grid_sha256": NATIVE_CUDA_GRID_CONTRACT_SHA256,
        "native_cuda_grid_route_sha256": NATIVE_GRID_ROUTE_CONTRACT_SHA256,
        "native_grid_point_count_sha256": NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256,
    }
    moved = [name for name, digest in contracts.items() if digest != expected[name]]
    if moved:
        raise RuntimeError("grid point-count contract changed: " + ", ".join(moved))
    return {
        **contracts,
        "point_count_definition": (
            "atom_count * radial_points * angular_polar * angular_azimuth"
        ),
    }


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
    except OSError as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except (NotImplementedError, TypeError, ValueError) as error:
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


def _build_report(
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
    basis_layout = _basis_layout_contract(repository)
    spd_expansion = _spd_expansion_contract(repository)
    _source_package_inventory(repository)
    public_route = _source_public_route(repository)
    grid_contract = _grid_count_contract(repository)
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
        native_grid_points = (
            len(atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        changed_native_grid_points = (
            len(changed_atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        if grid.npoint != native_grid_points or (
            changed_grid.npoint != changed_native_grid_points
        ):
            raise RuntimeError("source-only grid count disagrees with native contract")

        atom_pairs = shape["atom_count"] * (shape["atom_count"] - 1) // 2
        primitive_sum = shape["component_primitive_sum"]
        requirements = {
            "primitive_records": primitive_sum**4
            + (shape["atom_count"] + 2) * primitive_sum**2
            + atom_pairs,
            "grid_points": native_grid_points,
            "grid_pair_visits": (1 + 2 * native_grid_points) * atom_pairs,
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
            "shape": {**shape, "grid_points": native_grid_points},
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
        "python/vibeqc_compiler/dft/ao.py",
        "python/vibeqc_compiler/dft/grid.py",
        "python/vibeqc_compiler/xc/quadrature_cuda.py",
        "src/molecule/basis.cpp",
        "src/dft/ao_grid.cpp",
        "src/dft/bridge.cpp",
        "src/dft/grid.hpp",
        "src/dft/cuda_quadrature.cu",
        "src/methods/dft_method.cpp",
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
            **basis_layout,
            **spd_expansion,
            "component_primitive_sum_derivation": (
                "sum(shell primitive count * sparse public-AO Cartesian term count); "
                "s=1, p=3, spherical d=8 from src/molecule/basis.cpp"
            ),
        },
        "grid": grid_contract,
        "public_route": public_route,
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


def _clean_git_sha(
    repository: Path,
    *,
    ignored_path: Path | None = None,
) -> str:
    status_command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
    ]
    if ignored_path is not None:
        resolved_ignored = Path(ignored_path).resolve()
        if resolved_ignored.is_relative_to(repository):
            relative = resolved_ignored.relative_to(repository).as_posix()
            status_command.append(f":(exclude,literal){relative}")
    status = subprocess.run(
        status_command,
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


def build_report(
    repository: Path,
    *,
    aot_directory: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a report whose source identity is the clean tool-checkout HEAD."""

    repository = Path(repository).resolve()
    _assert_local_imports()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    return _build_report(
        repository,
        source_sha=_clean_git_sha(repository, ignored_path=output_path),
        aot_directory=aot_directory,
    )


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
    output_path = None if args.output is None else args.output.resolve()
    payload = build_report(
        repository,
        aot_directory=args.aot_directory,
        output_path=output_path,
    )
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
