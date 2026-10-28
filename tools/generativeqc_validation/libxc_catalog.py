"""Offline native Libxc receipt ingestion, separate from production admission.

Only authenticated producer results become evidence for the existing capability
DAG. Native execution scope remains an additional intersection: a stage pass is
never a promise about another geometry, precision, Hamiltonian or derivative.
"""

from __future__ import annotations

import json
import lzma
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.xc.capability_catalog import capability_summary
from generativeqc_compiler.xc.capability_resolution import CapabilityNotQualified
from generativeqc_compiler.xc.density_work import DENSITY_WORK_DOMAIN
from generativeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    resolve_endpoint_capability,
)
from generativeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    available_capabilities,
)
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN
from generativeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools.generate_build_identity import _inventory, _source_identity
from tools.generativeqc_validation.libxc_native_domain import native_domain_stage
from tools.qualify_libxc_forces import force_stage_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

SCHEMA = "generativeqc.libxc-native-catalog.v1"
BUNDLE_SCHEMA = "generativeqc.libxc-native-campaign.v1"
# These are input dependencies for stale-evidence detection, not source spelling
# tests. A change may deliberately demote claims, with an exact transition ack.
SOURCE_DIRS = (
    "src",
    "include",
    "cmake",
    "python/generativeqc",
    "python/generativeqc_compiler",
)
SOURCE_SUFFIXES = {
    ".py",
    ".json",
    ".cpp",
    ".hpp",
    ".h",
    ".cu",
    ".cuh",
    ".c",
    ".cmake",
    ".in",
}


def source_inventory(root: Path) -> dict[str, str]:
    """Bind native/compiler/frontend inputs, including additions and removals."""
    files = {root / "CMakeLists.txt"}
    for directory in SOURCE_DIRS:
        files.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
        )
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)}


def load_json(path: Path) -> Any:
    """Read a compact receipt without executing any retained driver."""
    data = path.read_bytes()
    if path.suffix == ".xz":
        data = lzma.decompress(data)
    return json.loads(data)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic compact evidence, preserving full floating precision."""
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(data) if path.suffix == ".xz" else data)


def require(condition: Any, reason: str) -> None:
    """Keep fail-closed validation active even under Python optimization."""
    if not condition:
        raise ValueError(reason)


def _close(actual: Any, expected: Any, tolerance: float, label: str) -> None:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(left.shape == right.shape and left.size > 0, f"{label}: shape mismatch")
    require(np.isfinite(left).all() and np.isfinite(right).all(), f"{label}: nonfinite")
    require(np.all(np.abs(left - right) <= tolerance), f"{label}: numerical gate")


def _work(row: Mapping[str, Any], *, backend: str) -> None:
    require(row.get("converged") is True, "unconverged native endpoint")
    require(
        row.get("executed_backend")
        == ("cuda" if backend == "cuda" else "cpu_reference"),
        "native backend mismatch",
    )
    diagnostic = row["scf"]
    work = diagnostic["coulomb_exchange_work"]
    require(
        work["complete"] is True
        and type(work["coulomb_applications"]) is int
        and work["coulomb_applications"] > 0
        and work["exchange_applications"] == 0,
        "incomplete or non-semilocal J/K work",
    )
    residual = diagnostic["physical_residual_max"]
    require(
        type(residual) in (int, float)
        and np.isfinite(residual)
        and 0 <= residual < 2e-9,
        "physical residual gate",
    )
    precision = row["semantic_work"] if backend == "cuda" else row["precision"]
    require(
        precision["requested_mode"] == "fp64" and precision["effective_bits"] == 64,
        "precision mismatch",
    )


def validate_cpu(
    name: str, record: Mapping[str, Any], library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Revalidate the existing independent full-vector CPU force producer."""
    result = record["result"]
    envelope = force_stage_evidence(name, result, evidence=record["evidence"])
    require(envelope["status"] == "pass", str(envelope["reason"]))
    require(
        result["library_identity"] == library["sha256"],
        "CPU compiled/loaded library mismatch",
    )
    for row in result["rows"]:
        endpoints = [
            row,
            *(entry[side] for entry in row["stencil"] for side in ("minus", "plus")),
        ]
        for endpoint in endpoints:
            _work(endpoint, backend="cpu")
            require(endpoint["scf"]["tile_points"] == 256, "CPU schedule mismatch")
            expected_domain = (
                DENSITY_WORK_DOMAIN if name.startswith("LDA_") else LIBXC_WORK_DOMAIN
            )
            require(
                endpoint["scf"]["scf_domain"] == expected_domain,
                "CPU native domain mismatch",
            )
            actual = endpoint["native_library"]
            require(
                all(actual[key] == library[key] for key in ("sha256", "bytes")),
                "CPU executed object mismatch",
            )
    return [
        {
            "backend": "cpu",
            "spin": spin,
            "products": ["energy", "forces"],
            "precision": "fp64",
            "integrals": "direct",
            "ecp": False,
            "basis": "sto-3g",
            "basis_max_l": 0,
            "elements": ["He", "H"] if spin == "polarized" else ["H", "H"],
            "charge": 0,
            "multiplicity": 2 if spin == "polarized" else 1,
            "tile_points": 256,
            "grid": [24, 8, 16],
            "partition_iterations": 3,
            "full_vector": True,
            "phases": ["cold", "warm-replay", "changed-geometry"],
            "coordinates": [
                row["coordinates"] for row in result["rows"] if row["spin"] == spin
            ],
            "point_domain": result["rows"][0]["scf"]["scf_domain"],
            "receipt_identity": result["identity"],
            "library": dict(library),
        }
        for spin in ("polarized", "unpolarized")
    ]


def validate_cuda(
    name: str, record: Mapping[str, Any], library: Mapping[str, Any], subject: str
) -> dict[str, Any]:
    """Replay CUDA numerical/lifecycle gates and bind all actual callback owners."""
    require(
        record["schema"] == "generativeqc.native-automatic-cuda-forces.v1",
        "CUDA receipt schema",
    )
    require(
        record["name"] == name and record.get("subject_identity") == subject,
        "CUDA scientific subject mismatch",
    )
    require(
        record["spins"] in (1, 2) and type(record["spins"]) is int, "CUDA spin mismatch"
    )
    require(bool(record["slurm_job_id"]), "missing scheduled CUDA execution")
    require(
        record["charge"] == record["spins"] - 1
        and record["multiplicity"] == record["spins"],
        "CUDA electron/spin context mismatch",
    )
    require(
        record["native_library_sha256"] == library["sha256"],
        "CUDA compiled/loaded library mismatch",
    )
    context = record["context"]
    require(
        context["precision"] == "fp64"
        and context["integrals"] == "direct"
        and context["ecp"] is False,
        "unsupported CUDA execution context",
    )
    require(
        context["reference"] == "PySCF-2.14.0/Libxc-7.0.0",
        "CUDA oracle version mismatch",
    )
    require(
        context["grid"] == [24, 8, 16]
        and context["partition_iterations"] == 3
        and context["tile_points"] == 37,
        "CUDA quadrature/schedule mismatch",
    )
    require(
        context["basis"] == "sto-3g"
        and context["fixture"] == "h4"
        and context["basis_max_l"] == 0
        and context["full_vector"] is True,
        "CUDA fixture scope mismatch",
    )
    rows = record["rows"]
    require(
        [r["phase"] for r in rows] == ["cold", "warm", "moved"],
        "incomplete CUDA prepared lifecycle",
    )
    require(
        rows[0]["coordinates"] == rows[1]["coordinates"]
        and rows[0]["coordinates"] != rows[2]["coordinates"],
        "CUDA lifecycle geometry mismatch",
    )
    for row in rows:
        _work(row, backend="cuda")
        _close(row["energy"], row["reference_energy"], 3e-9, "CUDA SCF")
        _close(row["energy"], row["fixed_density_energy"], 3e-10, "CUDA fixed density")
        for field, gate in (
            ("density_error", 2e-7),
            ("fock_error", 3e-9),
            ("directional_error", 5e-8),
        ):
            value = row[field]
            require(
                type(value) in (int, float)
                and np.isfinite(value)
                and 0 <= value <= gate,
                f"CUDA {field}",
            )
        artifact = row["artifact"]
        expected_domain = (
            DENSITY_WORK_DOMAIN if name.startswith("LDA_") else LIBXC_WORK_DOMAIN
        )
        require(
            row["scf"]["scf_domain"] == expected_domain
            and artifact["domain_version"] == (5 if name.startswith("LDA_") else 4),
            "CUDA native domain mismatch",
        )
        require(
            artifact["schema"] == "generativeqc.native-automatic-artifact.v2",
            "missing executed callback ABI",
        )
        require(
            artifact["identifier"] == name
            and artifact["spin_blocks"] == record["spins"],
            "CUDA descriptor mismatch",
        )
        require(
            artifact["force_completed"] is True and artifact["source_count"] == 7,
            "incomplete native force sources",
        )
        require(
            artifact["force_schedule"] == "shared-staged"
            and artifact["fallback_reason"] == "none",
            "CUDA schedule mismatch",
        )
        for owner in (
            "point_artifact",
            "force_artifact",
            "point_executable_artifact",
            "force_executable_artifact",
        ):
            require(
                all(
                    artifact[owner][key] == library[key] for key in ("sha256", "bytes")
                ),
                f"CUDA {owner} object mismatch",
            )
        for field in (
            "point_expression_identity",
            "point_source_identity",
            "force_plan_identity",
            "force_source_identity",
            "force_consumer_identity",
            "polarized_executable_identity",
            "restricted_executable_identity",
        ):
            value = artifact[field]
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value),
                f"invalid CUDA {field}",
            )
            require(value == rows[0]["artifact"][field], f"CUDA replay changed {field}")
        require(
            artifact["capacity"]["device_upper_bytes"]
            >= artifact["capacity"]["minimum_device_bytes"]
            > 0
            and artifact["stationary_work_v1"][3] > 0,
            "missing CUDA capacity/work",
        )
        forces = np.asarray(row["forces"], dtype=float)
        require(forces.shape == (4, 3), "CUDA force shape")
        # This catalog's force claim requires every Cartesian component to
        # pass both finite-difference oracles, including replay and movement.
        projected = forces.ravel()
        for field in ("native_fd", "independent_fd", "analytic_directions"):
            _close(row[field], projected, 3e-5, f"CUDA {field}")
        _close(forces.sum(axis=0), np.zeros(3), 2e-8, "CUDA translation")
    return {
        **dict(context),
        "backend": "cuda",
        "spin": "unpolarized" if record["spins"] == 1 else "polarized",
        "products": ["energy", "forces"],
        "elements": ["H"] * 4,
        "charge": record["charge"],
        "multiplicity": record["multiplicity"],
        "phases": [row["phase"] for row in rows],
        "coordinates": [row["coordinates"] for row in rows],
        "receipt_identity": canonical_hash(record),
        "library": dict(library),
        "point_domain": rows[0]["scf"]["scf_domain"],
        "fixed_density": ["energy", "fock", "density", "directional"],
    }


def _envelope(
    subject: str,
    stage: str,
    reference: str,
    coverage: list[dict[str, Any]],
    reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": subject,
        "stage": stage,
        "status": "pass" if reason is None else "fail",
        "reason": reason,
        "evidence": reference,
    }
    if coverage:
        result["qualification"] = {
            "schema": ENDPOINT_COVERAGE_SCHEMA,
            "coverage": coverage,
        }
    return result


def ingest(bundle: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Build the canonical DAG inventory from real receipts, demoting stale input.

    Malformed retained data is an error. A well-formed receipt for older inputs
    becomes an explicit blocker; its original identity is never overwritten.
    """
    require(bundle["schema"] == BUNDLE_SCHEMA, "native campaign schema mismatch")
    require(
        len(bundle["source_commit"]) == 40
        and all(c in "0123456789abcdef" for c in bundle["source_commit"]),
        "missing measured source revision",
    )
    require(
        type(bundle["source_input_count"]) is int
        and bundle["source_input_count"] > 0
        and len(bundle["source_identity"]) == 64,
        "native source inventory identity missing",
    )
    live = source_inventory(root)
    native_identity = _source_identity(
        root, _inventory(root, root / "cmake/GenerativeQCSourceIdentity.json")
    )
    verification = bundle["native_build_verification"]
    require(
        verification["library"] == bundle["build"]["library"],
        "native source query loaded a different ELF",
    )
    require(
        verification["source_identity"] == verification["measured_input_identity"],
        "native embedded build/source mismatch",
    )
    stale = (
        bundle["source_input_count"] != len(live)
        or bundle["source_identity"] != canonical_hash(live)
        or verification["source_identity"] != native_identity
    )
    build = bundle["build"]
    require(
        build["source_identity"] == bundle["source_identity"],
        "native build/source mismatch",
    )
    require(
        build["library"]["bytes"] > 0 and len(build["library"]["sha256"]) == 64,
        "native binary identity missing",
    )
    require(
        build["compiler_cache"]["version"]
        and build["compiler_cache"]["verified_commands"] > 0,
        "unverified compiler launcher",
    )
    require(
        build["toolchain"]
        and build["compile_commands_sha256"]
        and build["configuration_sha256"],
        "incomplete native build contract",
    )
    executions = bundle["executions"]
    require(bool(executions), "missing measured execution environment")
    for execution in executions:
        environment = execution["environment"]
        require(
            environment["pyscf"] == "2.14.0"
            and environment["libxc"] == "7.0.0"
            and environment["host"]
            and execution["names"],
            "native oracle/environment mismatch",
        )
        if execution["backend"] in ("cuda", "all"):
            require(
                environment["slurm_job_id"]
                and environment["cuda_visible_devices"]
                and environment["gpu"],
                "missing scheduled GPU provenance",
            )
    capabilities = {c.name: c for c in available_capabilities()}
    evidence, endpoints, blockers, observed = {}, [], [], []
    blockers.extend(
        {"name": row["name"], "backend": row["stage"], "reason": row["reason"]}
        for row in bundle["failures"]
    )
    require(
        isinstance(bundle["functionals"], dict) and bool(bundle["functionals"]),
        "missing campaign functional inventory",
    )
    for name, data in sorted(bundle["functionals"].items()):
        require(name in capabilities, f"unknown native registration: {name}")
        capability = capabilities[name]
        reference = f"native-catalog:{name}#sha256={canonical_hash(data)}"
        stages: dict[str, Any] = {}
        evidence[name] = stages
        if stale or data["subject_identity"] != capability.identity:
            reason = (
                "stale native/compiler/Graph/domain/ABI inputs; rerun qualification"
            )
            for stage in (
                "compiled-cpu",
                "compiled-cuda",
                "production-domain",
                "gpu-runtime",
                "molecular-scf",
                "forces",
            ):
                stages[stage] = _envelope(
                    capability.identity, stage, reference, [], reason
                )
            blockers.append({"name": name, "backend": "all", "reason": reason})
            continue
        domain = data.get("production-domain")
        if domain is not None:
            stages["production-domain"] = native_domain_stage(name, domain)
        coverage = []
        for backend in ("cpu", "cuda"):
            records = data.get(backend)
            if not records:
                blockers.append(
                    {
                        "name": name,
                        "backend": backend,
                        "reason": "missing native endpoint receipts",
                    }
                )
                continue
            if backend == "cpu":
                current = validate_cpu(name, records, build["library"])
            else:
                for record in records:
                    require(
                        any(
                            name in execution["names"]
                            and execution["backend"] in ("cuda", "all")
                            and execution["environment"]["slurm_job_id"]
                            == record["slurm_job_id"]
                            for execution in executions
                        ),
                        "CUDA receipt has no matching execution environment",
                    )
                require(
                    len({r["spins"] for r in records}) == len(records),
                    "duplicate CUDA spin receipt",
                )
                current = [
                    validate_cuda(name, r, build["library"], capability.identity)
                    for r in records
                ]
            coverage.extend(current)
            stages[f"compiled-{backend}"] = _envelope(
                capability.identity, f"compiled-{backend}", reference, current
            )
            if backend == "cuda":
                stages["gpu-runtime"] = _envelope(
                    capability.identity, "gpu-runtime", reference, current
                )
        if coverage:
            for stage in ("molecular-scf", "forces"):
                stages[stage] = _envelope(
                    capability.identity, stage, reference, coverage
                )
        for scope in coverage:
            observed.append({"name": name, "family": capability.family, **scope})
            for product in scope["products"]:
                try:
                    resolved = resolve_endpoint_capability(
                        name,
                        backend=scope["backend"],
                        spin=scope["spin"],
                        product=product,
                        evidence=stages,
                    )
                except CapabilityNotQualified as exc:
                    blockers.append(
                        {
                            "name": name,
                            "backend": scope["backend"],
                            "spin": scope["spin"],
                            "product": product,
                            "reason": str(exc),
                        }
                    )
                else:
                    endpoints.append(
                        {
                            "name": name,
                            "family": capability.family,
                            "product": product,
                            "resolution": resolved.to_payload(),
                            **scope,
                        }
                    )
    counts = Counter(
        (r["family"], r["backend"], r["spin"], r["product"]) for r in endpoints
    )
    return {
        "schema": SCHEMA,
        "measured_source_commit": bundle["source_commit"],
        "native_library": build["library"],
        "executions": executions,
        "source_identity": canonical_hash(live),
        "measured_source_identity": bundle["source_identity"],
        "native_source_identity": native_identity,
        "stale": stale,
        "structurally_selectable": len(AUTO_BULK_COMPONENTS),
        "unmeasured_functionals": sorted(
            set(capabilities) - set(bundle["functionals"])
        ),
        "structural_blockers": {
            name: c.production_domain_profile.blocker
            for name, c in sorted(capabilities.items())
            if not c.production_domain_profile.eligible
        },
        "execution_identity": canonical_hash(build),
        "summary": capability_summary(evidence),
        "evidence": evidence,
        "endpoints": endpoints,
        "observed_endpoints": observed,
        "blockers": blockers,
        "endpoint_counts": [
            {"family": k[0], "backend": k[1], "spin": k[2], "product": k[3], "count": v}
            for k, v in sorted(counts.items())
        ],
    }


def resolve_context(
    report: Mapping[str, Any],
    name: str,
    *,
    backend: str,
    spin: str,
    product: str,
    precision: str,
    integrals: str,
    ecp: bool,
    basis: str,
    basis_max_l: int,
    grid: list[int],
    elements: list[str],
    charge: int,
    multiplicity: int,
    tile_points: int,
    partition_iterations: int,
    coordinates: list[list[float]],
) -> dict[str, Any]:
    """Look up an exactly measured offline context, never grant runtime admission."""
    for row in report["endpoints"]:
        if (
            all(
                row.get(key) == value
                for key, value in {
                    "name": name,
                    "backend": backend,
                    "spin": spin,
                    "product": product,
                    "precision": precision,
                    "integrals": integrals,
                    "ecp": ecp,
                    "basis": basis,
                    "basis_max_l": basis_max_l,
                    "grid": grid,
                    "elements": elements,
                    "charge": charge,
                    "multiplicity": multiplicity,
                    "tile_points": tile_points,
                    "partition_iterations": partition_iterations,
                }.items()
            )
            and coordinates in row["coordinates"]
        ):
            return dict(row)
    raise ValueError("no qualified native receipt for the requested execution context")
