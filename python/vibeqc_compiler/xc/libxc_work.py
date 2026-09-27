"""Compiler-owned Libxc work-domain policy for automatic semilocal Graphs.

Imported Maple Graphs represent the interior functional algebra. Production point
lowering must reproduce Libxc's worker boundary *outside* that Graph: screen total
vacuum density, construct safe work coordinates, evaluate E/vxc at those work
coordinates, and never differentiate through the clipping operations.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

from . import libxc_bulk
from .libxc_maple import MapleImportError

LIBXC_WORK_DOMAIN = "libxc-7.0/work-semilocal-v1"
LIBXC_WORK_DOMAIN_VERSION = 4
_PINNED_LIBXC_GLOBAL_FHC = True

_POLARIZED_LAYOUTS = {
    "lda": ("rho_a", "rho_b"),
    "gga": ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"),
    "mgga": (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    ),
}


@dataclass(frozen=True)
class LibxcWorkPolicy:
    """One exact compiler-owned projection of Libxc worker semantics."""

    name: str
    family: str
    density_threshold: float
    sigma_floor: float | None
    tau_floor: float | None
    enforce_fhc: bool
    source_identity: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "vibeqc.libxc-work-policy.v1",
            "domain": LIBXC_WORK_DOMAIN,
            **asdict(self),
        }


@cache
def _catalog_index() -> dict[str, dict[str, Any]]:
    catalog = libxc_bulk.read_catalog()
    return {item["name"]: item for item in catalog["registrations"]}


def _record(name: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Libxc work policy requires a nonempty functional name")
    key = name.upper()
    try:
        return _catalog_index()[key]
    except KeyError as exc:
        raise MapleImportError(f"unknown bulk Libxc registration: {name!r}") from exc


@cache
def automatic_work_policy(name: str) -> LibxcWorkPolicy:
    """Return the worker policy for one imported LDA/GGA/MGGA registration."""

    record = _record(name)
    family = record["family"]
    if family not in _POLARIZED_LAYOUTS:
        raise MapleImportError(
            f"automatic Libxc work lowering does not support family {family!r}"
        )
    density = float(record["bindings"]["p_a_dens_threshold"])
    if not math.isfinite(density) or density <= 0.0:
        raise MapleImportError(
            "automatic Libxc work lowering requires a positive density threshold"
        )

    sigma_floor = density ** (8.0 / 3.0) if family != "lda" else None
    tau_floor = 1.0e-20 if family == "mgga" else None
    raw_flags = record.get("flags", "")
    if not isinstance(raw_flags, str):
        raise MapleImportError("bulk Libxc work policy found malformed flags")
    flags = set(re.split(r"\s*\|\s*", raw_flags))
    enforce_fhc = family == "mgga" and (
        "XC_FLAGS_ENFORCE_FHC" in flags or _PINNED_LIBXC_GLOBAL_FHC
    )

    root = asset_path("upstream/libxc/7.0.0")
    functionals = root / "functionals.c"
    required = [
        "func->sigma_threshold = pow(func->info->dens_threshold, 4.0/3.0);",
        "func->tau_threshold   = 1e-20;",
    ]
    if family == "mgga":
        required.append("func->info->flags = func->info->flags | XC_FLAGS_ENFORCE_FHC;")
    source = functionals.read_text(encoding="utf-8")
    if any(snippet not in source for snippet in required):
        raise MapleImportError("pinned Libxc threshold policy changed")

    sources = {"functionals.c": file_hash(functionals)}
    if family == "mgga":
        worker = root / "work_mgga_inc.c"
        worker_source = worker.read_text(encoding="utf-8")
        required_worker = (
            "if(dens < p->dens_threshold)",
            "my_rho[0] = m_max(p->dens_threshold, VAR(rho, ip, 0));",
            "my_sigma[0] = m_max(p->sigma_threshold * p->sigma_threshold, VAR(sigma, ip, 0));",
            "my_tau[0] = m_max(p->tau_threshold, VAR(tau, ip, 0));",
            "my_sigma[0] = m_min(my_sigma[0], 8.0*my_rho[0]*my_tau[0]);",
            "s_ave = 0.5*(my_sigma[0] + my_sigma[2]);",
        )
        if any(snippet not in worker_source for snippet in required_worker):
            raise MapleImportError("pinned Libxc MGGA work policy changed")
        sources["work_mgga_inc.c"] = file_hash(worker)

    source_identity = canonical_hash(
        {
            "schema": "vibeqc.libxc-work-source.v1",
            "compiler_owner": file_hash(Path(__file__)),
            "family": family,
            "sources": sources,
            "semantics": (
                "screen-total-density",
                "floor-spin-density",
                *(
                    ("floor-same-spin-sigma", "clamp-cross-spin-sigma")
                    if family != "lda"
                    else ()
                ),
                *(("floor-tau", "fermi-hole-curvature") if family == "mgga" else ()),
                "raw-vxc-at-work-inputs",
                "original-density-energy-weight",
            ),
        }
    )
    return LibxcWorkPolicy(
        name=record["name"],
        family=family,
        density_threshold=density,
        sigma_floor=sigma_floor,
        tau_floor=tau_floor,
        enforce_fhc=enforce_fhc,
        source_identity=source_identity,
    )


def polarized_work_setup(
    policy: LibxcWorkPolicy,
    features: tuple[str, ...],
    *,
    indent: str = "  ",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Emit backend-neutral C++ scalar setup and interior argument names.

    The returned statements intentionally contain no derivative of the work
    transformation. Callers evaluate already-differentiated interior roots at
    the returned work coordinates.
    """

    expected = _POLARIZED_LAYOUTS[policy.family]
    if tuple(features) != expected:
        raise MapleImportError(
            f"{policy.name} work lowering expected {expected!r}, got {tuple(features)!r}"
        )
    d = float(policy.density_threshold).hex()
    lines = [
        f"{indent}const double total_density = rho_a + rho_b;",
        f"{indent}if (total_density < {d}) return {{}};",
        f"{indent}const double work_rho_a = fmax({d}, rho_a);",
        f"{indent}const double work_rho_b = fmax({d}, rho_b);",
    ]
    arguments = ["work_rho_a", "work_rho_b"]
    if policy.family != "lda":
        if policy.sigma_floor is None:
            raise MapleImportError("GGA/MGGA work policy omitted sigma floor")
        s = float(policy.sigma_floor).hex()
        lines.extend(
            [
                f"{indent}double work_sigma_aa = fmax({s}, sigma_aa);",
                f"{indent}double work_sigma_bb = fmax({s}, sigma_bb);",
            ]
        )
        if policy.family == "mgga":
            if policy.tau_floor is None:
                raise MapleImportError("MGGA work policy omitted tau floor")
            t = float(policy.tau_floor).hex()
            lines.extend(
                [
                    f"{indent}const double work_tau_a = fmax({t}, tau_a);",
                    f"{indent}const double work_tau_b = fmax({t}, tau_b);",
                ]
            )
            if policy.enforce_fhc:
                lines.extend(
                    [
                        f"{indent}work_sigma_aa = fmin(work_sigma_aa, 8.0 * work_rho_a * work_tau_a);",
                        f"{indent}work_sigma_bb = fmin(work_sigma_bb, 8.0 * work_rho_b * work_tau_b);",
                    ]
                )
        lines.extend(
            [
                f"{indent}const double sigma_average = 0.5 * (work_sigma_aa + work_sigma_bb);",
                f"{indent}const double work_sigma_ab =",
                f"{indent}    fmax(-sigma_average, fmin(sigma_average, sigma_ab));",
            ]
        )
        arguments.extend(("work_sigma_aa", "work_sigma_ab", "work_sigma_bb"))
        if policy.family == "mgga":
            arguments.extend(("work_tau_a", "work_tau_b"))
    return tuple(lines), tuple(arguments)


__all__ = [
    "LIBXC_WORK_DOMAIN",
    "LIBXC_WORK_DOMAIN_VERSION",
    "LibxcWorkPolicy",
    "automatic_work_policy",
    "polarized_work_setup",
]
