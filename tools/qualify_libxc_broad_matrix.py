"""Run the broad automatic Libxc CPU promotion matrix outside routine PR CI.

The campaign satisfies the #1121 breadth target by evidence, not by a
hand-maintained functional list. It scans deterministic non-curated LDA/GGA/
tau-meta-GGA candidates, retains production-domain blockers, and advances only
admitted candidates through the remaining exact evidence producers:

production domain -> compiled CPU -> molecular SCF -> public method.

Scanning continues within each family until three LDA, five GGA, and three
tau-meta-GGA registrations reach the public endpoint, or the eligible inventory
is exhausted. This tool is intended for scheduled/manual qualification and does
not add per-functional scientific branches or runtime Libxc dependencies.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    available_capabilities,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools.qualify_libxc_compiled_cpu import qualify_compiled_cpu
from tools.qualify_libxc_molecular_scf import qualify_molecular_scf
from tools.qualify_libxc_production_domain import qualify_functional
from tools.qualify_libxc_public_method import qualify_public_method

MATRIX_SCHEMA = "vibeqc.libxc-broad-promotion-matrix/v2"
DEFAULT_QUOTAS = {"lda": 3, "gga": 5, "mgga": 3}
_FAMILY_ORDER = ("lda", "gga", "mgga")
_STAGE_ORDER = ("production-domain", "compiled-cpu", "molecular-scf", "public-method")
_EXPECTED_INGREDIENTS = {
    "lda": ("rho",),
    "gga": ("rho", "sigma"),
    "mgga": ("rho", "sigma", "tau"),
}


def _validate_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    expected = set(DEFAULT_QUOTAS)
    if set(quotas) != expected:
        raise ValueError(f"matrix quotas must contain exactly {sorted(expected)!r}")
    normalized: dict[str, int] = {}
    for family in _FAMILY_ORDER:
        value = quotas[family]
        if type(value) is not int or value <= 0:
            raise ValueError("matrix quotas must be positive integers")
        normalized[family] = value
    return normalized


def _eligible(capability: BulkFunctionalCapability) -> bool:
    expected = _EXPECTED_INGREDIENTS.get(capability.family)
    return (
        expected is not None
        and capability.name in AUTO_BULK_COMPONENTS
        and capability.production_domain_profile.eligible
        and capability.required_ingredients == expected
    )


def candidate_inventory(
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> tuple[BulkFunctionalCapability, ...]:
    """Return deterministic structurally eligible candidates for evidence scanning."""
    requested = _validate_quotas(quotas)
    groups: dict[str, list[BulkFunctionalCapability]] = {
        family: [] for family in _FAMILY_ORDER
    }
    for capability in sorted(available_capabilities(), key=lambda item: item.name):
        if _eligible(capability):
            groups[capability.family].append(capability)

    shortages = {
        family: (requested[family], len(groups[family]))
        for family in _FAMILY_ORDER
        if len(groups[family]) < requested[family]
    }
    if shortages:
        raise ValueError(f"insufficient broad-matrix candidate inventory: {shortages!r}")
    return tuple(
        capability
        for family in _FAMILY_ORDER
        for capability in groups[family]
    )


def _stage_status(payload: Mapping[str, Any]) -> str:
    envelope = payload.get("stage_evidence")
    if not isinstance(envelope, Mapping):
        raise TypeError("qualification payload omitted stage_evidence")
    status = envelope.get("status")
    if status not in ("pass", "fail", "not-run"):
        raise ValueError(f"unsupported qualification status {status!r}")
    return str(status)


def _stage_reason(payload: Mapping[str, Any]) -> str | None:
    envelope = payload.get("stage_evidence")
    if not isinstance(envelope, Mapping):
        raise TypeError("qualification payload omitted stage_evidence")
    reason = envelope.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise TypeError("qualification stage reason must be a string or null")
    return reason


def _write_stage(root: Path, stage: str, payload: Mapping[str, Any]) -> str:
    path = root / f"{stage}.json"
    atomic_json(path, dict(payload))
    return str(path)


def _record(
    capability: BulkFunctionalCapability,
    *,
    stages: Mapping[str, str],
    artifacts: Mapping[str, str],
    blocker_stage: str | None,
    blocker: str | None,
) -> dict[str, Any]:
    return {
        "name": capability.name,
        "family": capability.family,
        "required_ingredients": list(capability.required_ingredients),
        "capability_identity": capability.identity,
        "stages": dict(stages),
        "artifacts": dict(artifacts),
        "status": "pass" if stages.get("public-method") == "pass" else "blocked",
        "blocker_stage": blocker_stage,
        "blocker": blocker,
    }


def run_matrix(
    capabilities: tuple[BulkFunctionalCapability, ...],
    *,
    output: Path,
    evidence_prefix: str,
    build_dir: Path,
    pyscf_version: str,
    libxc: Any,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    cxx: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Scan candidates until exact public endpoint quotas are satisfied."""
    requested = _validate_quotas(quotas)
    if not isinstance(evidence_prefix, str) or not evidence_prefix.strip():
        raise ValueError("broad matrix requires a nonempty evidence prefix")
    prefix = evidence_prefix.rstrip("/")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    attempted: Counter[str] = Counter()
    production_passed: Counter[str] = Counter()
    public_passed: Counter[str] = Counter()

    for capability in capabilities:
        family = capability.family
        if family not in requested or public_passed[family] >= requested[family]:
            continue
        if not _eligible(capability):
            continue

        attempted[family] += 1
        name = capability.name
        functional_dir = output / name
        functional_dir.mkdir(parents=True, exist_ok=True)
        evidence_base = f"{prefix}/{name}"
        stages = {stage: "not-run" for stage in _STAGE_ORDER}
        artifacts: dict[str, str] = {}
        blocker_stage: str | None = None
        blocker: str | None = None

        try:
            production = qualify_functional(
                name,
                evidence=f"{evidence_base}/production-domain.json",
                pyscf_version=pyscf_version,
                libxc=libxc,
            )
            stages["production-domain"] = _stage_status(production)
            artifacts["production-domain"] = _write_stage(
                functional_dir, "production-domain", production
            )
            if stages["production-domain"] != "pass":
                blocker_stage = "production-domain"
                blocker = _stage_reason(production)
                rows.append(
                    _record(
                        capability,
                        stages=stages,
                        artifacts=artifacts,
                        blocker_stage=blocker_stage,
                        blocker=blocker,
                    )
                )
                continue
            production_passed[family] += 1

            compiled = qualify_compiled_cpu(
                name,
                evidence=f"{evidence_base}/compiled-cpu.json",
                cxx=cxx,
                timeout=timeout,
            )
            stages["compiled-cpu"] = _stage_status(compiled)
            artifacts["compiled-cpu"] = _write_stage(
                functional_dir, "compiled-cpu", compiled
            )
            if stages["compiled-cpu"] != "pass":
                blocker_stage = "compiled-cpu"
                blocker = _stage_reason(compiled)
                rows.append(
                    _record(
                        capability,
                        stages=stages,
                        artifacts=artifacts,
                        blocker_stage=blocker_stage,
                        blocker=blocker,
                    )
                )
                continue

            molecular = qualify_molecular_scf(
                name,
                compiled_cpu_evidence=compiled["stage_evidence"],
                production_domain_evidence=production["stage_evidence"],
                build_dir=build_dir,
                evidence=f"{evidence_base}/molecular-scf.json",
                cxx=cxx,
                timeout=timeout,
            )
            stages["molecular-scf"] = _stage_status(molecular)
            artifacts["molecular-scf"] = _write_stage(
                functional_dir, "molecular-scf", molecular
            )
            if stages["molecular-scf"] != "pass":
                blocker_stage = "molecular-scf"
                blocker = _stage_reason(molecular)
                rows.append(
                    _record(
                        capability,
                        stages=stages,
                        artifacts=artifacts,
                        blocker_stage=blocker_stage,
                        blocker=blocker,
                    )
                )
                continue

            public = qualify_public_method(
                name,
                compiled_cpu_evidence=compiled["stage_evidence"],
                production_domain_evidence=production["stage_evidence"],
                molecular_scf_evidence=molecular["stage_evidence"],
                evidence=f"{evidence_base}/public-method.json",
            )
            stages["public-method"] = _stage_status(public)
            artifacts["public-method"] = _write_stage(
                functional_dir, "public-method", public
            )
            if stages["public-method"] != "pass":
                blocker_stage = "public-method"
                blocker = _stage_reason(public)
            else:
                public_passed[family] += 1
        except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as exc:
            blocker_stage = next(
                (stage for stage in _STAGE_ORDER if stages[stage] == "not-run"),
                "runner",
            )
            blocker = f"{type(exc).__name__}: {exc}"

        rows.append(
            _record(
                capability,
                stages=stages,
                artifacts=artifacts,
                blocker_stage=blocker_stage,
                blocker=blocker,
            )
        )

    complete = all(
        public_passed[family] >= requested[family] for family in _FAMILY_ORDER
    )
    summary = {
        "schema": MATRIX_SCHEMA,
        "quotas": requested,
        "complete": complete,
        "attempted_counts": {
            family: attempted[family] for family in _FAMILY_ORDER
        },
        "production_pass_counts": {
            family: production_passed[family] for family in _FAMILY_ORDER
        },
        "public_pass_counts": {
            family: public_passed[family] for family in _FAMILY_ORDER
        },
        "functionals": rows,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-prefix", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cxx")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import pyscf
    from pyscf.dft import libxc

    summary = run_matrix(
        candidate_inventory(),
        output=args.output,
        evidence_prefix=args.evidence_prefix,
        build_dir=args.build_dir,
        pyscf_version=pyscf.__version__,
        libxc=libxc,
        cxx=args.cxx,
        timeout=args.timeout,
    )
    counts = summary["public_pass_counts"]
    print(
        "bulk Libxc broad matrix: "
        + ", ".join(
            f"{family}={counts.get(family, 0)}/{count}"
            for family, count in DEFAULT_QUOTAS.items()
        )
        + f" -> {args.output / 'summary.json'}"
    )
    return 1 if args.require_pass and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
