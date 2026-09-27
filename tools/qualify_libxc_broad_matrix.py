"""Run the broad automatic Libxc CPU promotion matrix outside routine PR CI.

The default deterministic matrix satisfies the #1121 breadth target: three
non-curated LDA, five non-curated GGA, and three non-curated tau meta-GGA
registrations. Each selected registration traverses the existing evidence
producers in order:

compiled CPU -> production domain -> molecular SCF -> public method.

This tool is intentionally suitable for scheduled/manual qualification. It does
not add per-functional scientific branches or runtime Libxc dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

MATRIX_SCHEMA = "vibeqc.libxc-broad-promotion-matrix/v1"
DEFAULT_QUOTAS = {"lda": 3, "gga": 5, "mgga": 3}
_STAGE_ORDER = ("compiled-cpu", "production-domain", "molecular-scf", "public-method")


def _validate_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    expected = set(DEFAULT_QUOTAS)
    if set(quotas) != expected:
        raise ValueError(f"matrix quotas must contain exactly {sorted(expected)!r}")
    normalized: dict[str, int] = {}
    for family in sorted(expected):
        value = quotas[family]
        if type(value) is not int or value <= 0:
            raise ValueError("matrix quotas must be positive integers")
        normalized[family] = value
    return normalized


def _eligible(capability: BulkFunctionalCapability) -> bool:
    if capability.name not in AUTO_BULK_COMPONENTS:
        return False
    if not capability.production_domain_profile.eligible:
        return False
    ingredients = capability.required_ingredients
    if capability.family == "lda":
        return ingredients == ("rho",)
    if capability.family == "gga":
        return ingredients == ("rho", "sigma")
    if capability.family == "mgga":
        return ingredients == ("rho", "sigma", "tau")
    return False


def _spread(values: list[BulkFunctionalCapability], count: int) -> tuple[BulkFunctionalCapability, ...]:
    if len(values) < count:
        raise ValueError(
            f"broad matrix needs {count} candidates but only {len(values)} are eligible"
        )
    if count == 1:
        return (values[len(values) // 2],)
    indices = tuple(index * (len(values) - 1) // (count - 1) for index in range(count))
    if len(set(indices)) != count:
        raise RuntimeError("deterministic broad-matrix sampling produced duplicates")
    return tuple(values[index] for index in indices)


def select_representatives(
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> tuple[BulkFunctionalCapability, ...]:
    """Select a deterministic catalog-spanning non-curated semilocal matrix."""
    requested = _validate_quotas(quotas)
    groups: dict[str, list[BulkFunctionalCapability]] = {
        family: [] for family in requested
    }
    for capability in sorted(available_capabilities(), key=lambda item: item.name):
        if _eligible(capability):
            groups[capability.family].append(capability)

    selected: list[BulkFunctionalCapability] = []
    for family in ("lda", "gga", "mgga"):
        selected.extend(_spread(groups[family], requested[family]))
    return tuple(selected)


def _stage_status(payload: Mapping[str, Any]) -> str:
    envelope = payload.get("stage_evidence")
    if not isinstance(envelope, Mapping):
        raise TypeError("qualification payload omitted stage_evidence")
    status = envelope.get("status")
    if status not in ("pass", "fail", "not-run"):
        raise ValueError(f"unsupported qualification status {status!r}")
    return str(status)


def _write_stage(root: Path, stage: str, payload: Mapping[str, Any]) -> str:
    path = root / f"{stage}.json"
    atomic_json(path, dict(payload))
    return str(path)


def run_matrix(
    capabilities: tuple[BulkFunctionalCapability, ...],
    *,
    output: Path,
    evidence_prefix: str,
    build_dir: Path,
    pyscf_version: str,
    libxc: Any,
    cxx: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run the exact promotion chain and retain every stage artifact."""
    if not isinstance(evidence_prefix, str) or not evidence_prefix.strip():
        raise ValueError("broad matrix requires a nonempty evidence prefix")
    prefix = evidence_prefix.rstrip("/")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for capability in capabilities:
        name = capability.name
        functional_dir = output / name
        functional_dir.mkdir(parents=True, exist_ok=True)
        evidence_base = f"{prefix}/{name}"
        stages: dict[str, str] = {}
        artifacts: dict[str, str] = {}
        blocker: str | None = None

        try:
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

            if (
                stages["compiled-cpu"] == "pass"
                and stages["production-domain"] == "pass"
            ):
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
            else:
                molecular = None
                stages["molecular-scf"] = "not-run"

            if molecular is not None and stages["molecular-scf"] == "pass":
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
            else:
                stages["public-method"] = "not-run"
        except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as exc:
            blocker = f"{type(exc).__name__}: {exc}"
            for stage in _STAGE_ORDER:
                stages.setdefault(stage, "not-run")

        rows.append(
            {
                "name": name,
                "family": capability.family,
                "required_ingredients": list(capability.required_ingredients),
                "capability_identity": capability.identity,
                "stages": stages,
                "artifacts": artifacts,
                "status": "pass" if stages.get("public-method") == "pass" else "fail",
                "blocker": blocker,
            }
        )

    passed = Counter(row["family"] for row in rows if row["status"] == "pass")
    selected = Counter(row["family"] for row in rows)
    summary = {
        "schema": MATRIX_SCHEMA,
        "selected_counts": dict(sorted(selected.items())),
        "public_pass_counts": dict(sorted(passed.items())),
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

    selected = select_representatives()
    summary = run_matrix(
        selected,
        output=args.output,
        evidence_prefix=args.evidence_prefix,
        build_dir=args.build_dir,
        pyscf_version=pyscf.__version__,
        libxc=libxc,
        cxx=args.cxx,
        timeout=args.timeout,
    )
    counts = summary["public_pass_counts"]
    passed = all(counts.get(family, 0) >= count for family, count in DEFAULT_QUOTAS.items())
    print(
        "bulk Libxc broad matrix: "
        + ", ".join(
            f"{family}={counts.get(family, 0)}/{count}"
            for family, count in DEFAULT_QUOTAS.items()
        )
        + f" -> {args.output / 'summary.json'}"
    )
    return 1 if args.require_pass and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
