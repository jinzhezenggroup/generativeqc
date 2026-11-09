"""Recompute the retained native Libxc catalog and enforce exact transition acks.

Run this command in CI against the checked-out source. It uses real retained
native measurements without loading a library, compiling code or probing a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.xc.capability_catalog import (
    capability_changes,
    render_capability_changes,
    render_capability_summary,
)
from generativeqc_compiler.xc.capability_transition import (
    capability_regressions,
    require_acknowledged_capability_regressions,
)

from tools.generativeqc_validation.libxc_catalog import (
    ingest,
    load_json,
    require,
    write_json,
)

MANIFEST = ROOT / "manifests/libxc/native-qualification.json"


def render(report: dict, previous: dict) -> str:
    """Render one current catalog; measured successes and DAG admission differ."""
    lines = [
        "# Native automatic Libxc qualification",
        "",
        f"Measured source revision: `{report['measured_source_commit']}`.",
        f"Retained measured source inputs: `{report['measured_source_identity']}`.",
        f"Current checked source inputs: `{report['source_identity']}`.",
        f"Current checked native input identity: `{report['native_source_identity']}`.",
        f"Retained evidence stale for checked inputs: `{report['stale']}`.",
        f"Retained executed native library SHA-256: `{report['native_library']['sha256']}`.",
        "Independent oracle: PySCF 2.14.0 / Libxc 7.0.0.",
        (
            f"Structurally selectable non-curated automatic registrations: {report['structurally_selectable']}; "
            f"registrations without a retained native campaign: {len(report['unmeasured_functionals'])}."
        ),
        "",
        "This offline qualification inventory does not restrict structural public selection.",
        "The automatic count excludes registrations already owned by curated components.",
        "Counts apply only to the exact retained native contexts; no response, Hessian,",
        "HVP, higher-l, mixed/Fast, ECP or DF scope is inferred.",
        "The `public-method` row counts the optional stronger admission receipt,",
        "not available public selectors. Its absence does not disable a method.",
        "A successful default check establishes archival consistency only. Use",
        "`python tools/check_libxc_native_catalog.py --require-current` for representative",
        "source-matched paired energy/force acceptance across families, backends and spins.",
        "",
        "```text",
        render_capability_summary(report["summary"]),
        "```",
        "",
        "| Family | Backend | Spin | Product | Qualified registrations |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in report["endpoint_counts"]:
        lines.append(
            "| {family} | {backend} | {spin} | {product} | {count} |".format(**row)
        )
    lines += [
        "",
        f"Current-source observed backend/spin lifecycle records: {len(report['observed_endpoints'])}.",
        "Historical measurements remain in the retained campaign; stale receipts are not",
        "counted as current-source admitted contexts. Admitted records retain exact",
        "geometry, grid, precision, binary and derivative scope.",
        "The [manifest](../../manifests/libxc/native-qualification.json) locates the",
        "compressed machine-readable snapshot and independent numerical receipts.",
        "",
        "## Qualified registrations",
        "",
        "| Registration | Backend | Product | Qualified spin layouts |",
        "| --- | --- | --- | --- |",
    ]
    scopes: dict[tuple[str, str, str], set[str]] = {}
    for row in report["endpoints"]:
        scopes.setdefault((row["name"], row["backend"], row["product"]), set()).add(
            row["spin"]
        )
    for (name, backend, product), spins in sorted(scopes.items()):
        lines.append(
            f"| `{name}` | {backend} | {product} | {', '.join(sorted(spins))} |"
        )
    lines += [
        "",
        "## Explicit blockers",
        "",
    ]
    for name, reason in report["structural_blockers"].items():
        lines.append(f"- `{name}`: {reason}")
    seen = set()
    for row in report["blockers"]:
        item = (row["name"], row["reason"])
        if item not in seen:
            lines.append(f"- `{row['name']}`: {row['reason']}")
            seen.add(item)
    lines += [
        "",
        "Physical empty-spin particle-birth continuation remains excluded;",
        "a successful molecular force measurement does not qualify that domain.",
        "",
        "## Revision diff",
        "",
        "```text",
        render_capability_changes(previous, report["summary"]),
        "```",
        "",
    ]
    return "\n".join(lines)


def require_current_acceptance(report: dict) -> None:
    """Require representative current E/F contexts, not a registration allow-list.

    Archival consistency permits stale or missing measurements. End-to-end
    acceptance additionally needs a source-matched energy and force receipt in
    each ordinary semilocal family, backend and spin layout. Pair products only
    within the same exact admitted context; observed but blocked results do not
    qualify. This does not grant unmeasured domains or public-method receipts.
    """
    require(
        not report["stale"], "current-source acceptance requires fresh native evidence"
    )
    contexts = {}
    for row in report["endpoints"]:
        context = {
            key: value
            for key, value in row.items()
            if key not in ("product", "resolution")
        }
        identity = canonical_hash(context)
        entry = contexts.setdefault(identity, (context, set()))
        entry[1].add(row["product"])
    covered = {
        (context["family"], context["backend"], context["spin"])
        for context, products in contexts.values()
        if {"energy", "forces"} <= products
    }
    missing = [
        f"{family}/{backend}/{spin}"
        for family in ("lda", "gga", "mgga")
        for backend in ("cpu", "cuda")
        for spin in ("unpolarized", "polarized")
        if (family, backend, spin) not in covered
    ]
    require(
        not missing,
        "current-source acceptance missing paired admitted energy/forces contexts: "
        + ", ".join(missing),
    )
    registrations = {}
    for context, products in contexts.values():
        if {"energy", "forces"} <= products:
            registrations.setdefault((context["family"], context["name"]), set()).add(
                (context["backend"], context["spin"])
            )
    matrix = {
        (backend, spin)
        for backend in ("cpu", "cuda")
        for spin in ("unpolarized", "polarized")
    }
    shared_families = {
        family
        for (family, _name), measured in registrations.items()
        if matrix <= measured
    }
    require(
        {"lda", "gga", "mgga"} <= shared_families,
        "current-source acceptance requires a common registration across backends/spins "
        "for each family; missing: "
        + ", ".join(sorted({"lda", "gga", "mgga"} - shared_families)),
    )


def check(
    manifest_path: Path, *, write: bool = False, require_current: bool = False
) -> dict:
    """Validate bytes, live inputs, exact report coverage and regression policy."""
    manifest = load_json(manifest_path)
    require(
        manifest["schema"] == "generativeqc.libxc-native-catalog-manifest.v1",
        "catalog manifest schema",
    )
    for field in ("campaign", "previous"):
        path = ROOT / manifest[field]["path"]
        require(
            path.resolve().is_relative_to(ROOT),
            "catalog evidence path escapes checkout",
        )
        require(
            file_hash(path) == manifest[field]["sha256"],
            f"{field} evidence bytes changed",
        )
    bundle = load_json(ROOT / manifest["campaign"]["path"])
    previous = load_json(ROOT / manifest["previous"]["path"])
    report = ingest(bundle, ROOT)
    report["historical"] = []
    for item in manifest.get("historical", []):
        path = ROOT / item["path"]
        require(
            file_hash(path) == item["sha256"], "retained historical manifest changed"
        )
        retained = load_json(path)
        report["historical"].append(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "source": retained.get("source", retained.get("source_commit")),
                "schema": retained["schema"],
                "current_claims": [],
                "reason": "historical execution/source identity; never relabelled as current evidence",
            }
        )
    report["changes"] = capability_changes(previous, report["summary"])
    report["execution_identity"] = canonical_hash(bundle["build"])
    acknowledgements = manifest.get("acknowledgements", [])
    snapshot_path = ROOT / manifest["snapshot"]
    if write and snapshot_path.exists():
        retained = load_json(snapshot_path)
        if report != retained:
            require(
                previous == retained["summary"],
                "retain the current summary as previous before changing the catalog; "
                "an older baseline cannot hide intervening demotions",
            )
    require_acknowledged_capability_regressions(
        previous, report["summary"], acknowledged=acknowledgements
    )
    if require_current:
        require_current_acceptance(report)
    rendered = render(report, previous)
    if write:
        write_json(ROOT / manifest["snapshot"], report)
        (ROOT / manifest["report"]).write_text(rendered)
    else:
        expected = load_json(ROOT / manifest["snapshot"])
        if report != expected:
            transitions = capability_regressions(expected["summary"], report["summary"])
            raise ValueError(
                "live native catalog differs from retained snapshot; requalify or explicitly demote claims: "
                + ", ".join(item.token for item in transitions)
            )
        require(
            (ROOT / manifest["report"]).read_text() == rendered,
            "rendered native catalog is stale",
        )
    print(render_capability_summary(report["summary"]))
    print(
        f"native observed contexts: {len(report['observed_endpoints'])}; qualified products: {len(report['endpoints'])}; stale: {report['stale']}"
    )
    if require_current:
        print("current-source representative energy/force acceptance: passed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate after reviewing exact regression acknowledgements",
    )
    parser.add_argument(
        "--require-current",
        action="store_true",
        help="require source-matched paired energy/forces contexts for LDA/GGA/tau-MGGA, CPU/CUDA and RKS/UKS",
    )
    args = parser.parse_args()
    check(args.manifest, write=args.write, require_current=args.require_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
