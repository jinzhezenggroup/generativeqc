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
    merge_catalogs,
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
        f"Some retained contexts require targeted revalidation: `{report['stale']}`.",
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
        "dependency-current paired energy/force acceptance across families, backends and spins.",
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
        f"Dependency-current backend/spin lifecycle records: {len(report['observed_endpoints'])}.",
        "Historical measurements remain in the retained campaign; stale receipts are not",
        "counted as dependency-current admitted contexts. Unchanged dependencies can",
        "inherit evidence; this does not claim a new binary was measured. Records retain exact",
        "geometry, grid, precision, binary and derivative scope.",
        "The [manifest](../../manifests/libxc/native-qualification.json) locates the",
        "compressed machine-readable snapshot and independent numerical receipts.",
        "",
        "## Retained executions",
        "",
        "Incremental batches supplement, rather than relabel, earlier measurements.",
        "Each context retains its own executed library identity in the snapshot.",
        "",
        "| Measured revision/base | Measured source inputs | Executed ELF SHA-256 |",
        "| --- | --- | --- |",
    ]
    for campaign in report.get("campaigns", [report]):
        lines.append(
            f"| `{campaign['measured_source_commit']}` | "
            f"`{campaign['measured_source_identity']}` | "
            f"`{campaign['native_library']['sha256']}` |"
        )
    lines += [
        "",
        "## Targeted revalidation",
        "",
        "Whole-tree/ELF hashes authenticate historical execution, not a global expiry switch.",
        "The machine-readable snapshot lists changed inputs for each affected backend/product.",
        "Contexts with unchanged dependencies retain acceptance; missing measurements are not failures.",
        "ABI, build and unknown shared inputs require compatibility review and targeted regression.",
        "",
        "| Registration | Backend | Products to revalidate | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["revalidation_plan"]:
        lines.append(
            f"| `{row['name']}` | {row['backend']} | {', '.join(row['products'])} | {row['reason']} |"
        )
    lines += [
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
    acceptance additionally needs a dependency-current energy and force receipt in
    each ordinary semilocal family, backend and spin layout. Pair products only
    within the same exact admitted context; observed but blocked results do not
    qualify. This does not grant unmeasured domains or public-method receipts.
    """
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
        "dependency-current acceptance missing paired admitted energy/forces contexts: "
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
        "dependency-current acceptance requires a common registration across backends/spins "
        "for each family; missing: "
        + ", ".join(sorted({"lda", "gga", "mgga"} - shared_families)),
    )


def validate_resource_acceptance(
    resource: dict, reports: list[dict], *, require_current: bool = False
) -> None:
    """Authenticate scheduled safety execution and require its actual outcomes.

    A hash binds receipt bytes, not their success. Keep incomplete or failed
    safety runs out of representative acceptance even when correctly archived.
    """
    environment = resource.get("environment", {})
    require(
        resource.get("schema") == "generativeqc.libxc-scoped-safety.v1"
        and all(
            isinstance(environment.get(field), str) and environment[field]
            for field in ("slurm_job_id", "cuda_visible_devices", "host", "gpu")
        )
        and all(
            resource.get(field) == environment[field]
            for field in ("slurm_job_id", "cuda_visible_devices", "host")
        ),
        "resource/lifetime receipt has inconsistent scheduled execution",
    )
    required_cases = {
        "test_native_automatic_force_minimum_and_below_budget[1]",
        "test_native_automatic_force_minimum_and_below_budget[2]",
        "test_native_automatic_force_ragged_failure_and_recovery",
    }
    passed = resource.get("pytest_passed")
    require(
        isinstance(passed, list)
        and all(isinstance(case, str) for case in passed)
        and len(passed) == len(required_cases)
        and set(passed) == required_cases,
        "resource/lifetime receipt lacks complete passed safety cases",
    )
    failure = resource.get("nonconvergence", {})
    require(
        isinstance(failure, dict)
        and type(failure.get("attempts")) is int
        and failure["attempts"] >= 2
        and type(failure.get("max_iterations")) is int
        and failure["max_iterations"] == 1
        and type(failure.get("spins")) is int
        and failure["spins"] in (1, 2)
        and all(
            failure.get(field) is True
            for field in ("no_forces", "no_warm_seed", "no_force_receipt")
        ),
        "resource/lifetime nonconvergence safety did not pass",
    )
    matched = [
        batch
        for batch in reports
        if batch["native_library"]["sha256"] == resource.get("native_library_sha256")
        and any(
            execution["backend"] in ("cuda", "all")
            and execution["environment"] == environment
            and failure.get("name") in execution["names"]
            for execution in batch["executions"]
        )
    ]
    require(
        bool(matched),
        "resource/lifetime receipt has no matching scheduled native execution",
    )
    if require_current:
        spin = "unpolarized" if failure["spins"] == 1 else "polarized"
        require(
            any(
                row["name"] == failure.get("name")
                and row["backend"] == "cuda"
                and row["spin"] == spin
                and row["product"] == "forces"
                and row["library"]["sha256"] == resource.get("native_library_sha256")
                for batch in matched
                for row in batch["endpoints"]
            ),
            "current acceptance requires a matching resource/lifetime receipt",
        )


def acceptance_view(report: dict) -> dict:
    """Do not make unrelated whole-tree provenance movement a CI failure.

    The returned view still compares all scientific identities, exact contexts,
    receipt hashes, scoped dependency deltas, blockers and transition evidence.
    """
    return {
        key: value
        for key, value in report.items()
        if key not in ("source_identity", "native_source_identity", "source_changed")
    }


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

    def measured_inputs(item: dict) -> dict | None:
        inputs = item.get("dependency_inputs")
        if inputs is None:
            return None
        path = ROOT / inputs["path"]
        require(
            path.resolve().is_relative_to(ROOT),
            "dependency input path escapes checkout",
        )
        require(
            file_hash(path) == inputs["sha256"], "dependency inventory bytes changed"
        )
        return load_json(path)

    reports = [ingest(bundle, ROOT, measured_inputs=measured_inputs(manifest))]
    for item in manifest.get("incremental_campaigns", []):
        path = ROOT / item["path"]
        require(
            path.resolve().is_relative_to(ROOT),
            "incremental evidence path escapes checkout",
        )
        require(file_hash(path) == item["sha256"], "incremental evidence bytes changed")
        reports.append(
            ingest(load_json(path), ROOT, measured_inputs=measured_inputs(item))
        )
    report = merge_catalogs(reports)
    resource_input = manifest.get("resource_acceptance")
    if resource_input is not None:
        path = ROOT / resource_input["path"]
        require(
            path.resolve().is_relative_to(ROOT),
            "resource evidence path escapes checkout",
        )
        require(
            file_hash(path) == resource_input["sha256"],
            "resource evidence bytes changed",
        )
        resource = load_json(path)
        validate_resource_acceptance(resource, reports, require_current=require_current)
        report["resource_acceptance"] = {
            **resource_input,
            "native_library_sha256": resource["native_library_sha256"],
            "slurm_job_id": resource["environment"]["slurm_job_id"],
        }
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
    acknowledgements = manifest.get("acknowledgements", [])
    snapshot_path = ROOT / manifest["snapshot"]
    if write and snapshot_path.exists():
        retained = load_json(snapshot_path)
        if acceptance_view(report) != acceptance_view(retained):
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
        require(
            resource_input is not None
            and resource["native_library_sha256"]
            in {
                row["library"]["sha256"]
                for row in report["endpoints"]
                if row["backend"] == "cuda" and row["product"] == "forces"
            },
            "current acceptance requires a matching resource/lifetime receipt",
        )
    rendered = render(report, previous)
    if write:
        write_json(ROOT / manifest["snapshot"], report)
        (ROOT / manifest["report"]).write_text(rendered)
    else:
        expected = load_json(ROOT / manifest["snapshot"])
        if acceptance_view(report) != acceptance_view(expected):
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
        print("dependency-current representative energy/force acceptance: passed")
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
        help="require dependency-current paired energy/forces for LDA/GGA/tau-MGGA, CPU/CUDA and RKS/UKS; unrelated stale rows do not veto acceptance",
    )
    args = parser.parse_args()
    check(args.manifest, write=args.write, require_current=args.require_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
