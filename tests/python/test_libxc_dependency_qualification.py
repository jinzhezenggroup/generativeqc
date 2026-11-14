"""Dependency deltas invalidate only their measured backend/product contexts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from test_libxc_native_catalog import recorded_contract

from tools.generativeqc_validation import libxc_catalog as catalog
from tools.generativeqc_validation.libxc_dependencies import (
    authenticated_inputs,
    changed_inputs,
)

if TYPE_CHECKING:
    from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
NAME = "LDA_C_VWN_4"


@pytest.fixture(scope="module")
def inputs() -> dict:
    return catalog.load_json(
        ROOT / "manifests/libxc/native-qualification-inputs.json.xz"
    )


def replay(inputs: dict, monkeypatch: pytest.MonkeyPatch, changes: dict) -> dict:
    """Replay immutable measured semantics, varying actual source inventory inputs."""
    with recorded_contract() as bundle:
        sources = {**inputs["sources"], **changes}
        native = {**inputs["native_sources"], **inputs["generator_sources"], **changes}
        monkeypatch.setattr(catalog, "source_inventory", lambda root: sources)
        monkeypatch.setattr(catalog, "native_inventory", lambda root: native)
        return catalog.ingest(bundle, ROOT, measured_inputs=inputs)


@pytest.mark.parametrize(
    "path",
    [
        "src/xtb/native/scf.cpp",
        "src/cc/new_solver.cpp",
        "src/posthf/mp2_energy.cpp",
        "src/integrals/AGENTS.md",
    ],
)
def test_unrelated_changes_retain_all_measured_contexts(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    report = replay(inputs, monkeypatch, {path: "1" * 64})
    assert report["source_changed"]
    assert not report["stale"]
    assert not report["revalidation_plan"]
    assert len(report["observed_endpoints"]) == 44
    assert len(report["endpoints"]) == 88


@pytest.mark.parametrize(
    "path",
    ["src/dft/new_xc_kernel.cu", "python/generativeqc_compiler/xc/spatial_cuda.py"],
)
def test_cuda_kernel_retains_cpu_qualification(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    report = replay(inputs, monkeypatch, {path: "1" * 64})
    assert report["stale"]
    assert len(report["observed_endpoints"]) == 22
    assert len(report["endpoints"]) == 44
    assert {row["backend"] for row in report["endpoints"]} == {"cpu"}
    assert {row["backend"] for row in report["revalidation_plan"]} == {"cuda"}


def test_cuda_force_change_retains_cuda_energy(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = replay(inputs, monkeypatch, {"src/dft/new_stationary_cuda.cu": "1" * 64})
    assert len(report["observed_endpoints"]) == 44
    assert len(report["endpoints"]) == 66
    assert len([row for row in report["endpoints"] if row["backend"] == "cpu"]) == 44
    assert {
        row["product"] for row in report["endpoints"] if row["backend"] == "cuda"
    } == {"energy"}
    assert all(row["products"] == ["forces"] for row in report["revalidation_plan"])


@pytest.mark.parametrize(
    "path",
    [
        "src/scf/ks.cpp",
        "src/integrals/direct_jk.hpp",
        "include/generativeqc/generativeqc.h",
        "cmake/new_flags.cmake",
        "src/unknown/new.hpp",
        "src/unknown/new.cu",
        "python/generativeqc_compiler/common/cuda_environment.py",
        "src/posthf/capacity.hpp",
        "src/posthf/raw_source.hpp",
        "python/generativeqc_compiler/xc/integration.py",
        "python/generativeqc_compiler/xc/spatial_program.py",
    ],
)
def test_shared_endpoint_abi_and_unknown_changes_remain_fail_closed(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    report = replay(inputs, monkeypatch, {path: "1" * 64})
    assert not report["endpoints"]
    assert len(report["revalidation_plan"]) == 22
    assert all(path in row["changed_inputs"] for row in report["revalidation_plan"])
    assert report["summary"]["qualified_counts"]["production-domain"] == 11


def test_graph_contract_change_is_registration_local(inputs: dict) -> None:
    from generativeqc_compiler.xc import libxc_bulk_capabilities as capabilities

    from tools.generativeqc_validation import libxc_native_domain as domain

    with recorded_contract() as bundle, pytest.MonkeyPatch.context() as monkeypatch:
        identity, point = capabilities._capability_identity, domain._point
        monkeypatch.setattr(
            capabilities,
            "_capability_identity",
            lambda record, source: (
                "1" * 64 if record["name"] == NAME else identity(record, source)
            ),
        )

        def changed_point(name: str, family: str, spin: str) -> SimpleNamespace:
            emitted = point(name, family, spin)
            if name == NAME:
                emitted.source_identity = "1" * 64
            return emitted

        monkeypatch.setattr(domain, "_point", changed_point)
        monkeypatch.setattr(catalog, "source_inventory", lambda root: inputs["sources"])
        monkeypatch.setattr(
            catalog,
            "native_inventory",
            lambda root: {**inputs["native_sources"], **inputs["generator_sources"]},
        )
        report = catalog.ingest(bundle, ROOT, measured_inputs=inputs)
    assert len(report["endpoints"]) == 80
    assert {row["name"] for row in report["revalidation_plan"]} == {NAME}


def test_aggregate_subject_drift_requires_exact_emitted_and_method_contracts(
    inputs: dict,
) -> None:
    from generativeqc_compiler.xc import libxc_bulk_capabilities as capabilities

    with recorded_contract() as bundle, pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            capabilities,
            "_capability_identity",
            lambda record, source: canonical_hash(record["name"]),
        )
        monkeypatch.setattr(catalog, "source_inventory", lambda root: inputs["sources"])
        monkeypatch.setattr(
            catalog,
            "native_inventory",
            lambda root: {**inputs["native_sources"], **inputs["generator_sources"]},
        )
        report = catalog.ingest(bundle, ROOT, measured_inputs=inputs)
    assert not report["stale"]
    assert len(report["endpoints"]) == 88
    assert report["native_library"] == bundle["build"]["library"]
    assert report["measured_source_identity"] == bundle["source_identity"]


@pytest.mark.parametrize("operation", ["added", "removed", "modified"])
def test_dependency_census_includes_additions_removals_and_edits(
    operation: str,
) -> None:
    path = "src/dft/kernel.cu"
    before = {} if operation == "added" else {path: "0" * 64}
    after = {} if operation == "removed" else {path: "1" * 64}
    assert changed_inputs(before, after, backend="cuda", product="energy") == [path]
    assert not changed_inputs(before, after, backend="cpu", product="energy")


@pytest.mark.parametrize("field", ["sources", "native_sources"])
def test_dependency_sidecar_cannot_relabel_measured_sources(
    inputs: dict, field: str
) -> None:
    with recorded_contract() as bundle:
        changed = deepcopy(inputs)
        changed[field]["src/scf/ks.cpp"] = "0" * 64
        with pytest.raises(ValueError, match="inventory/(source|embedded)"):
            authenticated_inputs(bundle, changed)


@pytest.mark.parametrize("corruption", ["missing", "revision"])
def test_native_generator_provenance_is_required(inputs: dict, corruption: str) -> None:
    with recorded_contract() as bundle:
        changed = deepcopy(inputs)
        if corruption == "missing":
            changed.pop("generator_sources")
        else:
            changed["source_commit"] = "0" * 40
        with pytest.raises(ValueError, match="generator"):
            authenticated_inputs(bundle, changed)


def test_live_emission_and_cpu_method_contracts_revalidate_historical_provenance() -> (
    None
):
    from tools.generativeqc_validation.libxc_native_domain import native_domain_stage

    manifest = catalog.load_json(ROOT / "manifests/libxc/native-qualification.json")
    bundle = catalog.load_json(ROOT / manifest["campaign"]["path"])
    record = bundle["functionals"][NAME]
    original = deepcopy(record)
    stage = native_domain_stage(
        NAME, record["production-domain"], measured_subject=record["subject_identity"]
    )
    assert stage["status"] == "pass"
    scopes = catalog.validate_cpu(
        NAME,
        record["cpu"],
        bundle["build"]["library"],
        measured_subject=record["subject_identity"],
    )
    assert len(scopes) == 2
    assert record == original


def test_checker_ignores_only_unrelated_whole_tree_provenance() -> None:
    from tools.check_libxc_native_catalog import acceptance_view

    before = {
        "source_identity": "old",
        "native_source_identity": "old",
        "source_changed": False,
        "endpoints": ["measured"],
    }
    after = {
        **before,
        "source_identity": "new",
        "native_source_identity": "new",
        "source_changed": True,
    }
    assert acceptance_view(before) == acceptance_view(after)
    after["endpoints"] = []
    assert acceptance_view(before) != acceptance_view(after)


@pytest.mark.parametrize("corruption", ["bytes", "library"])
def test_checker_resource_receipt_rejects_rebinding(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    from tools import check_libxc_native_catalog as checker

    manifest = catalog.load_json(checker.MANIFEST)
    path = ROOT / manifest["resource_acceptance"]["path"]
    if corruption == "bytes":
        original = checker.file_hash
        monkeypatch.setattr(
            checker,
            "file_hash",
            lambda candidate: "0" * 64 if candidate == path else original(candidate),
        )
        reason = "resource evidence bytes changed"
    else:
        original = checker.load_json

        def altered(candidate: Path) -> dict:
            data = original(candidate)
            if candidate == path:
                data["native_library_sha256"] = "0" * 64
            return data

        monkeypatch.setattr(checker, "load_json", altered)
        reason = "resource/lifetime receipt has no matching scheduled native execution"
    with pytest.raises(ValueError, match=reason):
        checker.check(checker.MANIFEST)


def test_current_resource_receipt_cannot_borrow_cpu_only_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import check_libxc_native_catalog as checker

    original = checker.merge_catalogs

    def different_cuda_owner(reports: list[dict]) -> dict:
        report = original(reports)
        for row in report["endpoints"]:
            if row["backend"] == "cuda":
                row["library"] = {**row["library"], "sha256": "0" * 64}
        return report

    monkeypatch.setattr(checker, "merge_catalogs", different_cuda_owner)
    with pytest.raises(ValueError, match="matching resource/lifetime receipt"):
        checker.check(checker.MANIFEST, require_current=True)


def test_documentation_edits_do_not_enter_the_execution_census(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("project(example)\n")
    (tmp_path / "docs").mkdir()
    document = tmp_path / "docs/contract.md"
    document.write_text("original\n")
    before = catalog.source_inventory(tmp_path)
    document.write_text("clarified\n")
    assert catalog.source_inventory(tmp_path) == before


def test_incremental_campaigns_supplement_other_backends_and_deduplicate_counts(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = replay(inputs, monkeypatch, {"src/dft/new_xc_kernel.cu": "1" * 64})
    current = replay(inputs, monkeypatch, {})
    selected = deepcopy(current)
    selected["evidence"] = {NAME: selected["evidence"][NAME]}
    for field in ("endpoints", "observed_endpoints"):
        selected[field] = [
            row
            for row in selected[field]
            if row["name"] == NAME and row["backend"] == "cuda"
        ]
    selected["unmeasured_functionals"] = sorted(
        set(current["summary"]["functionals"]) - {NAME}
    )
    with recorded_contract():
        merged = catalog.merge_catalogs([cpu, selected])
        duplicate = catalog.merge_catalogs([current, deepcopy(current)])
        refreshed = catalog.merge_catalogs([cpu, current])
    assert len(merged["endpoints"]) == 48
    assert len([row for row in merged["endpoints"] if row["backend"] == "cpu"]) == 44
    assert {row["name"] for row in merged["endpoints"] if row["backend"] == "cuda"} == {
        NAME
    }
    assert len(merged["campaigns"]) == 2
    assert len(merged["superseded_revalidation"]) == 1
    assert all(row["name"] != NAME for row in merged["revalidation_plan"])
    assert len(duplicate["endpoints"]) == 88
    assert duplicate["endpoint_counts"] == current["endpoint_counts"]
    assert not refreshed["stale"]
    assert not refreshed["revalidation_plan"]
    assert len(refreshed["superseded_revalidation"]) == 11
    from tools.check_libxc_native_catalog import render

    ledger = deepcopy(merged)
    ledger["campaigns"][1]["native_library"]["sha256"] = "1" * 64
    rendered = render(ledger, current["summary"])
    assert ledger["campaigns"][0]["native_library"]["sha256"] in rendered
    assert ledger["campaigns"][1]["native_library"]["sha256"] in rendered
    gap = {
        "name": NAME,
        "backend": "cuda",
        "reason": "missing native endpoint receipts",
    }
    partial = deepcopy(cpu)
    partial["blockers"].append(gap)
    one_spin = deepcopy(selected)
    one_spin["endpoints"] = [
        row for row in one_spin["endpoints"] if row["spin"] == "unpolarized"
    ]
    with recorded_contract():
        completed = catalog.merge_catalogs([partial, selected])
        incomplete = catalog.merge_catalogs([partial, one_spin])
    assert gap in completed["superseded_blockers"]
    assert gap not in completed["blockers"]
    assert gap in completed["campaigns"][0]["blockers"]
    assert gap in incomplete["blockers"]


@pytest.mark.parametrize("field", ["spin", "grid", "coordinates", "product"])
def test_incremental_refresh_does_not_cover_other_exact_contexts(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    stale = replay(inputs, monkeypatch, {"src/dft/new_xc_kernel.cu": "1" * 64})
    current = replay(inputs, monkeypatch, {})
    if field == "spin":
        current["endpoints"] = [
            row for row in current["endpoints"] if row["spin"] != "polarized"
        ]
    elif field == "product":
        current["endpoints"] = [
            row for row in current["endpoints"] if row["product"] != "forces"
        ]
    else:
        for row in current["endpoints"]:
            row[field] = [12, 6, 8] if field == "grid" else [[[0.0, 0.0, 0.0]]]
    with recorded_contract():
        merged = catalog.merge_catalogs([stale, current])
    assert merged["stale"]
    assert len(merged["revalidation_plan"]) == 11
    assert not merged["superseded_revalidation"]


def test_scf_changes_do_not_erase_scalar_compilation_proofs(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = replay(inputs, monkeypatch, {"src/scf/fock_prepared.cpp": "1" * 64})
    assert not report["endpoints"]
    assert report["summary"]["qualified_counts"]["compiled-cpu"] == 11
    assert report["summary"]["qualified_counts"]["compiled-cuda"] == 11
    assert report["summary"]["qualified_counts"]["production-domain"] == 11


def test_new_campaign_embeds_its_own_authenticated_dependency_census(
    inputs: dict,
) -> None:
    with recorded_contract() as original, pytest.MonkeyPatch.context() as patch:
        bundle = {**original, "dependency_inputs": inputs}
        patch.setattr(catalog, "source_inventory", lambda root: inputs["sources"])
        patch.setattr(
            catalog,
            "native_inventory",
            lambda root: {**inputs["native_sources"], **inputs["generator_sources"]},
        )
        report = catalog.ingest(bundle, ROOT)
    assert not report["stale"]
    assert len(report["endpoints"]) == 88


@pytest.mark.parametrize("corruption", ["method", "plan", "functional", "forces"])
def test_provenance_equivalence_never_waives_scientific_or_numerical_contracts(
    corruption: str,
) -> None:
    manifest = catalog.load_json(ROOT / "manifests/libxc/native-qualification.json")
    bundle = catalog.load_json(ROOT / manifest["campaign"]["path"])
    record = deepcopy(bundle["functionals"][NAME]["cpu"])
    result = record["result"]
    row = result["rows"][0]
    if corruption == "method":
        row["resolution"]["method_identifier"] = "different-method"
    elif corruption == "plan":
        row["resolution"]["plan_identity"] = "0" * 64
    elif corruption == "functional":
        row["state_identity"]["functional_identity"] = "0" * 64
    else:
        row["forces"][0][0] += 1
    result["identity"] = canonical_hash(
        {key: value for key, value in result.items() if key != "identity"}
    )
    with pytest.raises(ValueError):
        catalog.validate_cpu(
            NAME,
            record,
            bundle["build"]["library"],
            measured_subject=result["subject_identity"],
        )
