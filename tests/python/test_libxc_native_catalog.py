"""Real retained native receipts, stale-input gates and exact context isolation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.xc.capability_transition import (
    capability_regressions,
    require_acknowledged_capability_regressions,
)
from generativeqc_compiler.xc.libxc_bulk_capabilities import available_capabilities

from tools.generativeqc_validation import libxc_catalog as catalog
from tools.generativeqc_validation.libxc_native_domain import native_domain_stage

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
NAME = "LDA_C_VWN_4"


@pytest.fixture(scope="module")
def bundle() -> dict:
    manifest = catalog.load_json(ROOT / "manifests/libxc/native-qualification.json")
    return catalog.load_json(ROOT / manifest["campaign"]["path"])


@contextmanager
def recorded_contract() -> Iterator[dict]:
    """Isolate historical source contracts, never current-source admission.

    The manifest authenticates the unmodified campaign before this seam is
    installed. Expected identities/contracts come from that immutable receipt,
    not the test's possibly corrupted copy. Numerical values, context/ELF/work
    validation, profile/cases and exact corruption rejection are not stubbed.
    Current source identity and demotion are tested separately without this seam.
    """
    from generativeqc_compiler.common.provenance import file_hash
    from generativeqc_compiler.xc import libxc_bulk_capabilities as capabilities

    from tools import qualify_libxc_forces as force
    from tools.generativeqc_validation import libxc_native_domain as domain

    manifest = catalog.load_json(ROOT / "manifests/libxc/native-qualification.json")
    path = ROOT / manifest["campaign"]["path"]
    assert file_hash(path) == manifest["campaign"]["sha256"]
    original = catalog.load_json(path)
    original_identity = capabilities._capability_identity
    original_point = domain._point

    def identity(record: dict, source_identity: str) -> str:
        retained = original["functionals"].get(record["name"])
        return (
            retained["subject_identity"]
            if retained
            else original_identity(record, source_identity)
        )

    def point(name: str, family: str, spin: str) -> SimpleNamespace:
        retained = original["functionals"].get(name)
        if retained is None:
            return original_point(name, family, spin)
        row = next(
            r for r in retained["production-domain"]["records"] if r["spin"] == spin
        )
        return SimpleNamespace(
            source_identity=row["point_source_identity"],
            mathematical_contract=row["mathematics"],
        )

    def resolution(name: str, *, backend: str, spin: str) -> SimpleNamespace:
        assert backend == "cpu"
        rows = original["functionals"][name]["cpu"]["result"]["rows"]
        row = next(r for r in rows if r["spin"] == spin)
        state = row["state_identity"]
        return SimpleNamespace(
            to_payload=lambda: deepcopy(row["resolution"]),
            method=SimpleNamespace(spin=spin),
            required_ingredients=tuple(state["ingredients"]),
            plan=SimpleNamespace(
                semilocal=SimpleNamespace(
                    functional=SimpleNamespace(identity=state["functional_identity"])
                )
            ),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(capabilities, "_capability_identity", identity)
        patch.setattr(domain, "_point", point)
        patch.setattr(force, "resolve_bulk_ks", resolution)
        yield original


@pytest.fixture
def historical_contract() -> Iterator[None]:
    with recorded_contract():
        yield


def replay_recorded_source(bundle: dict) -> dict:
    """Replay authenticated historical receipts, not live-source qualification."""
    with recorded_contract() as original:
        recorded_inputs = dict.fromkeys(
            range(original["source_input_count"]), "recorded"
        )
        actual_hash = catalog.canonical_hash
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(catalog, "source_inventory", lambda root: recorded_inputs)
            patch.setattr(
                catalog,
                "canonical_hash",
                lambda value: (
                    original["source_identity"]
                    if value is recorded_inputs
                    else actual_hash(value)
                ),
            )
            patch.setattr(
                catalog,
                "_source_identity",
                lambda root, inventory: original["native_build_verification"][
                    "measured_input_identity"
                ],
            )
            return catalog.ingest(bundle, ROOT)


@pytest.fixture(scope="module")
def report(bundle: dict) -> dict:
    return replay_recorded_source(bundle)


def test_live_receipts_allow_truthful_source_demotion(bundle: dict) -> None:
    live = catalog.ingest(bundle, ROOT)
    if live["stale"]:
        assert not live["observed_endpoints"]
        assert not live["endpoints"]
        assert live["summary"]["qualified_counts"]["forces"] == 0
        assert live["summary"]["blocked_counts"]["forces"] == 11
    else:
        assert len(live["observed_endpoints"]) == 44
        assert live["summary"]["qualified_counts"]["forces"] == 11


def test_boundary_coordinates_replay_recorded_inputs(bundle: dict) -> None:
    from generativeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

    from tools.generativeqc_validation.libxc_native_domain import (
        packed_cases,
        same_fixture_inputs,
    )

    for name, data in bundle["functionals"].items():
        domain = data["production-domain"]
        profile = functional_capability(name).production_domain_profile
        for row in domain["records"]:
            ids, inputs = packed_cases(profile, row["spin"])
            assert row["case_ids"] == ids
            assert same_fixture_inputs(row["inputs"], inputs)
            if "tau/isoorbital" in ids:
                iso = inputs[ids.index("tau/isoorbital"), -2:]
                near = inputs[ids.index("tau/near-isoorbital"), -2:]
                assert (near > iso).all()


@pytest.mark.parametrize(
    "corruption", ["three-ulps", "zero", "sign", "shape", "nan", "boundary"]
)
def test_fixture_rounding_allowance_does_not_relax_domain(corruption: str) -> None:
    import numpy as np

    from tools.generativeqc_validation.libxc_native_domain import same_fixture_inputs

    expected = np.array([[0.0, -0.2, 0.013005319148936172]])
    changed = expected.copy()
    changed[0, 2] = np.nextafter(changed[0, 2], np.inf)
    assert same_fixture_inputs(changed, expected)
    if corruption == "three-ulps":
        changed[0, 2] = np.nextafter(np.nextafter(changed[0, 2], np.inf), np.inf)
    elif corruption == "zero":
        changed[0, 0] = np.nextafter(0.0, 1.0)
    elif corruption == "sign":
        changed[0, 1] *= -1
    elif corruption == "shape":
        changed = changed.flatten()
    elif corruption == "nan":
        changed[0, 1] = np.nan
    else:
        changed[0, 2] = expected[0, 2] * (1 + 1e-12)
    assert not same_fixture_inputs(changed, expected)


def test_real_receipts_reach_multifamily_dag_without_whitelisting(
    historical_contract: None, report: dict
) -> None:
    assert not report["stale"]
    assert report["summary"]["qualified_counts"]["forces"] == 11
    assert len(report["observed_endpoints"]) == 44
    assert len(report["endpoints"]) == 88
    assert {r["family"] for r in report["endpoints"]} == {"lda", "gga", "mgga"}
    assert report["summary"]["qualified_counts"]["response"] == 0
    assert report["summary"]["qualified_counts"]["public-method"] == 0
    # Evidence is opt-in. Ordinary inventory queries and structural selection
    # must not silently load this machine's native qualification campaign.
    assert all(
        c.qualified_stages == ("graph-imported", "pointwise-validated")
        for c in available_capabilities()
    )
    from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks

    assert resolve_bulk_ks("GGA_X_APBE", backend="cuda").method is not None


@pytest.mark.parametrize(
    "owner",
    [
        "point_artifact",
        "force_artifact",
        "point_executable_artifact",
        "force_executable_artifact",
    ],
)
def test_actual_loaded_callback_mismatch_is_rejected(
    historical_contract: None, bundle: dict, owner: str
) -> None:
    row = deepcopy(bundle["functionals"][NAME]["cuda"][0])
    row["rows"][0]["artifact"][owner]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="object mismatch"):
        catalog.validate_cuda(
            NAME, row, bundle["build"]["library"], row["subject_identity"]
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "backend",
        "spin",
        "schedule",
        "energy",
        "fock",
        "precision",
        "lifecycle",
        "geometry",
        "fd",
        "nonfinite",
        "j-work",
        "domain-abi",
        "full-vector",
    ],
)
def test_cuda_replays_raw_gates_instead_of_trusting_pass_labels(
    historical_contract: None, bundle: dict, corruption: str
) -> None:
    record = deepcopy(bundle["functionals"][NAME]["cuda"][0])
    row = record["rows"][0]
    if corruption == "backend":
        row["executed_backend"] = "cpu_reference"
    elif corruption == "spin":
        record["spins"] = 3 - record["spins"]
    elif corruption == "schedule":
        row["artifact"]["force_schedule"] = "fused"
    elif corruption == "energy":
        row["reference_energy"] += 1e-3
    elif corruption == "fock":
        row["fock_error"] = 1e-2
    elif corruption == "precision":
        row["semantic_work"]["effective_bits"] = 32
    elif corruption == "lifecycle":
        record["rows"].pop()
    elif corruption == "geometry":
        record["rows"][2]["coordinates"] = row["coordinates"]
    elif corruption == "fd":
        row["independent_fd"][0] += 1e-2
    elif corruption == "nonfinite":
        row["forces"][0][0] = float("nan")
    elif corruption == "j-work":
        row["scf"]["coulomb_exchange_work"]["complete"] = False
    elif corruption == "full-vector":
        record["context"]["full_vector"] = False
    else:
        row["artifact"]["schema"] = "generativeqc.native-automatic-artifact.v1"
    with pytest.raises(ValueError):
        catalog.validate_cuda(
            NAME, record, bundle["build"]["library"], record["subject_identity"]
        )


@pytest.mark.parametrize("corruption", ["library", "lifecycle", "energy", "work"])
def test_cpu_real_full_vector_receipt_is_revalidated(
    historical_contract: None, bundle: dict, corruption: str
) -> None:
    record = deepcopy(bundle["functionals"][NAME]["cpu"])
    result = record["result"]
    if corruption == "library":
        result["rows"][0]["native_library"]["sha256"] = "0" * 64
    elif corruption == "lifecycle":
        result["rows"].pop()
    elif corruption == "energy":
        result["rows"][0]["stencil"][0]["plus"]["reference_energy"] += 0.01
    else:
        result["rows"][0]["scf"]["coulomb_exchange_work"]["exchange_applications"] = 1
    # An edited result with a self-consistent hash still has to satisfy science,
    # compiled/loaded-object equality and complete executed work requirements.
    result["identity"] = canonical_hash(
        {k: v for k, v in result.items() if k != "identity"}
    )
    with pytest.raises(ValueError):
        catalog.validate_cpu(NAME, record, bundle["build"]["library"])


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "point_source_identity",
        "mathematics",
        "case_ids",
        "inputs",
        "reference",
    ],
)
def test_native_domain_rejects_stale_contract_or_numerical_evidence(
    historical_contract: None, bundle: dict, field: str
) -> None:
    domain = deepcopy(bundle["functionals"][NAME]["production-domain"])
    row = domain["records"][0]
    if field == "profile":
        domain["profile"]["identity"] = "0" * 64
    elif field == "point_source_identity":
        row[field] = "0" * 64
    elif field == "mathematics":
        row[field]["domain_version"] += 1
    elif field == "case_ids":
        row[field].pop()
    elif field == "inputs":
        row[field][0][0] += 1e-6
    else:
        row[field][0][0] += 1
    domain["identity"] = canonical_hash(
        {k: v for k, v in domain.items() if k != "identity"}
    )
    if field == "reference":
        assert native_domain_stage(NAME, domain)["status"] == "fail"
    else:
        with pytest.raises(ValueError):
            native_domain_stage(NAME, domain)


@pytest.mark.parametrize(
    "change",
    ["precision", "integrals", "ecp", "basis_max_l", "product", "grid", "coordinates"],
)
def test_unmeasured_execution_context_cannot_borrow_a_stage(
    historical_contract: None, report: dict, change: str
) -> None:
    row = next(
        r for r in report["endpoints"] if r["name"] == NAME and r["product"] == "forces"
    )
    arguments = {
        k: row[k]
        for k in (
            "backend",
            "spin",
            "product",
            "precision",
            "integrals",
            "ecp",
            "basis",
            "basis_max_l",
            "grid",
            "elements",
            "charge",
            "multiplicity",
            "tile_points",
            "partition_iterations",
        )
    }
    arguments["coordinates"] = row["coordinates"][0]
    assert catalog.resolve_context(report, NAME, **arguments)
    arguments[change] = {
        "precision": "fast",
        "integrals": "df",
        "ecp": True,
        "basis_max_l": 2,
        "product": "hvp",
        "grid": [99, 8, 16],
        "coordinates": [[0, 0, 0]],
    }[change]
    with pytest.raises(ValueError, match="no qualified native receipt"):
        catalog.resolve_context(report, NAME, **arguments)


def test_missing_domain_demotes_both_backends_and_exact_ack_is_required(
    bundle: dict, report: dict
) -> None:
    changed = deepcopy(bundle)
    del changed["functionals"][NAME]["production-domain"]
    after = replay_recorded_source(changed)
    assert not any(r["name"] == NAME for r in after["endpoints"])
    before_summary, after_summary = report["summary"], after["summary"]
    with pytest.raises(ValueError, match="unacknowledged"):
        require_acknowledged_capability_regressions(before_summary, after_summary)
    tokens = [r.token for r in capability_regressions(before_summary, after_summary)]
    require_acknowledged_capability_regressions(
        before_summary, after_summary, acknowledged=tokens
    )
    with pytest.raises(ValueError, match="stale or unknown"):
        require_acknowledged_capability_regressions(
            before_summary,
            after_summary,
            acknowledged=[*tokens[:-1], tokens[-1] + "stale"],
        )
    with pytest.raises(ValueError, match="duplicate"):
        require_acknowledged_capability_regressions(
            before_summary, after_summary, acknowledged=[*tokens, tokens[0]]
        )


def test_changed_native_inputs_never_relabel_historical_results(
    bundle: dict, report: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = catalog.source_inventory(ROOT)
    live["include/generativeqc/generativeqc.h"] = "0" * 64
    monkeypatch.setattr(catalog, "source_inventory", lambda root: live)
    after = catalog.ingest(bundle, ROOT)
    assert after["stale"] and not after["endpoints"]
    assert after["summary"]["qualified_counts"]["forces"] == 0
    assert after["summary"]["blocked_counts"]["forces"] == 11
    assert after["measured_source_identity"] == bundle["source_identity"]
    assert after["source_identity"] != bundle["source_identity"]
    before, current = report["summary"], after["summary"]
    with pytest.raises(ValueError, match="unacknowledged"):
        require_acknowledged_capability_regressions(before, current)
    tokens = [item.token for item in capability_regressions(before, current)]
    require_acknowledged_capability_regressions(before, current, acknowledged=tokens)


@pytest.mark.parametrize("field", ["pyscf", "libxc", "gpu", "slurm_job_id"])
def test_execution_provenance_is_required(bundle: dict, field: str) -> None:
    changed = deepcopy(bundle)
    changed["executions"][0]["environment"][field] = ""
    with pytest.raises(ValueError, match="oracle/environment|GPU provenance"):
        catalog.ingest(changed, ROOT)


def test_cuda_cannot_borrow_another_job_environment(bundle: dict) -> None:
    changed = deepcopy(bundle)
    changed["functionals"][NAME]["cuda"][0]["slurm_job_id"] = "unknown-job"
    with pytest.raises(ValueError, match="matching execution environment"):
        replay_recorded_source(changed)


def test_embedded_native_source_must_match_build_inputs(bundle: dict) -> None:
    changed = deepcopy(bundle)
    changed["native_build_verification"]["source_identity"] = "0" * 64
    with pytest.raises(ValueError, match="embedded build/source mismatch"):
        catalog.ingest(changed, ROOT)


@pytest.mark.parametrize("missing", ["cuda", "polarized", "forces"])
def test_other_backend_spin_or_product_cannot_fill_missing_coverage(
    historical_contract: None, report: dict, missing: str
) -> None:
    from generativeqc_compiler.xc.capability_resolution import CapabilityNotQualified
    from generativeqc_compiler.xc.endpoint_capability import resolve_endpoint_capability

    evidence = deepcopy(report["evidence"][NAME])
    if missing == "cuda":
        del evidence["compiled-cuda"]
        del evidence["gpu-runtime"]
    elif missing == "polarized":
        for stage in ("molecular-scf", "forces"):
            qualification = evidence[stage]["qualification"]
            qualification["coverage"] = [
                r for r in qualification["coverage"] if r["spin"] != "polarized"
            ]
    else:
        del evidence["forces"]
    with pytest.raises(CapabilityNotQualified):
        resolve_endpoint_capability(
            NAME, backend="cuda", spin="polarized", product="forces", evidence=evidence
        )


@pytest.mark.parametrize("product", ["response", "hessian", "hvp"])
def test_first_force_scope_never_grants_second_order(
    report: dict, product: str
) -> None:
    assert not any(row["product"] == product for row in report["endpoints"])


def test_live_checker_rejects_changed_toolchain_and_old_update_baseline(
    bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import check_libxc_native_catalog as checker

    actual_load = checker.load_json
    changed = deepcopy(bundle)
    changed["build"]["toolchain"]["CMAKE_CXX_COMPILER"]["version"] += " changed"
    manifest = actual_load(checker.MANIFEST)
    campaign_path = ROOT / manifest["campaign"]["path"]
    monkeypatch.setattr(
        checker,
        "load_json",
        lambda path: changed if path == campaign_path else actual_load(path),
    )
    with pytest.raises(ValueError, match="differs from retained snapshot"):
        checker.check(checker.MANIFEST)
    with pytest.raises(ValueError, match="older baseline cannot hide"):
        checker.check(checker.MANIFEST, write=True)


@pytest.mark.parametrize(
    "field", ["source", "subject", "worker", "resolution", "functional"]
)
def test_historical_seam_does_not_authenticate_mutated_provenance(
    bundle: dict, field: str
) -> None:
    changed = deepcopy(bundle)
    record = changed["functionals"][NAME]
    if field == "source":
        changed["source_identity"] = "0" * 64
        changed["build"]["source_identity"] = "0" * 64
        after = replay_recorded_source(changed)
        assert after["stale"] and not after["endpoints"]
    elif field == "subject":
        record["subject_identity"] = "0" * 64
        after = replay_recorded_source(changed)
        assert not any(row["name"] == NAME for row in after["endpoints"])
    else:
        if field == "worker":
            domain = record["production-domain"]
            domain["records"][0]["point_source_identity"] = "0" * 64
            domain["identity"] = canonical_hash(
                {k: v for k, v in domain.items() if k != "identity"}
            )
        else:
            result = record["cpu"]["result"]
            row = result["rows"][0]
            if field == "resolution":
                row["resolution"] = {}
            else:
                row["state_identity"]["functional_identity"] = "0" * 64
            result["identity"] = canonical_hash(
                {k: v for k, v in result.items() if k != "identity"}
            )
        with pytest.raises(ValueError):
            replay_recorded_source(changed)
