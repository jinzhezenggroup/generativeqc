"""Hash-valid failed safety receipts must never satisfy endpoint acceptance."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from tools import check_libxc_native_catalog as checker
from tools.generativeqc_validation.libxc_catalog import validate_cuda

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def retained() -> tuple[dict, list[dict]]:
    manifest = checker.load_json(checker.MANIFEST)
    resource = checker.load_json(checker.ROOT / manifest["resource_acceptance"]["path"])
    # Exercise safety validation with authenticated measured scopes, independent
    # of whether current generator dependencies permit inheriting those scopes.
    reports = []
    for item in manifest["incremental_campaigns"]:
        path = checker.ROOT / item["path"]
        assert checker.file_hash(path) == item["sha256"]
        campaign = checker.load_json(path)
        library = campaign["build"]["library"]
        endpoints = []
        for name, data in campaign["functionals"].items():
            for record in data.get("cuda", []):
                scope = validate_cuda(name, record, library, data["subject_identity"])
                endpoints.append({**scope, "name": name, "product": "forces"})
        reports.append(
            {
                "native_library": library,
                "executions": campaign["executions"],
                "endpoints": endpoints,
            }
        )
    return resource, reports


def test_retained_scheduled_safety_acceptance(
    retained: tuple[dict, list[dict]],
) -> None:
    checker.validate_resource_acceptance(*retained, require_current=True)


@pytest.mark.parametrize("mode", ["empty", "missing_spin", "duplicate", "wrong_case"])
def test_failed_or_incomplete_safety_cases_are_rejected(
    retained: tuple[dict, list[dict]], mode: str
) -> None:
    resource, reports = deepcopy(retained)
    if mode == "empty":
        resource["pytest_passed"] = []
    elif mode == "missing_spin":
        resource["pytest_passed"].pop(1)
    elif mode == "duplicate":
        resource["pytest_passed"][1] = resource["pytest_passed"][0]
    else:
        resource["pytest_passed"][2] = "unrelated_test"
    with pytest.raises(ValueError, match="complete passed safety cases"):
        checker.validate_resource_acceptance(resource, reports, require_current=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("no_forces", False),
        ("no_force_receipt", False),
        ("no_warm_seed", False),
        ("no_forces", 1),
        ("attempts", 1),
        ("attempts", True),
        ("max_iterations", 300),
        ("max_iterations", True),
        ("spins", 0),
        ("spins", True),
    ],
)
def test_nonconvergence_failures_and_invalid_controls_are_rejected(
    retained: tuple[dict, list[dict]], field: str, value: object
) -> None:
    resource, reports = deepcopy(retained)
    resource["nonconvergence"][field] = value
    with pytest.raises(ValueError, match="nonconvergence safety did not pass"):
        checker.validate_resource_acceptance(resource, reports, require_current=True)


@pytest.mark.parametrize("field", ["slurm_job_id", "cuda_visible_devices", "host"])
def test_conflicting_resource_execution_fields_are_rejected(
    retained: tuple[dict, list[dict]], field: str
) -> None:
    resource, reports = deepcopy(retained)
    resource[field] = "different"
    with pytest.raises(ValueError, match="inconsistent scheduled execution"):
        checker.validate_resource_acceptance(resource, reports, require_current=True)


@pytest.mark.parametrize(
    "mode", ["job", "visibility", "gpu", "library", "name", "spin", "product"]
)
def test_safety_cannot_borrow_another_execution_or_force_context(
    retained: tuple[dict, list[dict]], mode: str
) -> None:
    resource, reports = deepcopy(retained)
    if mode in ("job", "visibility", "gpu"):
        field = {
            "job": "slurm_job_id",
            "visibility": "cuda_visible_devices",
            "gpu": "gpu",
        }[mode]
        resource["environment"][field] = "different"
        if field in resource:
            resource[field] = "different"
    elif mode == "library":
        resource["native_library_sha256"] = "0" * 64
    elif mode == "name":
        resource["nonconvergence"]["name"] = "UNMEASURED"
    else:
        for report in reports:
            report["endpoints"] = [
                row
                for row in report["endpoints"]
                if (
                    row["spin"] != "unpolarized"
                    if mode == "spin"
                    else row["product"] != "forces"
                )
            ]
    with pytest.raises(ValueError, match="matching"):
        checker.validate_resource_acceptance(resource, reports, require_current=True)


def test_checker_rejects_new_hash_valid_failed_safety_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repinning a newly archived failed run is not scientific acceptance."""
    manifest = checker.load_json(checker.MANIFEST)
    path = checker.ROOT / manifest["resource_acceptance"]["path"]
    original_load, original_hash = checker.load_json, checker.file_hash
    resource = original_load(path)
    resource["pytest_passed"] = []
    # Model a correctly updated manifest for new bytes, not a stale hash failure.
    digest = checker.canonical_hash(resource)
    manifest["resource_acceptance"]["sha256"] = digest

    def load(candidate: Path) -> dict:
        if candidate == checker.MANIFEST:
            return deepcopy(manifest)
        if candidate == path:
            return deepcopy(resource)
        return original_load(candidate)

    monkeypatch.setattr(checker, "load_json", load)
    monkeypatch.setattr(
        checker,
        "file_hash",
        lambda candidate: digest if candidate == path else original_hash(candidate),
    )
    with pytest.raises(ValueError, match="complete passed safety cases"):
        checker.check(checker.MANIFEST, require_current=True)


def test_archival_safety_does_not_promote_or_veto_pending_contexts(
    retained: tuple[dict, list[dict]],
) -> None:
    resource, reports = deepcopy(retained)
    for report in reports:
        report["endpoints"] = []
    checker.validate_resource_acceptance(resource, reports)
    with pytest.raises(ValueError, match="matching"):
        checker.validate_resource_acceptance(resource, reports, require_current=True)
