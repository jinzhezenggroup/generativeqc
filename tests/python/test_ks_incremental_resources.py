"""Fail closed for unqualified incremental KS capacity without requiring a GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from generativeqc import (
    Calculator,
    ResourceBudget,
    _native,
    estimate_ks_resources,
    resources_ks,
)

if TYPE_CHECKING:
    from pathlib import Path

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
SELECTORS = (
    "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
    "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK",
)


@pytest.fixture(autouse=True)
def ordinary_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for selector in SELECTORS:
        monkeypatch.delenv(selector, raising=False)
        for suffix in ("REBUILD_INTERVAL", "DENSITY_RMS_THRESHOLD"):
            monkeypatch.delenv(f"{selector}_{suffix}", raising=False)


def _inventory_library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Only the ordinary shape query is stubbed; no CUDA execution is claimed."""
    library = SimpleNamespace(
        generativeqc_ks_resource_inventory_version_v1=lambda: 1,
        generativeqc_ks_options_version=lambda: 1,
    )
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    monkeypatch.setattr(
        resources_ks,
        "_cuda_item_inventory",
        lambda *args, **kwargs: {"state": 8, "xc": 8, "coulomb": 8, "setup": 24},
    )
    return library


def _forbidden(*_args: object, **_kwargs: object) -> None:
    pytest.fail("unsupported incremental KS plan reached native loading/preparation")


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("value", ("1", "on"))
def test_incremental_ks_exact_budget_is_unsupported_before_native_loading(
    monkeypatch: pytest.MonkeyPatch, method: str, selector: str, value: str
) -> None:
    library = _inventory_library(monkeypatch)
    options = {"method": method, "backend": "cuda", "max_iterations": 1}
    ordinary = estimate_ks_resources([H2, H2], library=library, **options)
    budget = ResourceBudget(
        host_bytes=ordinary.peak_bytes["host"],
        device_bytes=ordinary.peak_bytes["device"],
    )
    estimate_ks_resources(
        [H2, H2], library=library, budget=budget, **options
    ).require_feasible()
    # The complete ordinary budget includes the retained public force arena.
    assert ordinary.peak_bytes["device"] >= 512 << 20
    monkeypatch.setenv(selector, value)
    monkeypatch.setattr(_native, "load_library", _forbidden)
    monkeypatch.setattr(resources_ks, "_cuda_item_inventory", _forbidden)
    incremental = estimate_ks_resources([H2, H2], budget=budget, **options)
    assert incremental.status == "unsupported"
    assert not incremental.requests[0].candidates
    assert incremental.identity != ordinary.identity
    controls = json.loads(incremental.requests[0].identity.schedule)
    assert controls["incremental_direct_jk"][selector] is True
    with pytest.raises(NotImplementedError, match="incremental Direct-J/K"):
        incremental.require_feasible()


@pytest.mark.parametrize("backend,precision", (("cpu", "fp64"), ("cuda", "auto")))
def test_enabled_selector_does_not_advertise_excluded_cpu_or_mixed_mode(
    monkeypatch: pytest.MonkeyPatch, backend: str, precision: str
) -> None:
    monkeypatch.setenv(SELECTORS[0], "1")
    monkeypatch.setattr(_native, "load_library", _forbidden)
    plan = estimate_ks_resources([H2], backend=backend, precision=precision)
    assert plan.status == "unsupported"


@pytest.mark.parametrize("generic,legacy", (("off", "on"), ("on", "off"), ("1", "1")))
def test_selectors_combine_with_or_not_generic_precedence(
    monkeypatch: pytest.MonkeyPatch, generic: str, legacy: str
) -> None:
    monkeypatch.setenv(SELECTORS[0], generic)
    monkeypatch.setenv(SELECTORS[1], legacy)
    monkeypatch.setattr(_native, "load_library", _forbidden)
    assert estimate_ks_resources([H2], backend="cuda").status == "unsupported"


@pytest.mark.parametrize(
    "backend,precision", (("cpu", "fp64"), ("cuda", "fp64"), ("cuda", "auto"))
)
@pytest.mark.parametrize("generic,legacy", (("0", "off"), ("off", "0"), ("off", "off")))
def test_disabled_spellings_preserve_ordinary_plan_and_ignore_inactive_tuning(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    precision: str,
    generic: str,
    legacy: str,
) -> None:
    library = _inventory_library(monkeypatch)
    options = {"backend": backend, "precision": precision, "library": library}
    ordinary = estimate_ks_resources([H2], **options).require_feasible()
    monkeypatch.setenv(SELECTORS[0], generic)
    monkeypatch.setenv(SELECTORS[1], legacy)
    for selector in SELECTORS:
        for suffix in ("REBUILD_INTERVAL", "DENSITY_RMS_THRESHOLD"):
            monkeypatch.setenv(f"{selector}_{suffix}", "invalid-but-inactive")
    assert estimate_ks_resources([H2], **options) == ordinary


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("value", ("", "false", "true", "ON", "1 ", "0.0"))
def test_malformed_selectors_match_native_truth_parsing_without_loading(
    monkeypatch: pytest.MonkeyPatch, selector: str, value: str
) -> None:
    for name in SELECTORS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(selector, value)
    monkeypatch.setattr(_native, "load_library", _forbidden)
    with pytest.raises(ValueError, match=f"{selector} must be 0/off or 1/on"):
        estimate_ks_resources([H2], backend="cuda")


@pytest.mark.parametrize("method", ("pbe-rks", "pbe-uks"))
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize(
    "entrypoint",
    (
        "singlepoint_energy",
        "singlepoint_forces",
        "prepare_batch",
        "batch_singlepoint",
        "stale_plan",
    ),
)
def test_public_budget_and_stale_plan_reject_before_preparation(
    monkeypatch: pytest.MonkeyPatch, method: str, selector: str, entrypoint: str
) -> None:
    calculator = Calculator(method=method)
    # Exercise public Python routing with a real CPU-loaded ABI, replacing only
    # CUDA shape metadata. Native allocation/execution is forbidden below.
    calculator._device_name = "cuda"
    _inventory_library(monkeypatch)
    ordinary = calculator.estimate_resources([H2, H2]).require_feasible()
    calculator._resource_budget = ResourceBudget(
        host_bytes=ordinary.peak_bytes["host"],
        device_bytes=ordinary.peak_bytes["device"],
    )
    for name in (
        "generativeqc_context_create",
        "generativeqc_calculation_prepare",
        "generativeqc_batch_prepare",
        "generativeqc_resource_ledger_create_v1",
    ):
        monkeypatch.setattr(calculator._library, name, _forbidden)
    monkeypatch.setenv(selector, "1")
    assert calculator.estimate_resources([H2, H2]).status == "unsupported"
    if entrypoint == "stale_plan":
        # A supplied plan must also enforce the guard without a calculator budget.
        calculator._resource_budget = None
        with pytest.raises(
            ValueError, match="KS inputs differ from the global resource plan"
        ):
            calculator.prepare_batch([H2, H2], resource_plan=ordinary)
    else:
        with pytest.raises(NotImplementedError, match="incremental Direct-J/K"):
            if entrypoint.startswith("singlepoint"):
                properties = (
                    ("energy", "forces")
                    if entrypoint.endswith("forces")
                    else ("energy",)
                )
                calculator.singlepoint(H2, properties=properties)
            elif entrypoint == "batch_singlepoint":
                calculator.batch_singlepoint([H2, H2])
            else:
                calculator.prepare_batch([H2, H2])


def test_explicit_unbudgeted_experiment_keeps_native_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = Calculator(method="pbe-rks")
    calculator._device_name = "cuda"
    monkeypatch.setenv(SELECTORS[0], "1")
    monkeypatch.setattr(resources_ks, "ks_resource_request", _forbidden)

    def native_boundary(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("reached native context admission")

    monkeypatch.setattr(
        calculator._library, "generativeqc_context_create", native_boundary
    )
    with pytest.raises(RuntimeError, match="reached native context admission"):
        calculator.prepare_batch([H2])


def test_incremental_cli_returns_unsupported_without_native_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from generativeqc.__main__ import main

    path = tmp_path / "h2.xyz"
    path.write_text("2\nbohr\nH 0 0 -0.7\nH 0 0 0.7\n")
    monkeypatch.setenv(SELECTORS[0], "on")
    monkeypatch.setattr(_native, "load_library", _forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generativeqc",
            "resources",
            str(path),
            "--method",
            "pbe-uks",
            "--backend",
            "cuda",
            "--batch",
            "2",
            "--units",
            "bohr",
        ],
    )
    assert main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "unsupported"
    assert "incremental Direct-J/K" in output["diagnostic"]


@pytest.mark.parametrize(
    "generic,legacy", (("1", "0"), ("off", "on"), ("on", "off"), ("0", "false"))
)
def test_native_cpu_does_not_ignore_requested_or_invalid_selector(
    monkeypatch: pytest.MonkeyPatch, generic: str, legacy: str
) -> None:
    calculator = Calculator(method="pbe-rks")
    monkeypatch.setenv(SELECTORS[0], generic)
    monkeypatch.setenv(SELECTORS[1], legacy)
    with pytest.raises(RuntimeError, match="invalid argument"):
        calculator.prepare_batch([H2])
