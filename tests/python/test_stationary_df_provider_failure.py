"""Exercise fitted/direct provider admission through real CPU/CUDA consumers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from _stationary_provider_test_support import (
    cpu_provider_selection,
    cuda_provider_selection,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def separate_source_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most provider tests audit the explicit independent-source contract."""
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_REDUCTION", "separate")


def _select(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: object,
    *,
    budget: int = 32,
    ecp: bool = False,
    required: bool = False,
    host_reserve: int = 32,
    device_reserve: int = 32,
) -> object:
    return cuda_provider_selection(
        monkeypatch,
        tmp_path,
        source,
        budget=budget,
        ecp=ecp,
        required=required,
        host_reserve=host_reserve,
        device_reserve=device_reserve,
    )


@pytest.mark.parametrize(
    "failure", ["missing", "not-callable", "unavailable", "no-budget", "ecp"]
)
def test_fitted_derivative_failure_never_selects_exact_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    exact = Mock(
        side_effect=AssertionError("Direct must not execute for a fitted state")
    )
    source = SimpleNamespace(density_fitted=True, cuda_integral_derivatives=exact)
    if failure != "missing":
        source.density_fitted_integral_derivatives = (
            None if failure == "not-callable" else Mock(return_value=None)
        )
    with pytest.raises(NotImplementedError, match="change the Hamiltonian"):
        _select(
            monkeypatch,
            tmp_path,
            source,
            budget=0 if failure == "no-budget" else 32,
            ecp=failure == "ecp",
        )
    exact.assert_not_called()
    if failure in ("no-budget", "ecp"):
        source.density_fitted_integral_derivatives.assert_not_called()


def test_fitted_provider_errors_propagate_without_exact_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(
            side_effect=RuntimeError("stale DF response")
        ),
        cuda_integral_derivatives=Mock(),
    )
    with pytest.raises(RuntimeError, match="stale DF response"):
        _select(monkeypatch, tmp_path, source)
    source.cuda_integral_derivatives.assert_not_called()


@pytest.mark.parametrize("invalid", [np.zeros((2, 2, 3)), np.full((4, 2, 3), np.nan)])
def test_fitted_provider_rejects_invalid_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: np.ndarray
) -> None:
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(invalid, {})),
    )
    with pytest.raises(RuntimeError, match="invalid output"):
        _select(monkeypatch, tmp_path, source)


def test_complete_fitted_response_and_explicit_exact_fallback_remain_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(return_value=(output, {})),
        cuda_integral_derivatives=Mock(return_value=None),
    )
    np.testing.assert_array_equal(_select(monkeypatch, tmp_path, source), output)
    source.density_fitted_integral_derivatives.assert_called_once_with(2, 32)
    source.density_fitted = False
    assert _select(monkeypatch, tmp_path, source) is None
    source.cuda_integral_derivatives.assert_called_once_with(
        2, 32, range_exchange=False
    )


@pytest.mark.parametrize("failure", ["missing", "unavailable", "no-budget"])
def test_enlarged_direct_domain_cannot_select_ao_task_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    source = SimpleNamespace(density_fitted=False)
    if failure != "missing":
        source.cuda_integral_derivatives = Mock(return_value=None)
    with pytest.raises(NotImplementedError, match="native.*integral"):
        _select(
            monkeypatch,
            tmp_path,
            source,
            budget=0 if failure == "no-budget" else 32,
            required=True,
        )


def test_enlarged_domain_accepts_complete_native_owner_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(return_value=(output, {})),
    )
    np.testing.assert_array_equal(
        _select(monkeypatch, tmp_path, source, required=True), output
    )
    source.cuda_integral_derivatives.assert_called_once_with(
        2, 32, range_exchange=False
    )


@pytest.mark.parametrize("actual", [32, 33])
def test_native_host_staging_must_fit_its_concurrent_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, actual: int
) -> None:
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(
            return_value=(np.zeros((4, 2, 3)), {"one_electron_host_peak_bytes": actual})
        ),
    )
    if actual == 32:
        assert (
            _select(monkeypatch, tmp_path, source, required=True, host_reserve=32)
            is not None
        )
    else:
        with pytest.raises(RuntimeError, match="host staging exceeds"):
            _select(monkeypatch, tmp_path, source, required=True, host_reserve=32)


@pytest.mark.parametrize("actual", [31, 32, 33])
def test_fitted_device_staging_must_fit_its_concurrent_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, actual: int
) -> None:
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(
            return_value=(
                np.zeros((4, 2, 3)),
                {"one_electron_device_peak_bytes": actual},
            )
        ),
        cuda_integral_derivatives=Mock(),
    )
    if actual <= 32:
        assert (
            _select(monkeypatch, tmp_path, source, budget=64, device_reserve=32)
            is not None
        )
    else:
        with pytest.raises(RuntimeError, match="device staging exceeds"):
            _select(monkeypatch, tmp_path, source, budget=64, device_reserve=32)
    source.cuda_integral_derivatives.assert_not_called()


def test_combined_unavailable_retries_complete_separate_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v1 owner must remain available to enlarged domains using an older library."""
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_REDUCTION", "combined")
    output = np.zeros((4, 2, 3))
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(side_effect=[None, (output, {})]),
    )
    np.testing.assert_array_equal(
        _select(monkeypatch, tmp_path, source, required=True), output
    )
    assert source.cuda_integral_derivatives.call_count == 2
    assert source.cuda_integral_derivatives.call_args_list[0].kwargs == {
        "range_exchange": False,
        "combined_two_electron": True,
    }
    assert source.cuda_integral_derivatives.call_args_list[1].kwargs == {
        "range_exchange": False
    }


def test_combined_failure_does_not_retry_a_different_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_REDUCTION", "combined")
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(side_effect=RuntimeError("stale owner")),
    )
    with pytest.raises(RuntimeError, match="stale owner"):
        _select(monkeypatch, tmp_path, source)
    assert source.cuda_integral_derivatives.call_count == 1


def test_default_complete_direct_owner_combines_two_electron_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENERATIVEQC_DIRECT_FORCE_REDUCTION")
    output = np.zeros((3, 2, 3))
    source = SimpleNamespace(
        density_fitted=False,
        cuda_integral_derivatives=Mock(return_value=(output, {})),
    )
    np.testing.assert_array_equal(
        _select(monkeypatch, tmp_path, source, required=True), output
    )
    source.cuda_integral_derivatives.assert_called_once_with(
        2, 32, range_exchange=False, combined_two_electron=True
    )


def test_cpu_reference_derivatives_reject_a_fitted_hamiltonian(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(NotImplementedError, match="change the Hamiltonian"):
        cpu_provider_selection(
            monkeypatch, tmp_path, SimpleNamespace(density_fitted=True), "reference"
        )
    assert cpu_provider_selection(
        monkeypatch, tmp_path, SimpleNamespace(density_fitted=True), "native"
    )
    assert not cpu_provider_selection(
        monkeypatch, tmp_path, SimpleNamespace(density_fitted=False), "reference"
    )
    assert not cpu_provider_selection(
        monkeypatch, tmp_path, SimpleNamespace(), "native"
    )


@pytest.mark.parametrize(
    "failure", ("missing", "not-callable", "unavailable", "no-budget")
)
def test_small_direct_domain_retains_bounded_ao_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    source = SimpleNamespace(density_fitted=False)
    if failure != "missing":
        source.cuda_integral_derivatives = (
            None if failure == "not-callable" else Mock(return_value=None)
        )
    assert (
        _select(
            monkeypatch, tmp_path, source, budget=0 if failure == "no-budget" else 32
        )
        is None
    )
    if failure == "no-budget":
        source.cuda_integral_derivatives.assert_not_called()


@pytest.mark.parametrize("error_type", (MemoryError, ValueError, TypeError))
def test_fitted_provider_error_identity_survives_owner_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    failure = error_type("injected fitted provider failure")
    source = SimpleNamespace(
        density_fitted=True,
        density_fitted_integral_derivatives=Mock(side_effect=failure),
        cuda_integral_derivatives=Mock(
            side_effect=AssertionError("Direct fallback changed the Hamiltonian")
        ),
    )
    with pytest.raises(error_type) as caught:
        _select(monkeypatch, tmp_path, source)
    assert caught.value is failure
    source.cuda_integral_derivatives.assert_not_called()


@pytest.mark.parametrize("invalid", (np.zeros((4, 2, 3)), np.full((3, 2, 3), np.nan)))
def test_malformed_combined_output_does_not_retry_separate_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid: np.ndarray,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_REDUCTION", "combined")

    def provide(*args: object, **kwargs: object) -> object:
        assert kwargs.get("combined_two_electron") is True, (
            "malformed output retried a different ABI"
        )
        return invalid, {}

    source = SimpleNamespace(density_fitted=False, cuda_integral_derivatives=provide)
    with pytest.raises(RuntimeError, match="invalid output"):
        _select(monkeypatch, tmp_path, source, required=True)
