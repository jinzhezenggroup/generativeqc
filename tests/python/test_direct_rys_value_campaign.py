"""Host-only checks that an explicitly requested Rys campaign cannot test nothing."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "tests/python/test_direct_rys_values_cuda.py"


def _portable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_ARCHITECTURE", "sm_80")
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_PROFILE", "portable")
    monkeypatch.delenv("GENERATIVEQC_RYS_VALUE_MANIFEST", raising=False)


@pytest.mark.parametrize("family", [None, "value", "task"])
def test_requested_empty_rys_family_fails_closed(
    monkeypatch: pytest.MonkeyPatch, family: str | None
) -> None:
    _portable_environment(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_RESOURCE_CUDA_TEST", "1")
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_LIBRARY", "/dummy/not-loaded.so")
    if family is None:
        monkeypatch.delenv("GENERATIVEQC_RYS_VALUE_FAMILY", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_FAMILY", family)
    with pytest.raises(ValueError, match="empty kernel inventory"):
        runpy.run_path(str(CAMPAIGN))


@pytest.mark.parametrize(
    "requested,library", [(False, False), (False, True), (True, False)]
)
def test_unrequested_rys_campaign_keeps_optional_skip(
    monkeypatch: pytest.MonkeyPatch, requested: bool, library: bool
) -> None:
    _portable_environment(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_FAMILY", "task")
    monkeypatch.setenv("GENERATIVEQC_RESOURCE_CUDA_TEST", "1" if requested else "0")
    if library:
        monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_LIBRARY", "/dummy/not-loaded.so")
    else:
        monkeypatch.delenv("GENERATIVEQC_RYS_VALUE_LIBRARY", raising=False)
    namespace = runpy.run_path(str(CAMPAIGN))
    assert not namespace["CLASSES"]
    assert namespace["pytestmark"].args[0] is True


def test_portable_primary_campaign_retains_real_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_environment(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_RESOURCE_CUDA_TEST", "1")
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_LIBRARY", "/dummy/not-loaded.so")
    monkeypatch.setenv("GENERATIVEQC_RYS_VALUE_FAMILY", "primary")
    namespace = runpy.run_path(str(CAMPAIGN))
    assert len(namespace["CLASSES"]) == 21
    assert "psss" in namespace["CLASSES"]
    assert "dddd" in namespace["CLASSES"]
    assert namespace["pytestmark"].args[0] is False
