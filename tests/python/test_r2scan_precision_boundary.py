"""r2SCAN AUTO precision should reach CUDA runtime admission."""

import pytest
from generativeqc import Calculator, _native


class _RuntimeLoaded(Exception):
    pass


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_auto_cuda_reaches_runtime_loading(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def loaded(**kwargs: object) -> None:
        raise _RuntimeLoaded

    monkeypatch.setattr(_native, "load_library", loaded)
    with pytest.raises(_RuntimeLoaded):
        Calculator(method=method, device="cuda", precision="auto")
