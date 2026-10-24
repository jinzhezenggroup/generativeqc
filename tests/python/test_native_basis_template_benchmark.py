"""The endpoint harness must reject unsafe receipts before runtime imports."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from benchmarks import _retention, native_basis_template_endpoint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("existing", [False, True])
def test_endpoint_rejects_unsafe_output_before_science(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    destination = tmp_path / (
        "receipt.json" if existing else "benchmarks/results/raw.json"
    )
    if existing:
        destination.write_text("retained receipt")
    monkeypatch.delenv("GENERATIVEQC_LIBRARY", raising=False)
    monkeypatch.setitem(sys.modules, "generativeqc", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_basis_template_endpoint",
            "--source-sha",
            "test",
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        native_basis_template_endpoint.main()
    assert failure.value.code == 2
    assert ("overwrite" if existing else "raw_output_path") in capsys.readouterr().err
    if existing:
        assert destination.read_text() == "retained receipt"
    else:
        assert not destination.exists()
