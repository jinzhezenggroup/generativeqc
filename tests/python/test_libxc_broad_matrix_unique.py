"""Broad-matrix breadth requires distinct registrations, not repeated attempts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from test_libxc_broad_matrix import _candidate

from tools import qualify_libxc_broad_matrix as matrix

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("same_object", (True, False))
def test_duplicate_names_reject_before_producers_and_output(
    same_object: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate("repeated-gga", "gga")
    repeated = candidate if same_object else _candidate("repeated-gga", "gga")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("duplicate matrix started a qualification producer")

    monkeypatch.setattr(matrix, "_eligible", lambda capability: True)
    monkeypatch.setattr(matrix, "qualify_functional", forbidden)
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="unique functional names"):
        matrix.run_matrix(
            (candidate, repeated),
            output=output,
            evidence_prefix="test://unique-campaign",
            build_dir=tmp_path,
            pyscf_version="not-run",
            libxc=None,
            quotas={"lda": 1, "gga": 2, "mgga": 1},
        )
    assert not output.exists()
