"""Complete spherical orbital-f CPU ECP forces on one bounded worker."""

import typing
from pathlib import Path

import pytest
import test_ecp_public_cpu as gates

retain_endpoint_evidence = gates.retain_endpoint_evidence


@pytest.mark.parametrize(
    "method", ["lda-rks", "pbe-rks", "b3lyp-rks", "lda-uks", "pbe-uks", "b3lyp-uks"]
)
def test_cpu_f_ecp_forces_owned_by_prepared_provider(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: typing.Any,
    tmp_path: Path,
) -> None:
    """Retain the native-owner guard and full independent analytic/FD gate."""
    gates.check_cpu_f_ecp_forces_owned_by_prepared_provider(
        method, "spherical", monkeypatch, record_property, tmp_path
    )
