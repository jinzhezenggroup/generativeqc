"""Validate benchmark-only controls without constructing a CUDA context."""

from __future__ import annotations

import os
import subprocess

import pytest


@pytest.fixture(scope="module")
def endpoint_binary() -> str:
    """Use the explicitly built executable; invalid inputs stop before GPU setup."""
    binary = os.environ.get("GENERATIVEQC_DF_FORCE_ENDPOINT_BINARY")
    if not binary:
        pytest.skip("requires the compiled DF force endpoint benchmark")
    return binary


def invoke(endpoint_binary: str, *trailing: str) -> subprocess.CompletedProcess[str]:
    """Preserve all established positional slots while appending RHF controls."""
    return subprocess.run(
        [
            endpoint_binary,
            "/nonexistent/generativeqc-endpoint-input",
            "/nonexistent/generativeqc-endpoint-output",
            "1",
            "1",
            "1",
            "1",
            "8",
            "8",
            "8",
            "0",
            "1",
            "2",
            "1",
            "30",
            "0",
            "0",
            "1",
            "auto",
            "0",
            "1",
            *trailing,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("trailing", [(), ("1e-12",), ("1e-13",)])
def test_reference_tolerance_accepts_defaults_or_tighter_values(
    endpoint_binary: str, trailing: tuple[str, ...]
) -> None:
    completed = invoke(endpoint_binary, *trailing)
    assert completed.returncode != 0
    assert "invalid molecular probe dimensions" in completed.stderr


@pytest.mark.parametrize("tolerance", ["0", "-1e-13", "nan", "inf", "1e-11", "1e-13x"])
def test_reference_tolerance_rejects_invalid_or_looser_values(
    endpoint_binary: str, tolerance: str
) -> None:
    completed = invoke(endpoint_binary, tolerance)
    assert completed.returncode != 0
    assert "invalid reference tolerance" in completed.stderr


def test_reference_tolerance_rejects_extra_arguments(endpoint_binary: str) -> None:
    completed = invoke(endpoint_binary, "1e-13", "1e-13")
    assert completed.returncode != 0
    assert "usage: df-force-endpoint" in completed.stderr
