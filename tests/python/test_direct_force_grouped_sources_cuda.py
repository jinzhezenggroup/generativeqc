"""Spatial Direct queues preserve independent observable J'/K' channels."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_spatial_force_sources_match_libcint(
    method: str, representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the spatial owner even below its default size admission boundary."""
    from test_cuda_hybrid_snapshot import (
        test_separate_full_range_derivatives_match_libcint as check_sources,
    )

    monkeypatch.setenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "relative-density")
    check_sources(method, representation, "1", True, "bounded", monkeypatch)
