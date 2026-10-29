"""Exercise real RKS Hessian imports and public lazy exports."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("name", "owner"),
    (
        ("DirectionalRKSBatchResponse", "rks_hessian_directional"),
        ("directional_rks_responses", "rks_hessian_directional"),
        ("RKSHVPBatchResult", "rks_hessian"),
        ("RKSHessianResult", "rks_hessian"),
        ("rks_hvp_many", "rks_hessian"),
        ("rks_hessian", "rks_hessian"),
    ),
)
def test_public_api_retains_batch_and_full_hessian_exports(
    name: str, owner: str
) -> None:
    # Real imports catch missing directional dependencies in the molecular owner
    # without requiring definitions or import statements at fixed source locations.
    facade = importlib.import_module("tools.generativeqc_hessian")
    installed = importlib.import_module(f"generativeqc.{owner}")

    assert name in facade.__all__
    exported = getattr(facade, name)
    assert callable(exported)
    assert exported is getattr(installed, name)
