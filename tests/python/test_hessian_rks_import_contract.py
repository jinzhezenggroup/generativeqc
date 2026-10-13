"""Exercise real RKS Hessian imports and public compatibility exports."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("name", "compatibility", "owner"),
    (
        ("DirectionalRKSBatchResponse", "rks_directional", "rks_hessian_directional"),
        ("directional_rks_responses", "rks_directional", "rks_hessian_directional"),
        ("RKSHVPBatchResult", "rks_molecular", "rks_hessian"),
        ("RKSHessianResult", "rks_molecular", "rks_hessian"),
        ("rks_hvp_many", "rks_molecular", "rks_hessian"),
        ("rks_hessian", "rks_molecular", "rks_hessian"),
    ),
)
def test_public_api_retains_batch_and_full_hessian_exports(
    name: str, compatibility: str, owner: str
) -> None:
    # Real imports catch missing directional dependencies in the molecular owner
    # without requiring definitions or import statements at fixed source locations.
    facade = importlib.import_module("tools.generativeqc_hessian")
    legacy = importlib.import_module(f"tools.generativeqc_hessian.{compatibility}")
    installed = importlib.import_module(f"generativeqc.{owner}")

    assert name in facade.__all__
    exported = getattr(facade, name)
    assert callable(exported)
    assert exported is getattr(legacy, name)
    assert exported is getattr(installed, name)
