"""Manifest packaging scope is independent of portable algorithm legality."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_shared_rys_xy import (
    shared_rys_xy_companions,
)

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "python/generativeqc_compiler/integral/production_shell_classes.json"
)


@pytest.mark.parametrize(
    "architecture", ("sm_75", "sm_80", "sm_90", "sm_100", "sm_120")
)
def test_stock_weighted_packaging_retains_qualified_target_scope(
    architecture: str,
) -> None:
    """Portable schedule reuse must not silently expand compiled masks."""
    profile = resolve_production_profile(MANIFEST, architecture, "portable")
    for item in profile.selections:
        if item.spec.name in ("dppp", "pppp"):
            assert "shared_rys_weighted_z_fock" in item.capabilities
            assert item.has_capability("shared_rys_weighted_z_fock") == (
                architecture == "sm_90"
            )
    assert len(shared_rys_xy_companions(profile, weighted_z=True)) == (
        2 if architecture == "sm_90" else 0
    )
    assert len(shared_rys_xy_companions(profile)) == 2


@pytest.mark.parametrize("architecture", ("sm_80", "sm_120", "sm_121"))
def test_manifest_can_explicitly_package_weighted_companions_on_another_target(
    tmp_path: Path, architecture: str
) -> None:
    """An explicit declaration, not an SM-name check, admits the same plan."""
    payload = json.loads(MANIFEST.read_text())
    for row in payload["architectures"]["portable_cuda"]["kernels"]:
        if "shared_rys_weighted_z_fock" in row.get("capabilities", []):
            row["capability_architectures"]["shared_rys_weighted_z_fock"] = [
                architecture
            ]
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(payload))
    profile = resolve_production_profile(manifest, architecture, "portable")
    assert {
        item.incumbent.spec.name: item.plan.block_threads
        for item in shared_rys_xy_companions(profile, weighted_z=True)
    } == {"dppp": 192, "pppp": 96}


def test_unscoped_declaration_retains_cross_target_opt_in(tmp_path: Path) -> None:
    """Custom manifests need no hard-coded architecture or profile exception."""
    payload = json.loads(MANIFEST.read_text())
    for row in payload["architectures"]["portable_cuda"]["kernels"]:
        row.pop("capability_architectures", None)
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(payload))
    profile = resolve_production_profile(manifest, "sm_80", "portable")
    assert len(shared_rys_xy_companions(profile, weighted_z=True)) == 2


@pytest.mark.parametrize("origin,destination", (("sm_90", "sm_80"), ("sm_80", "sm_90")))
def test_retargeting_does_not_transfer_another_targets_packaging_scope(
    origin: str, destination: str
) -> None:
    """Scopes survive immutable selection copies used for target-plan probes."""
    profile = resolve_production_profile(MANIFEST, origin, "portable")
    for item in profile.selections:
        if item.spec.name in ("dppp", "pppp"):
            retargeted = replace(item, architecture=destination)
            assert retargeted.has_capability("shared_rys_weighted_z_fock") == (
                destination == "sm_90"
            )


def test_scope_can_exclude_another_optional_companion(tmp_path: Path) -> None:
    """The schema is capability policy, not a renamed weighted-Z SM check."""
    payload = json.loads(MANIFEST.read_text())
    for row in payload["architectures"]["portable_cuda"]["kernels"]:
        if "shared_rys_xy_fock" in row.get("capabilities", []):
            row["capability_architectures"]["shared_rys_xy_fock"] = ["sm_90"]
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(payload))
    profile = resolve_production_profile(manifest, "sm_80", "portable")
    assert not shared_rys_xy_companions(profile)
    assert not shared_rys_xy_companions(profile, weighted_z=True)


def test_scope_cannot_change_primary_wrapper_packaging(tmp_path: Path) -> None:
    """Only independently inventoried companions support target scoping."""
    payload = json.loads(MANIFEST.read_text())
    for row in payload["architectures"]["portable_cuda"]["kernels"]:
        if row["shell_class"] == "ssss":
            row["fock_route"] = "streaming"
            row["capability_architectures"] = {"streaming_fock": ["sm_90"]}
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only supports companion capabilities"):
        resolve_production_profile(manifest, "sm_80", "portable")


@pytest.mark.parametrize(
    "scope,error",
    (
        (None, TypeError),
        ([], TypeError),
        ({"shared_rys_weighted_z_fock": "sm_90"}, TypeError),
        ({"shared_rys_weighted_z_fock": []}, TypeError),
        ({"shared_rys_weighted_z_fock": [False]}, TypeError),
        ({"shared_rys_weighted_z_fock": ["typo"]}, ValueError),
        ({"undeclared_capability": ["sm_90"]}, ValueError),
    ),
)
def test_invalid_capability_scope_fails_before_emission(
    tmp_path: Path, scope: object, error: type[Exception]
) -> None:
    payload = json.loads(MANIFEST.read_text())
    for row in payload["architectures"]["portable_cuda"]["kernels"]:
        if "shared_rys_weighted_z_fock" in row.get("capabilities", []):
            row["capability_architectures"] = scope
    manifest = tmp_path / "profile.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(error):
        resolve_production_profile(manifest, "sm_80", "portable")
