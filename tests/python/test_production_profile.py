"""Production manifest/profile ownership stays separate from emission orchestration."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from generativeqc_compiler.integral import production_profile


def test_production_profile_avoids_emission_orchestration_dependencies() -> None:
    """Manifest/profile parsing must not depend on emission or bundle owners."""

    source = Path(production_profile.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    absolute_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert relative_modules.isdisjoint(
        {"production", "production_cost", "cuda_emitter", "cuda_lowering"}
    )
    assert absolute_roots.isdisjoint({"benchmarks", "tools", "generativeqc"})


def test_resolved_profile_is_owned_by_profile_module() -> None:
    """The canonical profile result type is no longer defined by the facade."""

    assert production_profile.ResolvedProductionProfile.__module__.endswith(
        ".production_profile"
    )


@pytest.mark.parametrize("invalid", (0, 1, "false", None))
def test_fock_packaging_requires_an_explicit_list(
    tmp_path: Path, invalid: object
) -> None:
    """Reject ambiguous build policy instead of silently omitting kernels."""
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "architectures": {
                    "portable_cuda": {
                        "kind": "portable",
                        "kernels": [],
                        "packaging": {"fock_variants": invalid},
                    }
                },
            }
        )
    )
    with pytest.raises(TypeError, match="fock_variants must be a list"):
        production_profile.resolve_production_profile(manifest, "sm_80")


@pytest.mark.parametrize(
    "obsolete", ("optional_fock_variants", "compile_derivative_shells")
)
def test_obsolete_live_packaging_is_not_a_compatibility_alias(
    tmp_path: Path, obsolete: str
) -> None:
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "architectures": {
                    "portable_cuda": {"kind": "portable", "kernels": [], obsolete: True}
                },
            }
        )
    )
    with pytest.raises(ValueError, match="use explicit packaging"):
        production_profile.resolve_production_profile(manifest, "sm_80")


@pytest.mark.parametrize(
    "preference, error, message",
    (
        ({"class_set": "missing"}, ValueError, "unknown Rys-task Fock class set"),
        (
            {"class_set": "bounded_spd", "classes": []},
            ValueError,
            "requires classes or class_set",
        ),
        (
            {"classes": [], "architectures": None},
            TypeError,
            "architectures must be a list",
        ),
        (
            {"classes": [], "architectures": "sm_90"},
            TypeError,
            "architectures must be a list",
        ),
        (
            {"classes": [], "architectures": [90]},
            TypeError,
            "architectures must be a list",
        ),
        (
            {"classes": [], "architectures": ["sm_90", "90-real"]},
            ValueError,
            "duplicate Rys-task preference architecture",
        ),
        (
            {"classes": [], "unexpected": True},
            ValueError,
            "unknown Rys-task preference field",
        ),
        ({"classes": ["psps", "psps"]}, ValueError, "contain duplicates"),
    ),
)
def test_scoped_rys_preferences_reject_ambiguous_evidence(
    tmp_path: Path, preference: object, error: type[Exception], message: str
) -> None:
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "architectures": {
                    "portable_cuda": {
                        "kind": "portable",
                        "kernels": [],
                        "preferred_rys_task_fock_shell_classes": preference,
                    }
                },
            }
        )
    )
    with pytest.raises(error, match=message):
        production_profile.resolve_production_profile(manifest, "sm_90")


@pytest.mark.parametrize("architecture", (None, "sm_80", "sm_120"))
@pytest.mark.parametrize("acceptance", (None, {}, {"architecture": "sm_90"}))
def test_legacy_manifest_cannot_invent_architecture_qualification(
    tmp_path: Path, architecture: str | None, acceptance: object
) -> None:
    """Retired manifests must be migrated, never silently relabeled as SM120."""
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shell_classes": ["dsss"],
                "acceptance": acceptance,
            }
        )
    )
    with pytest.raises(ValueError, match="use schema_version 2"):
        production_profile.resolve_production_profile(manifest, architecture)


def test_explicit_target_does_not_require_manifest_default(tmp_path: Path) -> None:
    """A target supplied by the build is authoritative even with many profiles."""
    manifest = tmp_path / "profile.json"
    payload = {
        "schema_version": 2,
        "architectures": {
            "sm_80": {"kind": "tuned", "kernels": []},
            "sm_90": {"kind": "tuned", "kernels": []},
            "portable_cuda": {"kind": "portable", "kernels": []},
        },
    }
    manifest.write_text(json.dumps(payload))
    resolved = production_profile.resolve_production_profile(manifest, "sm_90")
    assert resolved.profile == "sm_90" and resolved.tuned
    assert resolved.target.architecture == "sm_90"
    portable = production_profile.resolve_production_profile(manifest, "sm_100")
    assert portable.portable and not portable.tuned
    assert portable.target.architecture == "sm_100"
    with pytest.raises(ValueError, match="requires default_architecture"):
        production_profile.resolve_production_profile(manifest)


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90", "sm_120"))
def test_omitted_target_uses_manifest_evidence_only(
    tmp_path: Path, architecture: str
) -> None:
    manifest = tmp_path / "profile.json"
    payload = {
        "schema_version": 2,
        "architectures": {architecture: {"kind": "tuned", "kernels": []}},
    }
    manifest.write_text(json.dumps(payload))
    assert (
        production_profile.resolve_production_profile(manifest).target.architecture
        == architecture
    )
    payload["default_architecture"] = architecture
    payload["architectures"]["portable_cuda"] = {"kind": "portable", "kernels": []}
    manifest.write_text(json.dumps(payload))
    assert (
        production_profile.resolve_production_profile(manifest).target.architecture
        == architecture
    )


@pytest.mark.parametrize("architecture", ("", " "))
def test_invalid_explicit_target_cannot_select_manifest_default(
    tmp_path: Path, architecture: str
) -> None:
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_90",
                "architectures": {"sm_90": {"kind": "tuned", "kernels": []}},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid CUDA architecture"):
        production_profile.resolve_production_profile(manifest, architecture)
