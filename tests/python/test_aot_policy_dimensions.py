"""Build intent, compiled consumers and preferred routes are separate contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.integral.derivative_cuda_shell_aot import (
    cuda_derivative_shell_packages,
)
from generativeqc_compiler.integral.production_bundle import write_production_bundles
from generativeqc_compiler.integral.production_contracts import AotPackaging
from generativeqc_compiler.integral.production_k_block import direct_k_block_candidates
from generativeqc_compiler.integral.production_policy import (
    cuda_image_inventory,
    production_policy_inventory,
    profile_overrides,
    resolved_profile_inventory,
)
from generativeqc_compiler.integral.production_profile import (
    ProfileMatch,
    resolve_production_profile,
)
from generativeqc_compiler.integral.production_rys_tasks import (
    direct_rys_task_candidates,
    preferred_rys_task_candidates,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90", "sm_120"))
@pytest.mark.parametrize("family", ("rys_value", "k_block", "rys_task"))
@pytest.mark.parametrize("relabel", (False, True))
def test_optional_fock_families_are_independently_packaged(
    architecture: str,
    family: str,
    relabel: bool,
) -> None:
    """One requested alternative must not bring unrelated code into its module."""
    original = resolve_production_profile(MANIFEST, architecture)
    selected = replace(
        original,
        packaging=AotPackaging(fock_variants=frozenset({family})),
        prefer_rys_task_fock=False,
        preferred_rys_task_fock_shell_classes=None,
    )
    if relabel:
        selected = replace(
            selected, tuned=not selected.tuned, match=ProfileMatch.PORTABLE
        )
    for name, candidates in (
        ("rys_value", direct_rys_value_candidates),
        ("k_block", direct_k_block_candidates),
        ("rys_task", direct_rys_task_candidates),
    ):
        assert bool(candidates(selected)) == (name == family)
    assert selected.selections == original.selections
    assert preferred_rys_task_candidates(selected) == ()


@pytest.mark.parametrize(
    "policy",
    (
        {"fock_variants": ["unknown"]},
        {"fock_variants": ["rys_task", "rys_task"]},
        {"fock_variants": [None]},
        {"unexpected": True},
    ),
)
def test_invalid_family_packaging_is_rejected(tmp_path: Path, policy: dict) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["architectures"]["sm_120"]["packaging"] = policy
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises((TypeError, ValueError)):
        resolve_production_profile(manifest, "sm_120")


@pytest.mark.parametrize(
    "field", ("small_work_angular_queue", "require_bounded_spd_coverage")
)
@pytest.mark.parametrize("value", (1, "true", None))
def test_force_policy_requires_explicit_booleans(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["architectures"]["sm_120"]["force_policy"][field] = value
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="force execution"):
        resolve_production_profile(manifest, "sm_120")


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90", "sm_120"))
def test_portable_identity_is_not_a_compilation_capability(architecture: str) -> None:
    portable = resolve_production_profile(MANIFEST, architecture, "portable")
    assert not portable.packaging.derivative_shells
    assert not portable.prefer_rys_task_fock
    assert cuda_derivative_shell_packages(portable, ()) == ()
    # Opt-in compilation proves inventory construction, not device qualification.
    explicit = replace(
        portable, packaging=replace(portable.packaging, derivative_shells=True)
    )
    assert cuda_derivative_shell_packages(explicit, ())
    inventory = resolved_profile_inventory(explicit)
    assert inventory["selection"]["portable"]
    assert inventory["compiled_masks"]["exact_derivative_shells"] > 0
    assert inventory["default_preferences"]["rys_task_fock"] == 0


def test_metadata_relabeling_does_not_change_explicit_packaging() -> None:
    tuned = resolve_production_profile(MANIFEST, "sm_120")
    relabeled = replace(tuned, match=ProfileMatch.PORTABLE, tuned=False)
    assert cuda_derivative_shell_packages(tuned, ()) == cuda_derivative_shell_packages(
        relabeled, ()
    )
    assert (
        resolved_profile_inventory(tuned)["compiled_masks"]
        == resolved_profile_inventory(relabeled)["compiled_masks"]
    )
    lean = replace(
        tuned,
        packaging=replace(tuned.packaging, fock_variants=frozenset()),
        prefer_rys_task_fock=False,
    )
    inventory = resolved_profile_inventory(lean)
    assert (
        inventory["compiled_masks"]["force"]
        == resolved_profile_inventory(tuned)["compiled_masks"]["force"]
    )
    assert (
        inventory["compiled_masks"]["fock"]
        == resolved_profile_inventory(tuned)["compiled_masks"]["fock"]
    )
    assert inventory["compiled_masks"]["rys_task_fock"] == 0
    assert inventory["default_preferences"]["rys_task_fock"] == 0


@pytest.mark.parametrize("field", ("derivative_shells", "prefer_rys_task_fock"))
@pytest.mark.parametrize("value", (1, "true", None))
def test_policy_fields_require_explicit_booleans(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(MANIFEST.read_text())
    target = payload["architectures"]["sm_120"]
    if field == "derivative_shells":
        target["packaging"][field] = value
    else:
        target[field] = value
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match=field):
        resolve_production_profile(manifest, "sm_120")


def test_absent_policy_never_infers_admission_from_tuned_kind(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    for field in ("packaging", "prefer_rys_task_fock", "force_policy"):
        payload["architectures"]["sm_120"].pop(field)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    profile = resolve_production_profile(manifest, "sm_120")
    assert profile.tuned
    assert not profile.packaging.derivative_shells
    assert not profile.prefer_rys_task_fock
    assert profile.force_policy.small_work_angular_queue
    assert not profile.force_policy.require_bounded_spd_coverage


@pytest.mark.parametrize(
    "entries",
    (
        ("sm_80=auto", "80-real=portable"),
        ("sm_90=portable",),
        ("sm_80=",),
        ("sm_80= portable",),
    ),
)
def test_profile_overrides_fail_closed(entries: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        profile_overrides(("80", "120-real"), entries)


def test_images_are_independent_of_profile_and_device_qualification() -> None:
    assert cuda_image_inventory(
        ("sm_80", "sm_120"), ("80-real", "120-virtual", "120-real")
    ) == {"sm_80": ["80-real"], "sm_120": ["120-real", "120-virtual"]}
    assert cuda_image_inventory(("sm_120",), ()) == {"sm_120": []}
    with pytest.raises(ValueError, match="no configured target"):
        cuda_image_inventory(("sm_80",), ("120-real",))


def test_bundle_records_global_and_per_target_intent(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    for profile in payload["architectures"].values():
        profile["kernels"] = [
            item for item in profile["kernels"] if item["shell_class"] == "ssss"
        ]
        profile["preferred_rys_task_fock_shell_classes"] = []
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "output"
    targets = ("sm_120", "sm_90", "sm_80")
    images = ("120-real", "120-virtual", "90", "80-real")
    overrides = {"sm_120": "auto"}
    write_production_bundles(
        manifest,
        output,
        1,
        targets,
        overrides,
        default_profile="portable",
        cuda_images=images,
    )
    record = json.loads((output / "generativeqc_aot_build_policy.json").read_text())
    assert record["schema_version"] == 2
    by_target = {item["architecture"]: item for item in record["targets"]}
    assert by_target["sm_120"]["selection"] == {
        "requested": "auto",
        "request_source": "target-override",
        "resolved": "sm_120",
        "match": "exact",
        "tuned": True,
        "portable": False,
    }
    assert by_target["sm_120"]["cuda_images"] == ["120-real", "120-virtual"]
    for target in ("sm_80", "sm_90"):
        assert by_target[target]["selection"]["requested"] == "portable"
        assert by_target[target]["selection"]["request_source"] == "global-default"
        assert by_target[target]["selection"]["resolved"] == "portable_cuda"
        assert by_target[target]["compiled_masks"]["fock"] > 0
        assert not by_target[target]["packaging"]["derivative_shells"]
    profiles = tuple(
        resolve_production_profile(manifest, target, overrides.get(target, "portable"))
        for target in targets
    )
    assert record == production_policy_inventory(
        profiles, images=images, overrides=overrides
    )


@pytest.mark.parametrize(
    "generator", ("generate_shell_kernels.py", "generate_derivative_cuda_shell_aot.py")
)
@pytest.mark.parametrize(
    "entries", (("sm_90=portable",), ("sm_80=auto", "80-real=portable"))
)
def test_both_generators_reject_ambiguous_or_unused_target_overrides(
    tmp_path: Path, generator: str, entries: tuple[str, ...]
) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / generator),
        "--production-manifest",
        str(MANIFEST),
        "--target-architecture",
        "sm_80",
    ]
    command += (
        ["--output-directory", str(tmp_path / "out")]
        if generator == "generate_shell_kernels.py"
        else ["--output", str(tmp_path / "out.cuh")]
    )
    for entry in entries:
        command.extend(("--profile-map", entry))
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 2
    assert "unique configured target" in result.stderr


@pytest.mark.parametrize(
    "overrides,success",
    (
        ("sm_80=portable;sm_120=auto", True),
        ("sm_120;sm_120=portable", False),
        ("sm_90=portable", False),
        ("portable", False),
    ),
)
def test_cmake_target_override_collision_and_priority(
    tmp_path: Path, overrides: str, success: bool
) -> None:
    script = tmp_path / "profiles.cmake"
    result_file = tmp_path / "arguments.txt"
    script.write_text(f'''
cmake_minimum_required(VERSION 3.24)
include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")
set(_generativeqc_cuda_compile_architectures "120-real;80-real")
set(GENERATIVEQC_AOT_PROFILE "portable")
set(GENERATIVEQC_AOT_PROFILES "{overrides}")
generativeqc_cuda_profile_arguments(arguments)
file(WRITE "{result_file}" "${{arguments}}")
''')
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) == success
    if success:
        arguments = result_file.read_text().split(";")
        assert arguments[arguments.index("--profile") + 1] == "portable"
        assert "sm_120=auto" in arguments and "sm_80=portable" in arguments
