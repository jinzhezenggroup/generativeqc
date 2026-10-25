"""Guard the compiled stationary AOT profile inventory against drift."""

from pathlib import Path

from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_FUNCTIONALS,
    QUALIFIED_STATIONARY_AOT_PROFILES,
)

from tools import generate_stationary_aot_profiles as generator


def test_stationary_aot_manifest_generates_package_metadata() -> None:
    output = (
        Path(__file__).resolve().parents[2]
        / "python/generativeqc_compiler/method/_generated_stationary_aot_profiles.py"
    )
    assert output.read_text(encoding="utf-8") == generator.emit_python()


def test_aot_profile_legacy_transport_is_generated_without_name_branch() -> None:
    assert QUALIFIED_FUNCTIONALS == (0, 1, 2)
    assert len(QUALIFIED_STATIONARY_AOT_PROFILES) == 16
    assert {profile.name for profile in QUALIFIED_STATIONARY_AOT_PROFILES} == {
        "lda",
        "pbe",
        "r2scan",
        "pbe0",
        "b3lyp",
        "m06_2x",
        "mn15",
        "wb97mv",
    }
    assert all(profile.plan.identity for profile in QUALIFIED_STATIONARY_AOT_PROFILES)
