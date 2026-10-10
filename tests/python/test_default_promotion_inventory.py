"""Regression coverage for the #1598 default-promotion inventory."""

from __future__ import annotations

import copy
import shutil
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tools.check_default_promotion_inventory import (
    DEFAULT_INVENTORY,
    ROOT,
    load_and_validate,
    validate_inventory,
)


def _payload() -> dict:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    return payload


def _copy_audited_sources(payload: dict, destination_root: Path) -> None:
    sources = set(payload["scope"]["audited_sources"])
    for entry in payload["entries"]:
        sources.update(entry["sources"])
    for relative in sorted(sources):
        source = ROOT / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def test_current_default_promotion_inventory_is_complete() -> None:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    assert payload["tracking_issue"] == 1598
    assert len(payload["entries"]) >= 10


@pytest.mark.parametrize(
    "relative,before,after",
    [
        ("src/scf/cuda/reference_eri_policy.hpp", "!value ||", "false ||"),
        ("src/scf/cuda/reference_eri_policy.hpp", "if (!cold_reference)", "if (false)"),
        (
            "src/scf/cuda/reference_eri_policy.hpp",
            "maximum_angular != 3",
            "maximum_angular > 3",
        ),
        (
            "src/scf/cuda/reference_eri_policy.hpp",
            "direct_nbf < 128",
            "direct_nbf < 64",
        ),
        (
            "src/scf/cuda/reference_eri_policy.hpp",
            "maximum_iterations < 8",
            "maximum_iterations < 4",
        ),
        (
            "src/scf/cuda/rhf_resident_values.cpp",
            "budget, 8ULL << 30",
            "budget, 16ULL << 30",
        ),
        (
            "src/scf/cuda/rhf_resident_values.cpp",
            "device_overhead, 256ULL << 20",
            "device_overhead, 0",
        ),
    ],
)
def test_rhf_phase_value_default_domain_is_audited(
    tmp_path: Path, relative: str, before: str, after: str
) -> None:
    """Widening automatic construction needs renewed endpoint qualification."""
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / relative
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "RHF phase-value auto default or admission domain drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("guess.work_amortization_ratio < 1.0", "guess.work_amortization_ratio < 0.1"),
        ("preliminary_iterations = 32", "preliminary_iterations = 50"),
        ("guess.preparation_peak_bytes > guess.value_budget_bytes", "false"),
    ],
)
def test_df_rhf_guess_default_domain_is_audited(
    tmp_path: Path, before: str, after: str
) -> None:
    """A broader workload or additional provisional work needs a promotion decision."""
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/methods/df_hf_guess.cpp"
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "DF-RHF preconvergence default or admission domain drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


@pytest.mark.parametrize(
    "before,after",
    [
        (
            '"GENERATIVEQC_CUDA_XC_BATCH_TILES", 32',
            '"GENERATIVEQC_CUDA_XC_BATCH_TILES", 1',
        ),
        (
            '"GENERATIVEQC_CUDA_XC_BATCH_BYTES", 32 * 1024 * 1024',
            '"GENERATIVEQC_CUDA_XC_BATCH_BYTES", 64 * 1024 * 1024',
        ),
    ],
)
def test_native_xc_batch_defaults_are_audited(
    tmp_path: Path, before: str, after: str
) -> None:
    """Changing either default limit requires revisiting the recorded decision."""
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/dft/cuda_ks.cpp"
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "XC point-batch default drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


@pytest.mark.parametrize(
    "relative,before,after",
    [
        ("src/dft/cuda_ks_final_validation_policy.hpp", "aos >= 384", "aos >= 17"),
        ("src/dft/cuda_ks_final_validation_policy.hpp", "spins == 1", "spins <= 2"),
        (
            "src/dft/cuda_ks_final_validation_policy.hpp",
            "if (!setting) return default_eligible;",
            "if (!setting) return true;",
        ),
        (
            "src/dft/cuda_ks_final_validation_policy.hpp",
            "fock_exchange_coefficient == -0.125",
            "fock_exchange_coefficient == -0.25",
        ),
        (
            "src/dft/cuda_ks.cpp",
            "!final_stationary_weights_ready;",
            "true;",
        ),
    ],
)
def test_native_ks_final_validation_default_is_audited(
    tmp_path: Path, relative: str, before: str, after: str
) -> None:
    """Default promotion must not silently expand the measured scope or leases."""
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / relative
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "KS final-validation default" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


def test_fixture_copies_registered_sources_outside_the_audited_scope(
    tmp_path: Path,
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    for entry in payload["entries"]:
        for relative in entry["sources"]:
            assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes()
    assert not validate_inventory(payload, root=tmp_path)


@pytest.mark.parametrize(
    "before,after",
    [
        ("if profile.portable:", "if False:"),
        (
            "schedule_candidates(integral, profile.target)",
            "schedule_candidates(integral, fallback_target)",
        ),
        ("if schedule is None:", "if False:"),
        (
            "if profile.portable:",
            'if profile.target.architecture != "sm_120" or profile.portable:',
        ),
        (
            '"dpps", "dspp"}',
            '"dpps", "dspp", "ssss"}',
        ),
    ],
)
def test_rys_task_default_capability_admission_is_audited(
    tmp_path: Path, before: str, after: str
) -> None:
    """Retain shared classes, actual-target scheduling and portable fallback."""
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "python/generativeqc_compiler/integral/production_rys_tasks.py"
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "Rys-task default target/class admission drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


@pytest.mark.parametrize(
    "before,after",
    [
        (
            "return exchange ? DirectFockLowering::Default : DirectFockLowering::Incumbent;",
            "return exchange ? DirectFockLowering::RysTask : DirectFockLowering::Incumbent;",
        ),
        (
            "selection.rys_task_fock_mask &= class_mask;",
            "selection.rys_task_fock_mask |= class_mask;",
        ),
        (
            "selection.rys_task_fock_mask = generated::preferred_rys_task_fock_shell_class_mask();",
            "selection.rys_task_fock_mask = generated::enabled_rys_task_fock_shell_class_mask();",
        ),
    ],
)
def test_unified_k_selector_cannot_bypass_preference_or_coverage(
    tmp_path: Path, before: str, after: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/cuda/direct_fock_lowering.hpp"
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "Rys-task default selection/filter guard drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


def test_work_default_must_be_audited_independently_of_task_preference(
    tmp_path: Path,
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/cuda/direct_fock_lowering.hpp"
    original = source.read_text()
    source.write_text(
        original.replace(
            "return detail::GeneratedExchangeTaskSchedule::Work;",
            "return detail::GeneratedExchangeTaskSchedule::Fill;",
            1,
        )
    )
    assert any(
        "Direct K work schedule default/selection guard drifted" in error
        for error in validate_inventory(payload, root=tmp_path)
    )


def test_inventory_requires_owner_rationale_and_revisit_condition() -> None:
    payload = _payload()
    for field, value in (
        ("owner_issues", []),
        ("rationale", ""),
        ("revisit_condition", ""),
    ):
        candidate = copy.deepcopy(payload)
        candidate["entries"][0][field] = value
        errors = validate_inventory(candidate, root=ROOT, check_sources=False)
        assert any(field in error for error in errors)


def test_inventory_rejects_duplicate_control_registration() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    control = candidate["entries"][0]["controls"][0]
    candidate["entries"][1]["controls"].append(control)
    errors = validate_inventory(candidate, root=ROOT, check_sources=False)
    assert any("registered by both" in error for error in errors)


def test_inventory_rejects_missing_audited_control() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    target = "tensor-execution:cuda-graph"
    for entry in candidate["entries"]:
        if target in entry["controls"]:
            entry["controls"].remove(target)
            break
    errors = validate_inventory(candidate, root=ROOT)
    assert any(target in error and "unregistered" in error for error in errors)


def test_new_tensor_schedule_opt_in_must_be_registered(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)

    plan = tmp_path / "python/generativeqc_compiler/tensor/cuda_plan.py"
    source = plan.read_text()
    source = source.replace(
        "    direct_gemm: bool = True\n",
        "    new_default_off_path: bool = False\n    direct_gemm: bool = True\n",
        1,
    )
    assert "new_default_off_path" in source
    plan.write_text(source)

    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        "tensor-schedule:new_default_off_path" in error and "unregistered" in error
        for error in errors
    )


@pytest.mark.parametrize("prefix", ["experimental_", "incremental_"])
@pytest.mark.parametrize(
    "initializer", ["{}", "{false}", " = false", "{true}", " = true", " = policy()", ""]
)
def test_new_scf_control_must_be_registered_independently_of_initializer(
    tmp_path: Path, prefix: str, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    name = f"{prefix}new_path"
    source = tmp_path / "src/scf/types.hpp"
    original = source.read_text()
    changed = original.replace(
        "struct ScfOptions {", f"struct ScfOptions {{\n  bool {name}{initializer};", 1
    )
    assert changed != original
    source.write_text(changed)

    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        f"scf-option:{name}" in error and "unregistered" in error for error in errors
    )


def test_commented_scf_declarations_are_not_controls(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text()
        + "\n// bool experimental_removed_path{false};\n"
        + "/* bool incremental_removed_path = false; */\n"
    )
    assert not validate_inventory(payload, root=tmp_path)


def test_scf_comment_separates_type_and_control_name(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text().replace(
            "struct ScfOptions {",
            "struct ScfOptions {\n  bool/* explanation */experimental_new_path{false};",
            1,
        )
    )
    errors = validate_inventory(payload, root=tmp_path)
    assert any("scf-option:experimental_new_path" in error for error in errors)


@pytest.mark.parametrize("initializer", ["{false}", " = false", " = policy()"])
def test_registered_scf_control_survives_initializer_changes(
    tmp_path: Path, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text().replace(
            "bool incremental_direct_jk{};",
            f"bool incremental_direct_jk{initializer};",
            1,
        )
    )
    assert not validate_inventory(payload, root=tmp_path)


def test_tensor_schedule_classifications_preserve_existing_decisions() -> None:
    entries = {
        control: entry
        for entry in _payload()["entries"]
        for control in entry["controls"]
    }
    for name in ("stream_reductions", "streamed_gemm_reduction"):
        entry = entries[f"tensor-schedule:{name}"]
        assert entry["classification"] == "negative-evidence"
        assert any("generated-reduction-h100" in item for item in entry["evidence"])
    assert entries["tensor-schedule:direct_gemm"]["classification"] == "already-default"
    assert (
        entries["tensor-schedule:layouts"]["classification"]
        == "guarded-promotion-candidate"
    )


def test_policy_taxonomy_declares_scientific_choices() -> None:
    payload = _payload()
    assert "scientific-choice" in payload["classifications"]
    candidate = copy.deepcopy(payload)
    candidate["classifications"].remove("scientific-choice")
    errors = validate_inventory(candidate, root=ROOT, check_sources=False)
    assert any("policy taxonomy" in error for error in errors)


def test_scientific_model_and_projection_choices_remain_explicit() -> None:
    entries = {
        control: entry
        for entry in _payload()["entries"]
        for control in entry["controls"]
    }
    for control in (
        "public-model:density-fitting",
        "public-model:cosx-exchange",
        "initial-guess:basis-projection",
    ):
        assert entries[control]["classification"] == "scientific-choice"
    assert (
        entries["initial-guess:preliminary-scf"]["classification"]
        == "needs-qualification"
    )
    assert (
        entries["cc-option:packed_diis"]["classification"]
        == "guarded-promotion-candidate"
    )


def test_cc_batched_replay_registration_records_scoped_automatic_use() -> None:
    """One-slice internal initialization must not hide the CUDA energy owner."""
    entry = next(
        item
        for item in _payload()["entries"]
        if "cc-option:df_replay_auxiliary_batch" in item["controls"]
    )
    assert entry["classification"] == "already-default"
    assert {
        "src/cc/solver.hpp",
        "src/cc/cuda_solver.cu",
        "src/methods/rccsd_method.cpp",
        "tools/generate_df_ccsd_native.py",
    } <= set(entry["sources"])
    assert "tests/python/test_df_cc_batched_replay.py" in entry["evidence"]
    assert (
        "benchmarks/results/df-cc-batched-replay-20261010/publication.json"
        in entry["evidence"]
    )


def test_cc_input_residual_registration_records_scoped_automatic_use() -> None:
    """An internal false initializer must not hide the qualified automatic owner."""
    entry = next(
        item
        for item in _payload()["entries"]
        if "cc-option:diis_input_residual" in item["controls"]
    )
    assert entry["classification"] == "already-default"
    assert {
        "src/cc/solver.hpp",
        "src/cc/solver.cpp",
        "src/cc/cuda_solver.cu",
        "src/methods/rccsd_method.cpp",
        "src/methods/df_ccsdt_force.cu",
    } <= set(entry["sources"])
    assert (
        "benchmarks/results/df-cc-input-residual-20261010/publication.json"
        in entry["evidence"]
    )


def test_public_density_fitting_default_is_audited(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "python/generativeqc/calculator.py"
    changed = source.read_text().replace(
        'density_fitting: str | bool = "none"',
        'density_fitting: str | bool = "auto"',
        1,
    )
    source.write_text(changed)
    errors = validate_inventory(payload, root=tmp_path)
    assert any("density-fitting default drifted" in error for error in errors)


@pytest.mark.parametrize("approximation", ["DensityFitted", "SeminumericalCosx"])
@pytest.mark.parametrize("commented_exact", ["", "//", "/*"])
def test_fock_approximation_default_is_audited(
    tmp_path: Path, approximation: str, commented_exact: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/fock_build.hpp"
    original = source.read_text()
    exact = "FockApproximation approximation{FockApproximation::Exact};"
    replacement = (
        f"FockApproximation approximation{{FockApproximation::{approximation}}};"
    )
    if commented_exact:
        suffix = " */" if commented_exact == "/*" else ""
        replacement += f"\n  {commented_exact} {exact}{suffix}"
    changed = original.replace(exact, replacement, 1)
    assert changed != original
    # The provider domain retains an independent Exact default. Neither it nor
    # comments inside FockTermSpec may hide a changed mathematical request.
    assert exact in changed
    source.write_text(changed)
    errors = validate_inventory(payload, root=tmp_path)
    assert any("Fock approximation default drifted" in error for error in errors)


def test_preliminary_initial_guess_default_is_audited(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "python/generativeqc/calculator.py"
    changed = source.read_text().replace(
        'initial_guess: InitialGuessSpec | typing.Literal["auto"] | None = "auto"',
        'initial_guess: InitialGuessSpec | typing.Literal["auto"] | None = None',
        1,
    )
    source.write_text(changed)
    errors = validate_inventory(payload, root=tmp_path)
    assert any("initial-guess default drifted" in error for error in errors)


@pytest.mark.parametrize("initializer", ["{}", "{false}", " = false"])
def test_new_cc_default_off_option_must_be_registered(
    tmp_path: Path, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/cc/solver.hpp"
    original = source.read_text()
    changed = original.replace(
        "struct SolverOptions {",
        f"struct SolverOptions {{\n  bool new_default_off_path{initializer};",
        1,
    )
    assert changed != original
    source.write_text(changed)
    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        "cc-option:new_default_off_path" in error and "unregistered" in error
        for error in errors
    )


@pytest.mark.parametrize("initializer", ["{}", "{false}", " = false"])
def test_new_response_default_off_option_must_be_registered(
    tmp_path: Path, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/hf/rhf_frame_response.hpp"
    original = source.read_text()
    changed = original.replace(
        "struct RHFFrameResponseOptions {",
        f"struct RHFFrameResponseOptions {{\n  bool new_default_off_path{initializer};",
        1,
    )
    assert changed != original
    source.write_text(changed)
    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        "response-option:new_default_off_path" in error and "unregistered" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "relative,before,after",
    [
        ("src/scf/cuda/direct_md_j.hpp", "128U << 20", "256U << 20"),
        ("src/scf/cuda/direct_jk.cpp", "host.nbf < 8", "host.nbf < 4"),
        (
            "src/scf/cuda/direct_jk.cpp",
            "md_ready = !runtime::active_device_resource_ledger",
            "md_ready = true",
        ),
        (
            "src/scf/cuda/direct_jk.cpp",
            'std::strcmp(disabled, "1") == 0',
            'std::strcmp(disabled, "0") == 0',
        ),
    ],
)
def test_md_j_default_domain_and_optional_capacity_are_audited(
    tmp_path: Path, relative: str, before: str, after: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / relative
    original = source.read_text()
    assert before in original
    source.write_text(original.replace(before, after))
    assert any(
        "MD-J default" in error for error in validate_inventory(payload, root=tmp_path)
    )
