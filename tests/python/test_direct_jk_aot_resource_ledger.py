"""Fail-closed tests for the direct J/K AOT static resource ledger gate."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from tools import validate_direct_jk_aot_resources as validator


def _identity(**overrides: object) -> dict[str, object]:
    payload = {
        "architecture": validator.QUALIFIED_ARCHITECTURE,
        "unit_mode": validator.QUALIFIED_UNIT_MODE,
        "shard_source_sha256": "a" * 64,
        "preprocessed_sha256": "b" * 64,
        "header_sha256": {
            "scf/generated_shell_task.hpp": "c" * 64,
            "runtime/compensated_atomic.cuh": "d" * 64,
        },
        "manifest_sha256": "e" * 64,
        "generator_abi": validator.generator_abi(),
        "toolchain": {
            "nvcc": "nvcc-12.9",
            "ptxas": "ptxas-12.9",
            "cuobjdump": "cuobjdump-12.9",
        },
        "flags": validator.qualified_flags(),
    }
    payload.update(overrides)
    return payload


def _kernels() -> dict[str, object]:
    return {
        "generated_sm120_dsss_shell_class_fock_rhf_streaming_kernel": {
            "REG": 114,
            "STACK": 0,
            "SHARED": 128,
            "LOCAL": 0,
        },
        "generated_sm120_ddss_shell_class_fock_rhf_streaming_kernel": {
            "REG": 114,
            "STACK": 0,
            "SHARED": 128,
            "LOCAL": 0,
        },
        "generated_sm120_ddps_shell_class_fock_rhf_streaming_kernel": {
            "REG": 158,
            "STACK": 0,
            "SHARED": 128,
            "LOCAL": 0,
        },
    }


def _ledger(**overrides: object) -> dict[str, object]:
    payload = {
        "schema": validator.SCHEMA,
        "schema_version": validator.SCHEMA_VERSION,
        "coverage": validator.COVERAGE,
        "issue": validator.ISSUE,
        **_identity(),
        "kernels": _kernels(),
    }
    payload.update(overrides)
    return payload


def _measurement(**overrides: object) -> dict[str, object]:
    payload = {
        "identity": _identity(),
        "toolchain": {
            "nvcc": "nvcc-12.9",
            "ptxas": "ptxas-12.9",
            "cuobjdump": "cuobjdump-12.9",
        },
        "kernels": _kernels(),
    }
    payload.update(overrides)
    return payload


class TestCuobjdumpParser:
    def test_parses_complete_rows(self) -> None:
        output = (
            " Function generated_a:\nREG:114 STACK:0 SHARED:128 LOCAL:0\n"
            "\tFunction generated_b:\nREG:158 STACK:0 SHARED:128 LOCAL:0\n"
        )
        rows = validator.cuobjdump_resources(output)
        assert rows == {
            "generated_a": {"REG": 114, "STACK": 0, "SHARED": 128, "LOCAL": 0},
            "generated_b": {"REG": 158, "STACK": 0, "SHARED": 128, "LOCAL": 0},
        }

    def test_incomplete_row_fails_closed(self) -> None:
        output = "Function generated_a:\nREG:114 STACK:0 SHARED:128\n"
        with pytest.raises(ValueError, match="incomplete"):
            validator.cuobjdump_resources(output)

    def test_duplicate_row_fails_closed(self) -> None:
        output = (
            "Function generated_a:\nREG:114 STACK:0 SHARED:128 LOCAL:0\n"
            "Function generated_a:\nREG:114 STACK:0 SHARED:128 LOCAL:0\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            validator.cuobjdump_resources(output)

    def test_empty_report_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="no kernel resources"):
            validator.cuobjdump_resources("no functions here")


class TestLedgerValidation:
    def test_valid_ledger_passes(self) -> None:
        validator.validate_ledger(_ledger())

    @pytest.mark.parametrize(
        "field,value",
        [
            ("schema", "other.schema"),
            ("schema_version", 99),
            ("coverage", "something else"),
            ("issue", "#9999"),
            ("architecture", "sm_90"),
            ("unit_mode", "class"),
        ],
    )
    def test_contract_fields_fail_closed(self, field: str, value: object) -> None:
        with pytest.raises(validator.LedgerError):
            validator.validate_ledger(_ledger(**{field: value}))

    def test_missing_identity_field_fails_closed(self) -> None:
        payload = _ledger()
        payload.pop("preprocessed_sha256")
        with pytest.raises(validator.LedgerError, match="preprocessed_sha256"):
            validator.validate_ledger(payload)

    def test_partial_header_contract_fails_closed(self) -> None:
        payload = _ledger()
        payload["header_sha256"] = {"scf/generated_shell_task.hpp": "c" * 64}
        with pytest.raises(validator.LedgerError, match="header_sha256"):
            validator.validate_ledger(payload)

    def test_wrong_generator_abi_fails_closed(self) -> None:
        payload = _ledger()
        payload["generator_abi"] = validator.generator_abi() + 1
        with pytest.raises(validator.LedgerError, match="generator_abi"):
            validator.validate_ledger(payload)

    def test_kernel_missing_resource_field_fails_closed(self) -> None:
        payload = _ledger()
        name = next(iter(payload["kernels"]))
        payload["kernels"][name].pop("STACK")
        with pytest.raises(validator.LedgerError, match="REG/STACK/SHARED/LOCAL"):
            validator.validate_ledger(payload)

    def test_negative_resource_fails_closed(self) -> None:
        payload = _ledger()
        name = next(iter(payload["kernels"]))
        payload["kernels"][name]["REG"] = -1
        with pytest.raises(validator.LedgerError, match="non-negative"):
            validator.validate_ledger(payload)

    def test_empty_kernels_fail_closed(self) -> None:
        with pytest.raises(validator.LedgerError, match="non-empty"):
            validator.validate_ledger(_ledger(kernels={}))


class TestComparison:
    def test_exact_match_passes(self) -> None:
        assert validator.compare(_ledger(), _measurement()) == []

    def test_register_drift_is_attributed_to_the_kernel(self) -> None:
        measured = _measurement()
        name = "generated_sm120_dsss_shell_class_fock_rhf_streaming_kernel"
        measured["kernels"][name]["REG"] = 255
        failures = validator.compare(_ledger(), measured)
        assert len(failures) == 1
        assert failures[0] == f"kernel resource drift: {name} REG 114 -> 255"

    def test_stack_drift_is_attributed(self) -> None:
        measured = _measurement()
        name = "generated_sm120_ddps_shell_class_fock_rhf_streaming_kernel"
        measured["kernels"][name]["STACK"] = 544
        failures = validator.compare(_ledger(), measured)
        assert failures == [f"kernel resource drift: {name} STACK 0 -> 544"]

    def test_kernel_inventory_drift_is_attributed(self) -> None:
        measured = _measurement()
        measured["kernels"]["generated_sm120_new_kernel"] = {
            "REG": 64,
            "STACK": 0,
            "SHARED": 0,
            "LOCAL": 0,
        }
        removed = "generated_sm120_ddss_shell_class_fock_rhf_streaming_kernel"
        measured["kernels"].pop(removed)
        failures = validator.compare(_ledger(), measured)
        assert any("kernel inventory drift" in failure for failure in failures)
        assert any("generated_sm120_new_kernel" in failure for failure in failures)
        assert any(removed in failure for failure in failures)

    def test_shard_source_identity_drift_fails(self) -> None:
        measured = _measurement()
        measured["identity"]["shard_source_sha256"] = "f" * 64
        failures = validator.compare(_ledger(), measured)
        assert any("shard_source_sha256" in failure for failure in failures)

    def test_prelude_header_drift_fails_with_the_header_name(self) -> None:
        measured = _measurement()
        measured["identity"]["header_sha256"]["scf/generated_shell_task.hpp"] = "f" * 64
        failures = validator.compare(_ledger(), measured)
        assert any(
            "header scf/generated_shell_task.hpp" in failure for failure in failures
        )

    def test_preprocessed_identity_drift_fails(self) -> None:
        measured = _measurement()
        measured["identity"]["preprocessed_sha256"] = "f" * 64
        failures = validator.compare(_ledger(), measured)
        assert any("preprocessed_sha256" in failure for failure in failures)

    def test_toolchain_drift_fails_per_component(self) -> None:
        measured = _measurement()
        measured["toolchain"]["nvcc"] = "nvcc-12.9.1"
        failures = validator.compare(_ledger(), measured)
        assert any("toolchain drift (nvcc)" in failure for failure in failures)

    def test_flags_drift_fails(self) -> None:
        measured = _measurement()
        measured["identity"]["flags"]["split_compile_threads"] = 1
        failures = validator.compare(_ledger(), measured)
        assert any("flags.split_compile_threads" in failure for failure in failures)


class TestMeasuredLedger:
    def test_measured_ledger_is_complete_and_sorted(self) -> None:
        payload = validator.measured_ledger(_measurement())
        validator.validate_ledger(payload)
        names = list(payload["kernels"])
        assert names == sorted(names)

    def test_measured_ledger_never_targets_the_committed_path(self) -> None:
        relative = validator.LEDGER.relative_to(validator.ROOT).as_posix()
        assert relative == "manifests/maintenance/direct_jk_aot_resource_ledger.json"


class TestGenerationDeterminism:
    def test_generation_is_byte_deterministic(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="gqc-aot-ledger-test-"))
        try:
            first = root / "first"
            second = root / "second"
            shards_a = validator.generate_production_shards(first)
            shards_b = validator.generate_production_shards(second)
            assert len(shards_a) == validator.QUALIFIED_SHARDS
            for shard_a, shard_b in zip(shards_a, shards_b):
                assert shard_a.read_bytes() == shard_b.read_bytes()
            qualified = shards_a[validator.QUALIFIED_SHARD_INDEX]
            text = qualified.read_text()
            for shell_class in ("dsss", "ddss", "ddps"):
                assert shell_class in text
        finally:
            shutil.rmtree(root, ignore_errors=True)
