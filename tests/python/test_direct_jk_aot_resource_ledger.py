"""Fail-closed tests for the direct J/K AOT static resource ledger gate."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common import compiler_cache
from generativeqc_compiler.common.compiler_cache import CompilerCacheLauncher
from generativeqc_compiler.common.compiler_process import CompileResult

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

    @pytest.mark.parametrize("field", tuple(validator.qualified_flags()))
    def test_missing_flag_fails_closed(self, field: str) -> None:
        payload = _ledger()
        payload["flags"].pop(field)
        with pytest.raises(validator.LedgerError, match="flags"):
            validator.validate_ledger(payload)

    def test_unknown_flag_fails_closed(self) -> None:
        payload = _ledger()
        payload["flags"]["unknown"] = "option"
        with pytest.raises(validator.LedgerError, match="flags"):
            validator.validate_ledger(payload)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("architecture", "sm_90"),
            ("standard", "c++17"),
            ("optimization", "-O0"),
            ("split_compile_threads", 8),
            ("split_compile_threads", True),
            ("split_compile_threads", 1.0),
            ("definitions", []),
            ("definitions", tuple(validator.QUALIFIED_DEFINITIONS)),
            ("ptxas_verbose", False),
            ("ptxas_verbose", 1),
        ],
    )
    def test_wrong_flag_value_or_type_fails_closed(
        self, field: str, value: object
    ) -> None:
        payload = _ledger()
        payload["flags"][field] = value
        with pytest.raises(validator.LedgerError, match=f"flags.{field}"):
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

    def test_truncated_ledger_cannot_bypass_flag_comparison(self) -> None:
        ledger = _ledger(flags={"architecture": validator.QUALIFIED_ARCHITECTURE})
        measured = _measurement()
        measured["identity"]["flags"]["optimization"] = "-O0"
        with pytest.raises(validator.LedgerError, match="flags"):
            validator.compare(ledger, measured)

    @pytest.mark.parametrize("field", tuple(validator.qualified_flags()))
    def test_missing_measured_flag_fails(self, field: str) -> None:
        measured = _measurement()
        measured["identity"]["flags"].pop(field)
        assert any(
            f"flags.{field}" in failure
            for failure in validator.compare(_ledger(), measured)
        )

    @pytest.mark.parametrize("flags", [None, [], "sm_120"])
    def test_non_object_measured_flags_fail(self, flags: object) -> None:
        measured = _measurement()
        measured["identity"]["flags"] = flags
        assert validator.compare(_ledger(), measured) == [
            "identity drift (flags): measured flags must be an object"
        ]

    def test_extra_measured_flag_fails(self) -> None:
        measured = _measurement()
        measured["identity"]["flags"]["fast_compile"] = None
        assert any(
            "flags.fast_compile" in failure
            for failure in validator.compare(_ledger(), measured)
        )

    @pytest.mark.parametrize(
        "field,value", [("split_compile_threads", True), ("ptxas_verbose", 1)]
    )
    def test_measured_flag_type_drift_fails(self, field: str, value: object) -> None:
        measured = _measurement()
        measured["identity"]["flags"][field] = value
        assert any(
            f"flags.{field}" in failure
            for failure in validator.compare(_ledger(), measured)
        )

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
        measured["identity"]["flags"]["split_compile_threads"] = 8
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


class TestProductionCompile:
    def test_split_setting_matches_cmake_release_default(self) -> None:
        cmake = (validator.ROOT / "CMakeLists.txt").read_text()
        default = re.search(
            r'set\(GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS "(\d+)" CACHE STRING',
            cmake,
        )
        assert default is not None
        assert validator.QUALIFIED_SPLIT_COMPILE_THREADS == int(default[1]) == 1

    @pytest.mark.parametrize(
        "returncode,timed_out", [(0, False), (1, False), (124, True)]
    )
    @pytest.mark.parametrize("stats_returncode", [0, 1])
    def test_actual_ccache_launcher_is_observed_around_compile(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        returncode: int,
        timed_out: bool,
        stats_returncode: int,
    ) -> None:
        launcher = CompilerCacheLauncher(
            Path("/verified/ccache"), "ccache", "ccache 4.14"
        )
        selections: list[float] = []
        events: list[str] = []
        arguments = {
            "nvcc": Path("/fake/nvcc"),
            "shard": tmp_path / "shard.cu",
            "generated": tmp_path / "generated",
            "cache": tmp_path / "cache",
            "identity": _identity(),
            "timeout": 23,
        }

        def resolve(path: str, timeout: float) -> CompilerCacheLauncher:
            selections.append(timeout)
            return launcher

        def stats(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            assert command == [str(launcher.path), "--show-stats"]
            assert kwargs["timeout"] == 5
            events.append("stats")
            return subprocess.CompletedProcess(
                command, stats_returncode, f"stats {len(events)}", ""
            )

        def compile_run(
            command: list[str], timeout: float, *, label: str
        ) -> CompileResult:
            assert command[:2] == [str(launcher.path), str(arguments["nvcc"])]
            assert label == "NVCC" and 0 < timeout <= 23
            assert not any(item.startswith("--split-compile") for item in command)
            assert "-O3" in command and "-std=c++20" in command
            assert "-arch=sm_120" in command and "-Xptxas=-v" in command
            assert "-DGENERATIVEQC_HAS_CUDA=1" in command
            events.append("compile")
            pending_receipt = json.loads(
                next((tmp_path / "cache").glob("*/compiler-cache.json")).read_text()
            )
            assert pending_receipt["before"]["stdout"] == "stats 1"
            assert pending_receipt["after"]["status"] == "not-recorded"
            if returncode == 0:
                Path(command[command.index("-o") + 1]).write_bytes(b"compiled object")
            return CompileResult(
                returncode, timed_out, 0.1, "compile stdout", "compile stderr"
            )

        monkeypatch.setattr(compiler_cache, "_resolve_compiler_cache", resolve)
        monkeypatch.setattr(compiler_cache, "run_compiler", compile_run)
        monkeypatch.setattr(validator.subprocess, "run", stats)
        if returncode:
            with pytest.raises(RuntimeError, match="shard compilation failed"):
                validator.compile_shard(**arguments)
        else:
            _, _, metadata = validator.compile_shard(**arguments)
            assert metadata["cache_hit"] is False
        assert events == ["stats", "compile", "stats"]
        assert selections == [23, 23]
        directory = next((tmp_path / "cache").iterdir())
        receipt_path = directory / "compiler-cache.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["launcher"] == {
            "name": "ccache",
            "path": str(launcher.path),
            "version": "ccache 4.14",
        }
        assert receipt["before"]["stdout"] == "stats 1"
        assert receipt["after"]["stdout"] == "stats 3"
        assert receipt["after"]["status"] == ("fail" if stats_returncode else "pass")
        assert "cumulative" in receipt["after"]["scope"]
        metadata = json.loads((directory / "compile.json").read_text())
        assert metadata["returncode"] == returncode
        assert metadata["status"] == ("fail" if returncode else "pass")
        log = (directory / "nvcc.log").read_text()
        assert log.startswith("compile stdoutcompile stderr")
        if timed_out:
            assert "NVCC compilation timed out after 23 seconds" in log
            assert metadata["reason"] == "NVCC timeout"
        if returncode == 0:
            previous_receipt = receipt_path.read_bytes()
            _, _, metadata = validator.compile_shard(**arguments)
            assert metadata["cache_hit"] is True
            assert selections == [23, 23]
            assert events == ["stats", "compile", "stats"]
            assert receipt_path.read_bytes() == previous_receipt

    def test_private_sccache_does_not_query_shared_statistics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected_run(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("must not query or start an unrelated shared sccache server")

        monkeypatch.setattr(validator.subprocess, "run", unexpected_run)
        launcher = CompilerCacheLauncher(
            Path("/verified/sccache"), "sccache", "sccache 0.16.0"
        )
        snapshot = validator.cache_statistics(launcher)
        assert snapshot["status"] == "unavailable"
        assert "private sccache" in snapshot["reason"]

    @pytest.mark.parametrize(
        "error", [OSError("stats unavailable"), subprocess.TimeoutExpired("ccache", 5)]
    )
    def test_statistics_errors_are_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def failed_stats(*args: Any, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(validator.subprocess, "run", failed_stats)
        launcher = CompilerCacheLauncher(
            Path("/verified/ccache"), "ccache", "ccache 4.14"
        )
        snapshot = validator.cache_statistics(launcher)
        assert snapshot["status"] == "unavailable"
        assert snapshot["reason"] == str(error)

    def test_compile_exception_still_retains_statistics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        launcher = CompilerCacheLauncher(
            Path("/verified/ccache"), "ccache", "ccache 4.14"
        )
        monkeypatch.setattr(
            validator, "resolve_compiler_cache", lambda timeout: launcher
        )
        snapshots: list[int] = []

        def stats(selected: CompilerCacheLauncher) -> dict[str, Any]:
            assert selected == launcher
            snapshots.append(len(snapshots))
            return {"status": "pass", "stdout": str(snapshots[-1])}

        def compile_error(*args: Any, **kwargs: Any) -> Any:
            raise OSError("could not start compiler")

        monkeypatch.setattr(validator, "cache_statistics", stats)
        monkeypatch.setattr(validator.CudaCompilerAdapter, "compile", compile_error)
        with pytest.raises(OSError, match="could not start compiler"):
            validator.compile_shard(
                nvcc=Path("/fake/nvcc"),
                shard=tmp_path / "shard.cu",
                generated=tmp_path / "generated",
                cache=tmp_path / "cache",
                identity=_identity(),
                timeout=23,
            )
        receipt = json.loads(
            next((tmp_path / "cache").glob("*/compiler-cache.json")).read_text()
        )
        assert receipt["before"]["stdout"] == "0"
        assert receipt["after"]["stdout"] == "1"


class TestFailureReports:
    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("NVCC timeout"),
            OSError("compiler missing"),
            ValueError("bad resource dump"),
        ],
    )
    def test_measurement_failure_writes_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
    ) -> None:
        def failed_measurement(**kwargs: Any) -> Any:
            raise error

        nvcc = tmp_path / "nvcc"
        nvcc.touch()
        monkeypatch.setattr(validator, "run_measurement", failed_measurement)
        output = tmp_path / "report"
        assert (
            validator.main(
                [
                    "--nvcc",
                    str(nvcc),
                    "--cache",
                    str(tmp_path / "cache"),
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        report = json.loads((output / "report.json").read_text())
        assert report["status"] == "fail"
        assert report["measurement"] is None
        assert report["failures"] == [f"measurement failed: {error}"]


class TestWorkflowCache:
    def test_outer_cache_key_covers_transitive_header(self) -> None:
        workflow = (validator.ROOT / ".github/workflows/ci.yml").read_text()
        key = next(
            line for line in workflow.splitlines() if "key: direct-jk-aot-" in line
        )
        assert "'src/runtime/compensated_output.hpp'" in key

    def test_failure_artifacts_include_cache_receipts_and_compile_logs(self) -> None:
        workflow = (validator.ROOT / ".github/workflows/ci.yml").read_text()
        step = workflow.split(
            "- name: Preserve direct J/K AOT resource qualification report", 1
        )[1].split("- name:", 1)[0]
        assert "if: always()" in step
        for path in (
            "build/direct-jk-aot/report.json",
            "build/direct-jk-aot-cache/*/compiler-cache.json",
            "build/direct-jk-aot-cache/*/compile.json",
            "build/direct-jk-aot-cache/*/nvcc.log",
        ):
            assert path in step
