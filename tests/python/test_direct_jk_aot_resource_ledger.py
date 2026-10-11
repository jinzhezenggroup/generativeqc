"""Fail-closed tests for the direct J/K AOT static resource ledger gate."""

from __future__ import annotations

import json
import os
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

FAKE_NVCC = Path("/fake/nvcc").resolve()
FAKE_CCACHE = Path("/verified/ccache").resolve()


def _production_cache(**overrides: str) -> str:
    fields = {
        "CMAKE_CUDA_COMPILER": str(FAKE_NVCC),
        "CMAKE_CUDA_HOST_COMPILER": "",
        "GENERATIVEQC_AOT_SHARDS": "8",
        "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS": "1",
        "GENERATIVEQC_ENABLE_CUDA": "ON",
        "GENERATIVEQC_ENABLE_AOT_SHELLS": "ON",
        "GENERATIVEQC_AOT_UNIT_MODE": "stable-shards",
        "GENERATIVEQC_AOT_PROFILE": "sm_120",
        "GENERATIVEQC_AOT_PROFILES": "",
        "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES": "120",
        "GENERATIVEQC_CUDA_ARCHITECTURES": "",
        "GENERATIVEQC_CUDA_FAST_COMPILE": "OFF",
        "GENERATIVEQC_RESOLVED_COMPILER_CACHE": str(FAKE_CCACHE),
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "GENERATIVEQC_AOT_SHELL_MANIFEST": str(validator.MANIFEST),
    }
    fields.update(overrides)
    return "".join(f"{field}:STRING={value}\n" for field, value in fields.items())


def _write_production_rule(
    directory: Path, *, launcher: str = "/verified/ccache", nvcc: Path = FAKE_NVCC
) -> Path:
    rules = directory / "CMakeFiles" / "rules.ninja"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(
        "rule CUDA_COMPILER__generativeqc_aot_sm_120_Release\n"
        f"  command = /usr/bin/cmake -E env CCACHE_BASEDIR={directory.resolve()} "
        f"{launcher} {nvcc} $DEFINES $INCLUDES $FLAGS -c $in -o $out\n"
    )
    (directory / "build.ninja").write_text(
        "build CMakeFiles/generativeqc_aot_sm_120.dir/generated/generativeqc_generated_shell_sm120_shard_6.cu.o: CUDA_COMPILER__generativeqc_aot_sm_120_Release shard_6.cu\n"
        "  FLAGS = -O3 -DNDEBUG -std=c++20\n"
    )
    return rules


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

    def test_record_mode_preserves_the_stale_host_context_as_a_negative(self) -> None:
        payload = json.loads(validator.LEDGER.read_text(encoding="utf-8"))
        assert "host_compiler" not in payload["flags"]
        with pytest.raises(validator.LedgerError, match="flags"):
            validator.validate_ledger(payload)

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
            ("fast_compile", True),
            ("fast_compile", 0),
            ("host_compiler", "/usr/bin/clang++"),
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
            for shard_a, shard_b in zip(shards_a, shards_b, strict=True):
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
        _write_production_rule(tmp_path)
        original_directory = Path.cwd()
        original_basedir = os.environ.get("CCACHE_BASEDIR")
        events: list[str] = []
        arguments = {
            "nvcc": Path("/fake/nvcc"),
            "shard": tmp_path / "shard.cu",
            "generated": tmp_path / "generated",
            "cache": tmp_path / "cache",
            "identity": _identity(),
            "timeout": 23,
            "working_directory": tmp_path,
            "launcher": launcher,
        }

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
            assert "-DNDEBUG" in command
            assert command[command.index("-c") + 1] == "shard.cu"
            assert Path.cwd() == tmp_path.resolve()
            assert os.environ["CCACHE_BASEDIR"] == str(tmp_path.resolve())
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

        monkeypatch.setattr(
            compiler_cache,
            "resolve_compiler_cache",
            lambda timeout: pytest.fail(
                "CMake-selected launcher must bypass automatic selection"
            ),
        )
        monkeypatch.setattr(compiler_cache, "run_compiler", compile_run)
        monkeypatch.setattr(validator.subprocess, "run", stats)
        if returncode:
            with pytest.raises(RuntimeError, match="shard compilation failed"):
                validator.compile_shard(**arguments)
        else:
            _, _, metadata = validator.compile_shard(**arguments)
            assert metadata["cache_hit"] is False
        assert events == ["stats", "compile", "stats"]
        assert Path.cwd() == original_directory
        assert os.environ.get("CCACHE_BASEDIR") == original_basedir
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

    def test_preprocessing_uses_the_cmake_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        generated = tmp_path / "generated"
        shard = generated / "shard.cu"

        def preprocess(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[bytes]:
            assert command[-1] == str(Path("generated/shard.cu"))
            assert kwargs["cwd"] == tmp_path
            return subprocess.CompletedProcess(command, 0, b"preprocessed", b"")

        monkeypatch.setattr(validator.subprocess, "run", preprocess)
        digest = validator.preprocessed_identity_sha256(
            FAKE_NVCC,
            shard,
            generated,
            working_directory=tmp_path,
        )
        assert digest == validator.hashlib.sha256(b"preprocessed").hexdigest()

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
        _write_production_rule(tmp_path)
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
                working_directory=tmp_path,
                launcher=launcher,
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


class TestCompileImplementationIdentity:
    def test_adapter_change_misses_without_relabeling_scientific_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_production_rule(tmp_path, launcher="/fake/ccache")
        implementation = tmp_path / "cuda_adapter.py"
        implementation.write_text("original command implementation\n")
        monkeypatch.setattr(
            validator, "COMPILE_IMPLEMENTATIONS", (str(implementation),)
        )
        launcher = CompilerCacheLauncher(Path("/fake/ccache"), "ccache", "4.14")
        monkeypatch.setattr(
            validator, "cache_statistics", lambda launcher: {"status": "pass"}
        )
        executions: list[str] = []

        def compile_run(
            self: Any, source: Path, output: Path, **kwargs: Any
        ) -> CompileResult:
            command_version = implementation.read_text()
            executions.append(command_version)
            output.write_text(command_version)
            return CompileResult(0, False, 0.1, command_version, "")

        monkeypatch.setattr(validator.CudaCompilerAdapter, "compile", compile_run)
        identity = _identity()
        arguments = {
            "nvcc": Path("/fake/nvcc"),
            "shard": tmp_path / "shard.cu",
            "generated": tmp_path / "generated",
            "cache": tmp_path / "cache",
            "identity": identity,
            "timeout": 23,
            "working_directory": tmp_path,
            "launcher": launcher,
        }
        first_object, first_directory, first = validator.compile_shard(**arguments)
        _, _, replay = validator.compile_shard(**arguments)
        assert first["cache_hit"] is False and replay["cache_hit"] is True
        assert len(executions) == 1

        implementation.write_text("changed optimizer option\n")
        second_object, second_directory, second = validator.compile_shard(**arguments)
        assert second["cache_hit"] is False
        assert first_directory != second_directory
        assert first_object.read_text() == "original command implementation\n"
        assert second_object.read_text() == "changed optimizer option\n"
        assert first["identity"] == second["identity"] == identity
        assert (
            first["compile_implementation_sha256"]
            != second["compile_implementation_sha256"]
        )
        assert len(executions) == 2

        # Even a restored record in the current key must carry this execution
        # identity; matching scientific inputs and object/log bytes are not enough.
        metadata_path = second_directory / "compile.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["compile_implementation_sha256"]
        metadata_path.write_text(json.dumps(metadata))
        _, _, legacy = validator.compile_shard(**arguments)
        assert legacy["cache_hit"] is False
        assert len(executions) == 3

    def test_cache_identity_covers_both_command_builders(self) -> None:
        assert (
            "tools/validate_direct_jk_aot_resources.py"
            in validator.COMPILE_IMPLEMENTATIONS
        )
        assert (
            "python/generativeqc_compiler/common/cuda_adapter.py"
            in validator.COMPILE_IMPLEMENTATIONS
        )
        assert (
            "python/generativeqc_compiler/common/compiler_cache.py"
            in validator.COMPILE_IMPLEMENTATIONS
        )


class TestProductionContextBinding:
    def test_release_definition_is_in_preprocessing_and_flag_identity(self) -> None:
        command = validator._preprocess_command(
            Path("/fake/nvcc"), Path("shard.cu"), Path("generated")
        )
        assert "-DNDEBUG" in command
        assert "NDEBUG" in validator.qualified_flags()["definitions"]

    def test_historical_ledger_without_release_definition_fails_closed(self) -> None:
        ledger = _ledger()
        ledger["flags"]["definitions"].remove("NDEBUG")
        with pytest.raises(validator.LedgerError, match="flags.definitions"):
            validator.validate_ledger(ledger)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("GENERATIVEQC_CUDA_FAST_COMPILE", "ON"),
            ("CMAKE_CUDA_COMPILER", str(Path("/other/nvcc").resolve())),
            ("CMAKE_CUDA_HOST_COMPILER", str(Path("/usr/bin/clang++").resolve())),
        ],
    )
    def test_compile_context_override_fails_before_work(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache(**{field: value}))
        with pytest.raises(ValueError, match=field):
            validator.validate_production_shard_count(cache, FAKE_NVCC)

    def test_configured_cache_launcher_is_part_of_the_context(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache())
        working_directory, launcher = validator.validate_production_shard_count(
            cache, FAKE_NVCC
        )
        assert working_directory == tmp_path.resolve()
        assert launcher == FAKE_CCACHE

    @pytest.mark.parametrize("cache_type", ["STRING", "UNINITIALIZED"])
    def test_accepts_configured_qualified_shard_count(
        self, tmp_path: Path, cache_type: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache().replace(":STRING=", f":{cache_type}="))
        validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize("profile", ["auto", "sm_120"])
    @pytest.mark.parametrize("profiles", ["", "sm_120"])
    def test_equivalent_qualified_profile_selection(
        self, tmp_path: Path, profile: str, profiles: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(
            _production_cache(
                GENERATIVEQC_AOT_PROFILE=profile, GENERATIVEQC_AOT_PROFILES=profiles
            )
        )
        validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize("value", ["ON", "YES", "TRUE", "Y", "1", "true"])
    def test_enabled_cmake_boolean_spellings(self, tmp_path: Path, value: str) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(
            _production_cache(
                GENERATIVEQC_ENABLE_CUDA=value, GENERATIVEQC_ENABLE_AOT_SHELLS=value
            )
        )
        validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize("duplicate", [False, True])
    def test_profile_map_must_be_declared_once(
        self, tmp_path: Path, duplicate: bool
    ) -> None:
        entry = "GENERATIVEQC_AOT_PROFILES:STRING=\n"
        contents = _production_cache()
        contents = contents + entry if duplicate else contents.replace(entry, "")
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(contents)
        with pytest.raises(ValueError, match="GENERATIVEQC_AOT_PROFILES"):
            validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("GENERATIVEQC_ENABLE_CUDA", "OFF"),
            ("GENERATIVEQC_ENABLE_AOT_SHELLS", "OFF"),
            ("GENERATIVEQC_CUDA_FAST_COMPILE", "ON"),
            ("CMAKE_CUDA_HOST_COMPILER", str(Path("/usr/bin/clang++").resolve())),
            ("GENERATIVEQC_AOT_UNIT_MODE", "class"),
            ("GENERATIVEQC_AOT_PROFILE", "portable"),
            ("GENERATIVEQC_AOT_PROFILES", "portable"),
            ("GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES", "90"),
            ("GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES", "120;90"),
            ("GENERATIVEQC_AOT_SHELL_MANIFEST", "/other/production.json"),
            ("CMAKE_CUDA_COMPILER", str(Path("/other/nvcc").resolve())),
        ],
    )
    def test_generation_context_override_fails_before_work(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache(**{field: value}))

        def unexpected_generation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("production context mismatch must fail before generation")

        monkeypatch.setattr(
            validator, "generate_production_shards", unexpected_generation
        )
        with pytest.raises(ValueError, match="production"):
            validator.run_measurement(
                nvcc=Path("/fake/nvcc"),
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=cache,
                timeout=23,
            )

    @pytest.mark.parametrize("mode", ["missing", "duplicate", "empty"])
    @pytest.mark.parametrize(
        "field",
        [
            "GENERATIVEQC_ENABLE_CUDA",
            "GENERATIVEQC_ENABLE_AOT_SHELLS",
            "GENERATIVEQC_AOT_UNIT_MODE",
            "GENERATIVEQC_AOT_PROFILE",
            "GENERATIVEQC_AOT_SHELL_MANIFEST",
            "GENERATIVEQC_CUDA_FAST_COMPILE",
            "CMAKE_CUDA_COMPILER",
            "GENERATIVEQC_RESOLVED_COMPILER_CACHE",
        ],
    )
    def test_incomplete_generation_context_fails_closed(
        self, tmp_path: Path, field: str, mode: str
    ) -> None:
        contents = _production_cache()
        line = next(
            line for line in contents.splitlines(True) if line.startswith(field + ":")
        )
        if mode == "missing":
            contents = contents.replace(line, "")
        elif mode == "duplicate":
            contents += line
        else:
            contents = contents.replace(line, f"{field}:STRING=\n")
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(contents)
        with pytest.raises(ValueError, match=field):
            validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize(
        "field",
        [
            "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES",
            "GENERATIVEQC_CUDA_ARCHITECTURES",
            "CMAKE_CUDA_ARCHITECTURES",
        ],
    )
    def test_explicit_target_precedence(self, tmp_path: Path, field: str) -> None:
        targets = dict.fromkeys(
            [
                "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES",
                "GENERATIVEQC_CUDA_ARCHITECTURES",
                "CMAKE_CUDA_ARCHITECTURES",
            ],
            "",
        )
        targets[field] = "120"
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache(**targets))
        validator.validate_production_shard_count(cache, FAKE_NVCC)
        targets[field] = ""
        cache.write_text(_production_cache(**targets))
        with pytest.raises(ValueError, match="explicit CUDA target"):
            validator.validate_production_shard_count(cache, FAKE_NVCC)

    def test_target_precedence_ignores_unused_lower_priority_values(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(
            _production_cache(
                GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES="120",
                GENERATIVEQC_CUDA_ARCHITECTURES="90",
                CMAKE_CUDA_ARCHITECTURES="90",
            )
        )
        validator.validate_production_shard_count(cache, FAKE_NVCC)

    @pytest.mark.parametrize(
        "split_entry",
        [
            "",
            "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=\n",
            "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=2\n",
            "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=8\n",
            "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=0\n",
            "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=1;8\n",
            (
                "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=1\n"
                "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS:STRING=1\n"
            ),
        ],
    )
    def test_split_override_fails_before_generation_or_compilation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, split_entry: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text("GENERATIVEQC_AOT_SHARDS:STRING=8\n" + split_entry)

        def unexpected_generation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("split mismatch must fail before generation/compilation")

        monkeypatch.setattr(
            validator, "generate_production_shards", unexpected_generation
        )
        with pytest.raises(
            ValueError, match="GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=1"
        ):
            validator.run_measurement(
                nvcc=Path("/fake/nvcc"),
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=cache,
                timeout=23,
            )

    @pytest.mark.parametrize(
        "contents",
        [
            "",
            "// GENERATIVEQC_AOT_SHARDS:STRING=8\n",
            "GENERATIVEQC_AOT_SHARDS:STRING=10\n",
            "GENERATIVEQC_AOT_SHARDS:STRING=0\n",
            "GENERATIVEQC_AOT_SHARDS:STRING=8;10\n",
            "GENERATIVEQC_AOT_SHARDS:STRING=8\nGENERATIVEQC_AOT_SHARDS:STRING=8\n",
        ],
    )
    def test_bad_config_fails_before_generation_or_compilation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(contents)

        def unexpected_generation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("production mismatch must fail before generation/compilation")

        monkeypatch.setattr(
            validator, "generate_production_shards", unexpected_generation
        )
        with pytest.raises(ValueError, match="GENERATIVEQC_AOT_SHARDS=8"):
            validator.run_measurement(
                nvcc=Path("/fake/nvcc"),
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=cache,
                timeout=23,
            )

    def test_missing_config_is_not_silently_assumed(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validator.validate_production_shard_count(
                tmp_path / "CMakeCache.txt", FAKE_NVCC
            )

    def test_cli_reports_config_override_failure(self, tmp_path: Path) -> None:
        nvcc = tmp_path / "nvcc"
        nvcc.touch()
        configured = tmp_path / "configured.txt"
        configured.write_text("GENERATIVEQC_AOT_SHARDS:STRING=10\n")
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
                    "--cmake-cache",
                    str(configured),
                ]
            )
            == 1
        )
        report = json.loads((output / "report.json").read_text())
        assert report["status"] == "fail" and report["measurement"] is None
        assert "GENERATIVEQC_AOT_SHARDS=8" in report["failures"][0]

    def test_workflow_binds_the_configured_production_cache(self) -> None:
        workflow = (validator.ROOT / ".github/workflows/ci.yml").read_text()
        configure = workflow.split(
            "- name: Configure direct J/K AOT release context", 1
        )[1].split("- name:", 1)[0]
        assert "-DGENERATIVEQC_CUDA_FAST_COMPILE=OFF" in configure
        step = workflow.split(
            "- name: Qualify direct J/K AOT static resources (shard 6)", 1
        )[1].split("- name:", 1)[0]
        assert "--cmake-cache build/cuda-release-sm120/CMakeCache.txt" in step
        assert "--record-output build/direct-jk-aot/ledger.candidate.json" in step

    def test_cmake_records_the_resolved_compiler_cache(self) -> None:
        cmake = (validator.ROOT / "CMakeLists.txt").read_text()
        assert "GENERATIVEQC_RESOLVED_COMPILER_CACHE" in cmake


class TestProductionLauncherBinding:
    @pytest.mark.parametrize("suffix", ["Release", "unscanned_Release"])
    def test_literal_cmake_launcher_is_bound(self, tmp_path: Path, suffix: str) -> None:
        rules = _write_production_rule(tmp_path)
        rules.write_text(
            rules.read_text().replace("_sm_120_Release", f"_sm_120_{suffix}")
        )
        build = tmp_path / "build.ninja"
        build.write_text(
            build.read_text().replace("_sm_120_Release", f"_sm_120_{suffix}")
        )
        assert validator.production_cache_launcher(tmp_path, FAKE_NVCC) == Path(
            "/verified/ccache"
        )

    @pytest.mark.parametrize("code_check", [False, True])
    def test_edge_launcher_is_bound_to_the_qualified_shard(
        self, tmp_path: Path, code_check: bool
    ) -> None:
        rules = _write_production_rule(tmp_path)
        prefix = f"/usr/bin/cmake -E env CCACHE_BASEDIR={tmp_path.resolve()} /verified/ccache "
        variables = "${LAUNCHER}${CODE_CHECK}" if code_check else "${LAUNCHER}"
        rules.write_text(rules.read_text().replace(prefix, variables))
        (tmp_path / "build.ninja").write_text(
            "build CMakeFiles/generativeqc_aot_sm_120.dir/generated/generativeqc_generated_shell_sm120_shard_6.cu.o: CUDA_COMPILER__generativeqc_aot_sm_120_Release shard_6.cu\n"
            f"  LAUNCHER = {prefix}\n"
            "  FLAGS = -O3 -DNDEBUG -std=c++20\n"
            "  CODE_CHECK = \n"
            "build unrelated.o: unrelated source.cu\n"
            "  LAUNCHER = /other/wrapper\n"
        )
        assert validator.production_cache_launcher(tmp_path, FAKE_NVCC) == Path(
            "/verified/ccache"
        )

    @pytest.mark.parametrize(
        "old,new",
        [
            ("/verified/ccache", "/verified/sccache"),
            ("/verified/ccache", "ccache"),
            ("/usr/bin/cmake", "/usr/bin/wrapper"),
            ("/verified/ccache", "/verified/extra /verified/ccache"),
            (str(FAKE_NVCC), "/other/nvcc"),
            ("_sm_120_Release", "_sm_120_Debug"),
            ("/verified/ccache", "/verified/ccache$ "),
        ],
    )
    def test_unknown_or_changed_rule_fails_closed(
        self, tmp_path: Path, old: str, new: str
    ) -> None:
        rules = _write_production_rule(tmp_path)
        rules.write_text(rules.read_text().replace(old, new))
        with pytest.raises(ValueError, match="production"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    @pytest.mark.parametrize(
        "flag",
        [
            "-ccbin=/usr/bin/clang++",
            "-ccbin /usr/bin/clang++",
            "--compiler-bindir=/usr/bin/clang++",
            "--compiler-bindir /usr/bin/clang++",
        ],
    )
    @pytest.mark.parametrize("location", ["rule", "edge"])
    @pytest.mark.parametrize("per_edge_launcher", [False, True])
    def test_normal_variable_host_compiler_fails_before_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        flag: str,
        location: str,
        per_edge_launcher: bool,
    ) -> None:
        cache = tmp_path / "CMakeCache.txt"
        cache.write_text(_production_cache())
        monkeypatch.delenv("CUDAHOSTCXX", raising=False)
        rules = _write_production_rule(tmp_path)
        build = tmp_path / "build.ninja"
        if per_edge_launcher:
            prefix = f"/usr/bin/cmake -E env CCACHE_BASEDIR={tmp_path.resolve()} /verified/ccache "
            rules.write_text(rules.read_text().replace(prefix, "${LAUNCHER}"))
            build.write_text(build.read_text() + f"  LAUNCHER = {prefix}\n")
        if location == "rule":
            rules.write_text(rules.read_text().replace("$DEFINES", f"{flag} $DEFINES"))
        else:
            build.write_text(build.read_text().replace("FLAGS = ", f"FLAGS = {flag} "))

        def unexpected_generation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("normal-variable host selection must fail before generation")

        monkeypatch.setattr(
            validator, "generate_production_shards", unexpected_generation
        )
        with pytest.raises(ValueError, match="generated CUDA host compiler override"):
            validator.run_measurement(
                nvcc=FAKE_NVCC,
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=cache,
                timeout=23,
            )

    def test_edge_flags_are_checked_with_a_per_edge_launcher(
        self, tmp_path: Path
    ) -> None:
        rules = _write_production_rule(tmp_path)
        prefix = f"/usr/bin/cmake -E env CCACHE_BASEDIR={tmp_path.resolve()} /verified/ccache "
        rules.write_text(rules.read_text().replace(prefix, "${LAUNCHER}"))
        build = tmp_path / "build.ninja"
        build.write_text(
            build.read_text()
            + f"  LAUNCHER = {prefix}\n"
            + "  FLAGS = -ccbin=/usr/bin/clang++\n"
        )
        # Reject duplicate FLAGS rather than silently choosing a safe-looking one.
        with pytest.raises(ValueError, match="literal FLAGS"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)
        build.write_text(
            build.read_text().replace("  FLAGS = -O3 -DNDEBUG -std=c++20\n", "")
        )
        with pytest.raises(ValueError, match="generated CUDA host compiler override"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    @pytest.mark.parametrize(
        "flags",
        [None, "$HOST_FLAGS", "${HOST_FLAGS}", "-O3 $\n    -ccbin=/usr/bin/clang++"],
    )
    def test_nonliteral_or_missing_edge_flags_fail_closed(
        self, tmp_path: Path, flags: str | None
    ) -> None:
        _write_production_rule(tmp_path)
        build = tmp_path / "build.ninja"
        build.write_text(
            build.read_text().replace(
                "  FLAGS = -O3 -DNDEBUG -std=c++20\n",
                "" if flags is None else f"  FLAGS = {flags}\n",
            )
        )
        with pytest.raises(ValueError, match="literal FLAGS"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    def test_normal_edge_flags_do_not_borrow_another_targets_host_override(
        self, tmp_path: Path
    ) -> None:
        _write_production_rule(tmp_path)
        build = tmp_path / "build.ninja"
        build.write_text(
            build.read_text().replace(
                "-O3 -DNDEBUG -std=c++20",
                "-O3 -DNDEBUG --generate-code=arch=compute_120,code=[compute_120,sm_120] -Xcompiler=-fPIC -std=c++20",
            )
            + "build unrelated.o: unrelated source.cu\n"
            + "  FLAGS = -ccbin=/usr/bin/clang++\n"
        )
        assert validator.production_cache_launcher(tmp_path, FAKE_NVCC) == Path(
            "/verified/ccache"
        )

    @pytest.mark.parametrize(
        "flag",
        [
            "-optf=/tmp/host-options",
            "-optf /tmp/host-options",
            "--options-file=/tmp/host-options",
            "--options-file /tmp/host-options",
        ],
    )
    def test_edge_flags_cannot_hide_a_host_override_in_an_options_file(
        self, tmp_path: Path, flag: str
    ) -> None:
        _write_production_rule(tmp_path)
        build = tmp_path / "build.ninja"
        build.write_text(build.read_text().replace("FLAGS = ", f"FLAGS = {flag} "))
        with pytest.raises(ValueError, match="options-file indirection"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    def test_different_base_directory_fails_closed(self, tmp_path: Path) -> None:
        rules = _write_production_rule(tmp_path)
        rules.write_text(
            rules.read_text().replace(
                f"CCACHE_BASEDIR={tmp_path.resolve()}", "CCACHE_BASEDIR=/other"
            )
        )
        with pytest.raises(ValueError, match="canonical"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    @pytest.mark.parametrize("kind", ["rule", "command"])
    def test_duplicate_rule_or_command_is_rejected(
        self, tmp_path: Path, kind: str
    ) -> None:
        rules = _write_production_rule(tmp_path)
        text = rules.read_text()
        rules.write_text(text + (text if kind == "rule" else text.splitlines(True)[1]))
        with pytest.raises(ValueError, match="one"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    @pytest.mark.parametrize(
        "edge",
        [
            "missing",
            "duplicate",
            "no_launcher",
            "duplicate_launcher",
            "code_check",
            "missing_separator",
        ],
    )
    def test_ambiguous_edge_launcher_is_rejected(
        self, tmp_path: Path, edge: str
    ) -> None:
        rules = _write_production_rule(tmp_path)
        prefix = f"/usr/bin/cmake -E env CCACHE_BASEDIR={tmp_path.resolve()} /verified/ccache "
        rules.write_text(rules.read_text().replace(prefix, "${LAUNCHER}${CODE_CHECK}"))
        launcher = f"  LAUNCHER = {prefix}\n"
        body = (
            "build CMakeFiles/generativeqc_aot_sm_120.dir/generated/generativeqc_generated_shell_sm120_shard_6.cu.o: CUDA_COMPILER__generativeqc_aot_sm_120_Release shard_6.cu\n"
            + launcher
            + "  FLAGS = -O3 -DNDEBUG -std=c++20\n"
        )
        if edge == "missing":
            body = ""
        elif edge == "duplicate":
            body *= 2
        elif edge == "no_launcher":
            body = body.replace(launcher, "")
        elif edge == "duplicate_launcher":
            body += launcher
        elif edge == "missing_separator":
            body = body.replace(prefix + "\n", prefix.rstrip() + "\n")
        else:
            body += "  CODE_CHECK = /other/wrapper\n"
        (tmp_path / "build.ninja").write_text(body)
        with pytest.raises(ValueError, match="production"):
            validator.production_cache_launcher(tmp_path, FAKE_NVCC)

    @pytest.mark.parametrize(
        "path,name", [("/verified/sccache", "sccache"), ("/different/ccache", "ccache")]
    )
    def test_adapter_launcher_mismatch_fails_before_stats_or_compile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str, name: str
    ) -> None:
        _write_production_rule(tmp_path)
        original_directory = Path.cwd()
        original_basedir = os.environ.get("CCACHE_BASEDIR")
        launcher = CompilerCacheLauncher(Path(path), name, "verified")

        def unexpected_work(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("mismatched launcher must not collect statistics or compile")

        monkeypatch.setattr(validator, "cache_statistics", unexpected_work)
        monkeypatch.setattr(validator.CudaCompilerAdapter, "compile", unexpected_work)
        with pytest.raises(ValueError, match="compiler-cache launcher mismatch"):
            validator.compile_shard(
                nvcc=FAKE_NVCC,
                shard=tmp_path / "shard.cu",
                generated=tmp_path / "generated",
                cache=tmp_path / "cache",
                identity=_identity(),
                timeout=23,
                working_directory=tmp_path,
                launcher=launcher,
            )
        assert Path.cwd() == original_directory
        assert os.environ.get("CCACHE_BASEDIR") == original_basedir

    def test_missing_rules_fail_before_generation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        configured = tmp_path / "CMakeCache.txt"
        configured.write_text(_production_cache())

        def unexpected_generation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("missing production rules must fail before generation")

        monkeypatch.setattr(
            validator, "generate_production_shards", unexpected_generation
        )
        with pytest.raises(FileNotFoundError, match="rules.ninja"):
            validator.run_measurement(
                nvcc=FAKE_NVCC,
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=configured,
                timeout=23,
            )

    def test_effective_explicit_launcher_supersedes_project_candidate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        configured = tmp_path / "CMakeCache.txt"
        configured.write_text(
            _production_cache(
                GENERATIVEQC_RESOLVED_COMPILER_CACHE="/project/ccache",
                CMAKE_CUDA_COMPILER_LAUNCHER="/usr/bin/cmake;-E;env;CCACHE_BASEDIR="
                + str(tmp_path.resolve())
                + ";/explicit/ccache",
            )
        )
        _write_production_rule(tmp_path, launcher="/explicit/ccache")
        selected: list[Path] = []

        def verify(path: Path, timeout: float) -> Any:
            selected.append(path)
            raise RuntimeError("stop after effective launcher selection")

        monkeypatch.setattr(validator, "verify_compiler_cache", verify)
        with pytest.raises(
            RuntimeError, match="stop after effective launcher selection"
        ):
            validator.run_measurement(
                nvcc=FAKE_NVCC,
                cache=tmp_path / "objects",
                output=tmp_path / "report",
                generated_directory=tmp_path / "generated",
                cmake_cache=configured,
                timeout=23,
            )
        assert selected == [Path("/explicit/ccache")]

    def test_ambient_host_override_fails_before_generation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        configured = tmp_path / "CMakeCache.txt"
        configured.write_text(_production_cache())
        monkeypatch.setenv("CUDAHOSTCXX", "/usr/bin/clang++")
        with pytest.raises(ValueError, match="CUDAHOSTCXX"):
            validator.validate_production_shard_count(configured, FAKE_NVCC)
