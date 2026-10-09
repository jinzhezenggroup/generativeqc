"""Resolve actual emitted Fock includes in standalone checkout/wheel commands."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tomllib
from generativeqc_compiler.common import paths
from generativeqc_compiler.common.cuda_adapter import (
    CudaCompilerAdapter,
    CudaCompileResult,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.integral import benchmark
from generativeqc_compiler.integral.autotune import supported_schedule_trials
from generativeqc_compiler.integral.shell_spec import DPDS_SPEC
from generativeqc_compiler.integral.tuning.process import _compile_trial

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("installed", (False, True))
@pytest.mark.parametrize("entry", ("benchmark", "tuning"))
def test_standalone_commands_preprocess_runtime_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: bool, entry: str
) -> None:
    """Use real preprocessing with the command's include paths, without CUDA."""
    host = shutil.which("c++")
    if host is None:
        pytest.skip("host preprocessor unavailable")
    if installed:
        # Materialize exactly the wheel's declared runtime header assets.
        mapping = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
            "scikit-build"
        ]["wheel"]["force-include"]
        for relative in (
            "src/runtime/compensated_atomic.cuh",
            "src/runtime/compensated_output.hpp",
        ):
            destination = tmp_path / "site" / mapping[relative]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        monkeypatch.setattr(paths, "PACKAGE", tmp_path / "site/generativeqc_compiler")
    cuda = tmp_path / "cuda"
    cuda.mkdir()
    (cuda / "cuda_runtime.h").touch()
    run = subprocess.run
    calls = []

    def preprocess(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        source = next(item for item in command if str(item).endswith(".cu"))
        assert '#include "runtime/compensated_atomic.cuh"' in Path(source).read_text()
        result = run(
            [
                host,
                "-E",
                "-x",
                "c++",
                f"-I{cuda}",
                *(item for item in command if item.startswith("-I")),
                source,
                "-o",
                str(tmp_path / "preprocessed.ii"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        calls.append(command)
        # Stop the benchmark before execution; this is an include check only.
        return subprocess.CompletedProcess(command, 17, "", "")

    if entry == "benchmark":
        monkeypatch.setattr(benchmark.subprocess, "run", preprocess)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark",
                "--nvcc",
                "nvcc",
                "--architecture",
                "sm_120",
                "--shell-class",
                "dpds",
                "--consumer",
                "fock",
                "--tasks",
                "1",
            ],
        )
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 17
    else:
        trial = supported_schedule_trials(
            DPDS_SPEC, "fock", target=cuda_target_info("sm_120")
        )[0]
        source = benchmark.emit_shell_class_benchmark_cuda(
            trial.spec,
            1,
            1,
            0,
            1,
            1,
            consumer="fock",
            target=trial.target,
        )
        (tmp_path / f"{trial.spec.name}_{trial.schedule_id}.cu").write_text(source)

        def compile_command(self: Any, command: list[str]) -> CudaCompileResult:
            preprocess(command)
            return CudaCompileResult(17, False, 0.0, "", "")

        monkeypatch.setattr(CudaCompilerAdapter, "_run_compiler", compile_command)
        _compile_trial(Path("nvcc"), "sm_120", tmp_path, trial)
    assert len(calls) == 1
