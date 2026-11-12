"""Keep CI lowering reuse separate from binary caches and correctness gates."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> str:
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    return re.split(
        r"\n  [a-zA-Z0-9_-]+:\n", source.split(f"\n  {name}:\n", 1)[1], maxsplit=1
    )[0]


def _step(job: str, name: str) -> str:
    return _job(job).split(f"      - name: {name}\n", 1)[1].split("      - ", 1)[0]


def test_source_snapshot_is_exact_data_only_and_excludes_bytecode() -> None:
    source = (ROOT / ".github/actions/cache-stationary-sources/action.yml").read_text()
    assert "uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9" in source
    assert "path: ${{ runner.temp }}/stationary-aot-source-cache" in source
    assert "key: stationary-aot-source-v2-${{ hashFiles(" in source
    for dependency in (
        "python/generativeqc_compiler/**",
        "src/integrals/**",
        "src/dft/**",
        "src/runtime/**",
        "tools/generate_stationary_force_aot.py",
        "tools/generate_native_stationary_cuda.py",
        "!**/__pycache__/**",
        "!**/*.pyc",
        "!**/*.pyo",
    ):
        assert repr(dependency) in source
    assert "restore-keys:" not in source
    assert "run:" not in source
    assert "CCACHE_DIR" not in source


@pytest.mark.parametrize("job", ("cuda-compile", "native-cli-artifact"))
def test_optional_transport_precedes_unchanged_required_compilation(job: str) -> None:
    restore = _step(job, "Restore stationary AOT semantic sources")
    assert "uses: ./.github/actions/cache-stationary-sources" in restore
    assert "continue-on-error: true" in restore
    assert "timeout-minutes: 2" in restore
    configure = (
        "Configure production CUDA compile check"
        if job == "cuda-compile"
        else "Build monolithic C++ CUDA CLI (release or fast PR)"
    )
    build = _step(job, configure)
    assert (
        '-DGENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE="$RUNNER_TEMP/'
        'stationary-aot-source-cache"'
    ) in build
    assert "continue-on-error:" not in build
    assert _job(job).index(restore) < _job(job).index(build)
    if job == "cuda-compile":
        assert "-DGENERATIVEQC_BUILD_TESTS=ON" in build
        required = _step(job, "Compile production CUDA and runtime CTest")
        assert "--target generativeqc_cuda_runtime_tests --parallel 3" in required
        assert "continue-on-error:" not in required
    else:
        assert '-DGENERATIVEQC_CUDA_FAST_COMPILE="$CLI_FAST_COMPILE"' in build
        # The executable CLI cache contract owns the effective parallelism cap.
        assert "cmake --build --preset cuda-release-cli" in build


def test_master_seed_and_release_share_sources_not_compiler_flag_identities() -> None:
    seed = _step("native-cli-artifact", "Seed fast CUDA CI objects in the shared cache")
    assert "github.event_name == 'push' && github.ref == 'refs/heads/master'" in seed
    assert "-DGENERATIVEQC_CUDA_FAST_COMPILE=ON" in seed
    assert "-DGENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE=" in seed
    assert "--target generativeqc_cuda_runtime_tests --parallel 3" in seed


def test_routine_reports_keep_junit_without_reenabling_coverage() -> None:
    report = _step("python", "Preserve Python test timings")
    assert "if: always() && steps.python_tests.outcome != ''" in report
    assert "steps.python_tests.outcome != 'skipped'" in report
    assert "name: pytest-timing-${{ matrix.shard }}" in report
    assert "path: .artifacts/pytest-${{ matrix.shard }}.xml" in report
    assert "include-hidden-files: true" in report
    assert "coverage-" not in report
    assert "retention-days: 7" in report
