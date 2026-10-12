"""Exercise shared compiler-cache ownership and recoverable master failures."""

import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    section = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [A-Za-z0-9_-]+:\n", section, maxsplit=1)[0]


def _step(job: str, name: str) -> str:
    return _job(job).split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def _compiler_restore(job: str) -> str:
    return next(
        step
        for step in re.split(r"\n      - ", _job(job))
        if "uses: actions/cache/restore@" in step and "path: ~/.cache/ccache" in step
    )


def _compiler_saves(job: str) -> list[str]:
    return [
        step
        for step in re.split(r"\n      - ", _job(job))
        if "uses: actions/cache/save@" in step and "path: ~/.cache/ccache" in step
    ]


def test_cpu_consumers_share_master_namespace_without_publishing() -> None:
    producer = _compiler_restore("master-cpu-cache")
    key = next(line.strip() for line in producer.splitlines() if "key:" in line)
    prefix = "ccache-linux-cpu-v4-ubuntu24-x86_64-"
    assert key.startswith(f"key: {prefix}")
    assert "restore-keys: |\n            " + prefix in producer
    header = _job("master-cpu-cache").split("\n    steps:\n", 1)[0]
    assert "github.event_name == 'push' && github.ref == 'refs/heads/master'" in header
    for job in ("cpu", "python", "cpu-benchmark"):
        restore = _compiler_restore(job)
        assert key in restore
        assert "restore-keys: |\n            " + prefix in restore
        assert "CCACHE_MAXSIZE: 2G" in _job(job).split("\n    steps:\n", 1)[0]
        assert not _compiler_saves(job)


def test_master_cpu_seed_executes_consumer_compiler_and_coverage_modes(
    tmp_path: Path,
) -> None:
    build = _step("master-cpu-cache", "Compile GCC, instrumented GCC, and Clang")
    script = textwrap.dedent(build.split("        run: |\n", 1)[1])
    calls = tmp_path / "calls.jsonl"
    cmake = tmp_path / "cmake"
    cmake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps([os.environ.get('CXX'), sys.argv[1:]]) + '\\n')\n"
    )
    cmake.chmod(0o755)
    cache = tmp_path / "ccache"
    cache.write_text("#!/bin/sh\nexit 0\n")
    cache.chmod(0o755)
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS": str(calls),
        },
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,  # The seed resets "build"; never delete the test runner's build.
        timeout=10,
    )
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    configures = [(compiler, args) for compiler, args in records if "-S" in args]
    assert [compiler for compiler, _ in configures] == ["g++", "g++", "clang++"]
    assert len(records) == 6
    assert [args[args.index("-B") + 1] for _, args in configures] == [
        "build", "build", "build"
    ]
    builds = [args for _, args in records if "--build" in args]
    assert [args[args.index("--build") + 1] for args in builds] == [
        "build", "build", "build"
    ]
    for _, args in configures:
        assert "-DGENERATIVEQC_ENABLE_CUDA=OFF" in args
        assert "-DCMAKE_BUILD_TYPE=Release" in args
        assert "-DGENERATIVEQC_COMPILER_CACHE=ccache" in args
    plain, coverage, clang = [args for _, args in configures]
    assert "-DCMAKE_CXX_FLAGS=" in plain and "-DCMAKE_CXX_FLAGS=" in clang
    flags = "-fprofile-arcs -ftest-coverage -fprofile-update=atomic"
    assert f"-DCMAKE_CXX_FLAGS={flags}" in coverage
    assert flags in _step("cpu", "Configure")
    assert "-DCMAKE_EXE_LINKER_FLAGS=--coverage" in coverage
    assert "-DCMAKE_SHARED_LINKER_FLAGS=--coverage" in coverage
    # Profile output directories are compiler identity. Never normalize them away.
    consumer_directory = re.search(r"-B (\S+)", _step("cpu", "Configure")).group(1)
    assert coverage[coverage.index("-B") + 1] == consumer_directory
    assert "ctest" not in _job("master-cpu-cache")


def test_master_python_seed_matches_python_shard_compilation_paths(
    tmp_path: Path,
) -> None:
    """Exercise the seed with a fake CMake, without deleting the active build."""
    seed = _step("master-cpu-cache", "Build Python native cache seed")
    consumer = _step("python", "Build the CPU native library")
    script = textwrap.dedent(seed.split("        run: |\n", 1)[1])
    calls = tmp_path / "calls.jsonl"
    cmake = tmp_path / "cmake"
    cmake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    cmake.chmod(0o755)
    ccache = tmp_path / "ccache"
    ccache.write_text("#!/bin/sh\nexit 0\n")
    ccache.chmod(0o755)
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS": str(calls),
            "GENERATIVEQC_XTB_CPU_LINALG_LIBRARY": str(tmp_path / "openblas.so"),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 2
    assert records[0][records[0].index("-B") + 1] == "build"
    assert records[1] == ["--build", "build", "--parallel"]
    assert "-DGENERATIVEQC_BUILD_TESTS=OFF" in records[0]
    assert "-DGENERATIVEQC_PYTHON_WHEEL=ON" in records[0]
    assert "-DGENERATIVEQC_COMPILER_CACHE=ccache" in records[0]
    # The real consumer uses the same path; compiler identity remains CMake's.
    assert "cmake -S . -B build -G Ninja" in consumer


def test_python_test_timeout_leaves_finite_setup_headroom() -> None:
    job = _job("python")
    tests = _step("python", "Run Python tests with coverage")
    pattern = (
        r"timeout-minutes: \$\{\{ \(github.event_name == 'schedule' \|\| "
        r"github.event_name == 'workflow_dispatch'\) && (\d+) \|\| (\d+) \}\}"
    )
    job_limits = re.search(pattern, job.split("\n    steps:\n", 1)[0])
    test_limits = re.search(pattern, tests)
    assert job_limits and test_limits
    assert tuple(map(int, test_limits.groups())) == (90, 30)
    assert tuple(map(int, job_limits.groups())) == (100, 40)
    assert "id: python_tests" in tests
    assert "continue-on-error:" not in tests
    assert "-m pytest" in tests and '"${test_targets[@]}"' in tests


@pytest.mark.parametrize(
    ("job", "restore_id", "first_id", "last_id", "save_name", "complete_name"),
    [
        (
            "master-cpu-cache",
            "master_cpu_ccache",
            "cpu_seed",
            "python_seed",
            "Preserve partial master CPU compiler work",
            "Save complete CPU compiler cache",
        ),
        (
            "native-cli-artifact",
            "cli_ccache",
            "fast_cuda_seed",
            "cli_smoke",
            "Preserve partial master CUDA compiler work",
            "Save complete shared CUDA ccache on master",
        ),
    ],
)
@pytest.mark.parametrize(
    ("event", "ref", "hit", "primary", "first", "last", "expected"),
    [
        ("push", "refs/heads/master", "false", "key", "success", "failure", True),
        ("push", "refs/heads/master", "", "key", "failure", "skipped", True),
        ("push", "refs/heads/master", "false", "key", "cancelled", "skipped", True),
        ("push", "refs/heads/master", "false", "key", "success", "cancelled", True),
        ("push", "refs/heads/master", "false", "key", "success", "success", False),
        ("push", "refs/heads/master", "true", "key", "success", "failure", False),
        ("push", "refs/heads/master", "false", "", "success", "failure", False),
        ("push", "refs/heads/master", "false", "key", "", "skipped", False),
        ("push", "refs/heads/master", "false", "key", "skipped", "skipped", False),
        (
            "pull_request",
            "refs/heads/master",
            "false",
            "key",
            "success",
            "failure",
            False,
        ),
        (
            "merge_group",
            "refs/heads/master",
            "false",
            "key",
            "success",
            "failure",
            False,
        ),
        (
            "workflow_dispatch",
            "refs/heads/master",
            "false",
            "key",
            "success",
            "failure",
            False,
        ),
        ("push", "refs/tags/v1.0.0", "false", "key", "success", "failure", False),
        ("push", "refs/heads/feature", "false", "key", "success", "failure", False),
    ],
)
def test_only_started_master_failures_preserve_distinct_recovery_keys(
    job: str,
    restore_id: str,
    first_id: str,
    last_id: str,
    save_name: str,
    complete_name: str,
    event: str,
    ref: str,
    hit: str,
    primary: str,
    first: str,
    last: str,
    expected: bool,
) -> None:
    save = _step(job, save_name)
    assert "always()" in save
    assert "timeout-minutes: 2" in save
    assert "continue-on-error: true" in save
    expression = save.split("        if: ", 1)[1].splitlines()[0].strip()
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("always()", "true")
    values = {
        "github.event_name": event,
        "github.ref": ref,
        f"steps.{restore_id}.outputs.cache-primary-key": primary,
        f"steps.{restore_id}.outputs.cache-hit": hit,
        f"steps.{first_id}.outcome": first,
        f"steps.{last_id}.outcome": last,
    }
    for index, (source, _) in enumerate(values.items()):
        expression = expression.replace(source, f'"$VALUE_{index}"')
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={
            **os.environ,
            **{f"VALUE_{i}": value for i, value in enumerate(values.values())},
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == ("run" if expected else "skip")
    assert (
        "format('{0}-partial-{1}-{2}', "
        f"steps.{restore_id}.outputs.cache-primary-key, github.run_id, github.run_attempt)"
    ) in save
    complete = _step(job, complete_name)
    assert f"key: ${{{{ steps.{restore_id}.outputs.cache-primary-key }}}}" in complete
    assert "always()" not in complete and "failure()" not in complete
    assert "continue-on-error: true" in complete


def test_cuda_complete_snapshots_follow_both_variants_and_smoke() -> None:
    job = _job("native-cli-artifact")
    sequence = (
        "Seed fast CUDA CI objects in the shared cache",
        "Build monolithic C++ CUDA CLI (release or fast PR)",
        "Smoke-test release-mode native command",
        "Save complete shared CUDA ccache on master",
        "Publish trusted master CUDA ccache snapshot",
    )
    positions = [job.index("name: " + name) for name in sequence]
    assert positions == sorted(positions)
    for name in sequence[-2:]:
        step = _step("native-cli-artifact", name)
        assert (
            "github.event_name == 'push' && github.ref == 'refs/heads/master'" in step
        )
        assert "always()" not in step and "failure()" not in step
    producer = _compiler_restore("native-cli-artifact")
    consumer = _compiler_restore("cuda-compile")
    key = next(line.strip() for line in producer.splitlines() if "key:" in line)
    assert key in consumer
    for restore in (producer, consumer):
        assert "ccache-linux-cuda-v4-12.9-sm120-" in restore
        assert "ccache-cuda-v2-12.9-sm120-" in restore
        assert "ccache-native-cli-v1-cuda12.9-sm120-" in restore
    assert not _compiler_saves("cuda-compile")
    assert "needs.cuda-compile.outputs.publish_cache_snapshot == 'true'" in job


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_codspeed_non_pr_events_initialize_all_setup_guards(event: str) -> None:
    header = _job("cpu-benchmark").split("\n    steps:\n", 1)[0]
    assert 'GENERATIVEQC_CODSPEED_RUN: "1"' in header
    selector = _step("cpu-benchmark", "Select change-aware PR CodSpeed coverage")
    assert "if: github.event_name == 'pull_request'" in selector
    assert event != "pull_request"
    for name in (
        "Install uv and Python 3.11",
        "Install ccache 4.14",
        "Restore shared Linux CPU ccache (CodSpeed)",
        "Reset CPU benchmark ccache statistics",
        "Build CPU benchmark library",
        "Show CPU benchmark ccache statistics",
        "Install benchmark dependencies",
    ):
        assert "if: env.GENERATIVEQC_CODSPEED_RUN != '0'" in _step(
            "cpu-benchmark", name
        )
    run = _step("cpu-benchmark", "Run CPU CodSpeed endpoint suite")
    assert "github.event_name != 'pull_request'" in run
    assert "steps.baseline.outputs.qualified == 'true'" in run


@pytest.mark.parametrize(
    ("changed", "expected_run"),
    [
        ("docs/index.md\n", "0"),
        ("tests/python/test_foo.py\n", "0"),
        ("manifests/maintenance/record.json\n", "0"),
        ("src/scf/rhf.cpp\n", "1"),
        ("src/cc/df_triples_cuda.cu\n", "1"),
        ("tools/generate_df_occupied_triples.py\n", "1"),
        ("CMakeLists.txt\n", "1"),
        (".github/workflows/ci.yml\n", "1"),
    ],
)
def test_pr_codspeed_selector_only_overrides_the_run_guard(
    tmp_path: Path, changed: str, expected_run: str
) -> None:
    selector = _step("cpu-benchmark", "Select change-aware PR CodSpeed coverage")
    script = selector.split("        run: |\n", 1)[1]
    script = "\n".join(line[10:] for line in script.splitlines())
    script = script.replace(
        "${{ github.event.pull_request.base.sha }}...${{ github.event.pull_request.head.sha }}",
        "base...head",
    )
    git = tmp_path / "git"
    git.write_text('#!/bin/sh\nprintf "%s" "$CHANGED_FILES"\n')
    git.chmod(0o755)
    output = tmp_path / "environment"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_ENV": str(output),
            "CHANGED_FILES": changed,
            "GENERATIVEQC_CODSPEED_RUN": "1",
            "GENERATIVEQC_CODSPEED_EXTRA_CASES": "wb97mv",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    updates = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert updates == {"GENERATIVEQC_CODSPEED_RUN": expected_run}
    effective = {"GENERATIVEQC_CODSPEED_EXTRA_CASES": "wb97mv", **updates}
    assert effective["GENERATIVEQC_CODSPEED_EXTRA_CASES"] == "wb97mv"
