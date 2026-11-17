"""Exercise wheel resource limits at their container and build-tool boundaries."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from cibuildwheel.environment import parse_environment
from cibuildwheel.oci_container import OCIContainerEngineConfig
from test_ci_cpu_ccache_snapshot import fields

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/wheels.yml").read_text()


def _step(name: str) -> str:
    section = WORKFLOW.split(f"      - name: {name}\n", 1)[1]
    return re.split(r"\n      - (?:name|uses):", section, maxsplit=1)[0]


def _block(section: str, key: str) -> str:
    """Read the fixed workflow's literal/folded blocks without a YAML dependency."""
    match = re.search(rf"(?m)^( +){key}: ([>|]-?)\n((?:(?:\1  [^\n]*)?\n)+)", section)
    assert match is not None
    value = textwrap.dedent(match[3]).strip()
    return " ".join(value.splitlines()) if match[2].startswith(">") else value


def _build_environment() -> dict[str, str]:
    """Resolve runner expressions before handing options to cibuildwheel."""
    build = _step("Build and test wheel")
    return {
        key: re.sub(
            r"\$\{\{.*?\}\}",
            "123",
            _block(build, key) if value in (">-", "|") else str(value).strip('"'),
        )
        for key, value in fields(build, 8)["env"].items()
    }


def test_wheel_container_reserves_host_memory_without_disabling_cache_mount() -> None:
    """Use the pinned builder's parser, not string matching against YAML."""
    engine = OCIContainerEngineConfig.from_config_string(
        _build_environment()["CIBW_CONTAINER_ENGINE"]
    )
    assert engine.name == "docker"
    assert not engine.disable_host_mount
    options = dict(argument.split("=", 1) for argument in engine.create_args)
    assert options["--memory"] == "12g"
    assert options["--memory-swap"] == options["--memory"]
    assert options["--label"] == "gqc-wheel-run=123-123"


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ninja") is None,
    reason="build parallelism probe requires CMake and Ninja",
)
def test_wheel_environment_bounds_actual_cmake_build_parallelism(
    tmp_path: Path,
) -> None:
    """Observe overlapping build tasks without compiling or requiring a GPU."""
    environment = parse_environment(
        _build_environment()["CIBW_ENVIRONMENT"]
    ).as_dictionary(prev_environment=os.environ)
    limit = int(environment["CMAKE_BUILD_PARALLEL_LEVEL"])
    assert limit == 1
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, pathlib, sys, time\n"
        "start = time.monotonic()\n"
        "time.sleep(0.2)\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([start, time.monotonic()]))\n"
    )
    project = ["cmake_minimum_required(VERSION 3.24)", "project(WheelBudget NONE)"]
    for index in range(6):
        project.append(
            f'add_custom_target(probe_{index} ALL COMMAND "{sys.executable}" '
            f'"{probe}" "{tmp_path / f"timing-{index}.json"}" VERBATIM)'
        )
    (tmp_path / "CMakeLists.txt").write_text("\n".join(project) + "\n")
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build"), "-G", "Ninja"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["cmake", "--build", str(tmp_path / "build")],
        env=environment,
        check=True,
        capture_output=True,
        timeout=30,
    )
    intervals = [
        json.loads(path.read_text()) for path in tmp_path.glob("timing-*.json")
    ]
    assert len(intervals) == 6
    events = sorted(
        event for start, end in intervals for event in ((start, 1), (end, -1))
    )
    active = maximum = 0
    for _, change in events:
        active += change
        maximum = max(maximum, active)
    assert 1 <= maximum <= limit


@pytest.mark.parametrize("inspect_failure", [False, True])
def test_diagnostics_remove_only_owned_containers_even_when_inspection_fails(
    tmp_path: Path, inspect_failure: bool
) -> None:
    """Optional diagnostics cannot leak retained containers or touch unrelated work."""
    build_environment = _build_environment()
    assert build_environment["CIBW_DEBUG_KEEP_CONTAINER"] == "1"
    diagnostics_source = _step("Report wheel container resources and clean up")
    diagnostics = fields(diagnostics_source, 8)
    assert diagnostics["continue-on-error"] is True
    assert diagnostics["timeout-minutes"] <= 2
    assert "always()" in diagnostics["if"]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "with pathlib.Path(os.environ['DOCKER_CALLS']).open('a') as output:\n"
        "    output.write(json.dumps(arguments) + '\\n')\n"
        "if arguments[0] == 'ps':\n"
        "    assert arguments[-1] == 'label=gqc-wheel-run=123-2'\n"
        "    print('owned-build\\nowned-test')\n"
        "elif arguments[0] == 'inspect':\n"
        "    sys.exit(int(os.environ['INSPECT_FAILURE']))\n"
        "elif arguments[0] != 'rm':\n"
        "    sys.exit(17)\n"
    )
    docker.chmod(0o755)
    calls_path = tmp_path / "docker-calls.jsonl"
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _block(diagnostics_source, "run")],
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "GQC_WHEEL_RUN": "123-2",
            "DOCKER_CALLS": str(calls_path),
            "INSPECT_FAILURE": str(int(inspect_failure)),
        },
        check=True,
        capture_output=True,
        timeout=20,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert {call[-1] for call in calls if call[0] == "inspect"} == {
        "owned-build",
        "owned-test",
    }
    assert {tuple(call[1:]) for call in calls if call[0] == "rm"} == {
        ("--force", "--volumes", "owned-build"),
        ("--force", "--volumes", "owned-test"),
    }
