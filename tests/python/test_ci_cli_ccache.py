"""Exercise strict compiler caching for the CI-only native CUDA CLI build."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _step(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("\n  native-cli-artifact:\n", 1)[1]
    job = re.split(r"\n  [A-Za-z0-9_-]+:\n", job, maxsplit=1)[0]
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def _fake_tools(tmp_path: Path, failure: str = "") -> dict[str, str]:
    script = (
        "#!" + sys.executable + "\n"
        "import json, os, pathlib, sys\n"
        "tool = pathlib.Path(sys.argv[0]).name\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps([tool, args, "
        "os.environ.get('CCACHE_SLOPPINESS')]) + '\\n')\n"
        "if tool == 'cmake':\n"
        "    stage = 'build' if '--build' in args else 'configure'\n"
        "    if stage == os.environ['FAIL_STAGE']:\n"
        "        sys.exit(23)\n"
        "elif '--show-stats' in args and '--verbose' in args:\n"
        "    print('Uncacheable calls: 2 / 3 (66.67%)')\n"
        "    print('  Could not use precompiled header: 2 / 2 (100.0%)')\n"
    )
    for name in ("cmake", "ccache"):
        executable = tmp_path / name
        executable.write_text(script)
        executable.chmod(0o755)
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if key != "CCACHE_SLOPPINESS"
        },
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "FAIL_STAGE": failure,
    }


@pytest.mark.parametrize("fast", ["ON", "OFF"])
@pytest.mark.parametrize("failure", ["", "configure", "build"])
def test_cli_ci_configures_non_pch_strict_caching_and_preserves_failures(
    tmp_path: Path, fast: str, failure: str
) -> None:
    step = _step("Build monolithic C++ CUDA CLI (release or fast PR)")
    assert "continue-on-error:" not in step
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env={**_fake_tools(tmp_path, failure), "CLI_FAST_COMPILE": fast},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == (23 if failure else 0), result.stdout + result.stderr
    records = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert records[0][:2] == ["ccache", ["--zero-stats"]]
    assert all(sloppiness is None for _, _, sloppiness in records)
    configures = [
        args for tool, args, _ in records if tool == "cmake" and "--build" not in args
    ]
    assert len(configures) == 1
    configure = configures[0]
    preset_name = configure[configure.index("--preset") + 1]
    presets = json.loads((ROOT / "CMakePresets.json").read_text())["configurePresets"]
    preset = next(preset for preset in presets if preset["name"] == preset_name)
    # The normal preset keeps its optional cold-build PCH policy. Only this
    # cache-oriented CI invocation overrides it, for both compiler tiers.
    assert preset["cacheVariables"]["GENERATIVEQC_ENABLE_CXX_PCH"] == "ON"
    options = dict(preset["cacheVariables"])
    options.update(arg[2:].split("=", 1) for arg in configure if arg.startswith("-D"))
    assert options["GENERATIVEQC_ENABLE_CXX_PCH"] == "OFF"
    assert options["GENERATIVEQC_COMPILER_CACHE"] == "ccache"
    assert options["GENERATIVEQC_CUDA_FAST_COMPILE"] == fast
    builds = [
        args for tool, args, _ in records if tool == "cmake" and "--build" in args
    ]
    assert builds == (
        []
        if failure == "configure"
        else [["--build", "--preset", preset_name, "--parallel", "3"]]
    )


def test_cli_verbose_statistics_include_uncacheable_reasons(tmp_path: Path) -> None:
    step = _step("Show detailed CLI compiler cache statistics")
    script = step.split("        run: ", 1)[1].strip()
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=_fake_tools(tmp_path),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "Could not use precompiled header: 2 / 2" in result.stdout
    # Diagnostics must remain available after a failed build without converting
    # that failure into success or making a diagnostics failure a new gate.
    assert "continue-on-error: true" in step
    assert "always()" in step


@pytest.mark.parametrize("tier", ["skip", "fast", "release"])
def test_cli_verbose_statistics_respect_skipped_builds(tier: str) -> None:
    step = _step("Show detailed CLI compiler cache statistics")
    expression = step.split("        if: ", 1)[1].splitlines()[0]
    expression = expression.replace("always()", "true")
    expression = expression.replace("steps.cli_scope.outputs.tier", '"$TIER"')
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={**os.environ, "TIER": tier},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == ("skip" if tier == "skip" else "run")
