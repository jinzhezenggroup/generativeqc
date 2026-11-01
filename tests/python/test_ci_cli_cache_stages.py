"""Exercise stage attribution and failure isolation without native compilation."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEED = "Seed fast CUDA CI objects in the shared cache"
SEED_REPORT = "Report fast CUDA seed compiler cache"
CLI = "Build monolithic C++ CUDA CLI (release or fast PR)"


def _steps() -> dict[str, str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("\n  native-cli-artifact:\n", 1)[1]
    job = re.split(r"\n  [A-Za-z0-9_-]+:\n", job, maxsplit=1)[0]
    return {part.splitlines()[0]: part for part in job.split("      - name: ")[1:]}


def _script(step: str) -> str:
    marker = "        run: "
    value = step.split(marker, 1)[1]
    if value.startswith("|\n"):
        return textwrap.dedent(value[2:])
    if value.startswith(">-\n"):
        return " ".join(textwrap.dedent(value[3:]).splitlines())
    return value.strip()


def _tools(tmp_path: Path, failure: str = "") -> dict[str, str]:
    script = f"#!{sys.executable}\n" + textwrap.dedent(
        """\
        import json, os, pathlib, sys
        tool, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
        state = pathlib.Path(os.environ['COUNTERS'])
        with open(os.environ['CALLS'], 'a') as stream:
            stream.write(json.dumps([tool, args]) + '\\n')
        fail = os.environ['FAIL_STAGE']
        if tool == 'ccache':
            if '--zero-stats' in args:
                if fail == 'reset':
                    sys.exit(23)
                state.write_text('0')
            else:
                if fail == 'report':
                    print('Cacheable calls: 99 / 99\\nHits: 99\\nMisses: 0')
                    sys.exit(23)
                calls = int(state.read_text())
                print(f'Cacheable calls: {calls} / {calls}\\nHits: 0\\nMisses: {calls}')
                if '--verbose' in args:
                    print('  Detailed reason: fixture')
        else:
            stage = 'build' if '--build' in args else 'configure'
            if stage == 'build':
                state.write_text('7' if args[args.index('--preset') + 1] == 'cuda-release' else '3')
            if fail == stage:
                sys.exit(23)
        """
    )
    for name in ("cmake", "ccache"):
        executable = tmp_path / name
        executable.write_text(script)
        executable.chmod(0o755)
    (tmp_path / "counters").write_text("99")
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "COUNTERS": str(tmp_path / "counters"),
        "FAIL_STAGE": failure,
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "CLI_FAST_COMPILE": "ON",
    }


def _run(
    step: str, tmp_path: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    script = _script(step).replace(
        "python3 tools/report_ci_ccache.py",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tools/report_ci_ccache.py'))}",
    )
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_stage_reports_precede_next_reset_and_keep_counters_separate(
    tmp_path: Path,
) -> None:
    steps = _steps()
    names = list(steps)
    assert names.index(SEED) < names.index(SEED_REPORT) < names.index(CLI)
    env = _tools(tmp_path)
    for name in (SEED, SEED_REPORT, CLI, "Summarize compiler cache"):
        result = _run(steps[name], tmp_path, env)
        assert result.returncode == 0, result.stdout + result.stderr
    summary = (tmp_path / "summary").read_text()
    seed, cli = summary.split("### Compiler cache: CUDA native CLI monolithic build")
    assert "fast seed (shared library)" in seed and "| Cache misses | 7 |" in seed
    assert "| Cache misses | 3 |" in cli and "| Cache misses | 7 |" not in cli
    records = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    resets = [
        i for i, record in enumerate(records) if record == ["ccache", ["--zero-stats"]]
    ]
    assert len(resets) == 2
    assert ["ccache", ["--show-stats", "--verbose"]] in records[resets[0] : resets[1]]
    assert ["ccache", ["--show-stats"]] in records[resets[1] :]


@pytest.mark.parametrize("name", [SEED, CLI])
@pytest.mark.parametrize("failure", ["reset", "configure", "build"])
def test_build_failure_is_not_masked_by_reporting(
    tmp_path: Path, name: str, failure: str
) -> None:
    steps = _steps()
    env = _tools(tmp_path, failure)
    result = _run(steps[name], tmp_path, env)
    assert result.returncode == 23
    output = tmp_path / "outputs"
    assert output.exists() == (failure != "reset")
    assert "continue-on-error:" not in steps[name]
    if failure != "reset":
        report = SEED_REPORT if name == SEED else "Summarize compiler cache"
        diagnostic = _run(steps[report], tmp_path, env)
        assert diagnostic.returncode == 0
        assert "cache_stats_reset=true" in output.read_text()
        assert result.returncode == 23


@pytest.mark.parametrize(
    "name",
    [
        SEED_REPORT,
        "Show detailed CLI compiler cache statistics",
        "Summarize compiler cache",
    ],
)
@pytest.mark.parametrize("tier", ["skip", "fast", "release"])
@pytest.mark.parametrize("reset", ["", "false", "true"])
def test_diagnostics_require_their_own_successful_reset(
    name: str, tier: str, reset: str
) -> None:
    step = _steps()[name]
    expression = step.split("        if: ", 1)[1].splitlines()[0]
    expression = expression.replace("always()", "true")
    expression = expression.replace("steps.cli_scope.outputs.tier", '"$TIER"')
    owner = "fast_cuda_seed" if name == SEED_REPORT else "native_cli_build"
    expression = expression.replace(
        f"steps.{owner}.outputs.cache_stats_reset", '"$RESET"'
    )
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={**os.environ, "TIER": tier, "RESET": reset},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == ("run" if tier != "skip" and reset == "true" else "skip")
    assert "continue-on-error: true" in step


def test_failed_statistics_are_reported_unavailable_not_as_hits(tmp_path: Path) -> None:
    env = _tools(tmp_path, "report")
    result = _run(_steps()[SEED_REPORT], tmp_path, env)
    assert result.returncode == 0
    assert "statistics unavailable" in result.stdout
    assert "| Cache hits |" not in result.stdout
    assert "Detailed reason" not in result.stdout


def test_unwritable_summary_does_not_hide_log_statistics(tmp_path: Path) -> None:
    env = _tools(tmp_path)
    env["GITHUB_STEP_SUMMARY"] = str(tmp_path)
    result = _run(_steps()[SEED_REPORT], tmp_path, env)
    assert result.returncode == 0
    assert "Unable to append compiler-cache summary" in result.stdout
    assert "| Cache misses | 99 |" in result.stdout
    assert "Detailed reason: fixture" in result.stdout
