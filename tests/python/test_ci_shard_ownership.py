"""Exercise the real CI shell selection without building or running workloads."""

import shutil
import subprocess
import textwrap
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ("core-a", "core-b", "runtime-heavy", "posthf", "ecp-forces", "compiler-heavy")


def _selection(event: str, shard: str) -> tuple[str, list[str], list[str]]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("CI selection uses Bash")
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    step = source.split("- name: Run Python tests with coverage", 1)[1]
    body = step.split("        run: |\n", 1)[1]
    body = body.split("          .venv/bin/python -m pytest", 1)[0]
    script = textwrap.dedent(body)
    script = script.replace("${{ github.event_name }}", event)
    script = script.replace("${{ matrix.shard }}", shard)
    script += '\nprintf "%s\\0" "$dist_mode" "${test_targets[@]}" --extra-- "${extra_args[@]}"\n'
    result = subprocess.run(
        [bash, "-eu", "-c", script], check=True, capture_output=True, cwd=ROOT
    )
    values = result.stdout.decode().split("\0")[:-1]
    boundary = values.index("--extra--")
    return values[0], values[1:boundary], values[boundary + 1 :]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
def test_every_python_test_file_has_exactly_one_ci_shard(event: str) -> None:
    selections = {shard: _selection(event, shard) for shard in SHARDS}
    expected = Counter(
        test.relative_to(ROOT).as_posix()
        for test in (ROOT / "tests/python").rglob("test_*.py")
    )
    selected = Counter(
        target
        for _, targets, arguments in selections.values()
        for target in targets
        if f"--ignore={target}" not in arguments
    )
    # Also reject duplicate entries within one shard and nonexistent paths.
    assert selected == expected


@pytest.mark.parametrize("event", ("pull_request", "merge_group", "schedule"))
def test_native_codegen_tail_keeps_expensive_module_fixtures_local(event: str) -> None:
    distribution, targets, _ = _selection(event, "compiler-heavy")
    assert distribution == "loadfile"
    for filename in (
        "test_native_contraction_binding.py",
        "test_rccsd_arena_liveness.py",
        "test_cc_cuda_scalar_reduction_codegen.py",
        "test_rccsd_response_admission.py",
        "test_rccsd_iteration_reuse.py",
        "test_stationary_cuda_lowering.py",
        "test_rccsdt_native_triples_response_owner.py",
        "test_becke_normalize_cooperative.py",
        "test_rccsd_numeric_capacity.py",
        "test_stationary_aot_generation.py",
        "test_cc_shared_iteration_driver.py",
        "test_rccsd_codegen_no_numpy.py",
    ):
        assert f"tests/python/{filename}" in targets


@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
@pytest.mark.parametrize("shard", SHARDS)
def test_full_qualification_is_not_deselected(event: str, shard: str) -> None:
    distribution, _, arguments = _selection(event, shard)
    assert "--deselect" not in arguments
    assert distribution == (
        "worksteal" if shard in {"core-a", "core-b"} else "loadfile"
    )


def test_python_ci_preserves_per_test_timings_with_debug_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '--junitxml=".artifacts/pytest-${{ matrix.shard }}.xml"' in source
    assert "path: .artifacts/" in source
