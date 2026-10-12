"""Exercise the real CI shell selection without building or running workloads."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARDS = (
    "core-a",
    "core-b",
    "core-c",
    "runtime-heavy",
    "posthf",
    "ecp-forces",
    "compiler-heavy",
)


def _selection(
    event: str, shard: str, *, root: Path = ROOT
) -> tuple[str, list[str], list[str]]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("CI selection uses Bash")
    selector = ROOT / ".github/scripts/ci_select_python_shard.sh"
    assert selector.is_file()
    script = f"source {shlex.quote(str(selector))}\n"
    script += 'printf "%s\\0" "$dist_mode" "${test_targets[@]}" --extra-- "${extra_args[@]}"\n'
    result = subprocess.run(
        [bash, "-eu", "-c", script],
        check=True,
        capture_output=True,
        cwd=root,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": event,
            "GENERATIVEQC_CI_SHARD": shard,
        },
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


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
@pytest.mark.parametrize(
    "filename", ("test_libxc_automatic_forces.py", "test_mixed_rsh_df_forces.py")
)
def test_complete_force_tail_uses_tracked_ownership_and_preserves_core_buckets(
    event: str, presence: str, filename: str, tmp_path: Path
) -> None:
    # Exercise both the earlier PR tree and a merge tree containing master's
    # force module without mutating the checkout or relying on its current base.
    force = f"tests/python/{filename}"
    probes = [
        "tests/python/test_aaa.py",
        "tests/python/test_bbb.py",
        "tests/python/test_libxc_aaa.py",
        "tests/python/test_libxc_zzz.py",
        "tests/python/test_mixed_rsh_df_aaa.py",
        "tests/python/test_mixed_rsh_df_zzz.py",
        "tests/python/test_zzz.py",
    ]
    tracked = presence in {"tracked", "missing"}
    inventory = sorted([*probes, *([force] if tracked else [])])
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for relative in inventory:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(["git", "add", "tests"], check=True, cwd=tmp_path)
    if presence == "untracked":
        (tmp_path / force).touch()
    elif presence == "missing":
        # A missing tracked file must remain selected so pytest fails closed.
        (tmp_path / force).unlink()
    selections = {shard: _selection(event, shard, root=tmp_path) for shard in SHARDS}
    owners = [
        shard for shard, (_, targets, _) in selections.items() if force in targets
    ]
    assert owners == (["ecp-forces"] if tracked else [])
    assert selections["ecp-forces"][0] == "loadfile"
    for bucket, shard in enumerate(("core-a", "core-b", "core-c")):
        assert selections[shard][1] == [
            path for path in inventory[bucket::3] if path != force
        ]
    for _, _, arguments in selections.values():
        assert not any(force in argument for argument in arguments)


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
        "test_dft_mp_v1_capacity.py",
        "test_df_lambda_frontier.py",
    ):
        assert f"tests/python/{filename}" in targets


@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
@pytest.mark.parametrize("shard", SHARDS)
def test_full_qualification_is_not_deselected(event: str, shard: str) -> None:
    distribution, _, arguments = _selection(event, shard)
    assert "--deselect" not in arguments
    assert distribution == (
        "worksteal" if shard in {"core-a", "core-b", "core-c"} else "loadfile"
    )


@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
def test_nightly_heavy_tests_move_without_duplicate_or_missing_owners(
    event: str,
) -> None:
    selections = {shard: _selection(event, shard)[1] for shard in SHARDS}
    wb97mv = "tests/python/test_wb97mv_complete.py"
    assert wb97mv in selections["compiler-heavy"]
    assert wb97mv not in selections["core-a"]
    assert wb97mv not in selections["core-b"]
    assert wb97mv not in selections["core-c"]
    for path in (
        "tests/python/test_hessian_analytic.py",
        "tests/python/test_hessian_directional.py",
        "tests/python/test_hessian_reference.py",
        "tests/python/test_hessian_hvp.py",
    ):
        assert path in selections["runtime-heavy"]
        assert path not in selections["posthf"]


@pytest.mark.parametrize("event", ("pull_request", "merge_group", "push"))
def test_routine_shards_keep_expensive_files_with_single_owners(event: str) -> None:
    selections = {shard: _selection(event, shard)[1] for shard in SHARDS}
    wb97mv = "tests/python/test_wb97mv_complete.py"
    assert sum(wb97mv in targets for targets in selections.values()) == 1
    assert wb97mv not in selections["compiler-heavy"]
    for path in (
        "tests/python/test_hessian_analytic.py",
        "tests/python/test_hessian_directional.py",
        "tests/python/test_hessian_reference.py",
        "tests/python/test_hessian_hvp.py",
    ):
        assert path in selections["posthf"]
        assert path not in selections["runtime-heavy"]


@pytest.mark.parametrize("shard", ("core-a", "core-b", "core-c"))
@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
def test_nightly_rebalance_preserves_unrelated_core_file_parity(
    shard: str, event: str
) -> None:
    _, routine, _ = _selection("pull_request", shard)
    _, complete, _ = _selection(event, shard)
    wb97mv = "tests/python/test_wb97mv_complete.py"
    assert complete == [path for path in routine if path != wb97mv]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
def test_measured_hotspots_preserve_dedicated_shard_precedence(event: str) -> None:
    selections = {shard: set(_selection(event, shard)[1]) for shard in SHARDS}
    expected = {
        # Landed dedicated transfers supersede historical core pins.
        "ecp-forces": ("test_mixed_rsh_df_forces.py",),
        "compiler-heavy": (
            "test_dft_mp_v1_capacity.py",
            "test_df_lambda_frontier.py",
            "test_cc_native_tensor_cpu.py",
            "test_uccsd_native.py",
        ),
        "core-a": (
            "test_uccsd_gradient.py",
            "test_cpu_hf_high_l.py",
        ),
        "core-b": (
            "test_uccsd_triples_fock_response.py",
            "test_df_lambda_core_reuse.py",
            "test_df_triples_distinct_moments.py",
            "test_df_lambda_matrix_audit.py",
        ),
        "core-c": (
            "test_uccsd_t_lambda.py",
            "test_cc_canonical_denominators.py",
            "test_direct_order5_sources.py",
        ),
    }
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "tests/python"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.splitlines()
    )
    for owner, filenames in expected.items():
        for filename in filenames:
            path = f"tests/python/{filename}"
            assert (path in selections[owner]) == (path in tracked)
            assert all(
                path not in targets
                for shard, targets in selections.items()
                if shard != owner
            )


def test_python_ci_preserves_per_test_timings_with_debug_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '--junitxml=".artifacts/pytest-${{ matrix.shard }}.xml"' in source
    assert "path: .artifacts/" in source
    assert "source .github/scripts/ci_select_python_shard.sh" in source
    assert 'pytest "${test_targets[@]}" -v' in source
    assert "-o faulthandler_timeout=900" in source


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
def test_uccsd_owner_tail_keeps_complete_coverage_and_core_parity(
    event: str, presence: str, tmp_path: Path
) -> None:
    parallel = {
        "tests/python/test_uccsd_gradient.py": "core-a",
        "tests/python/test_cpu_hf_high_l.py": "core-a",
        "tests/python/test_uccsd_t_lambda.py": "core-c",
        "tests/python/test_uccsd_triples_fock_response.py": "core-b",
    }
    owners = {
        **parallel,
        "tests/python/test_cc_native_tensor_cpu.py": "compiler-heavy",
        "tests/python/test_uccsd_native.py": "compiler-heavy",
        "tests/python/test_uccsd_lambda.py": "posthf",
        "tests/python/test_uccsd_t_owner.py": "posthf",
        "tests/python/test_uccsd_physical_response.py": "posthf",
        "tests/python/test_uccsd_triples_response.py": "runtime-heavy",
    }
    probes = [
        "tests/python/test_aaa.py",
        "tests/python/test_cc_native_tensor_cpu_next.py",
        "tests/python/test_cpu_hf_high_l_next.py",
        "tests/python/test_uccsd_aaa.py",
        "tests/python/test_uccsd_lambda_next.py",
        "tests/python/test_uccsd_native_next.py",
        "tests/python/test_uccsd_physical_response_next.py",
        "tests/python/test_uccsd_t_lambda_next.py",
        "tests/python/test_uccsd_t_owner_next.py",
        "tests/python/test_uccsd_triples_fock_response_next.py",
        "tests/python/test_uccsd_triples_response_next.py",
        "tests/python/test_zzz.py",
    ]
    tracked = presence in {"tracked", "missing"}
    inventory = sorted([*probes, *(owners if tracked else [])])
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for relative in inventory:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(["git", "add", "tests"], check=True, cwd=tmp_path)
    for relative in owners:
        if presence == "untracked":
            (tmp_path / relative).touch()
        elif presence == "missing":
            (tmp_path / relative).unlink()
    selections = {shard: _selection(event, shard, root=tmp_path) for shard in SHARDS}
    for path, owner in owners.items():
        actual = [
            shard for shard, (_, targets, _) in selections.items() if path in targets
        ]
        assert actual == ([owner] if tracked else [])
        assert selections[owner][0] == (
            "worksteal" if owner.startswith("core-") else "loadfile"
        )
        assert not any(
            path in argument
            for _, _, arguments in selections.values()
            for argument in arguments
            if not argument.startswith("--ci-fresh-priority=")
        )
        priority = {
            "tests/python/test_uccsd_gradient.py": "core-a",
            "tests/python/test_uccsd_t_lambda.py": "core-c",
        }
        for shard, (_, _, arguments) in selections.items():
            selected_priority = [
                argument.split("=", 1)[1]
                for argument in arguments
                if argument.startswith("--ci-fresh-priority=")
            ]
            assert selected_priority == [
                target
                for target, owner in priority.items()
                if tracked and owner == shard
            ]
    for bucket, shard in enumerate(("core-a", "core-b", "core-c")):
        assert selections[shard][1] == [
            path
            for index, path in enumerate(inventory)
            if parallel.get(path) == shard
            or (path not in owners and index % 3 == bucket)
        ]


@pytest.mark.parametrize("workers", (1, 2, 4, 8))
@pytest.mark.parametrize(
    "priority",
    (
        (),
        (False,),
        (True,),
        (False, True),
        (True, False, False, True, False, True),
        tuple(index % 3 == 0 for index in range(19)),
        (True,) * 37,
    ),
)
def test_fresh_frontiers_preserve_inventory_and_ordinary_order(
    workers: int, priority: tuple[bool, ...]
) -> None:
    """Ordering cannot hide cases or perturb ordinary relative collection order."""
    from tools.pytest_ci_fresh_frontier import _frontier_indices

    actual = _frontier_indices(priority, workers)
    assert sorted(actual) == list(range(len(priority)))
    ordinary = [index for index, selected in enumerate(priority) if not selected]
    assert [index for index in actual if not priority[index]] == ordinary
    if workers == 1 or not any(priority):
        assert actual == list(range(len(priority)))
        return
    active = min(workers, len(priority))
    remaining, cursor = len(priority), 0
    for worker in range(active):
        capacity = remaining // (active - worker)
        group = actual[cursor : cursor + capacity]
        if worker < min(sum(priority), active):
            assert priority[group[0]]
        cursor += capacity
        remaining -= capacity


@pytest.mark.parametrize("workers", (0, -1))
def test_fresh_frontiers_reject_invalid_worker_counts(workers: int) -> None:
    """An invalid distribution cannot silently masquerade as serial execution."""
    from tools.pytest_ci_fresh_frontier import _frontier_indices

    with pytest.raises(ValueError, match="positive worker count"):
        _frontier_indices((True, False), workers)


@pytest.mark.parametrize("filtered", (False, True))
@pytest.mark.parametrize("mode", ("worksteal", "load", "serial"))
def test_real_worksteal_seeds_every_worker_without_changing_selected_cases(
    filtered: bool, mode: str, tmp_path: Path
) -> None:
    """Observe parallel startup and unchanged serial/-k selection behavior."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\n\n"
        "def pytest_runtest_setup(item):\n"
        "    worker = os.environ.get('PYTEST_XDIST_WORKER', 'serial')\n"
        "    with Path('worker-' + worker + '.jsonl').open('a') as output:\n"
        "        output.write(json.dumps(item.nodeid) + '\\n')\n",
        encoding="utf-8",
    )
    normal = tmp_path / "test_frontier_ordinary.py"
    normal.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(32))\n"
        "def test_ordinary(value):\n    assert value >= 0\n"
        "@pytest.mark.parametrize('value', range(8))\n"
        "def test_drop(value):\n    assert value >= 0\n",
        encoding="utf-8",
    )
    fresh = tmp_path / "test_frontier_fresh.py"
    fresh.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(8))\n"
        "def test_fresh(value):\n    assert value >= 0\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "xdist.plugin",
        "-p",
        "tools.pytest_ci_fresh_frontier",
        "-c",
        "pytest.ini",
        "-n",
        "0" if mode == "serial" else "4",
        f"--dist={'no' if mode == 'serial' else mode}",
        "--ci-fresh-priority=test_frontier_fresh.py",
        "-q",
        str(normal),
        str(fresh),
    ]
    if filtered:
        command.extend(("-k", "not test_drop"))
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=90,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
            "PYTHONPATH": str(ROOT),
        },
    )
    if mode == "load":
        assert result.returncode != 0
        assert "CI fresh frontiers require --dist=worksteal" in result.stderr
        assert not list(tmp_path.glob("worker-*.jsonl"))
        return
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in tmp_path.glob("worker-*.jsonl")
    ]
    assert len(records) == (1 if mode == "serial" else 4)
    if mode == "worksteal":
        assert all(
            entries[0].startswith("test_frontier_fresh.py::") for entries in records
        )
    actual = Counter(node for entries in records for node in entries)
    expected = Counter(
        f"test_frontier_ordinary.py::test_ordinary[{value}]" for value in range(32)
    )
    if not filtered:
        expected.update(
            f"test_frontier_ordinary.py::test_drop[{value}]" for value in range(8)
        )
    expected.update(
        f"test_frontier_fresh.py::test_fresh[{value}]" for value in range(8)
    )
    assert actual == expected
    if mode == "serial":
        assert records == [list(expected.elements())]
