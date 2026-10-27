"""Exercise the real CI shell selection without building or running workloads."""

import ast
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
    "uccsd-forces",
    "runtime-heavy",
    "posthf",
    "ecp-forces",
    "compiler-heavy",
)


def _selection(
    event: str,
    shard: str,
    *,
    root: Path = ROOT,
    environment: dict[str, str] | None = None,
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
            **(environment or {}),
        },
    )
    values = result.stdout.decode().split("\0")[:-1]
    boundary = values.index("--extra--")
    return values[0], values[1:boundary], values[boundary + 1 :]


def test_core_shard_selection_ignores_host_locale(tmp_path: Path) -> None:
    """Exercise non-C collation even on CI hosts lacking installed locales."""
    system_sort = shutil.which("sort")
    if system_sort is None:
        pytest.skip("CI selection uses sort")
    probe = tmp_path / "sort"
    probe.write_text(
        "#!/bin/sh\n"
        'if [ "$LC_ALL" != C ]; then\n'
        f'  exec {shlex.quote(system_sort)} -r "$@"\n'
        "fi\n"
        f'exec {shlex.quote(system_sort)} "$@"\n',
        encoding="utf-8",
    )
    probe.chmod(0o755)
    shards = ("core-a", "core-b", "core-c")
    reference = {
        shard: _selection(
            "pull_request", shard, environment={"LC_ALL": "C", "LANG": "C"}
        )
        for shard in shards
    }
    non_c = {
        shard: _selection(
            "pull_request",
            shard,
            environment={
                "LC_ALL": "en_US.UTF-8",
                "LANG": "en_US.UTF-8",
                "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
            },
        )
        for shard in shards
    }
    assert non_c == reference


def test_tests_never_replace_process_environ_mapping() -> None:
    for name in ("test_issue206_df_matrix.py", "test_bulk_aot_cuda_closure.py"):
        path = ROOT / "tests/python" / name
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders = [
            node.lineno
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
            and (
                getattr(node.func, "id", None) == "setattr"
                or getattr(node.func, "attr", None) == "setattr"
            )
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "environ"
        ]
        assert not offenders, f"{name} replaces os.environ at lines {offenders}"


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
        "worksteal"
        if shard in {"core-a", "core-b", "core-c", "uccsd-forces"}
        else "loadfile"
    )


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
@pytest.mark.parametrize(
    ("module", "owner"),
    (
        ("tests/python/test_df_shell_units.py", "ecp-forces"),
        ("tests/python/test_treutler_endpoints.py", "posthf"),
    ),
)
def test_measured_module_keeps_one_loadfile_owner_and_original_core_slots(
    event: str, presence: str, module: str, owner: str, tmp_path: Path
) -> None:
    ordinary = [
        "tests/python/test_aaa.py",
        "tests/python/test_df_aaa.py",
        "tests/python/test_df_zzz.py",
        *[f"tests/python/test_probe_{index}.py" for index in range(3)],
        "tests/python/test_treutler_aaa.py",
        "tests/python/test_treutler_zzz.py",
        "tests/python/test_zzz.py",
    ]
    tracked = presence in {"tracked", "missing"}
    inventory = sorted([*ordinary, *([module] if tracked else [])])
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for relative in inventory:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(["git", "add", "tests"], check=True, cwd=tmp_path)
    if presence == "untracked":
        (tmp_path / module).touch()
    elif presence == "missing":
        (tmp_path / module).unlink()
    selections = {shard: _selection(event, shard, root=tmp_path) for shard in SHARDS}
    owners = [
        shard for shard, (_, targets, _) in selections.items() if module in targets
    ]
    assert owners == ([owner] if tracked else [])
    assert selections[owner][0] == "loadfile"
    for bucket, shard in enumerate(("core-a", "core-b", "core-c")):
        assert selections[shard][1] == [
            path for path in inventory[bucket::3] if path != module
        ]
    assert not any(
        module in argument
        for _, _, arguments in selections.values()
        for argument in arguments
    )


@pytest.mark.parametrize("event", ("pull_request", "schedule"))
@pytest.mark.parametrize(
    ("module", "owner"),
    (
        ("tests/python/test_df_shell_units.py", "ecp-forces"),
        ("tests/python/test_treutler_endpoints.py", "posthf"),
    ),
)
def test_measured_module_executes_once_with_one_shared_fixture_owner(
    event: str, module: str, owner: str, tmp_path: Path
) -> None:
    """Qualify the selected scheduler with fast competing files and a slow fixture."""
    distribution, targets, arguments = _selection(event, owner)
    assert module in targets
    assert not any(module in argument for argument in arguments)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (tmp_path / module).write_text(
        "import json, os, time\nfrom pathlib import Path\nimport pytest\n"
        "@pytest.fixture(scope='module')\n"
        "def shared():\n"
        "    with Path('setups.jsonl').open('a') as output:\n"
        "        output.write(json.dumps(os.environ['PYTEST_XDIST_WORKER']) + '\\n')\n"
        "    time.sleep(0.15)\n    return 7\n"
        "@pytest.mark.parametrize('value', range(10))\n"
        "def test_complete(shared, value):\n"
        "    assert shared == 7\n"
        "    with Path('cases.jsonl').open('a') as output:\n"
        "        output.write(json.dumps([os.environ['PYTEST_XDIST_WORKER'], value]) + '\\n')\n",
        encoding="utf-8",
    )
    for index in range(7):
        (directory / f"test_other_{index}.py").write_text(
            "import pytest\n@pytest.mark.parametrize('value', range(12))\n"
            "def test_other(value):\n    assert value >= 0\n",
            encoding="utf-8",
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-n",
            "4",
            f"--dist={distribution}",
            "-q",
            "tests/python",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    setups = [
        json.loads(line)
        for line in (tmp_path / "setups.jsonl").read_text().splitlines()
    ]
    cases = [
        json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()
    ]
    assert len(setups) == 1
    assert {worker for worker, _ in cases} == set(setups)
    assert Counter(value for _, value in cases) == Counter(range(10))
    assert "94 passed" in result.stdout


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
            "test_uccsd_triples_fock_response.py",
        ),
        "uccsd-forces": (
            "test_wb97mv_hessian.py",
            "test_uccsd_gradient.py",
            "test_uccsd_t_gradient.py",
            "test_uccsd_t_lambda.py",
        ),
        "posthf": ("test_ccsdt_cbs_native.py",),
        "runtime-heavy": (
            "test_bulk_aot_clangxx_closure.py",
            "test_cpu_hf_high_l.py",
            "test_cuda_xc_point_contract.py",
            "test_uccsd_solver.py",
            "test_wb97mv_ecp_hessian.py",
        ),
        "core-a": (
            "test_cc_triples_tiles.py",
            "test_df_lambda_core_reuse.py",
        ),
        "core-b": (
            "test_correlation_convergence.py",
            "test_cpu_semilocal_force_contract.py",
            "test_wb97mv_uks_hessian.py",
            "test_uccsd_response.py",
            "test_df_triples_distinct_moments.py",
            "test_df_lambda_matrix_audit.py",
            "test_df_lambda_mixed_precision.py",
            "test_df_occupied_triples.py",
        ),
        "core-c": (
            "test_direct_exchange_queue.py",
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


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
@pytest.mark.parametrize(
    "insertions",
    (
        (),
        ("test_000.py", "test_lambda_shift.py"),
        ("test_000.py", "test_array_api_take_default_axis.py", "test_lambda_shift.py"),
    ),
)
def test_post_force_split_hotspots_keep_stable_owners_and_ordinary_slots(
    event: str, presence: str, insertions: tuple[str, ...], tmp_path: Path
) -> None:
    owners = {
        "tests/python/test_bulk_aot_clangxx_closure.py": "runtime-heavy",
        "tests/python/test_range_separated_df.py": "core-c",
        "tests/python/test_ccsdt_cbs_native.py": "posthf",
        "tests/python/test_cpu_hf_high_l.py": "runtime-heavy",
        "tests/python/test_cuda_xc_point_contract.py": "runtime-heavy",
        "tests/python/test_uccsd_solver.py": "runtime-heavy",
        "tests/python/test_wb97mv_ecp_hessian.py": "runtime-heavy",
        "tests/python/test_cc_triples_tiles.py": "core-a",
        "tests/python/test_df_lambda_mixed_precision.py": "core-b",
        "tests/python/test_df_lambda_core_reuse.py": "core-a",
        "tests/python/test_correlation_convergence.py": "core-b",
        "tests/python/test_cpu_semilocal_force_contract.py": "core-b",
        "tests/python/test_direct_exchange_queue.py": "core-c",
        "tests/python/test_df_occupied_triples.py": "core-b",
    }
    probes = [
        f"tests/python/{name}"
        for name in (
            "test_aaa.py",
            "test_bulk_aot_clangxx_closure_next.py",
            "test_range_separated_df_next.py",
            "test_cc_triples_tiles_next.py",
            "test_ccsdt_cbs_aaa.py",
            "test_ccsdt_cbs_zzz.py",
            "test_cpu_hf_high_l_next.py",
            "test_cuda_xc_point_contract_next.py",
            "test_correlation_convergence_next.py",
            "test_cpu_semilocal_force_contract_next.py",
            "test_df_lambda_core_reuse_next.py",
            "test_df_lambda_mixed_precision_next.py",
            "test_df_occupied_triples_next.py",
            "test_direct_exchange_queue_next.py",
            "test_uccsd_solver_next.py",
            "test_wb97mv_ecp_hessian_next.py",
            "test_zzz.py",
            *insertions,
        )
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
            # Missing tracked tests stay selected so pytest can fail closed.
            (tmp_path / relative).unlink()
    selections = {shard: _selection(event, shard, root=tmp_path) for shard in SHARDS}
    assert selections["runtime-heavy"][0] == "loadfile"
    assert selections["posthf"][0] == "loadfile"
    for path, owner in owners.items():
        actual = [
            shard for shard, (_, targets, _) in selections.items() if path in targets
        ]
        assert actual == ([owner] if tracked else [])
        assert sum(targets.count(path) for _, targets, _ in selections.values()) == (
            1 if tracked else 0
        )
        assert not any(
            path in argument
            for _, _, arguments in selections.values()
            for argument in arguments
        )
    for bucket, shard in enumerate(("core-a", "core-b", "core-c")):
        distribution, targets, _ = selections[shard]
        assert distribution == "worksteal"
        assert targets == [
            path
            for index, path in enumerate(inventory)
            if owners.get(path) == shard or (path not in owners and index % 3 == bucket)
        ]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
def test_complete_forces_and_rks_hessian_share_prioritized_worksteal_owner(
    event: str,
) -> None:
    distribution, targets, arguments = _selection(event, "uccsd-forces")
    expected = [
        "tests/python/test_wb97mv_hessian.py",
        "tests/python/test_uccsd_gradient.py",
        "tests/python/test_uccsd_t_gradient.py",
        "tests/python/test_uccsd_t_lambda.py",
    ]
    assert distribution == "worksteal"
    assert targets == expected
    assert "-p" in arguments
    assert "tools.pytest_ci_fresh_frontier" in arguments
    assert [
        item.split("=", 1)[1]
        for item in arguments
        if item.startswith("--ci-fresh-priority=")
    ] == expected
    # Neither the ordinary PR exclusions nor nightly qualification removes
    # any complete molecular oracle or alters existing failure checks.
    assert not any(
        any(path in arg for path in expected) and arg.startswith("--deselect")
        for arg in arguments
    )


def test_python_ci_preserves_per_test_timings_with_debug_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '--junitxml=".artifacts/pytest-${{ matrix.shard }}.xml"' in source
    assert "path: .artifacts/" in source
    assert "source .github/scripts/ci_select_python_shard.sh" in source
    assert 'pytest "${collection_targets[@]}" -v' in source
    assert "source .github/scripts/ci_collect_python_shard.sh" in source
    assert '"${collection_args[@]}"' in source
    _assert_timed_diagnostics_contract(source)


def _assert_timed_diagnostics_contract(source: str) -> None:
    step = source.split("      - name: Run Python tests with coverage\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    command = step.split('.venv/bin/python -m pytest "${collection_targets[@]}"', 1)[1]
    command = command.replace("\\\n", " ").splitlines()[0]
    arguments = shlex.split(command)
    assert arguments.count("-o") == 1 and arguments.count("-p") == 1
    assert arguments[arguments.index("-o") + 1] == "faulthandler_timeout=0"
    assert arguments[arguments.index("-p") + 1] == "tools.pytest_safe_diagnostics"
    assert "--diagnostic-timeout=900" in arguments
    assert "--diagnostics-dir=$PWD/.artifacts/python-diagnostics" in arguments
    assert "${diagnostic_args[@]}" in arguments
    assert "-p tools.pytest_native_diagnostics" in step
    assert (
        '--native-diagnostics-library="$RUNNER_TEMP/libnative-fault-record.so"' in step
    )
    preparation = source.split("      - name: Prepare bounded native diagnostics\n", 1)[
        1
    ].split("\n      - name:", 1)[0]
    assert (
        'tools/native_fault_record.c -o "$RUNNER_TEMP/native-fault-record.o"'
        in preparation
    )
    assert '-o "$RUNNER_TEMP/libnative-fault-record.so"' in preparation
    assert "&& 90 || 30" in step
    assert "&& 100 || 40" in source


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("-p tools.pytest_safe_diagnostics", ""),
        ("-p tools.pytest_safe_diagnostics", "-p no:tools.pytest_safe_diagnostics"),
        ("--diagnostic-timeout=900", "--diagnostic-timeout=0"),
        ("--diagnostic-timeout=900", "--diagnostic-timeout=901"),
        ("-o faulthandler_timeout=0", "-o faulthandler_timeout=900"),
        ('"${diagnostic_args[@]}"', '""'),
        ("-p tools.pytest_native_diagnostics", "-p no:tools.pytest_native_diagnostics"),
        ("tools/native_fault_record.c", "tools/not-the-fatal-recorder.c"),
        ("&& 90 || 30", "&& 90 || 31"),
        ("&& 100 || 40", "&& 100 || 41"),
    ],
)
def test_timed_diagnostics_contract_rejects_disabled_or_changed_guards(
    before: str, after: str
) -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert before in source
    with pytest.raises(AssertionError):
        _assert_timed_diagnostics_contract(source.replace(before, after))


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
def test_uccsd_owner_tail_keeps_complete_coverage_and_core_parity(
    event: str, presence: str, tmp_path: Path
) -> None:
    parallel = {
        "tests/python/test_uccsd_response.py": "core-b",
        "tests/python/test_uccsd_t_lambda.py": "uccsd-forces",
    }
    owners = {
        **parallel,
        "tests/python/test_cpu_hf_high_l.py": "runtime-heavy",
        "tests/python/test_uccsd_gradient.py": "uccsd-forces",
        "tests/python/test_uccsd_t_gradient.py": "uccsd-forces",
        "tests/python/test_cc_native_tensor_cpu.py": "compiler-heavy",
        "tests/python/test_uccsd_native.py": "compiler-heavy",
        "tests/python/test_uccsd_triples_fock_response.py": "compiler-heavy",
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
        "tests/python/test_uccsd_response_next.py",
        "tests/python/test_uccsd_t_gradient_next.py",
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
            "worksteal"
            if owner.startswith("core-") or owner == "uccsd-forces"
            else "loadfile"
        )
        assert not any(
            path in argument
            for _, _, arguments in selections.values()
            for argument in arguments
            if not argument.startswith("--ci-fresh-priority=")
        )
        priority = {
            "tests/python/test_uccsd_gradient.py": "uccsd-forces",
            "tests/python/test_uccsd_t_gradient.py": "uccsd-forces",
            "tests/python/test_uccsd_t_lambda.py": "uccsd-forces",
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


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("bucket", range(3))
@pytest.mark.parametrize("presence", ("absent", "untracked", "tracked", "missing"))
@pytest.mark.parametrize("family", ("rks", "uks", "lambda", "all"))
def test_fresh_frontier_files_keep_distinct_owners_and_ordinary_slots(
    event: str, bucket: int, presence: str, family: str, tmp_path: Path
) -> None:
    """Inventory insertions cannot collide fresh owners or shift ordinary slots."""
    frontiers = {
        "rks": ("tests/python/test_wb97mv_hessian.py", "uccsd-forces"),
        "uks": ("tests/python/test_wb97mv_uks_hessian.py", "core-b"),
        "lambda": ("tests/python/test_uccsd_t_lambda.py", "uccsd-forces"),
        "order5": ("tests/python/test_direct_order5_sources.py", "core-c"),
    }
    owners = dict(value for name, value in frontiers.items() if family in {name, "all"})
    ordinary = [
        f"tests/python/test_early_{index:02d}.py" for index in range(6 + bucket)
    ]
    ordinary.extend(f"{path[:-3]}_next.py" for path in owners)
    ordinary.extend(f"tests/python/test_zlater_{index}.py" for index in range(6))
    tracked = presence in {"tracked", "missing"}
    inventory = sorted([*ordinary, *(owners if tracked else [])])
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
            # Missing tracked tests stay selected so pytest fails closed.
            (tmp_path / relative).unlink()
    selections = {shard: _selection(event, shard, root=tmp_path) for shard in SHARDS}
    selected = Counter(
        target
        for _, targets, _ in selections.values()
        for target in targets
        if target in inventory or target in owners
    )
    assert selected == Counter(inventory)
    for path, owner in owners.items():
        assert [
            shard for shard, (_, targets, _) in selections.items() if path in targets
        ] == ([owner] if tracked else [])
    for owner, shard in enumerate(("core-a", "core-b", "core-c")):
        distribution, targets, extra = selections[shard]
        assert distribution == "worksteal"
        assert targets == [
            path
            for index, path in enumerate(inventory)
            if owners.get(path) == shard or (path not in owners and index % 3 == owner)
        ]
        priorities = [
            argument.split("=", 1)[1]
            for argument in extra
            if argument.startswith("--ci-fresh-priority=")
        ]
        assert priorities == [
            path for path, expected in owners.items() if tracked and expected == shard
        ]
        assert ("tools.pytest_ci_fresh_frontier" in extra) == bool(priorities)
    _, force_targets, force_extra = selections["uccsd-forces"]
    force_priorities = [
        argument.split("=", 1)[1]
        for argument in force_extra
        if argument.startswith("--ci-fresh-priority=")
    ]
    assert force_priorities == [path for path in force_targets if path in owners]
    assert ("tools.pytest_ci_fresh_frontier" in force_extra) == bool(force_priorities)
    for _, _, extra in selections.values():
        assert all(
            not any(path in argument for path in owners)
            for argument in extra
            if not argument.startswith("--ci-fresh-priority=")
        )


@pytest.mark.parametrize("filtered", (False, True))
@pytest.mark.parametrize("mode", ("worksteal", "load", "serial"))
@pytest.mark.parametrize(
    "fresh_suites",
    (
        (("test_frontier_fresh.py", 8),),
        (
            ("test_uccsd_t_lambda.py", 17),
            ("test_wb97mv_hessian.py", 10),
            ("test_wb97mv_uks_hessian.py", 4),
        ),
    ),
)
def test_real_worksteal_seeds_every_worker_without_changing_selected_cases(
    filtered: bool,
    mode: str,
    fresh_suites: tuple[tuple[str, int], ...],
    tmp_path: Path,
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
    for name, count in fresh_suites:
        (tmp_path / name).write_text(
            "import pytest\n"
            f"@pytest.mark.parametrize('value', range({count}))\n"
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
        *(f"--ci-fresh-priority={name}" for name, _ in fresh_suites),
        "-q",
        str(normal),
        *(str(tmp_path / name) for name, _ in fresh_suites),
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
        prefixes = tuple(f"{name}::" for name, _ in fresh_suites)
        assert all(entries[0].startswith(prefixes) for entries in records)
    actual = Counter(node for entries in records for node in entries)
    expected = Counter(
        f"test_frontier_ordinary.py::test_ordinary[{value}]" for value in range(32)
    )
    if not filtered:
        expected.update(
            f"test_frontier_ordinary.py::test_drop[{value}]" for value in range(8)
        )
    expected.update(
        f"{name}::test_fresh[{value}]"
        for name, count in fresh_suites
        for value in range(count)
    )
    assert actual == expected
    if mode == "serial":
        assert records == [list(expected.elements())]


def _collection_arguments(
    shard: str, targets: list[str], root: Path
) -> subprocess.CompletedProcess[str]:
    """Exercise the separate adapter without changing the selector's outputs."""
    adapter = ROOT / ".github/scripts/ci_collect_python_shard.sh"
    script = 'test_targets=("$@")\n'
    script += f"source {shlex.quote(str(adapter))}\n"
    script += (
        'printf "%s\\0" "${collection_targets[@]}" --ignore-- "${collection_args[@]}"\n'
    )
    return subprocess.run(
        ["bash", "-eu", "-c", script, "collection", *targets],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "GENERATIVEQC_CI_SHARD": shard},
    )


def _split_collection(result: subprocess.CompletedProcess[str]) -> tuple[list, list]:
    assert result.returncode == 0, result.stdout + result.stderr
    parts = result.stdout.split("\0")[:-1]
    boundary = parts.index("--ignore--")
    return parts[:boundary], parts[boundary + 1 :]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("shard", SHARDS)
def test_collection_adapter_preserves_exact_shard_file_inventory(
    event: str, shard: str
) -> None:
    """No ownership or event-dependent science selection moves into the adapter."""
    _, selected, _ = _selection(event, shard)
    targets, arguments = _split_collection(_collection_arguments(shard, selected, ROOT))
    if shard.startswith("core-"):
        assert targets == ["tests/python"]
        ignored = {item.removeprefix("--ignore=") for item in arguments}
        discovered = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests/python").rglob("test_*.py")
        }
        assert discovered - ignored == set(selected)
        assert not (set(selected) & ignored)
    else:
        assert targets == selected
        assert arguments == []


@pytest.mark.parametrize("mode", ("priority", "duplicate", "nodeid", "nested"))
def test_collection_adapter_preserves_explicit_order_fallback(
    mode: str, tmp_path: Path
) -> None:
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    for name in ("test_a.py", "test_b.py"):
        (directory / name).write_text("def test_ok(): pass\n", encoding="utf-8")
    nested = directory / "test_nested"
    nested.mkdir()
    (nested / "test_x.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    targets = {
        "nested": ["tests/python/test_nested/test_x.py"],
        "priority": ["tests/python/test_b.py", "tests/python/test_a.py"],
        "duplicate": ["tests/python/test_a.py", "tests/python/test_a.py"],
        "nodeid": ["tests/python/test_a.py::test_ok"],
    }[mode]
    actual, arguments = _split_collection(
        _collection_arguments("core-b", targets, tmp_path)
    )
    assert actual == targets
    assert arguments == []


@pytest.mark.parametrize("targets", ([], ["tests/python/test_missing.py"]))
def test_collection_adapter_fails_closed_for_empty_or_missing_selection(
    targets: list[str], tmp_path: Path
) -> None:
    (tmp_path / "tests/python").mkdir(parents=True)
    result = _collection_arguments("core-b", targets, tmp_path)
    assert result.returncode != 0
    assert "::error::" in result.stderr


@pytest.mark.parametrize("workers", (0, 4))
@pytest.mark.parametrize("fresh", (False, True))
def test_single_discovery_preserves_nodes_fixtures_and_unselected_import_boundary(
    workers: int, fresh: bool, tmp_path: Path
) -> None:
    """Real pytest/xdist keeps the complete cases and never imports foreign files."""
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def pytest_collection_finish(session):\n"
        "    worker = os.environ.get('PYTEST_XDIST_WORKER', 'serial')\n"
        "    if session.items:\n"
        "        Path(worker + '.json').write_text(json.dumps([i.nodeid for i in session.items]))\n",
        encoding="utf-8",
    )
    for name in ("a", "z"):
        (directory / f"test_{name}.py").write_text(
            "import pytest\n"
            "@pytest.fixture(scope='module', params=[1, 2])\n"
            "def shared(request):\n    yield request.param\n"
            "@pytest.mark.parametrize('value', [3, 4])\n"
            "def test_value(shared, value):\n    assert shared + value in (4, 5, 6)\n"
            "@pytest.mark.skip(reason='preserve skip')\n"
            "def test_skip(shared):\n    raise AssertionError('skipped')\n",
            encoding="utf-8",
        )
    # Neither tracked nor untracked foreign tests may execute import side effects.
    (directory / "test_unselected.py").write_text(
        "raise AssertionError('foreign import')\n"
    )
    (directory / "untracked_test.py").write_text(
        "raise AssertionError('untracked import')\n"
    )
    foreign = directory / "foreign"
    foreign.mkdir()
    (foreign / "conftest.py").write_text("raise AssertionError('foreign conftest')\n")
    external = tmp_path / "external"
    external.mkdir()
    (external / "conftest.py").write_text("raise AssertionError('symlink conftest')\n")
    (directory / "linked").symlink_to(external, target_is_directory=True)
    selected = ["tests/python/test_a.py", "tests/python/test_z.py"]
    targets, arguments = _split_collection(
        _collection_arguments("core-b", selected, tmp_path)
    )
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "xdist.plugin",
        "-n",
        str(workers),
        "--dist=worksteal",
    ]
    if fresh:
        command.extend(
            (
                "-p",
                "tools.pytest_ci_fresh_frontier",
                "--ci-fresh-priority=tests/python/test_z.py",
            )
        )
    baseline = subprocess.run(
        command + selected,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    expected = {path.name: path.read_text() for path in tmp_path.glob("*.json")}
    for path in tmp_path.glob("*.json"):
        path.unlink()
    adapted = subprocess.run(
        command + targets + arguments,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert adapted.returncode == 0, adapted.stdout + adapted.stderr
    actual = {path.name: path.read_text() for path in tmp_path.glob("*.json")}
    assert actual == expected
    assert len(actual) == (workers or 1)
    assert "8 passed, 4 skipped" in baseline.stdout
    assert "8 passed, 4 skipped" in adapted.stdout
    # An ordinary developer invocation is unchanged and still sees foreign files.
    ordinary = subprocess.run(
        command + ["tests/python"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert ordinary.returncode != 0
    assert "foreign import" in ordinary.stdout
    assert "untracked import" in ordinary.stdout
    assert "foreign conftest" in ordinary.stdout
    assert "symlink conftest" in ordinary.stdout


def test_collection_adapter_rejects_partial_file_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (directory / "test_a.py").write_text("def test_ok(): pass\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    find = binary / "find"
    find.write_text("#!/bin/sh\nprintf 'tests/python/test_a.py\\0'\nexit 17\n")
    find.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    result = _collection_arguments("core-b", ["tests/python/test_a.py"], tmp_path)
    assert result.returncode != 0
    assert "could not enumerate Python collection exclusions" in result.stderr
    assert not result.stdout


def test_collection_adapter_is_repeatable_without_broadening_selection(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (directory / "test_a.py").write_text("def test_ok(): pass\n")
    adapter = shlex.quote(str(ROOT / ".github/scripts/ci_collect_python_shard.sh"))
    script = (
        "test_targets=(tests/python/test_a.py)\n"
        f"source {adapter}\n"
        "printf 'raise AssertionError(42)\\n' > tests/python/test_late.py\n"
        f"source {adapter}\n"
        '[[ "${test_targets[*]}" == tests/python/test_a.py ]]\n'
        'printf "%s\\0" "${collection_targets[@]}" --ignore-- "${collection_args[@]}"\n'
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "GENERATIVEQC_CI_SHARD": "core-b"},
        check=False,
        capture_output=True,
        text=True,
    )
    targets, arguments = _split_collection(result)
    assert targets == ["tests/python"]
    assert "--ignore=tests/python/test_late.py" in arguments
    assert "--ignore=tests/python/test_a.py" not in arguments


def _prioritized_posthf_targets(targets: list[str], root: Path) -> list[str]:
    helper = ROOT / ".github/scripts/ci_prioritize_posthf_files.sh"
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            (
                'test_targets=("$@")\n'
                f"source {shlex.quote(str(helper))}\n"
                'for target in "${test_targets[@]}"; do printf "%s\\0" "$target"; done\n'
            ),
            "priority",
            *targets,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split("\0")[:-1]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
@pytest.mark.parametrize("shard", SHARDS)
def test_measured_file_priority_only_changes_routine_posthf_scheduling(
    event: str, shard: str
) -> None:
    distribution, selected, extra = _selection(event, shard)
    prioritized = shard == "posthf" and event in {"pull_request", "push", "merge_group"}
    assert ("--no-loadscope-reorder" in extra) == prioritized
    if prioritized:
        assert distribution == "loadfile"
        assert selected[:5] == [
            "tests/python/test_treutler_endpoints.py",
            "tests/python/test_uccsd_lambda.py",
            "tests/python/test_uccsd_t_owner.py",
            "tests/python/test_uccsd_physical_response.py",
            "tests/python/test_dft_complete_cpu.py",
        ]
        complete_cc = selected.index("tests/python/test_cc_complete_gradient.py")
        assert selected[complete_cc : complete_cc + 3] == [
            "tests/python/test_cc_complete_gradient.py",
            "tests/python/test_ccsdt_cbs_native.py",
            "tests/python/test_ecp_heavy.py",
        ]
        # The shared collection adapter must not undo the measured file order.
        targets, arguments = _split_collection(
            _collection_arguments(shard, selected, ROOT)
        )
        assert targets == selected
        assert arguments == []


@pytest.mark.parametrize(
    "targets",
    (
        [],
        ["tests/python/test_future.py", "tests/python/test_missing.py"],
        [
            "tests/python/test_future.py",
            "tests/python/test_uccsd_physical_response.py",
            "tests/python/test_other.py::test_case[space in id]",
            "tests/python/test_uccsd_lambda.py",
            "tests/python/test_future.py",
            "tests/python/test_uccsd_lambda.py",
        ],
    ),
)
def test_posthf_order_preserves_unknown_missing_and_duplicate_targets(
    targets: list[str], tmp_path: Path
) -> None:
    actual = _prioritized_posthf_targets(targets, tmp_path)
    assert Counter(actual) == Counter(targets)
    assert _prioritized_posthf_targets(actual, tmp_path) == actual
    unknown = [path for path in targets if "test_uccsd_" not in path]
    assert [path for path in actual if "test_uccsd_" not in path] == unknown
    if "tests/python/test_uccsd_lambda.py" in targets:
        assert actual[:2] == ["tests/python/test_uccsd_lambda.py"] * 2


@pytest.mark.parametrize("parent_worker", (None, "gw0"))
def test_real_loadfile_prioritizes_whole_files_without_changing_nodes_or_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_worker: str | None
) -> None:
    """Exercise the installed xdist scheduler, not a model of its behavior."""
    # A nested controller inherits its parent's environment but is not a worker.
    # Exercise this even when the outer regression suite runs without xdist.
    if parent_worker is None:
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    else:
        monkeypatch.setenv("PYTEST_XDIST_WORKER", parent_worker)
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "worker = None\n"
        "def pytest_configure(config):\n"
        "    global worker\n"
        "    worker = getattr(config, 'workerinput', {}).get('workerid')\n"
        "def pytest_runtest_logreport(report):\n"
        "    if worker and (report.when == 'call' or (report.when == 'setup' and report.skipped)):\n"
        "        with Path(worker + '.jsonl').open('a') as stream:\n"
        "            stream.write(json.dumps([report.nodeid, report.outcome]) + '\\n')\n",
        encoding="utf-8",
    )
    priority = [
        "test_treutler_endpoints.py",
        "test_uccsd_lambda.py",
        "test_uccsd_t_owner.py",
        "test_uccsd_physical_response.py",
    ]
    ordinary = [f"test_future_{index}.py" for index in range(4)]
    for name in [*priority, *ordinary]:
        count = 3 if name in priority else 8
        (directory / name).write_text(
            "import os, pytest\nfrom pathlib import Path\n"
            "@pytest.fixture(scope='module')\n"
            "def shared():\n"
            "    with Path('fixtures-' + os.environ['PYTEST_XDIST_WORKER'] + '.txt').open('a') as stream:\n"
            "        stream.write(__file__ + '\\n')\n"
            "    return 42\n"
            f"@pytest.mark.parametrize('value', range({count}))\n"
            "def test_value(shared, value):\n    assert shared == 42\n"
            "@pytest.mark.skip(reason='preserve skip')\n"
            "def test_skip(shared):\n    raise AssertionError('skipped')\n",
            encoding="utf-8",
        )
    selected = [f"tests/python/{name}" for name in [*ordinary, *reversed(priority)]]
    ordered = _prioritized_posthf_targets(selected, tmp_path)
    adapted, collection_args = _split_collection(
        _collection_arguments("posthf", ordered, tmp_path)
    )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "xdist.plugin",
        "-n",
        "4",
        "--dist=loadfile",
    ]
    inventories = []
    first_files = []
    for targets, arguments in (
        (selected, []),
        (adapted, [*collection_args, "--no-loadscope-reorder"]),
    ):
        result = subprocess.run(
            command + targets + arguments,
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "44 passed, 8 skipped" in result.stdout
        records = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in sorted(tmp_path.glob("gw*.jsonl"))
        ]
        inventories.append(Counter(tuple(row) for rows in records for row in rows))
        assert sum(inventories[-1].values()) == 52
        assert (
            sum(
                count
                for (_, outcome), count in inventories[-1].items()
                if outcome == "skipped"
            )
            == 8
        )
        first_files.append(
            {Path(rows[0][0].split("::", 1)[0]).name for rows in records}
        )
        owners = {}
        for worker, rows in enumerate(records):
            for node, _ in rows:
                owners.setdefault(node.split("::", 1)[0], set()).add(worker)
        assert all(len(workers) == 1 for workers in owners.values())
        fixtures = [
            line
            for path in tmp_path.glob("fixtures-*.txt")
            for line in path.read_text().splitlines()
        ]
        assert Counter(Path(path).name for path in fixtures) == Counter(
            [*priority, *ordinary]
        )
        for path in [*tmp_path.glob("gw*.jsonl"), *tmp_path.glob("fixtures-*.txt")]:
            path.unlink()
    assert inventories[0] == inventories[1]
    assert first_files == [set(ordinary), set(priority)]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "merge_group", "schedule", "workflow_dispatch")
)
def test_real_force_lane_starts_intact_rks_hessian_before_force_queues(
    event: str, tmp_path: Path
) -> None:
    """Execute the selector's real order and plugin without scientific workloads."""
    names = (
        "test_wb97mv_hessian.py",
        "test_uccsd_gradient.py",
        "test_uccsd_t_gradient.py",
        "test_uccsd_t_lambda.py",
    )
    counts = (10, 24, 24, 17)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    directory = tmp_path / "tests/python"
    directory.mkdir(parents=True)
    (tmp_path / "conftest.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\n\n"
        "def pytest_runtest_setup(item):\n"
        "    worker = os.environ['PYTEST_XDIST_WORKER']\n"
        "    with Path('worker-' + worker + '.jsonl').open('a') as output:\n"
        "        output.write(json.dumps(item.nodeid) + '\\n')\n",
        encoding="utf-8",
    )
    for name, count in zip(names, counts, strict=True):
        (directory / name).write_text(
            "import json, os\nfrom pathlib import Path\nimport pytest\n"
            "@pytest.fixture\n"
            "def fresh(request):\n"
            "    with Path('fresh-setups.jsonl').open('a') as output:\n"
            "        output.write(json.dumps([request.node.nodeid, "
            "os.environ['PYTEST_XDIST_WORKER']]) + '\\n')\n"
            "    return object()\n"
            f"@pytest.mark.parametrize('value', range({count}))\n"
            "def test_case(value, fresh):\n    assert value >= 0\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "tests"], check=True, cwd=tmp_path)
    distribution, targets, extra = _selection(event, "uccsd-forces", root=tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-c",
            "pytest.ini",
            "-n",
            "4",
            f"--dist={distribution}",
            "-q",
            *targets,
            *extra,
        ],
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
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in tmp_path.glob("worker-*.jsonl")
    ]
    assert len(records) == 4
    prefix = "tests/python/test_wb97mv_hessian.py::"
    assert all(entries[0].startswith(prefix) for entries in records)
    # The real module's water HVP is the fifth collected case. It must start
    # after at most one cheap admission check, before any force case on its lane.
    hvp = prefix + "test_case[4]"
    owner = next(entries for entries in records if hvp in entries)
    assert owner.index(hvp) <= 1
    assert all(node.startswith(prefix) for node in owner[: owner.index(hvp)])
    expected = Counter(
        f"tests/python/{name}::test_case[{value}]"
        for name, count in zip(names, counts, strict=True)
        for value in range(count)
    )
    assert Counter(node for entries in records for node in entries) == expected
    setups = [
        json.loads(line)
        for line in (tmp_path / "fresh-setups.jsonl").read_text().splitlines()
    ]
    # Every selected case owns one fresh setup; no duplicate or shared fixture
    # is introduced by adding a complete module to the existing priority lane.
    assert Counter(node for node, _ in setups) == expected
