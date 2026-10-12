"""Exercise the real CI shell selection without building or running workloads."""

import os
import shlex
import shutil
import subprocess
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
        ),
        "core-a": ("test_uccsd_gradient.py",),
        "core-b": (
            "test_df_lambda_core_reuse.py",
            "test_df_triples_distinct_moments.py",
            "test_df_lambda_matrix_audit.py",
        ),
        "core-c": (
            "test_cc_native_tensor_cpu.py",
            "test_cpu_hf_high_l.py",
            "test_cc_canonical_denominators.py",
            "test_direct_order5_sources.py",
        ),
    }
    for owner, filenames in expected.items():
        for filename in filenames:
            path = f"tests/python/{filename}"
            assert path in selections[owner]
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
    gradient = "tests/python/test_uccsd_gradient.py"
    owners = {
        gradient: "core-a",
        "tests/python/test_uccsd_lambda.py": "posthf",
        "tests/python/test_uccsd_t_owner.py": "posthf",
        "tests/python/test_uccsd_physical_response.py": "posthf",
        "tests/python/test_uccsd_triples_response.py": "runtime-heavy",
    }
    probes = [
        "tests/python/test_aaa.py",
        "tests/python/test_uccsd_aaa.py",
        "tests/python/test_uccsd_lambda_next.py",
        "tests/python/test_uccsd_physical_response_next.py",
        "tests/python/test_uccsd_t_owner_next.py",
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
            "worksteal" if owner == "core-a" else "loadfile"
        )
        assert not any(
            path in argument
            for _, _, arguments in selections.values()
            for argument in arguments
        )
    for bucket, shard in enumerate(("core-a", "core-b", "core-c")):
        assert selections[shard][1] == [
            path
            for index, path in enumerate(inventory)
            if (path == gradient and bucket == 0)
            or (path not in owners and index % 3 == bucket)
        ]
