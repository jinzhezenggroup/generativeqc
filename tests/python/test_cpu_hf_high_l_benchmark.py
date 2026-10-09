"""Source-work receipts must count the same canonical domain as CPU ERIs."""

import builtins
import os
import shutil
import subprocess
import sys
from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import _retention
from benchmarks import cpu_hf_high_l as benchmark
from benchmarks.cpu_hf_high_l import quartet_census


@pytest.mark.parametrize(
    "angular", ((0,), (2,), (3,), (0, 1, 2, 3), (3, 2, 1, 0), (0, 4), (0, 3, 4))
)
def test_census_matches_independent_ao_pair_enumeration(
    angular: tuple[int, ...],
) -> None:
    """Include mixed/high-l fallbacks, contracted shells and reversed shell order."""
    shells = tuple(
        SimpleNamespace(angular_momentum=order, primitives=tuple(range(index % 2 + 1)))
        for index, order in enumerate(angular)
    )
    ao_shells = [
        index
        for index, order in enumerate(angular)
        for _ in range((order + 1) * (order + 2) // 2)
    ]
    pairs = tuple(combinations_with_replacement(range(len(ao_shells)), 2))
    expected = {
        "primitive_components": 0,
        "spd_primitive_components": 0,
        "f_primitive_components": 0,
        "fallback_primitive_components": 0,
    }
    for bra, ket in combinations_with_replacement(pairs, 2):
        selected = tuple(shells[ao_shells[ao]] for ao in (*bra, *ket))
        primitives = 1
        for shell in selected:
            primitives *= len(shell.primitives)
        expected["primitive_components"] += primitives
        role = (
            "spd_primitive_components"
            if all(shell.angular_momentum <= 2 for shell in selected)
            else "fallback_primitive_components"
        )
        expected[role] += primitives
        if max(shell.angular_momentum for shell in selected) == 3:
            expected["f_primitive_components"] += primitives
    actual = quartet_census(shells)
    assert {key: actual[key] for key in expected} == expected
    shell_pairs = len(shells) * (len(shells) + 1) // 2
    assert actual["shell_quartets"] == shell_pairs * (shell_pairs + 1) // 2
    expected_shared = {"spd_shared_geometries": 0, "f_shared_geometries": 0}
    pairs = tuple(combinations_with_replacement(range(len(shells)), 2))
    for bra, ket in combinations_with_replacement(pairs, 2):
        selected = tuple(shells[index] for index in (*bra, *ket))
        maximum = max(shell.angular_momentum for shell in selected)
        if maximum > 3:
            continue
        primitives = 1
        for shell in selected:
            primitives *= len(shell.primitives)
        role = "spd_shared_geometries" if maximum <= 2 else "f_shared_geometries"
        expected_shared[role] += primitives
    assert {key: actual[key] for key in expected_shared} == expected_shared
    if angular == (2,):
        assert actual["spd_shared_geometries"] == 1
        assert actual["spd_primitive_components"] == 231
    if angular == (3,):
        assert actual["f_shared_geometries"] == 1
        assert actual["f_primitive_components"] == 1540


def test_empty_census_has_no_work() -> None:
    assert all(value == 0 for value in quartet_census(()).values())


@pytest.mark.parametrize("kind", ("direct", "alias", "traversal"))
def test_retained_output_is_rejected_before_runtime_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    destination = retained / "existing.json"
    destination.write_text("reviewed evidence", encoding="utf-8")
    if kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(retained, target_is_directory=True)
        destination = alias / "existing.json"
    elif kind == "traversal":
        destination = tmp_path / "scratch/../benchmarks/results/existing.json"

    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("generativeqc", "pyscf"):
            pytest.fail("benchmark runtime imported before output rejection")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(destination)])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert (retained / "existing.json").read_text(
        encoding="utf-8"
    ) == "reviewed evidence"
    assert not (tmp_path / "scratch").exists()


def test_scratch_output_reaches_runtime_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    destination = tmp_path / ".artifacts/benchmarks/hf.json"
    original_import = builtins.__import__

    def stop_before_runtime(name: str, *args: object, **kwargs: object) -> object:
        if name == "generativeqc":
            raise RuntimeError("accepted output reached runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_before_runtime)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(destination)])
    with pytest.raises(RuntimeError, match="accepted output reached runtime"):
        benchmark.main()
    assert not destination.parent.exists()


def test_direct_script_rejects_retained_output(tmp_path: Path) -> None:
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for filename in ("cpu_hf_high_l.py", "_retention.py"):
        shutil.copyfile(
            Path(benchmark.__file__).with_name(filename), directory / filename
        )
    destination = directory / "results/existing.json"
    destination.parent.mkdir()
    destination.write_text("reviewed evidence", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(directory / "cpu_hf_high_l.py"),
            "--output",
            str(destination),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "raw_output_path" in completed.stderr
    assert destination.read_text(encoding="utf-8") == "reviewed evidence"
