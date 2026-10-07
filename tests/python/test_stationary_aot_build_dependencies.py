"""Post-link AOT identity dependencies cover every file hashed by its contract."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.common import paths as common_paths
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.method import stationary_cuda

ROOT = Path(__file__).resolve().parents[2]


def test_aot_manifest_dependency_filter_matches_compatibility_hashes(
    tmp_path: Path,
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for dependency evaluation")
    tree = ast.parse(
        (ROOT / "python/generativeqc_compiler/method/stationary_cuda.py").read_text()
    )
    assets = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "STATIONARY_AOT_ASSETS"
            for t in node.targets
        )
    )
    expected = source_hashes(
        "common", "integral", "xc", "dft", "method", "tensor", assets=assets
    )
    expected["python/generativeqc_compiler/method/stationary_resources.py"] = (
        "explicit-resource-module"
    )
    workflow = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    selection = workflow.split("set(_generativeqc_stationary_contract_assets", 1)[1]
    selection = (
        "set(_generativeqc_stationary_contract_assets"
        + selection.split("endforeach()", 1)[0]
        + "endforeach()"
    )
    unrelated = ("tools/unrelated.py",)
    entries = "\n".join(f'  "{path}"' for path in (*expected, *unrelated))
    output = tmp_path / "dependencies.txt"
    script = tmp_path / "evaluate.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f"set(_generativeqc_identity_inputs\n{entries}\n)\n"
        + selection
        + f'\nfile(WRITE "{output.as_posix()}" "${{_generativeqc_stationary_contract_inputs}}")\n'
    )
    subprocess.run(
        [cmake, "-P", str(script)], check=True, capture_output=True, timeout=20
    )
    actual = {
        Path(p).relative_to(ROOT).as_posix() for p in output.read_text().split(";")
    }
    assert actual == set(expected)
    assert 'OUTPUTS "${_generativeqc_stationary_manifest}"' in workflow
    assert (
        'GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"'
        in workflow
    )


@pytest.mark.parametrize("changed_scope", ["common", "stationary_resources"])
@pytest.mark.parametrize("functional", [0, 1, 2])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_shared_compiler_source_changes_invalidate_aot_contract(
    functional: int, spin: str, changed_scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new shared compiler source revision must reject an older artifact."""
    original_hash = common_paths.file_hash
    revision = {"changed": False}

    def source_revision_hash(path: Path) -> str:
        digest = original_hash(path)
        selected = (
            path.is_relative_to(common_paths.PACKAGE / "common")
            if changed_scope == "common"
            else path.name == "stationary_resources.py"
        )
        if revision["changed"] and selected:
            return "0" * 64
        return digest

    monkeypatch.setattr(common_paths, "file_hash", source_revision_hash)
    monkeypatch.setattr(stationary_cuda, "file_hash", source_revision_hash)
    identity = stationary_cuda.stationary_aot_contract_identity
    identity.cache_clear()
    try:
        before = identity(functional, spin=spin)
        revision["changed"] = True
        identity.cache_clear()
        after = identity(functional, spin=spin)
        assert after != before
    finally:
        identity.cache_clear()
