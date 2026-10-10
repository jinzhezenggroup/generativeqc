"""Post-link AOT identity dependencies cover every file hashed by its contract."""

import ast
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from generativeqc_compiler.common import paths as common_paths
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.method import stationary_cuda

ROOT = Path(__file__).resolve().parents[2]


def _contract_dependency_selection(workflow: str) -> str:
    selection = workflow.split("set(_generativeqc_stationary_contract_assets", 1)[1]
    return (
        "set(_generativeqc_stationary_contract_assets"
        + selection.split("endforeach()", 1)[0]
        + "endforeach()"
    )


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
    selection = _contract_dependency_selection(workflow)
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


@pytest.mark.parametrize(
    "asset", ("residency_boundaries.hpp", "residency_observer.hpp")
)
def test_residency_asset_changes_regenerate_aot_identity(
    asset: str, tmp_path: Path
) -> None:
    """Real CMake dependencies must rebuild the contract after a header edit."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for dependency evaluation")
    relative = f"src/runtime/{asset}"
    header = tmp_path / relative
    header.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / relative, header)
    unrelated = tmp_path / "unrelated.hpp"
    unrelated.write_text("// not part of the contract\n")
    build = tmp_path / "build"
    output = build / "identity.txt"
    runs = build / "runs.txt"
    generator = tmp_path / "identity.py"
    generator.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
        "from generativeqc_compiler.common import paths\n"
        "from generativeqc_compiler.method.stationary_cuda import stationary_aot_contract_identity\n"
        "original = paths.asset_path\n"
        f"paths.asset_path = lambda name: Path({str(header)!r}) if name == {relative!r} else original(name)\n"
        f"Path({str(output)!r}).write_text(stationary_aot_contract_identity(0))\n"
        f"with Path({str(runs)!r}).open('a') as stream: stream.write('run\\n')\n"
    )
    selection = _contract_dependency_selection(
        (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\nproject(identity_probe NONE)\n"
        f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCGenerated.cmake")\n'
        f'set(_generativeqc_identity_inputs "{relative}" "unrelated.hpp")\n'
        + selection
        + "\ngenerativeqc_register_generated_sources(\n"
        '  NAME identity_probe GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/identity.py"\n'
        '  OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/identity.txt"\n'
        "  DEPENDS ${_generativeqc_stationary_contract_inputs})\n"
    )

    def run(*args: str) -> None:
        subprocess.run(
            [cmake, *args], check=True, capture_output=True, text=True, timeout=30
        )

    run("-S", str(tmp_path), "-B", str(build))
    command = ("--build", str(build), "--target", "identity_probe")
    run(*command)
    before = output.read_text()
    run(*command)
    assert runs.read_text() == "run\n"
    # Keep the test reliable on filesystems with whole-second timestamps.
    time.sleep(1.1)
    unrelated.write_text("// unrelated edit must not regenerate the contract\n")
    run(*command)
    assert runs.read_text() == "run\n"
    header.write_text(header.read_text() + "\n// identity dependency mutation\n")
    run(*command)
    assert output.read_text() != before
    assert runs.read_text() == "run\nrun\n"
    run(*command)
    assert runs.read_text() == "run\nrun\n"


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
