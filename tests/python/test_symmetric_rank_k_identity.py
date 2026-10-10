"""Qualification identity includes the staged and actual native build inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rank_k_generator", ROOT / "tools/generate_symmetric_rank_k_cuda.py"
)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "source"
    toolkit = tmp_path / "cuda"
    host = tmp_path / "host-cxx"
    (root / "cmake").mkdir(parents=True)
    (root / "cmake/GenerativeQCSourceIdentity.json").write_text(
        json.dumps({"schema_version": 1, "recursive_groups": [], "files": []})
    )
    for relative in (*generator._TOOLCHAIN_FILES, "include/nested/cuda_runtime.h"):
        path = toolkit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    harness = root / "tests/native/test_symmetric_rank_k_cuda.cu"
    harness.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/native/test_symmetric_rank_k_cuda.cu", harness)
    host.write_text("host compiler fixture")
    manifest = tmp_path / "host-toolchain.json"
    digest = hashlib.sha256(host.read_bytes()).hexdigest()
    roles = (
        "program:driver",
        "program:cc1plus",
        "program:as",
        "program:collect2",
        "program:ld",
        "program:lto-wrapper",
        "linker-plugin:liblto_plugin.so",
        "config:gcc-specs",
        "config:ld-default-script",
        "link-input:libstdc++.so",
        "link-input:libgcc.a",
        "link-input:crtbeginS.o",
        "header:0:fixture.hpp",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "generativeqc.rank-k-host-toolchain.v2",
                "compiler_sha256": digest,
                "target": "x86_64-linux-gnu",
                "version": "11.4.0",
                "entries": [{"role": role, "sha256": digest} for role in roles],
            }
        )
    )
    return root, toolkit, host, manifest


def test_compiled_harness_mutation_invalidates_identity(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    root, *_ = inputs
    harness = root / generator._COMPILATION_SOURCES[0]
    before = generator.compiler_identity(*inputs)
    harness.write_text(harness.read_text() + "\n// changed compile input\n")
    assert generator.compiler_identity(*inputs) != before


def test_missing_compiled_harness_fails_closed(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    root, *_ = inputs
    (root / generator._COMPILATION_SOURCES[0]).unlink()
    with pytest.raises(FileNotFoundError, match="test_symmetric_rank_k_cuda"):
        generator.compiler_identity(*inputs)


def test_uncompiled_test_source_does_not_change_identity(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    root, *_ = inputs
    before = generator.compiler_identity(*inputs)
    unrelated = root / "tests/native/uncompiled_rank_k_probe.cu"
    unrelated.write_text("uncompiled")
    assert generator.compiler_identity(*inputs) == before


@pytest.mark.parametrize("target", ["host", "header"])
def test_native_input_mutation_invalidates_identity(
    inputs: tuple[Path, Path, Path, Path],
    target: str,
) -> None:
    _, toolkit, host, _ = inputs
    before = generator.compiler_identity(*inputs)
    path = host if target == "host" else toolkit / "include/nested/cuda_runtime.h"
    path.write_text("changed native input")
    if target == "host":
        with pytest.raises(ValueError, match="different compiler bytes"):
            generator.compiler_identity(*inputs)
    else:
        assert generator.compiler_identity(*inputs) != before


@pytest.mark.parametrize("relative", generator._TOOLCHAIN_FILES)
def test_fixed_recipe_input_mutation_invalidates_identity(
    inputs: tuple[Path, Path, Path, Path], relative: str
) -> None:
    _, toolkit, _, _ = inputs
    before = generator.compiler_identity(*inputs)
    (toolkit / relative).write_text("changed fixed recipe input")
    assert generator.compiler_identity(*inputs) != before


def test_host_closure_mutation_invalidates_identity(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    *_, manifest = inputs
    before = generator.compiler_identity(*inputs)
    data = json.loads(manifest.read_text())
    data["entries"][-1]["sha256"] = "f" * 64
    manifest.write_text(json.dumps(data))
    assert generator.compiler_identity(*inputs) != before


def test_missing_or_incomplete_host_closure_fails_closed(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    *_, manifest = inputs
    original = manifest.read_text()
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        generator.compiler_identity(*inputs)
    manifest.write_text(original)
    data = json.loads(original)
    data["entries"] = data["entries"][:-1]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        generator.compiler_identity(*inputs)


@pytest.mark.parametrize(
    "name",
    [
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
        "GCC_COMPARE_DEBUG",
        "DEPENDENCIES_OUTPUT",
        "SUNPRO_DEPENDENCIES",
        "LD_PRELOAD",
    ],
)
def test_ambient_build_environment_is_rejected(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.delenv(name, raising=False)
    before = generator.compiler_identity(*inputs)
    monkeypatch.setenv(
        name, "--use_fast_math" if name.startswith("NVCC") else "/extra/input"
    )
    with pytest.raises(ValueError, match="unsupported by the fixed rank-k"):
        generator.compiler_identity(*inputs)
    monkeypatch.delenv(name)
    assert generator.compiler_identity(*inputs) == before


def test_relocated_inputs_and_launcher_do_not_change_identity(
    inputs: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, toolkit, host, manifest = inputs
    before = generator.compiler_identity(*inputs)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copytree(root, relocated / "source")
    shutil.copytree(toolkit, relocated / "cuda")
    shutil.copyfile(host, relocated / "host-cxx")
    shutil.copyfile(manifest, relocated / "host-toolchain.json")
    monkeypatch.setenv("SCCACHE_DIR", str(relocated / "cache"))
    assert (
        generator.compiler_identity(
            relocated / "source",
            relocated / "cuda",
            relocated / "host-cxx",
            relocated / "host-toolchain.json",
        )
        == before
    )


@pytest.mark.parametrize("kind", ["header", "response-file"])
def test_same_path_external_file_mutations_are_rejected(
    inputs: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    external = tmp_path / ("external.h" if kind == "header" else "options.rsp")
    if kind == "header":
        monkeypatch.setenv("CPATH", str(tmp_path))
    else:
        monkeypatch.setenv("NVCC_APPEND_FLAGS", f"--options-file={external}")
    for content in ("version one", "version two"):
        external.write_text(content)
        with pytest.raises(ValueError, match="unsupported by the fixed rank-k"):
            generator.compiler_identity(*inputs)


def test_generation_only_hashes_inputs_without_executing_compilers(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_process(*args: object, **kwargs: object) -> None:
        pytest.fail("source generation must not probe a compiler or GPU")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    rendered = generator.render(generator.compiler_identity(*inputs))
    assert "rank_k_bound_old_output" in rendered
    assert "rank_k_density_row_overwrite_candidates" in rendered
    assert "rank_k_density_row_update_candidates" in rendered
    assert "rank_k_density_n17_k9_row_update_candidates" in rendered
    assert "rank_k_compiled_batches" in rendered


def test_checkout_generation_needs_no_installed_python_packages(
    inputs: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    _, toolkit, host, manifest = inputs
    output = tmp_path / "isolated.cuh"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "tools/generate_symmetric_rank_k_cuda.py"),
            "--toolkit-root",
            str(toolkit),
            "--host-compiler",
            str(host),
            "--host-toolchain-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rendered = output.read_text()
    assert "rank_k_bound_old_output" in rendered
    assert "rank_k_density_row_overwrite_candidates" in rendered
    assert "rank_k_density_row_update_candidates" in rendered
    assert "rank_k_density_n17_k9_row_update_candidates" in rendered
    assert "rank_k_compiled_batches" in rendered


def test_missing_native_inputs_fail_closed(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    _, toolkit, host, _ = inputs
    host.unlink()
    with pytest.raises(FileNotFoundError):
        generator.compiler_identity(*inputs)
    host.write_text("host compiler fixture")
    shutil.rmtree(toolkit / "include")
    with pytest.raises(FileNotFoundError, match="toolkit headers"):
        generator.compiler_identity(*inputs)


@pytest.mark.parametrize("relative", ["nvvm/bin/cicc", "bin/nvcc.profile"])
def test_missing_device_subtool_fails_closed(
    inputs: tuple[Path, Path, Path, Path], relative: str
) -> None:
    _, toolkit, _, _ = inputs
    (toolkit / relative).unlink()
    with pytest.raises(FileNotFoundError, match=relative):
        generator.compiler_identity(*inputs)


def test_qualifier_verifies_staged_identity_before_compilation() -> None:
    script = (ROOT / "tools/qualify_symmetric_rank_k_cuda.sh").read_text()
    for relative in generator._TOOLCHAIN_FILES:
        assert f'"$toolkit_root/{relative}"' in script
    assert script.index("for variable in NVCC_PREPEND_FLAGS") < script.index(
        "cache_version="
    )
    host_inventory = script.index(
        '"$python_exe" -I -S tools/generate_rank_k_host_toolchain_manifest.py'
    )
    final_path = script.index('export PATH="$toolkit_root/bin:$PATH"')
    host_compare = script.index(
        'if ! cmp "$snapshot_root/host-toolchain-manifest.json"'
    )
    verify = script.index('"$python_exe" -I -S tools/generate_symmetric_rank_k_cuda.py')
    compare = script.index(
        'if ! cmp "$output_dir/generated/generated_symmetric_rank_k.cuh"'
    )
    compile_command = script.index('"$cache_exe" "$nvcc_exe" -std=c++20')
    link_command = script.index('"$cache_exe" "$nvcc_exe" --cudart shared')
    final_stats = script.rindex('"$cache_exe" --show-stats')
    assert (
        final_path
        < host_inventory
        < host_compare
        < verify
        < compare
        < compile_command
        < link_command
        < final_stats
    )
    assert '--host-compiler "$host_exe"' in script[verify:compare]
    assert (
        '--host-toolchain-manifest "$output_dir/host-toolchain-manifest.json"'
        in script[verify:compare]
    )
    assert "exit 2" in script[compare:compile_command]
    assert script.count('"-ccbin=$host_exe"') == 2
    assert script.count("export PATH=") == 1
    assert "BASH_REMATCH[2] < 16" in script
    assert "0.1[7-9]" not in script


def test_qualifier_rejects_ambient_response_file_before_tool_probes(
    tmp_path: Path,
) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is required")
    response = tmp_path / "options.rsp"
    response.write_text("--compiler-bindir=/unqualified/compiler")
    environment = dict(os.environ)
    environment["NVCC_APPEND_FLAGS"] = f"--options-file={response}"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "tools/qualify_symmetric_rank_k_cuda.sh"),
            str(tmp_path / "output"),
            "/missing/cache",
            "unqualified-commit",
            "/missing/toolkit",
            "/missing/python",
            "/missing/host",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unsupported by the fixed rank-k qualification recipe" in result.stderr
    assert not (tmp_path / "output").exists()
