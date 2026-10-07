"""Stationary AOT builds lower the shared primitive inventory only once."""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from generativeqc_compiler.method import stationary_cuda

from tools import generate_stationary_force_aot as generator
from tools import write_stationary_aot_manifest as manifest_writer

ROOT = Path(__file__).resolve().parents[2]


def test_all_shards_preserve_sources_and_unchanged_output_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources = tuple(
        ((), f"// primitive shard {index}\n")
        for index in range(generator.QUALIFIED_SPD_AOT_SHARDS)
    )
    lower = Mock(return_value=sources)
    monkeypatch.setattr(generator, "derivative_cuda_sources", lower)
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_stationary_force_aot.py",
            "--output",
            str(output),
            "--component-domain",
            "spd",
            "--all-shards",
        ],
    )
    generator.main()
    lower.assert_called_once_with(generator.QUALIFIED_SPD_COMPONENTS)
    expected = {
        f"generativeqc_stationary_spd_primitive_{index}.cu": source
        for index, (_, source) in enumerate(sources)
    }
    assert {path.name: path.read_text() for path in output.iterdir()} == expected
    times = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    generator.main()
    assert {path.name: path.stat().st_mtime_ns for path in output.iterdir()} == times


@pytest.mark.parametrize(
    "arguments",
    [
        ["--all-shards"],
        ["--all-shards", "--component-domain", "spd", "--shard-index", "0"],
        ["--all-shards", "--component-domain", "spd", "--functional", "0"],
        ["--all-shards", "--component-domain", "spd", "--spin", "unpolarized"],
        ["--component-domain", "spd", "--shard-index", "-1"],
        ["--component-domain", "spd", "--shard-index", "23"],
        ["--primitive-only", "--profile", "pbe0_rks"],
        ["--primitive-only", "--component-domain", "spd"],
        ["--primitive-only", "--all-shards"],
        ["--wrapper-only"],
        ["--wrapper-only", "--profile", "pbe0_rks", "--component-domain", "spd"],
    ],
)
def test_invalid_shard_arguments_fail_before_lowering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str]
) -> None:
    lower = Mock(side_effect=AssertionError("invalid arguments reached lowering"))
    monkeypatch.setattr(generator, "derivative_cuda_sources", lower)
    output = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["generate", "--output", str(output), *arguments])
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2
    lower.assert_not_called()
    assert not output.exists()


def test_sp_primitive_generation_has_no_functional_or_spin_specialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lower = Mock(return_value="// shared s/p primitives\n")
    monkeypatch.setattr(generator, "emit_first_derivative_cuda", lower)
    output = tmp_path / "primitives.cu"
    monkeypatch.setattr(
        sys, "argv", ["generate", "--output", str(output), "--primitive-only"]
    )
    generator.main()
    lower.assert_called_once_with(generator.qualified_sp_requests())
    assert output.read_text() == lower.return_value


@pytest.mark.parametrize("profile", generator.QUALIFIED_STATIONARY_AOT_PROFILE_NAMES)
def test_split_sp_wrappers_never_regenerate_integral_primitives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str
) -> None:
    lower = Mock(side_effect=AssertionError("profile wrapper regenerated primitives"))
    emit = Mock(return_value=f"// wrapper {profile}\n")
    monkeypatch.setattr(generator, "emit_first_derivative_cuda", lower)
    monkeypatch.setattr(generator, "emit_stationary_profile_aot_wrapper_cuda", emit)
    output = tmp_path / "wrapper.cu"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--output",
            str(output),
            "--wrapper-only",
            "--profile",
            profile,
        ],
    )
    generator.main()
    lower.assert_not_called()
    emit.assert_called_once_with(profile, iterations=3)
    assert output.read_text() == emit.return_value


@pytest.mark.parametrize("profile", ["pbe0_rks", "pbe0_uks", "b3lyp_rks", "b3lyp_uks"])
def test_split_sp_wrapper_preserves_generated_scientific_source(profile: str) -> None:
    """The only source difference is the separable primitive declaration."""
    primitive = "// shared primitive inventory\n"
    wrapper = stationary_cuda.emit_stationary_profile_aot_wrapper_cuda(profile)
    monolithic = stationary_cuda.emit_stationary_profile_aot_cuda(
        profile, primitive_source=primitive
    )
    assert (
        primitive + wrapper.removeprefix(stationary_cuda._FIRST_DERIVATIVE_DECLARATION)
        == monolithic
    )


@pytest.mark.parametrize("corrupt", [None, "wrapper", "primitive"])
def test_shared_sp_manifest_checks_both_source_halves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corrupt: str | None
) -> None:
    """Neither an old wrapper nor a substituted shared primitive may be packaged."""
    monkeypatch.setattr(
        manifest_writer, "emit_first_derivative_cuda", lambda *args: "primitive"
    )
    monkeypatch.setattr(
        manifest_writer,
        "emit_stationary_profile_aot_wrapper_cuda",
        lambda *args, **kwargs: "wrapper",
    )
    wrapper, primitive, library, output = (
        tmp_path / name
        for name in ("wrapper.cu", "primitive.cu", "lib.so", "manifest.json")
    )
    wrapper.write_text("bad" if corrupt == "wrapper" else "wrapper")
    primitive.write_text("bad" if corrupt == "primitive" else "primitive")
    library.write_bytes(b"test binary, never loaded")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest",
            "--library",
            str(library),
            "--source",
            str(wrapper),
            "--primitive-source",
            str(primitive),
            "--output",
            str(output),
            "--profile",
            "pbe0_rks",
            "--architecture",
            "sm_120",
            "--compile-architecture",
            "120-real",
        ],
    )
    if corrupt is not None:
        with pytest.raises(ValueError, match="s/p generated source identity mismatch"):
            manifest_writer.main()
        assert not output.exists()
    else:
        manifest_writer.main()
        import json

        metadata = json.loads(output.read_text())
        assert metadata["schema"] == "generativeqc.stationary-cuda-aot.v2"
        assert metadata["primitive_source_sha256"] == [
            manifest_writer.file_hash(primitive)
        ]
        # Independently reconstruct the complete seal, rather than testing the
        # writer against the same helper the writer just called.
        import hashlib

        seal = metadata.pop("manifest_integrity_sha256")
        expected = hashlib.sha256(
            json.dumps(
                {
                    "schema": "generativeqc.stationary-cuda-aot.manifest-integrity.v1",
                    "manifest": metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        assert seal == expected


def test_manifest_integrity_covers_every_payload_field() -> None:
    """No partial allowlist can leave new build-record fields unbound."""
    metadata = {
        "schema": "generativeqc.stationary-cuda-aot.v3",
        "functional": 1,
        "spin": "polarized",
        "profile": "pbe0_uks",
        "plan_identity": "plan",
        "partition_iterations": 3,
        "weight_programs": {"exact_exchange": "a" * 64},
        "component_domain": ["", "x", "xx"],
        "primitive_shard_width": 16,
        "primitive_shards": 23,
        "primitive_source_sha256": ["primitives"],
        "architectures": ["sm_120"],
        "compile_architectures": ["120-real"],
        "code_objects": [{"architecture": "sm_120", "kind": "cubin"}],
        "source_identity": "source",
        "source_sha256": "wrapper",
        "contract_identity": "compiler-contract",
        "binary_sha256": "binary",
        "binary_bytes": 123,
        "compile_contract": {"fp64": True, "fmad": False},
        "future_build_field": "also bound",
    }
    identity = stationary_cuda.stationary_aot_manifest_integrity(metadata)
    for field in metadata:
        assert (
            stationary_cuda.stationary_aot_manifest_integrity(
                {**metadata, field: "changed"}
            )
            != identity
        ), field
    assert (
        stationary_cuda.stationary_aot_manifest_integrity(
            {**metadata, "manifest_integrity_sha256": identity}
        )
        == identity
    )
    assert (
        stationary_cuda.stationary_aot_manifest_integrity(
            dict(reversed(tuple(metadata.items())))
        )
        == identity
    )


@pytest.mark.parametrize(
    "requested", [None, "pbe0_uks;b3lyp_rks;pbe0_uks", "", "unknown_rks"]
)
def test_stationary_packaging_profiles_are_bounded_and_deterministic(
    tmp_path: Path, requested: str | None
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for package policy evaluation")
    output = tmp_path / "selected.txt"
    script = tmp_path / "profiles.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        + (
            f'set(GENERATIVEQC_STATIONARY_AOT_PROFILES "{requested}")\n'
            if requested is not None
            else ""
        )
        + f'include("{ROOT}/cmake/GenerativeQCStationaryProfiles.cmake")\n'
        + "generativeqc_select_stationary_aot_profiles(selected)\n"
        + f'file(WRITE "{output}" "${{selected}}")\n'
    )
    result = subprocess.run(
        [cmake, "-P", str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if requested == "unknown_rks":
        assert result.returncode != 0
        assert "Unknown stationary CUDA AOT profile" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        expected = (
            sorted(set(requested.split(";")))
            if requested
            else (
                sorted(generator.QUALIFIED_STATIONARY_AOT_PROFILE_NAMES)
                if requested is None
                else []
            )
        )
        assert output.read_text() == ";".join(expected)


@pytest.mark.parametrize("backend", ["Ninja", "Unix Makefiles"])
def test_cmake_build_lowers_each_primitive_request_once(
    tmp_path: Path, backend: str
) -> None:
    """Exercise the real multi-output build rule with only CUDA emission stubbed."""
    cmake = shutil.which("cmake")
    builder = "ninja" if backend == "Ninja" else "make"
    if cmake is None or shutil.which(builder) is None:
        pytest.skip(f"CMake and {builder} are required for build graph evaluation")

    calls = tmp_path / "lowering.txt"
    stub = tmp_path / "generate.py"
    stub.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'python')!r}]\n"
        "from tools import generate_stationary_force_aot as generator\n"
        "from generativeqc_compiler.integral import first_derivative_schedule as schedule\n"
        "def emit(requests, *, symbol):\n"
        f"    with Path({str(calls)!r}).open('a') as log:\n"
        "        log.write(f'{symbol} {len(requests)}\\n')\n"
        "    return f'// {symbol} {len(requests)}\\n'\n"
        "schedule.emit_first_derivative_cuda = emit\n"
        "generator.main()\n"
    )
    workflow = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    start = workflow.index("    set(_generativeqc_stationary_spd_primitive_sources)")
    stop = workflow.index(
        "    add_library(generativeqc_stationary_spd_primitives", start
    )
    registration = workflow[start:stop].replace(
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py",
        stub.as_posix(),
    )
    output = tmp_path / "generated"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(stationary_codegen NONE)\n"
        f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(GENERATIVEQC_STATIONARY_AOT_DIRECTORY "{output.as_posix()}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCGenerated.cmake")\n'
        + registration
        + "add_custom_target(all_primitives ALL "
        "DEPENDS ${_generativeqc_stationary_spd_primitive_sources})\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), "-G", backend],
        check=True,
        capture_output=True,
        timeout=30,
    )

    def build_sources() -> None:
        subprocess.run(
            [cmake, "--build", str(build), "--parallel", "8"],
            check=True,
            capture_output=True,
            timeout=30,
        )

    build_sources()
    expected = [f"first_derivative_shard_{index} 16" for index in range(22)]
    expected.append("first_derivative_shard_22 10")
    assert calls.read_text().splitlines() == expected
    assert len(list(output.glob("*.cu"))) == 23
    build_sources()
    assert calls.read_text().splitlines() == expected

    # A missing secondary output must retrigger the one shared generation rule.
    missing = output / "generativeqc_stationary_spd_primitive_11.cu"
    missing.unlink()
    build_sources()
    assert missing.exists()
    assert calls.read_text().splitlines() == expected * 2


@pytest.mark.parametrize("profile", ["pbe0_rks", "b3lyp_uks"])
@pytest.mark.parametrize("component_domain", ["sp", "spd"])
def test_profile_wrapper_generation_uses_exact_profile_emitter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    component_domain: str,
) -> None:
    output = tmp_path / f"{profile}-{component_domain}.cu"
    monkeypatch.setattr(
        generator,
        "emit_first_derivative_cuda",
        Mock(return_value="// primitives\n"),
    )
    sp = Mock(return_value=f"// profile-sp {profile}\n")
    spd = Mock(return_value=f"// profile-spd {profile}\n")
    legacy_sp = Mock(
        side_effect=AssertionError("profile generation used legacy emitter")
    )
    legacy_spd = Mock(
        side_effect=AssertionError("profile generation used legacy component emitter")
    )
    monkeypatch.setattr(generator, "emit_stationary_profile_aot_cuda", sp)
    monkeypatch.setattr(
        generator, "emit_stationary_profile_component_aot_wrapper_cuda", spd
    )
    monkeypatch.setattr(generator, "emit_stationary_aot_cuda", legacy_sp)
    monkeypatch.setattr(
        generator, "emit_stationary_component_aot_wrapper_cuda", legacy_spd
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--output",
            str(output),
            "--profile",
            profile,
            "--component-domain",
            component_domain,
        ],
    )

    generator.main()

    if component_domain == "sp":
        sp.assert_called_once_with(
            profile, primitive_source="// primitives\n", iterations=3
        )
        spd.assert_not_called()
        assert output.read_text() == f"// profile-sp {profile}\n"
    else:
        sp.assert_not_called()
        spd.assert_called_once_with(profile, iterations=3)
        assert output.read_text() == f"// profile-spd {profile}\n"
    legacy_sp.assert_not_called()
    legacy_spd.assert_not_called()
