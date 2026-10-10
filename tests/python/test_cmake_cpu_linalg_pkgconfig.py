"""Link real and synthetic pkg-config providers through the production CMake owner."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="ELF link-order regression uses --as-needed"
)


def _run(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _environment(pcdir: Path, native_cxx: object) -> dict[str, str]:
    for tool in ("cmake", "pkg-config", "ar"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required")
    return {
        **os.environ,
        "PKG_CONFIG_PATH": str(pcdir),
        "PKG_CONFIG_LIBDIR": str(pcdir),
        "CMAKE_PREFIX_PATH": "",
        "CMAKE_CXX_COMPILER_LAUNCHER": native_cxx.cache,
    }


def _configure(
    root: Path,
    pcdir: Path,
    native_cxx: object,
    *,
    real: bool = False,
    fallback: Path | None = None,
    mode: str = "openblas",
) -> tuple[Path, dict[str, str]]:
    source = root / "consumer"
    source.mkdir()
    if real:
        sources = f'"{ROOT / "src/tensor/cpu_linalg.cpp"}"'
        executable = f'"{ROOT / "tests/native/test_cpu_linalg.cpp"}"'
    else:
        (source / "provider.cpp").write_text(
            '#include <cblas.h>\nextern "C" int provider_value() {\n'
            "#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX\n"
            "return scipy_openblas_get_num_threads();\n#else\n"
            "return openblas_get_num_threads();\n#endif\n}\n"
        )
        (source / "main.cpp").write_text(
            'extern "C" int provider_value();\n'
            "int main() { return provider_value() != 1; }\n"
        )
        sources, executable = "provider.cpp", "main.cpp"
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(PkgConfigProvider LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n"
        + (
            f'set(OpenBLAS_DIR "{fallback}")\n'
            if fallback
            else "set(CMAKE_DISABLE_FIND_PACKAGE_OpenBLAS TRUE)\n"
        )
        + "set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH FALSE)\n"
        f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        f"add_library(native_owner SHARED {sources})\n"
        f'target_include_directories(native_owner PUBLIC "{ROOT / "src"}")\n'
        # The same module/PRIVATE provider edge used by generativeqc SHARED.
        "generativeqc_configure_cpu_linalg(native_owner)\n"
        "target_link_options(native_owner PRIVATE -Wl,--as-needed -Wl,-z,defs)\n"
        f"add_executable(consumer {executable})\n"
        "target_link_libraries(consumer PRIVATE native_owner)\n"
    )
    env = _environment(pcdir, native_cxx)
    build = root / "build"
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DGENERATIVEQC_CPU_LINALG_PROVIDER={mode}",
        f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
    ]
    result = _run(command, env=env)
    (root / "configure.log").write_text(result.stdout + result.stderr)
    return build, env


def _capabilities(build: Path) -> tuple[bool, ...]:
    cache = (build / "CMakeCache.txt").read_text()
    result = []
    for name in ("LOCAL_THREADS", "GLOBAL_THREADS", "LAPACKE"):
        match = re.search(
            rf"^GENERATIVEQC_OPENBLAS_HAS_{name}:INTERNAL=(.*)$", cache, re.MULTILINE
        )
        assert match is not None, name
        result.append(match.group(1) == "1")
    return tuple(result)


def _synthetic_provider(
    root: Path,
    native_cxx: object,
    *,
    spelling: str,
    scipy: bool,
    missing: str = "",
    extra_flags: str = "",
) -> Path:
    root.mkdir()
    prefix = "scipy_" if scipy else ""
    (root / "cblas.h").write_text(
        "#ifndef PROVIDER_COMPILE_FLAG\n#error missing provider compile option\n#endif\n"
        'extern "C" {\n'
        f"int {prefix}openblas_set_num_threads_local(int);\n"
        f"int {prefix}openblas_get_num_threads();\n"
        f"void {prefix}openblas_set_num_threads(int);\n}}\n"
    )
    (root / "lapacke.h").write_text(
        '#define LAPACK_ROW_MAJOR 101\nextern "C" {\n'
        f"int {prefix}LAPACKE_dpotrf(int,char,int,double*,int);\n"
        f"int {prefix}LAPACKE_dsyevd(int,char,char,int,double*,int,double*);\n}}\n"
    )
    dependent = spelling in {"absolute-named", "named-absolute", "reversed"}
    value = "dependency()" if dependent else "1"
    code = 'extern "C" {\n' + ("int dependency();\n" if dependent else "")
    if missing != "local":
        code += f"int {prefix}openblas_set_num_threads_local(int) {{return {value};}}\n"
    code += (
        f"int {prefix}openblas_get_num_threads() {{return {value};}}\n"
        f"void {prefix}openblas_set_num_threads(int) {{}}\n"
        f"int {prefix}LAPACKE_dpotrf(int,char,int,double*,int) {{return {value}-1;}}\n"
    )
    if missing != "lapack":
        code += (
            f"int {prefix}LAPACKE_dsyevd(int,char,char,int,double*,int,double*) "
            f"{{return {value}-1;}}\n"
        )
    code += "}\n"
    shared = spelling in {"shared", "named"}

    def library(name: str, content: str) -> Path:
        source = root / f"{name}.cpp"
        source.write_text(content)
        output = root / f"lib{name}.{'so' if shared else 'a'}"
        if shared:
            native_cxx.build_shared([source], output)
        else:
            obj = root / f"{name}.o"
            native_cxx.compile_object(source, obj, args=("-fPIC",))
            _run(["ar", "rcs", str(output), str(obj)], env=dict(os.environ))
        return output

    main = library("provider", code)
    flags = str(main)
    if dependent:
        dependency = library("dependency", 'extern "C" int dependency(){return 1;}\n')
        if spelling == "absolute-named":
            flags = f"{main} -L{root} -ldependency"
        elif spelling == "named-absolute":
            flags = f"-L{root} -lprovider {dependency}"
        else:
            flags = f"{dependency} {main}"
    elif spelling == "named":
        flags = f"-L{root} -lprovider"
    module = "scipy-openblas" if scipy else "openblas"
    (root / f"{module}.pc").write_text(
        "Name: test-openblas\nDescription: link-order fixture\nVersion: 1.0\n"
        f"Libs: {extra_flags} {flags} -Wl,-rpath,{root}\n"
        f"Cflags: -I{root} -DPROVIDER_COMPILE_FLAG=1\n"
    )
    return root


@pytest.mark.parametrize("scipy", [False, True])
@pytest.mark.parametrize(
    "spelling", ["shared", "static", "named", "absolute-named", "named-absolute"]
)
def test_pkgconfig_libraries_link_after_objects(
    tmp_path: Path, native_cxx: object, spelling: str, scipy: bool
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata", native_cxx, spelling=spelling, scipy=scipy
    )
    build, env = _configure(tmp_path, provider, native_cxx)
    assert _capabilities(build) == (True, True, True)
    result = _run(["cmake", "--build", str(build), "--verbose"], env=env)
    (tmp_path / "build.log").write_text(result.stdout + result.stderr)
    _run([str(build / "consumer")], env=env)


@pytest.mark.parametrize("missing", ["local", "lapack"])
def test_pkgconfig_missing_symbols_stay_unavailable(
    tmp_path: Path, native_cxx: object, missing: str
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata",
        native_cxx,
        spelling="shared",
        scipy=True,
        missing=missing,
    )
    build, _env = _configure(tmp_path, provider, native_cxx)
    assert _capabilities(build) == (missing != "local", True, missing != "lapack")


def test_pkgconfig_preserves_wrong_archive_order(
    tmp_path: Path, native_cxx: object
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata", native_cxx, spelling="reversed", scipy=True
    )
    build, env = _configure(tmp_path, provider, native_cxx)
    assert _capabilities(build) == (False, False, False)
    result = subprocess.run(
        ["cmake", "--build", str(build)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "undefined reference" in result.stdout + result.stderr


def test_real_scipy_pkgconfig_links_native_owner(
    tmp_path: Path, native_cxx: object
) -> None:
    scipy = pytest.importorskip("scipy_openblas32")
    pcdir = tmp_path / "metadata"
    pcdir.mkdir()
    (pcdir / "scipy-openblas.pc").write_text(scipy.get_pkg_config())
    build, env = _configure(tmp_path, pcdir, native_cxx, real=True)
    # Symbol availability comes from the actual loaded library, not a version label.
    library = scipy.dll
    expected = (
        hasattr(library, "scipy_openblas_set_num_threads_local"),
        all(
            hasattr(library, name)
            for name in (
                "scipy_openblas_get_num_threads",
                "scipy_openblas_set_num_threads",
            )
        ),
        all(
            hasattr(library, name)
            for name in ("scipy_LAPACKE_dpotrf", "scipy_LAPACKE_dsyevd")
        ),
    )
    assert _capabilities(build) == expected
    result = _run(["cmake", "--build", str(build), "--verbose"], env=env)
    (tmp_path / "build.log").write_text(result.stdout + result.stderr)
    _run([str(build / "consumer")], env=env)


@pytest.mark.parametrize(
    "flags",
    [
        "-Wl,--start-group",
        "-Wl,--end-group",
        "-Wl,--whole-archive",
        "-Wl,--no-whole-archive",
        "-Wl,--push-state",
        "-Wl,--pop-state",
        "-Wl,--as-needed",
        "-Wl,--no-as-needed",
        "-Xlinker --whole-archive",
        "-Wl,-rpath,/tmp,--whole-archive",
    ],
)
def test_positional_pkgconfig_flags_fail_closed(
    tmp_path: Path, native_cxx: object, flags: str
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata",
        native_cxx,
        spelling="shared",
        scipy=True,
        extra_flags=flags,
    )
    with pytest.raises(AssertionError, match=r"unsupported\s+linker argument"):
        _configure(tmp_path, provider, native_cxx)


@pytest.mark.parametrize("fallback", [False, True])
def test_unsupported_pkgconfig_uses_independent_fallback(
    tmp_path: Path, native_cxx: object, fallback: bool
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata",
        native_cxx,
        spelling="shared",
        scipy=True,
        extra_flags="-Wl,--whole-archive",
    )
    config = tmp_path / "cmake-provider"
    config.mkdir()
    (config / "OpenBLASConfig.cmake").write_text(
        "set(OpenBLAS_FOUND TRUE)\n"
        "add_library(scipy_openblas_fallback INTERFACE IMPORTED)\n"
        "set_property(TARGET scipy_openblas_fallback PROPERTY "
        f'INTERFACE_LINK_LIBRARIES "{provider / "libprovider.so"}")\n'
        "set_property(TARGET scipy_openblas_fallback PROPERTY "
        "INTERFACE_COMPILE_DEFINITIONS PROVIDER_COMPILE_FLAG=1)\n"
        "set(OpenBLAS_LIBRARIES scipy_openblas_fallback)\n"
        f'set(OpenBLAS_INCLUDE_DIRS "{provider}")\n'
    )
    build, env = _configure(
        tmp_path,
        provider,
        native_cxx,
        fallback=config if fallback else None,
        mode="auto",
    )
    output = (tmp_path / "configure.log").read_text()
    assert "unsupported" in output
    if fallback:
        assert _capabilities(build) == (True, True, True)
        _run(["cmake", "--build", str(build)], env=env)
        _run([str(build / "consumer")], env=env)
    else:
        assert "scalar fallback" in output
        assert (
            "GENERATIVEQC_OPENBLAS_HAS_" not in (build / "CMakeCache.txt").read_text()
        )


def test_pkgconfig_absolute_libraries_survive_reconfigure(
    tmp_path: Path, native_cxx: object
) -> None:
    provider = _synthetic_provider(
        tmp_path / "metadata",
        native_cxx,
        spelling="shared",
        scipy=True,
    )
    build, env = _configure(tmp_path, provider, native_cxx)
    assert _capabilities(build) == (True, True, True)
    _run(["cmake", "-S", str(tmp_path / "consumer"), "-B", str(build)], env=env)
    assert _capabilities(build) == (True, True, True)
    _run(["cmake", "--build", str(build)], env=env)
    _run([str(build / "consumer")], env=env)
