"""Build the shared LP64 packaging owner and inspect its installed ELF contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux ELF wheel contract"
)


def run(command: list[str]) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("mode", ["bundled", "custom", "disabled"])
def test_private_provider_packaging(
    tmp_path: Path, native_cxx: NativeCxx, mode: str
) -> None:
    """A reviewed-provider double isolates packaging from network/distribution state."""
    for tool in ("cmake", "readelf", "nm"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} unavailable")
    provider_source = tmp_path / "provider.cpp"
    provider_source.write_text('extern "C" int test_provider() { return 23; }\n')
    provider = native_cxx.build_shared(
        [provider_source],
        tmp_path / "libreviewed_test_provider.so",
        link_args=["-Wl,-soname,libreviewed_test_provider.so"],
    )
    shim_source = tmp_path / "src/tensor/cpu/openblas_lp64_shim.cpp"
    shim_source.parent.mkdir(parents=True)
    shim_source.symlink_to(ROOT / "src/tensor/cpu/openblas_lp64_shim.cpp")
    resolver = tmp_path / "tools/xtb/resolve-openblas-wheel.py"
    resolver.parent.mkdir(parents=True)
    resolver.write_text(
        "print("
        + repr(
            json.dumps(
                {
                    "provider_path": str(provider),
                    "expected_config_prefix": "OpenBLAS reviewed-test",
                }
            )
        )
        + ")\n"
    )
    (tmp_path / "owner.cpp").write_text(
        "#include <cstring>\n"
        'extern "C" int configured_provider() {\n'
        "#if defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS)\n"
        "return std::strcmp(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX, "
        '"OpenBLAS reviewed-test") == 0;\n'
        "#elif defined(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME)\n"
        "return std::strcmp(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME, "
        f"{json.dumps(str(provider))}) == 0;\n"
        "#else\nreturn 0;\n#endif\n}\n"
    )
    (tmp_path / "main.cpp").write_text(
        'extern "C" int configured_provider();\n'
        f"int main() {{ return configured_provider() != {int(mode != 'disabled')}; }}\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(PrivateProvider LANGUAGES CXX)\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        "add_library(owner SHARED owner.cpp)\n"
        "generativeqc_configure_private_cpu_linalg(owner)\n"
        "add_executable(probe main.cpp)\n"
        "target_link_libraries(probe PRIVATE owner)\n"
    )
    build = tmp_path / "build"
    options = {
        "bundled": [
            "-DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=ON",
            "-DGENERATIVEQC_PYTHON_WHEEL=ON",
        ],
        "custom": [f"-DGENERATIVEQC_XTB_CPU_LINALG_LIBRARY={provider}"],
        "disabled": [],
    }[mode]
    run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
            *options,
        ]
    )
    run(["cmake", "--build", str(build), "--verbose"])
    run([str(build / "probe")])
    owner_elf = run(["readelf", "-d", str(build / "libowner.so")])
    assert "libreviewed_test_provider" not in owner_elf
    assert "openblas_lp64_shim" not in owner_elf
    shim_name = "libgenerativeqc_xtb_openblas_lp64_shim.so"
    if mode != "bundled":
        assert not (build / shim_name).exists()
        return
    shim_elf = run(["readelf", "-d", str(build / shim_name)])
    assert f"Library soname: [{shim_name}]" in shim_elf
    assert "Shared library: [libreviewed_test_provider.so]" in shim_elf
    assert not run(["nm", "-D", "--defined-only", str(build / shim_name)]).strip()
    prefix = tmp_path / "installed"
    run(["cmake", "--install", str(build), "--prefix", str(prefix)])
    installed = list(prefix.rglob(shim_name))
    assert len(installed) == 1
    installed_elf = run(["readelf", "-d", str(installed[0])])
    assert str(tmp_path) not in installed_elf
    assert "Shared library: [libreviewed_test_provider.so]" in installed_elf


def test_bundled_provider_requires_wheel(tmp_path: Path) -> None:
    """Reject an accidental hard provider bundle in an ordinary native build."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(InvalidBundle LANGUAGES CXX)\n"
        f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        "set(GENERATIVEQC_BUNDLE_XTB_OPENBLAS ON)\n"
        "generativeqc_configure_private_cpu_linalg(unused)\n"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "requires GENERATIVEQC_PYTHON_WHEEL=ON" in result.stderr
