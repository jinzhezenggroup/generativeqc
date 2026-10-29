"""Exercise manifest-bound host-linked CUDA packaging without a GPU/toolkit."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="NVIDIA Linux ELF contract"
)


def checked(*argv: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a finite host build or ABI command and retain its diagnostic output."""
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=90, **kwargs
    )


@pytest.mark.parametrize("install_rpath", ["", "$ORIGIN"])
def test_installed_plugin_retains_manifest_bytes_and_public_abi(
    tmp_path: Path, native_cxx: Any, install_rpath: str
) -> None:
    """The real host-link helper must seal the final ELF before hashing it.

    A C++ provider forces a genuine automatic build RPATH. Only the helper's
    NVIDIA branch is selected; no CUDA compiler or device is probed/executed.
    """
    (tmp_path / "provider.cpp").write_text(
        'extern "C" int install_provider() { return 41; }\n'
    )
    (tmp_path / "entry.cpp").write_text(
        'extern "C" int install_provider();\n'
        'extern "C" int installed_entry() { return install_provider() + 1; }\n'
    )
    (tmp_path / "manifest.cmake").write_text(
        'file(SHA256 "${LIBRARY}" digest)\nfile(WRITE "${OUTPUT}" "${digest}")\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.24)
project(InstallIdentity LANGUAGES CXX)
include("{ROOT / "cmake/GenerativeQCCudaArchive.cmake"}")
set(CMAKE_CUDA_COMPILER_ID NVIDIA)
set(GENERATIVEQC_PRUNE_CUDA_ARCHIVES OFF)
add_library(install_provider SHARED provider.cpp)
set_target_properties(install_provider PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/provider")
generativeqc_add_cuda_shared_library(stationary owner entry.cpp)
target_link_libraries(${{owner}} PRIVATE install_provider)
add_custom_command(TARGET stationary POST_BUILD
  COMMAND "${{CMAKE_COMMAND}}" "-DLIBRARY=$<TARGET_FILE:stationary>"
          "-DOUTPUT=${{CMAKE_BINARY_DIR}}/binding.sha256"
          -P "${{CMAKE_SOURCE_DIR}}/manifest.cmake" VERBATIM)
install(TARGETS stationary install_provider LIBRARY DESTINATION lib)
install(FILES "${{CMAKE_BINARY_DIR}}/binding.sha256" DESTINATION lib)
""")
    build, prefix = tmp_path / "build", tmp_path / "installed"
    checked(native_cxx.cache, "--version")
    checked(
        "cmake",
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
        f"-DCMAKE_INSTALL_RPATH={install_rpath}",
    )
    checked("cmake", "--build", str(build), "--parallel", "2")
    sealed = (build / "binding.sha256").read_text()
    original = (build / "libstationary.so").read_bytes()
    assert hashlib.sha256(original).hexdigest() == sealed
    checked("cmake", "--install", str(build), "--prefix", str(prefix))
    library = prefix / "lib/libstationary.so"
    assert (prefix / "lib/binding.sha256").read_text() == sealed
    assert hashlib.sha256(library.read_bytes()).hexdigest() == sealed
    assert library.read_bytes() == original
    env = dict(os.environ)
    if install_rpath:
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["LD_LIBRARY_PATH"] = str(prefix / "lib")
    probe = checked(
        sys.executable,
        "-c",
        "import ctypes,sys; f=ctypes.CDLL(sys.argv[1]).installed_entry; "
        "f.restype=ctypes.c_int; assert f()==42",
        str(library),
        env=env,
    )
    assert probe.returncode == 0
