"""Link the SciPy LAPACKE ABI independently of its header spelling."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from test_cmake_cpu_linalg_cache import _capability, _probe_project

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


@pytest.mark.skipif(os.name == "nt", reason="POSIX static-provider ABI fixture")
@pytest.mark.parametrize("prefixed_header", (False, True))
@pytest.mark.parametrize("missing_symbol", (False, True))
def test_scipy_lapacke_header_spelling_does_not_hide_linked_symbols(
    tmp_path: Path,
    native_cxx: NativeCxx,
    prefixed_header: bool,
    missing_symbol: bool,
) -> None:
    """Available symbols are admitted; a missing LAPACKE symbol still rejects."""
    archive_tool = shutil.which("ar")
    if archive_tool is None:
        pytest.skip("ar is required for the provider fixture")
    provider = tmp_path / "scipy-provider"
    include = provider / "include"
    include.mkdir(parents=True)
    (include / "cblas.h").write_text(
        "#pragma once\n"
        "inline int scipy_openblas_set_num_threads_local(int){return 1;}\n"
        "inline int scipy_openblas_get_num_threads(){return 1;}\n"
        "inline void scipy_openblas_set_num_threads(int){}\n"
    )
    prefix = "scipy_" if prefixed_header else ""
    (include / "lapacke.h").write_text(
        "#pragma once\n#define LAPACK_ROW_MAJOR 101\n"
        f'extern "C" int {prefix}LAPACKE_dpotrf(int,char,int,double*,int);\n'
        f'extern "C" int {prefix}LAPACKE_dsyevd(int,char,char,int,double*,int,double*);\n'
    )
    source = provider / "provider.cpp"
    source.write_text(
        'extern "C" int scipy_LAPACKE_dpotrf(int,char,int,double*,int){return 0;}\n'
        + (
            ""
            if missing_symbol
            else 'extern "C" int scipy_LAPACKE_dsyevd(int,char,char,int,double*,int,double*){return 0;}\n'
        )
    )
    obj = provider / "provider.o"
    subprocess.run(
        [native_cxx.cache, native_cxx.compiler, "-c", str(source), "-o", str(obj)],
        check=True,
        timeout=60,
    )
    library = provider / "libscipy_openblas_fixture.a"
    subprocess.run(
        [archive_tool, "rcs", str(library), str(obj)], check=True, timeout=30
    )
    (provider / "OpenBLASConfig.cmake").write_text(
        "set(OpenBLAS_FOUND TRUE)\n"
        f'set(OpenBLAS_LIBRARIES "{library.as_posix()}")\n'
        f'set(OpenBLAS_INCLUDE_DIRS "{include.as_posix()}")\n'
    )
    env = {
        **os.environ,
        "CXX": native_cxx.compiler,
        "CMAKE_CXX_COMPILER_LAUNCHER": native_cxx.cache,
    }
    _, cache, configure = _probe_project(tmp_path, env=env)
    configure(provider_path=provider)
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LAPACKE") is not missing_symbol
