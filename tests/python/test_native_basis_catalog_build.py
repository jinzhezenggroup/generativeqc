"""Bundled native data must retain build identity and redistribution notices."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc import autotune

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "tools/generate_cli_basis_pack.py",
        "python/generativeqc/data/basis_pack.json",
    ],
)
def test_bundled_catalog_inputs_change_native_source_identity(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    before = autotune.source_identity(ROOT)
    original = autotune.file_hash
    changed = ROOT / relative
    monkeypatch.setattr(
        autotune,
        "file_hash",
        lambda path: "changed-input" if path == changed else original(path),
    )
    assert autotune.source_identity(ROOT) != before


@pytest.mark.skipif(sys.platform != "linux", reason="ELF installation fixture")
def test_cli_disabled_library_installs_bundled_data_notice(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the native install contract")
    build, prefix = tmp_path / "build", tmp_path / "install"
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DGENERATIVEQC_ENABLE_CUDA=OFF",
            "-DGENERATIVEQC_BUILD_CLI=OFF",
            "-DGENERATIVEQC_BUILD_TESTS=OFF",
            "-DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar",
            "-DGENERATIVEQC_PYTHON_WHEEL=ON",
            "-DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF",
            "-DCMAKE_INSTALL_DATADIR=share",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Exercise the real install graph without rebuilding scientific code. The
    # wheel layout provides one unversioned ELF artifact; /bin/true is only an
    # explicit packaging fixture, never executed or treated as a native library.
    shutil.copyfile("/bin/true", build / "libgenerativeqc.so")
    for component in (None, "Unspecified"):
        destination = prefix / (component or "default")
        command = [cmake, "--install", str(build), "--prefix", str(destination)]
        if component is not None:
            command.extend(["--component", component])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        notice = destination / "share/generativeqc/licenses/bse-data-BSD-3-Clause.txt"
        assert (
            notice.read_bytes()
            == (ROOT / "LICENSES/bse-data-BSD-3-Clause.txt").read_bytes()
        )
        assert list(destination.rglob("libgenerativeqc.so"))
        assert (destination / "include/generativeqc/generativeqc.h").is_file()
