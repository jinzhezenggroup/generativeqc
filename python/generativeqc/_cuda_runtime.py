"""Discover optional NVIDIA user-space runtimes for installed CUDA wheels."""

from __future__ import annotations

import ctypes
import functools
import os
import platform
import shutil
import site
import sys
import sysconfig
from pathlib import Path

# Keep this list in dependency-friendly load order. libcuda is deliberately
# absent: the NVIDIA kernel driver must come from the system, never a toolkit
# stub or a Python package.
_CUDA_RUNTIME_LIBRARY_GROUPS = (
    ("libnvJitLink.so.12",),
    ("libcudart.so.12",),
    ("libcublasLt.so.12",),
    ("libcublas.so.12",),
    ("libcusparse.so.12",),
    ("libcusolver.so.11",),
)

_cuda_runtime_handles: dict[str, ctypes.CDLL] = {}


def _runtime_search_dirs() -> list[Path]:
    """Return known locations for optional CUDA user-space providers."""
    if not sys.platform.startswith("linux"):
        return []

    site_packages = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site and (site.ENABLE_USER_SITE or user_site in sys.path):
        site_packages.append(user_site)

    # Explicit deployment locations take precedence over bundled providers.
    # Empty loader-path components must not introduce the working directory.
    dirs = [
        Path(entry).expanduser()
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    machine = platform.machine()
    target = "aarch64" if machine == "arm64" else machine
    roots: list[str | None] = []
    # A compiler selected outside PATH also identifies its runtime prefix.
    # This is filesystem discovery only; no compiler or CUDA context is started.
    configured_compiler = os.environ.get("GENERATIVEQC_NVCC") or os.environ.get(
        "CUDACXX"
    )
    if configured_compiler and (
        compiler := shutil.which(str(Path(configured_compiler).expanduser()))
    ):
        roots.append(str(Path(compiler).resolve().parent.parent))
    roots.extend(
        os.environ.get(name) for name in ("CUDAToolkit_ROOT", "CUDA_PATH", "CUDA_HOME")
    )
    for root in roots:
        if root:
            prefix = Path(root).expanduser()
            dirs.extend(
                (
                    prefix / "lib64",
                    prefix / "lib",
                    prefix / "targets" / f"{target}-linux" / "lib",
                )
            )
    # Conda/venv installs need not live below /usr/local or site-packages/nvidia.
    dirs.append(Path(sys.prefix) / "lib")
    for entry in dict.fromkeys(site_packages):
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        dirs.extend(
            provider / "lib"
            for provider in nvidia_root.iterdir()
            if (provider / "lib").is_dir()
        )

    multiarch = sysconfig.get_config_var("MULTIARCH")
    for candidate in (
        "/usr/local/cuda/lib64",
        f"/usr/local/cuda/targets/{target}-linux/lib",
        "/usr/local/lib",
        f"/usr/lib/{multiarch}" if multiarch else "/usr/lib",
    ):
        path = Path(candidate)
        if path.is_dir():
            dirs.append(path)

    # Preserve discovery order while avoiding repeated dlopen attempts.
    return list(dict.fromkeys(path.resolve() for path in dirs if path.is_dir()))


def preload_cuda_runtime_libraries() -> tuple[str, ...]:
    """Register optional CUDA providers by SONAME before libgenerativeqc is loaded."""
    if not sys.platform.startswith("linux"):
        return ()

    loaded: list[str] = []
    # Retained handles already keep successfully loaded providers alive. Only
    # discover directories when a group is missing; retry missing providers on
    # every call so a later installation or repaired library remains visible.
    search_dirs: list[Path] | None = None
    for alternatives in _CUDA_RUNTIME_LIBRARY_GROUPS:
        if any(name in _cuda_runtime_handles for name in alternatives):
            continue
        if search_dirs is None:
            search_dirs = _runtime_search_dirs()
        for name in alternatives:
            for directory in search_dirs:
                candidate = directory / name
                if not candidate.is_file():
                    continue
                try:
                    handle = ctypes.CDLL(str(candidate), mode=os.RTLD_LOCAL)
                except OSError:
                    continue
                _cuda_runtime_handles[name] = handle
                loaded.append(name)
                break
            if name in _cuda_runtime_handles:
                break
    return tuple(loaded)


def install_native_loader() -> None:
    """Wrap ``generativeqc._native.load_library`` with CUDA provider discovery."""
    from . import _native

    original = _native.load_library
    if getattr(original, "_generativeqc_cuda_runtime_loader", False):
        return

    @functools.wraps(original)
    def load_library(*, device: str | None = None, device_id: int = 0) -> ctypes.CDLL:
        preload_cuda_runtime_libraries()
        return original(device=device, device_id=device_id)

    load_library._generativeqc_cuda_runtime_loader = True  # type: ignore[attr-defined]
    _native.load_library = load_library
