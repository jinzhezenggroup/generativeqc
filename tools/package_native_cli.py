#!/usr/bin/env python3
"""Package an installed Python-free CUDA CLI without the native SDK headers.

The GenerativeQC core is statically linked in the distribution preset. CUDA
providers and normal Linux runtime libraries remain external dependencies.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


def _dynamic_dependencies(executable: Path) -> list[str]:
    dynamic = subprocess.run(
        ["readelf", "-d", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(set(re.findall(r"Shared library: \[([^\]]+)\]", dynamic)))


def _copy_runtime_tree(prefix: Path, staging: Path) -> list[str]:
    executable = prefix / "bin" / "generativeqc"
    if not executable.is_file():
        raise ValueError(f"Expected an installed Linux ELF CLI at {executable}")
    with executable.open("rb") as binary:
        elf_magic = binary.read(4)
    if elf_magic != b"\x7fELF":
        raise ValueError(f"Expected an installed Linux ELF CLI at {executable}")
    dependencies = _dynamic_dependencies(executable)
    if any(name.startswith("libgenerativeqc.so") for name in dependencies):
        raise ValueError("Distribution CLI still requires the GenerativeQC shared core")
    shutil.copy2(executable, staging / "bin" / "generativeqc")

    # Supplemental stationary AOT kernels are loadable plug-ins. Keep their
    # SONAME symlinks and manifests, but never ship unused SDK static archives.
    for lib_root in ("lib", "lib64"):
        source_dir = prefix / lib_root
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.rglob("*")):
            if source.is_dir():
                continue
            if ".so" not in source.name and source.suffix != ".json":
                continue
            if source.name.startswith("libgenerativeqc.so"):
                raise ValueError(
                    "Static-core distribution must not ship libgenerativeqc.so"
                )
            rel = source.relative_to(prefix)
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(source.readlink())
            else:
                shutil.copy2(source, target)

    licenses = prefix / "share" / "generativeqc" / "licenses"
    if licenses.is_dir():
        shutil.copytree(licenses, staging / "share" / "generativeqc" / "licenses")

    return dependencies


def _archive_tree(source: Path, destination: Path, epoch: int) -> None:
    # Fixed metadata and gzip headers make identical installation trees
    # reproducible across workflow attempts (without copying archive-sized RAM).
    with destination.open("wb") as output:
        with gzip.GzipFile(
            fileobj=output, mode="wb", filename="", mtime=epoch
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                entries = [source, *sorted(source.rglob("*"))]
                for path in entries:
                    info = archive.gettarinfo(
                        str(path), arcname=str(path.relative_to(source.parent))
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    if info.isfile():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def package_native_cli(
    *,
    prefix: Path,
    source_root: Path,
    output_dir: Path,
    version: str,
    commit: str,
    release: bool,
    source_date_epoch: int,
) -> Path:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("A CMake numeric X.Y.Z project version is required")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full lowercase Git commit SHA is required")
    if source_date_epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be nonnegative")

    suffix = "" if release else f"+g{commit[:12]}"
    basename = f"generativeqc-v{version}{suffix}-linux-x86_64-cuda12.9-sm120"
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / basename
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "bin").mkdir(parents=True)
    dependencies = _copy_runtime_tree(prefix, staging)

    for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(source_root / notice, staging / notice)
    shutil.copytree(source_root / "LICENSES", staging / "LICENSES")
    metadata = {
        "version": version,
        "commit": commit,
        "platform": "linux-x86_64",
        "cuda_toolkit": "12.9",
        "cuda_architecture": "sm_120",
        "core_linkage": "static",
        "generativeqc_shared_core_required": False,
        "elf_needed": dependencies,
        "source_date_epoch": source_date_epoch,
    }
    (staging / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "README.txt").write_text(
        f"GenerativeQC v{version}  ({commit})\n"
        "Native CUDA 12.9 / sm_120 CLI for Linux x86_64\n\n"
        "Run: ./bin/generativeqc --version\n"
        "     ./bin/generativeqc methods --json\n\n"
        "The C++/CUDA GenerativeQC core is linked into bin/generativeqc;\n"
        "Python is NOT required at runtime. The external Linux libraries\n"
        "shown by 'readelf -d bin/generativeqc' must be installed on the host.\n"
        "In particular, install CUDA 12.9 runtime/math libraries and a\n"
        "compatible NVIDIA driver (libcuda.so.1). System OpenBLAS, OpenSSL,\n"
        "libarchive, glibc and libstdc++ may also be required. This package\n"
        "does not bundle NVIDIA or other system-provider libraries.\n"
        "If your CUDA installation is non-standard, configure the loader's\n"
        "library search path before launching. sm_120 targets Blackwell GPUs;\n"
        "this artifact is NOT a universal CUDA fat binary.\n"
        "Any shipped lib/ objects contain separately compiled stationary AOT\n"
        "plug-ins/manifests; keep these next to the executable.\n"
        "License and third-party notices are included in this archive.\n",
        encoding="utf-8",
    )

    archive_path = output_dir / f"{basename}.tar.gz"
    _archive_tree(staging, archive_path, source_date_epoch)
    checksum = hashlib.sha256()
    with archive_path.open("rb") as archive_stream:
        for chunk in iter(lambda: archive_stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    (output_dir / "SHA256SUMS").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    shutil.rmtree(staging)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    output = package_native_cli(
        prefix=args.prefix,
        source_root=args.source_root,
        output_dir=args.output_dir,
        version=args.version,
        commit=args.commit,
        release=args.release,
        source_date_epoch=args.source_date_epoch,
    )
    print(f"Prepared {output} and {output.parent / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
