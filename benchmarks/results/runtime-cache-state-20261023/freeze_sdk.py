"""Bind completed CPU-built SDK and independently retained scientific inputs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import lzma
import os
import runpy
import subprocess
import sys
from pathlib import Path

from benchmarks.runtime_cache_state import (
    SCHEMA,
    digest,
    identity,
    save,
    validate_config,
)


def file_record(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qm7", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source"
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("requires real owned CPU Slurm allocation")
    if not (root / "receipts/build-complete").is_file():
        raise ValueError("native SDK build is incomplete")
    output = root / "frozen-inputs"
    output.mkdir(exist_ok=False)
    from benchmarks.readme_cbs import geometry, setup

    atoms = geometry(args.qm7, 2)
    protocol, orbital, auxiliary = setup(atoms, 2, 32 << 30, "df")
    basis_paths = {}
    for name, bases in (("orbital", orbital), ("auxiliary", auxiliary)):
        basis_paths[name + "_paths"] = []
        for index, basis in enumerate(bases):
            path = output / f"{name}-{index}.json"
            basis.write(path)
            basis_paths[name + "_paths"].append(str(path))
    summary_path = (
        source / "benchmarks/results/cbs-fast-lambda-20261021/summary.json.xz"
    )
    summary = json.loads(lzma.decompress(summary_path.read_bytes()))
    reference = summary["points"][1]["reference"]
    if protocol != reference["protocol"]:
        raise ValueError(
            "frozen CBS scientific input differs from independent reference"
        )
    reference_root = source / "benchmarks/results/readme-final-20261022"
    oracles = json.loads(
        lzma.decompress((reference_root / "oracles.json.xz").read_bytes())
    )
    water = [
        r
        for r in oracles
        if r["method"] in ("pbe0", "wb97m-v") and r["atoms"] in (3, 12, 48)
    ]
    source_manifest = json.loads((root / "source-ececb48-manifest.json").read_text())
    if (
        json.loads((root / "receipts/source-verified.json").read_text())["status"]
        != "PASS"
    ):
        raise ValueError("full Git LF source verification incomplete")
    # Configuration/code generation must not silently turn the measured SDK
    # into a dirty source snapshot while retaining the original Git revision.
    for item in source_manifest["files"]:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or ".env" in relative.parts
            or relative.name.startswith(".env.")
        ):
            raise ValueError("unsafe source inventory member")
        path = source / relative
        if path.is_symlink():
            raise ValueError("source inventory unexpectedly contains a symbolic link")
        data = path.read_bytes()
        oid = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
        ).hexdigest()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if (
            hashlib.sha256(data).hexdigest() != item["sha256"]
            or oid != item["git_oid"]
            or mode != item["mode"]
        ):
            raise ValueError(f"post-build Git LF/source mode mismatch: {relative}")
    save(
        root / "receipts/source-postbuild-verified.json",
        {
            "revision": source_manifest["revision"],
            "files": len(source_manifest["files"]),
            "status": "PASS",
        },
    )
    paths = [
        source / item["path"]
        for item in source_manifest["files"]
        if Path(item["path"]).parts[0]
        in ("python", "src", "include", "manifests", "cmake")
    ]
    sdk = root / "sdk/lib"
    libraries = {
        path.resolve() for path in sdk.glob("libgenerativeqc*.so*") if path.is_file()
    }
    manifests = {
        path.resolve()
        for path in sdk.glob("generativeqc_stationary*.json")
        if path.is_file()
    }
    profiles = (
        subprocess.check_output(
            [
                sys.executable,
                str(source / "tools/generate_native_stationary_cuda.py"),
                "--list-aot-profiles",
            ],
            text=True,
        )
        .strip()
        .split(";")
    )
    expected_manifests = {
        f"generativeqc_stationary_{profile}{suffix}.json"
        for profile in profiles
        for suffix in ("", "_spd")
    }
    if {path.name for path in manifests} != expected_manifests:
        raise ValueError(
            "installed ordinary/SPD AOT manifests differ from compiler profile catalog"
        )
    if {"lib" + name.removesuffix(".json") + ".so" for name in expected_manifests} - {
        path.name for path in libraries
    }:
        raise ValueError("missing installed stationary AOT libraries")
    from generativeqc_compiler.method.stationary_cuda import (
        stationary_aot_manifest_integrity,
    )

    for path in sorted(manifests):
        manifest = json.loads(path.read_text())
        binary = path.with_name("lib" + path.stem + ".so")
        if manifest.get(
            "manifest_integrity_sha256"
        ) != stationary_aot_manifest_integrity(manifest):
            raise ValueError("stationary AOT manifest integrity mismatch")
        if (
            manifest["binary_sha256"] != digest(binary)
            or manifest["binary_bytes"] != binary.stat().st_size
        ):
            raise ValueError("stationary AOT manifest/binary mismatch")
        if manifest["architectures"] != ["sm_120"] or manifest[
            "compile_architectures"
        ] != ["120-real"]:
            raise ValueError(
                "stationary AOT architecture differs from frozen sm120 build"
            )
    cache_file = source / "build/cuda-release-sm120/CMakeCache.txt"
    controls = {}
    for line in cache_file.read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("/", "#")):
            name, value = line.split("=", 1)
            controls[name.split(":", 1)[0]] = value
    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "GENERATIVEQC_CUDA_FAST_COMPILE": "OFF",
        "GENERATIVEQC_ENABLE_AOT_SHELLS": "ON",
        "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA": "ON",
        "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT": "ON",
        "GENERATIVEQC_AOT_PROFILE": "sm_120",
    }
    if any(controls.get(name) != value for name, value in required.items()):
        raise ValueError("production build controls differ from frozen contract")
    command_path = root / "receipts/compiler-commands.txt"
    commands = command_path.read_text().splitlines()
    compiles = [
        line
        for line in commands
        if " -c " in line and ("/nvcc " in line or "/c++ " in line)
    ]
    if not compiles or any(
        str(root / "runtime-env/bin/sccache") not in line for line in compiles
    ):
        raise ValueError(
            "Ninja translation-unit commands bypass selected compiler cache"
        )
    paths.extend(libraries | manifests)
    library = (sdk / "libgenerativeqc.so").resolve()
    native = ctypes.CDLL(str(library))
    native.generativeqc_get_source_identity.argtypes = []
    native.generativeqc_get_source_identity.restype = ctypes.c_char_p
    native_identity = native.generativeqc_get_source_identity().decode()
    build_identity = runpy.run_path(str(source / "tools/generate_build_identity.py"))
    expected = build_identity["_source_identity"](
        source,
        build_identity["_inventory"](
            source, source / "cmake/GenerativeQCSourceIdentity.json"
        ),
    )
    if native_identity != expected:
        raise ValueError("loaded core identity differs from frozen source inventory")
    input_paths = [
        args.qm7,
        summary_path,
        reference_root / "oracles.json.xz",
        reference_root / "reference-reuse.json",
        *(Path(p) for name in basis_paths.values() for p in name),
    ]
    config = {
        "schema": SCHEMA,
        "policy": "standard",
        "production_source_revision": source_manifest["revision"],
        "install_kind": "build-tree",
        "installation_layout": "Git-frozen source-tree Python with CMake-installed native core/AOT SDK",
        "python_paths": [str(source / "python")],
        "library": str(library),
        "library_sha256": digest(library),
        "native_source_identity": native_identity,
        "sdk_files": [file_record(path) for path in sorted(set(paths))],
        "sdk_provenance": {
            "root": str(root),
            "source_manifest": file_record(root / "source-ececb48-manifest.json"),
            "build_commands": file_record(root / "receipts/compiler-commands.txt"),
            "aot_library_count": len(libraries) - 1,
            "aot_manifest_count": len(manifests),
            "build_controls": {name: controls[name] for name in required},
            "cache_wrapped_translation_units": len(compiles),
        },
        "input_files": [file_record(path) for path in input_paths],
        "cbs_inputs": {"protocol": protocol, **basis_paths},
        "cbs_reference": reference,
        "cbs_reference_sha256": identity(reference),
        "water_references": water,
        "reference_reuse": json.loads(
            (reference_root / "reference-reuse.json").read_text()
        ),
    }
    environment_path = root / "receipts/scientific-environment.json"
    environment = json.loads(environment_path.read_text())
    config["scientific_environment"] = {
        "manifest": file_record(environment_path),
        "python_executable_sha256": environment["python_executable_sha256"],
        "libxc_version": environment["libxc_version"],
        "packages": environment["packages"],
    }
    validate_config(config)
    save(output / "config.json", config)
    print(
        json.dumps(
            {
                "config_sha256": digest(output / "config.json"),
                "sdk_files": len(config["sdk_files"]),
                "source_revision": source_manifest["revision"],
                "native_source_identity": native_identity,
                "library_sha256": config["library_sha256"],
                "gpu_qualification": "NOT_RUN",
            }
        )
    )


if __name__ == "__main__":
    main()
