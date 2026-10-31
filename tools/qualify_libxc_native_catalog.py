"""Run the existing native CPU/CUDA producers and retain one offline DAG bundle.

CUDA requires a finite Slurm allocation. Independent PySCF/Libxc references are
qualification dependencies only. Failed campaigns are retained without promotion.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT), str(ROOT / "tests/python")]

from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.xc.libxc_bulk_capabilities import available_capabilities

from tools.generate_build_identity import _inventory, _source_identity
from tools.generativeqc_validation.libxc_catalog import (
    BUNDLE_SCHEMA,
    load_json,
    require,
    source_inventory,
    write_json,
)
from tools.generativeqc_validation.libxc_native_domain import (
    native_domain_stage,
    qualify_native_domain,
)
from tools.qualify_libxc_forces import qualify_forces


def verify_native_build(build: Path) -> dict[str, Any]:
    """Read the embedded native owner's source identity from the actual ELF."""
    path = (build / "libgenerativeqc.so").resolve(strict=True)
    library = ctypes.CDLL(str(path))
    query = library.generativeqc_get_source_identity
    query.argtypes = []
    query.restype = ctypes.c_char_p
    actual = query().decode()
    expected = _source_identity(
        ROOT, _inventory(ROOT, ROOT / "cmake/GenerativeQCSourceIdentity.json")
    )
    require(
        actual == expected,
        "loaded native library was built from different source inputs",
    )
    return {
        "source_identity": actual,
        "measured_input_identity": expected,
        "library": {"sha256": file_hash(path), "bytes": path.stat().st_size},
    }


def build_receipt(build: Path, source_identity: str) -> dict:
    """Retain the actual cache-wrapped compiler commands and executed ELF identity."""
    library = (build / "libgenerativeqc.so").resolve(strict=True)
    # CMake's compilation database omits launchers. Ninja's executable command
    # stream is the provenance of cache use; verify every actual compile there.
    command_stream = subprocess.run(
        [
            next(
                line.split("=", 1)[1]
                for line in (build / "CMakeCache.txt").read_text().splitlines()
                if line.startswith("CMAKE_MAKE_PROGRAM:FILEPATH=")
            ),
            "-C",
            str(build),
            "-t",
            "commands",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commands = [
        line
        for line in command_stream.splitlines()
        if " -c " in line
        and any(token in line for token in ("/g++ ", "/nvcc ", "/c++ "))
    ]
    require(
        commands and all("sccache" in row or "ccache" in row for row in commands),
        "native compile commands did not all use compiler cache",
    )
    launchers = {
        token
        for command in commands
        for token in shlex.split(command)
        if Path(token).name in ("sccache", "ccache")
    }
    require(
        len(launchers) == 1, "native build must identify one verified compiler launcher"
    )
    cache = subprocess.run(
        [launchers.pop(), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    compilers = {}
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            values[key.split(":")[0]] = value
    for key in ("CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER"):
        if key not in values or not values[key] or values[key].endswith("-NOTFOUND"):
            continue
        compiler = Path(values[key])
        compilers[key] = {
            "sha256": file_hash(compiler),
            "version": subprocess.run(
                [str(compiler), "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        }
    return {
        "source_identity": source_identity,
        "library": {"sha256": file_hash(library), "bytes": library.stat().st_size},
        "toolchain": compilers,
        "compiler_cache": {"version": cache, "verified_commands": len(commands)},
        "compile_commands_sha256": canonical_hash(commands),
        "configuration_sha256": file_hash(build / "CMakeCache.txt"),
        "configuration": {
            k: v for k, v in values.items() if k.startswith("GENERATIVEQC_")
        },
    }


def execution_environment(*, cuda: bool) -> dict[str, Any]:
    """Record the actual oracle and assigned hardware before measurement starts."""
    import numpy
    import pyscf
    from pyscf.dft import libxc

    require(
        pyscf.__version__ == "2.14.0" and libxc.__version__ == "7.0.0",
        "qualification requires the pinned PySCF 2.14.0 / Libxc 7.0.0 oracle",
    )
    result = {
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pyscf": pyscf.__version__,
        "libxc": libxc.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if cuda:
        require(
            result["slurm_job_id"] and result["cuda_visible_devices"],
            "CUDA qualification requires scheduler visibility",
        )
        # Query only the allocation; never change Slurm's visibility mapping.
        result["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                f"--id={result['cuda_visible_devices']}",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--build-receipt",
        type=Path,
        help="reuse an authenticated build receipt when executing the identical ELF on another node",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--name", action="append")
    parser.add_argument("--verify-build-only", action="store_true")
    args = parser.parse_args()
    from test_libxc_automatic_point_native import NAMES

    args.output.mkdir(parents=True, exist_ok=True)
    native_verification = verify_native_build(args.build_dir)
    if args.verify_build_only:
        write_json(args.output / "native-build-verification.json", native_verification)
        return 0
    sources = source_inventory(ROOT)
    source_id = canonical_hash(sources)
    # The full census is a local diagnostic. Published receipts need its count
    # and digest; live ingestion recomputes those from the measured source.
    write_json(args.output / "source-inputs.json.xz", sources)
    build = (
        load_json(args.build_receipt)
        if args.build_receipt
        else build_receipt(args.build_dir, source_id)
    )
    require(
        build["source_identity"] == source_id,
        "build receipt is for different source inputs",
    )
    require(
        build["library"]["sha256"]
        == file_hash((args.build_dir / "libgenerativeqc.so").resolve()),
        "build receipt is for a different native ELF",
    )
    os.environ["GENERATIVEQC_LIBRARY"] = str(
        (args.build_dir / "libgenerativeqc.so").resolve()
    )
    inventory = {c.name: c for c in available_capabilities()}
    names = args.name or NAMES
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "source_commit": args.source_commit,
        "source_input_count": len(sources),
        "source_identity": source_id,
        "build": build,
        "native_build_verification": native_verification,
        "executions": [
            {
                "names": list(names),
                "backend": args.backend,
                "environment": execution_environment(cuda=args.backend != "cpu"),
            }
        ],
        "functionals": {},
        "failures": [],
    }
    for name in names:
        data: dict[str, Any] = {"subject_identity": inventory[name].identity}
        bundle["functionals"][name] = data
        for stage in ("production-domain", "cpu", "cuda"):
            if stage != "production-domain" and args.backend not in (stage, "all"):
                continue
            try:
                if stage == "production-domain":
                    data[stage] = qualify_native_domain(
                        name, ROOT, args.output / "domain"
                    )
                    domain = native_domain_stage(name, data[stage])
                    require(domain["status"] == "pass", domain["reason"])
                elif stage == "cpu":
                    evidence = f"native-catalog/{name}/cpu"
                    data[stage] = {
                        **qualify_forces(name, evidence=evidence),
                        "evidence": evidence,
                    }
                else:
                    require(
                        bool(os.environ.get("SLURM_JOB_ID"))
                        and bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
                        "CUDA qualification requires scheduler visibility",
                    )
                    import pytest

                    producer = importlib.import_module("test_libxc_native_cuda_forces")
                    directory = args.output / "cuda"
                    os.environ["GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR"] = str(
                        directory
                    )
                    data[stage] = []
                    for spins in (1, 2):
                        with pytest.MonkeyPatch.context() as patch:
                            producer._lifecycle(
                                name, spins, patch, full_vector=True, replay=True
                            )
                        data[stage].append(
                            load_json(
                                directory / f"{name}-{spins}-True-True-False-h4.json"
                            )
                        )
                print(f"{name} {stage}: retained", flush=True)
            except Exception as exc:  # noqa: BLE001 -- persist each failure and continue the matrix
                import traceback

                traceback.print_exc()
                bundle["failures"].append(
                    {
                        "name": name,
                        "stage": stage,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            require(
                source_inventory(ROOT) == sources, "source changed during qualification"
            )
            require(
                file_hash((args.build_dir / "libgenerativeqc.so").resolve())
                == build["library"]["sha256"],
                "native binary changed during qualification",
            )
            write_json(args.output / "campaign.json.xz", bundle)
    return int(bool(bundle["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
