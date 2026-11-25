"""Build the staged release library and stationary profiles with sccache on qz."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from tools.generate_build_identity import _inventory, _source_identity

if TYPE_CHECKING:
    from types import FrameType

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BUILD = ROOT / "build"
RECEIPTS = ROOT / "build-receipts"
BASE = ROOT.parent / "h100-portability/20261027-fd06"
PYTHON = BASE / "env/bin/python"
CUDA = Path("/opt/generativeqc-reuse/cuda-12.9")
PROVIDER = BASE / "env/lib/python3.11/site-packages/scipy_openblas32/lib"
LAUNCHER = PYTHON.with_name("sccache")
PROFILES = ("pbe_rks", "r2scan_rks", "pbe0_rks", "wb97mv_rks")
REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stop(signum: int, _frame: FrameType | None) -> NoReturn:
    raise SystemExit(128 + signum)


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> None:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    RECEIPTS.mkdir()
    os.environ.update(
        SCCACHE_DIR=str(BASE / "cache/sccache"),
        SCCACHE_SERVER_UDS=str(ROOT / "build-cache.sock"),
        SCCACHE_IDLE_TIMEOUT="1200",
        NVCC_CCBIN="/usr/bin/g++-12",
    )
    archive_sha = sha256(ROOT / "source.tar.gz")
    if archive_sha != (ROOT / "source.sha256").read_text().split()[0]:
        raise ValueError("staged source archive checksum differs")
    native = _source_identity(
        SOURCE, _inventory(SOURCE, SOURCE / "cmake/GenerativeQCSourceIdentity.json")
    )
    version = subprocess.check_output(
        [LAUNCHER, "--version"], text=True, timeout=20
    ).strip()
    if version != "sccache 0.16.0":
        raise ValueError(f"unexpected compiler cache version: {version}")
    receipt = {
        "status": "running",
        "source_revision": REVISION,
        "source_archive_sha256": archive_sha,
        "native_source_identity": native,
        "sccache": version,
        "profiles": PROFILES,
        "stages": [],
    }

    def save() -> None:
        temporary = RECEIPTS / "build.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(RECEIPTS / "build.json")

    def run(argv: list, name: str, timeout: float) -> None:
        row = {"stage": name, "argv": list(map(str, argv)), "status": "running"}
        receipt["stages"].append(row)
        save()
        started = time.monotonic()
        print(f"START build {name}", flush=True)
        with (RECEIPTS / f"{name}.txt").open("x") as log:
            process = subprocess.Popen(
                row["argv"],
                cwd=SOURCE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout)
            finally:
                terminate(process)
        row.update(
            status="pass" if code == 0 else "fail",
            returncode=code,
            seconds=time.monotonic() - started,
        )
        save()
        print(f"END build {name} {code}", flush=True)
        if code:
            raise RuntimeError(f"build {name} failed with status {code}")

    save()
    with (RECEIPTS / "cache-server.txt").open("x") as log:
        server = subprocess.Popen(
            [LAUNCHER],
            env={**os.environ, "SCCACHE_START_SERVER": "1", "SCCACHE_NO_DAEMON": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(100):
                if Path(os.environ["SCCACHE_SERVER_UDS"]).exists():
                    break
                if server.poll() is not None:
                    raise RuntimeError("owned cache server exited")
                time.sleep(0.1)
            else:
                raise TimeoutError("owned cache server never bound its socket")
            run([LAUNCHER, "--show-stats"], "cache-before", 30)
            configure = [
                PYTHON,
                "-m",
                "cmake",
                "-S",
                SOURCE,
                "-B",
                BUILD,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
                f"-DCMAKE_CUDA_COMPILER={CUDA}/bin/nvcc",
                "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={LAUNCHER}",
                f"-DCMAKE_CUDA_COMPILER_LAUNCHER={LAUNCHER}",
                f"-DPython3_EXECUTABLE={PYTHON}",
                f"-DOpenBLAS_DIR={PROVIDER}/cmake/openblas",
                "-DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas",
                "-DCMAKE_CUDA_ARCHITECTURES=90",
                "-DGENERATIVEQC_CUDA_ARCHITECTURES=90",
                "-DGENERATIVEQC_ENABLE_CUDA=ON",
                "-DGENERATIVEQC_ENABLE_AOT_SHELLS=ON",
                "-DGENERATIVEQC_AOT_PROFILE=portable_cuda",
                "-DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON",
                "-DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON",
                "-DGENERATIVEQC_STATIONARY_AOT_PROFILES=" + ";".join(PROFILES),
                "-DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF",
                "-DGENERATIVEQC_BUILD_TESTS=OFF",
                "-DGENERATIVEQC_BUILD_CLI=OFF",
                "-DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=ON",
                "-DGENERATIVEQC_CUDA_COMPILE_JOBS=4",
                "-DGENERATIVEQC_AOT_COMPILE_JOBS=4",
                "-DGENERATIVEQC_AOT_SHARDS=8",
            ]
            run(configure, "configure", 1200)
            targets = ["generativeqc"] + [
                f"generativeqc_stationary_{profile}{suffix}_manifest"
                for profile in PROFILES
                for suffix in ("", "_spd")
            ]
            ninja = [PYTHON, "-m", "ninja", "-C", BUILD]
            commands = subprocess.check_output(
                [*ninja, "-t", "commands", *targets], text=True, timeout=120
            )
            selected = [
                line
                for line in commands.splitlines()
                if " -c " in line and ("g++-12" in line or "/nvcc" in line)
            ]
            if not selected or any(str(LAUNCHER) not in line for line in selected):
                raise ValueError("compiler invocation without verified cache launcher")
            receipt["compiler_commands_sha256"] = hashlib.sha256(
                commands.encode()
            ).hexdigest()
            receipt["cached_compiler_commands"] = len(selected)
            run([*ninja, "-j12", *targets], "compile", 10800)
            core = BUILD / "libgenerativeqc.so"
            library = ctypes.CDLL(str(core))
            library.generativeqc_get_source_identity.restype = ctypes.c_char_p
            if library.generativeqc_get_source_identity().decode() != native:
                raise ValueError(
                    "built native source identity differs from staged source"
                )
            audit = [
                PYTHON,
                SOURCE / "tools/audit_stationary_aot_package.py",
                "--directory",
                BUILD,
                "--architecture",
                "sm_90",
                "--native-library",
                core,
                "--require-native-cubin",
                "--output",
                RECEIPTS / "aot-audit.json",
            ]
            for profile in PROFILES:
                audit += ["--profile", profile]
            run(audit, "aot-audit", 600)
            run([LAUNCHER, "--show-stats"], "cache-after", 30)
            receipt.update(status="complete", library_sha256=sha256(core))
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()
            terminate(server)


if __name__ == "__main__":
    sys.exit(main())
