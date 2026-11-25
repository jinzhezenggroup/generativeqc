"""Build the pinned master on qz H100 with the paper's SM90/AOT settings."""

import ctypes
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from acquire import digest, interrupt, save, stop_process

PROFILES = ("pbe_rks", "r2scan_rks", "wb97mv_rks")


def source_identity(source):
    location = source / "tools/generate_build_identity.py"
    spec = importlib.util.spec_from_file_location("identity_owner", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = module._inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    return module._source_identity(source, inputs), len(inputs)


def build(root):
    source = root / "source"
    contract = json.loads((root / "source.json").read_text())
    paper = json.loads((root / "paper-config.json").read_text())
    old_environment = paper["environment"]
    python = Path(paper["interpreters"]["standard"])
    launcher = python.with_name("sccache")
    cuda = Path("/opt/generativeqc-reuse/cuda-12.9")
    provider = (
        python.parent.parent / "lib/python3.11/site-packages/scipy_openblas32/lib"
    )
    directory = root / "build"
    receipts = root / "build-receipts"
    receipts.mkdir()
    socket = root / "cc.s"
    env = {**os.environ, **old_environment}
    env.update(
        PYTHONPATH=f"{source}:{source}/python",
        LD_LIBRARY_PATH=f"{provider}:{cuda}/lib64:{cuda}/targets/x86_64-linux/lib",
        SCCACHE_SERVER_UDS=str(socket),
        SCCACHE_IDLE_TIMEOUT="1200",
        NVCC_CCBIN="/usr/bin/g++-12",
    )
    os.environ.update(env)
    report = {
        "status": "checking_source",
        "source_revision": contract["revision"],
        "source_archive_sha256": contract["source_archive_sha256"],
        "stages": [],
    }
    journal = receipts / "build.json"
    save(journal, report)

    def run(name, argv, timeout):
        row = {"name": name, "argv": list(map(str, argv)), "status": "running"}
        report["stages"].append(row)
        save(journal, report)
        print("START", name, flush=True)
        started = time.monotonic()
        with (receipts / f"{name}.txt").open("xb") as log:
            child = subprocess.Popen(
                row["argv"],
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                child.wait(timeout=timeout)
            finally:
                if child.poll() is None:
                    stop_process(child)
                row.update(
                    seconds=time.monotonic() - started, returncode=child.returncode
                )
                row["status"] = "complete" if child.returncode == 0 else "failed"
                save(journal, report)
        if child.returncode:
            raise RuntimeError(f"{name} failed ({child.returncode})")
        print("END", name, round(row["seconds"], 2), flush=True)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    server = None
    try:
        if digest(root / "source.tar.gz") != contract["source_archive_sha256"]:
            raise ValueError("source archive checksum differs")
        identity, count = source_identity(source)
        if (identity, count) != (
            contract["native_source_identity"],
            contract["native_identity_inputs"],
        ):
            raise ValueError("extracted native source identity differs")
        for name, expected in {
            **contract["inputs_sha256"],
            "geometries.json": contract["geometry_sha256"],
            "source/benchmarks/readme_defaults.py": contract["paper_runner_sha256"],
            "source/benchmarks/readme_defaults_accuracy.py": contract[
                "cpu_oracle_driver_sha256"
            ],
        }.items():
            if digest(root / name) != expected:
                raise ValueError(f"source or benchmark input differs: {name}")
        version = subprocess.check_output(
            [launcher, "--version"], text=True, timeout=20
        ).strip()
        if version != "sccache 0.16.0":
            raise ValueError(f"unexpected compiler cache version {version}")
        report.update(
            status="building", native_source_identity=identity, cache_version=version
        )
        with (receipts / "cache-server.txt").open("xb") as log:
            server = subprocess.Popen(
                [launcher],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**env, "SCCACHE_START_SERVER": "1", "SCCACHE_NO_DAEMON": "1"},
            )
        deadline = time.monotonic() + 15
        while not socket.exists():
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("private sccache server did not start")
            time.sleep(0.1)
        run("cache-before", [launcher, "--show-stats"], 30)
        configure = [
            python,
            "-m",
            "cmake",
            "-S",
            source,
            "-B",
            directory,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
            f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc",
            "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
            f"-DCMAKE_CUDA_COMPILER_LAUNCHER={launcher}",
            f"-DPython3_EXECUTABLE={python}",
            f"-DOpenBLAS_DIR={provider}/cmake/openblas",
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
        run("configure", configure, 1200)
        targets = ["generativeqc"] + [
            f"generativeqc_stationary_{profile}{variant}_manifest"
            for profile in PROFILES
            for variant in ("", "_spd")
        ]
        commands = subprocess.check_output(
            [python, "-m", "ninja", "-C", directory, "-t", "commands", *targets],
            text=True,
            timeout=120,
        )
        (receipts / "compiler-commands.txt").write_text(commands)
        compiles = [
            line
            for line in commands.splitlines()
            if " -c " in line and ("g++-12" in line or "/nvcc" in line)
        ]
        if not compiles or any(str(launcher) not in command for command in compiles):
            raise ValueError("compiler launches without the verified cache")
        run("build", [python, "-m", "ninja", "-C", directory, "-j8", *targets], 8400)
        library_path = directory / "libgenerativeqc.so"
        lib = ctypes.CDLL(str(library_path))
        lib.generativeqc_get_source_identity.restype = ctypes.c_char_p
        if lib.generativeqc_get_source_identity().decode() != identity:
            raise ValueError("loaded native source identity differs")
        audit = [
            python,
            source / "tools/audit_stationary_aot_package.py",
            "--directory",
            directory,
            "--architecture",
            "sm_90",
            "--native-library",
            library_path,
            "--require-native-cubin",
            "--output",
            receipts / "aot-audit.json",
        ]
        for profile in PROFILES:
            audit.extend(("--profile", profile))
        run("aot-audit", audit, 600)
        run("cache-after", [launcher, "--show-stats"], 30)
        report.update(status="complete", native_library_sha256=digest(library_path))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if server is not None:
                stop_process(server)
        finally:
            save(journal, report)


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())
