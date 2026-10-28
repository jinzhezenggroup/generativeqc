"""Measure real cold/hit compilation of the complete selected point/force package.

Only active AOT objects are removed between trials. Retain source generation,
shared compiler caches and other core objects; the measured compile scope is
reported explicitly, and linked/installed core size is inspected separately.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--scope", choices=["narrow", "full"], required=True)
p.add_argument("--label", required=True)
p.add_argument("--build", type=Path)
a = p.parse_args()
root = a.root.resolve()
driver = Path(__file__).resolve().parent
build = (
    (a.build if a.build.is_absolute() else root / a.build)
    if a.build
    else root / "build" / ("pro6000-cuda" if a.scope == "narrow" else "full-auto-cuda")
)
out = root / "evidence" / a.label
out.mkdir(parents=True, exist_ok=True)
for filename in ("qualify-build.py", "measure-build.py", "aot-resource-options.cmake"):
    (out / filename).write_bytes((driver / filename).read_bytes())
venv = Path("/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin")
env = {
    **os.environ,
    "PATH": f"{venv}:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin",
    "LD_LIBRARY_PATH": "/group/software/cuda-12.9.1/lib64:"
    + os.environ.get("LD_LIBRARY_PATH", ""),
}
subprocess.run([str(venv / "sccache"), "--version"], env=env, check=True)
args = [
    "cmake",
    "-S",
    str(root),
    "-B",
    str(build),
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
    "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++",
    "-DCMAKE_PREFIX_PATH=/home/jzzeng/miniconda3",
    "-DOPENSSL_ROOT_DIR=/home/jzzeng/miniconda3",
    "-Dnlohmann_json_DIR=/home/jzzeng/miniconda3/share/cmake/nlohmann_json",
    "-DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc",
    f"-DCMAKE_CXX_COMPILER_LAUNCHER={venv}/sccache",
    f"-DCMAKE_CUDA_COMPILER_LAUNCHER={venv}/sccache",
    f"-DPython3_EXECUTABLE={venv}/python",
    "-DGENERATIVEQC_ENABLE_CUDA=ON",
    "-DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real",
    "-DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF",
    "-DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON",
    "-DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF",
    "-DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF",
    "-DGENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON",
    "-DGENERATIVEQC_CUDA_COMPILE_JOBS=4",
    "-DGENERATIVEQC_AOT_COMPILE_JOBS=4",
    "-DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=2",
    "-DGENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS=2",
    f"-DCMAKE_PROJECT_INCLUDE={driver}/aot-resource-options.cmake",
    f"-DCMAKE_INSTALL_PREFIX={root}/install-{a.scope}",
]
if a.scope == "full":
    profiles = subprocess.check_output(
        [
            str(venv / "python"),
            str(root / "tools/generate_native_stationary_cuda.py"),
            "--list-automatic-aot-profiles",
        ],
        env=env,
        cwd=root,
        text=True,
    ).strip()
    args += [
        "-DGENERATIVEQC_BUILD_TESTS=OFF",
        "-DGENERATIVEQC_BUILD_CLI=ON",
        "-DGENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS=",
        "-DGENERATIVEQC_STATIONARY_AOT_PROFILES=" + profiles,
    ]
else:
    # Pin the narrow inventory even when reusing a previously full build tree.
    names = "LDA_C_VWN_4;LDA_C_CHACHIYO;LDA_C_PZ_MOD;GGA_X_APBE;GGA_X_PBE_SOL;GGA_X_RPBE;GGA_X_B86;GGA_C_APBE;MGGA_X_LTA;MGGA_X_GVT4;MGGA_C_TPSS"
    codes = (
        196639,
        196895,
        196618,
        196792,
        196724,
        196725,
        196711,
        196794,
        196809,
        196812,
        196839,
    )
    profiles = [
        "pbe_rks",
        "pbe_uks",
        *(f"libxc_{code}_{spin}" for code in codes for spin in ("rks", "uks")),
    ]
    args += [
        "-DGENERATIVEQC_BUILD_TESTS=ON",
        "-DGENERATIVEQC_BUILD_CLI=ON",
        "-DGENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS=" + names,
        "-DGENERATIVEQC_STATIONARY_AOT_PROFILES=" + ";".join(profiles),
    ]
with (out / "configure.log").open("w") as log:
    subprocess.run(
        args, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
    )
for tool in ("nvcc", "g++", "sccache", "cmake", "ninja"):
    (out / (tool + "-version.txt")).write_text(
        subprocess.check_output([tool, "--version"], env=env, text=True, timeout=30)
    )
(out / "host.json").write_text(
    subprocess.check_output(["lscpu", "--json"], env=env, text=True, timeout=30)
)
actual_commands = subprocess.check_output(
    [
        "ninja",
        "-C",
        str(build),
        "-t",
        "commands",
        "generativeqc_native_stationary",
        "generativeqc_libxc_semilocal_cuda",
    ],
    env=env,
    text=True,
    timeout=30,
)
(out / "commands.txt").write_text(actual_commands)
for command in actual_commands.splitlines():
    if " -c " in command and ("/bin/nvcc " in command or "/usr/bin/g++ " in command):
        assert "sccache " in command, command

# Generate before the timed compiler phases; retain its complete wall time.
t = time.perf_counter()
with (out / "generation.log").open("w") as log:
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "generativeqc_native_stationary_codegen",
            "generativeqc_libxc_semilocal_cuda_codegen",
            "-j",
            "4",
        ],
        env=env,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=7200,
    )
(out / "generation.json").write_text(
    json.dumps(
        {
            "seconds": time.perf_counter() - t,
            "scope": "selected force and XC generators; includes dependency generators when needed",
        },
        indent=2,
    )
    + "\n"
)
commands = json.loads((build / "compile_commands.json").read_text())
objects = []
for row in commands:
    cmd = row.get("arguments") or shlex.split(row["command"])
    target = row.get("output") or cmd[cmd.index("-o") + 1]
    if any(
        x in target
        for x in (
            "generativeqc_native_stationary.dir/",
            "generativeqc_libxc_semilocal_cuda.dir/",
            "generativeqc_stationary_sp_primitives.dir/",
        )
    ):
        objects.append(str(Path(row["directory"]) / target))
assert objects
(out / "scope.json").write_text(
    json.dumps(
        {
            "object_paths": objects,
            "cuda_pool": 4,
            "jobs": 20,
            "split_threads": 2,
            "generator_seconds_excluded": True,
        },
        indent=2,
    )
    + "\n"
)
cache = out / "compiler-cache"
if cache.exists():
    raise RuntimeError(
        "cold qualification cache already exists; choose a new label, never erase a cache"
    )
for mode in ("cold", "hit"):
    for path in objects:
        Path(path).unlink(missing_ok=True)
    subprocess.run(
        [
            str(venv / "python"),
            str(driver / "measure-build.py"),
            "--cache",
            str(cache),
            "--output",
            str(out / f"{mode}.json"),
            "--seconds",
            "7200",
            "--",
            "cmake",
            "--build",
            str(build),
            "-j",
            "20",
            "--target",
            "generativeqc_native_stationary",
            "generativeqc_libxc_semilocal_cuda",
        ],
        env=env,
        cwd=root,
        check=True,
    )
    (out / f"{mode}-link-cache-before.txt").write_text(
        subprocess.check_output(
            [str(venv / "sccache"), "--show-stats"], env=env, text=True, timeout=30
        )
    )
    t = time.perf_counter()
    with (out / f"{mode}-link.log").open("w") as log:
        subprocess.run(
            ["cmake", "--build", str(build), "-j", "20", "--target", "generativeqc"],
            env=env,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=7200,
        )
    (out / f"{mode}-link-cache-after.txt").write_text(
        subprocess.check_output(
            [str(venv / "sccache"), "--show-stats"], env=env, text=True, timeout=30
        )
    )
    (out / f"{mode}-link.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - t,
                "scope": "complete core target; includes unrelated core compilation on first build",
            },
            indent=2,
        )
        + "\n"
    )
if a.scope == "narrow":
    # Install the current front end too; the measured compiler scope above is
    # unchanged and excludes this qualification-driver preparation.
    with (out / "drivers.log").open("w") as log:
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "-j",
                "12",
                "--target",
                "generativeqc_cli",
                "generativeqc_dft_cuda_tests",
                "generativeqc_ks_cuda_tests",
            ],
            env=env,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=7200,
        )
if a.scope == "full":
    with (out / "drivers.log").open("w") as log:
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "-j",
                "12",
                "--target",
                "generativeqc_cli",
            ],
            env=env,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=7200,
        )
with (out / "install.log").open("w") as log:
    subprocess.run(
        ["cmake", "--install", str(build)],
        env=env,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=7200,
    )
print(a.label, "completed", flush=True)
