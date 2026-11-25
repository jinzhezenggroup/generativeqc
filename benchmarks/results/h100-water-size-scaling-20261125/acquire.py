"""Run the cold GPU pairs serially on one H100, retaining every attempt."""

import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from geometry import SIZES

METHODS = ("hf", "pbe", "r2scan", "wb97m-v")
ARMS = ("standard", "gpu4pyscf")
MODES = ("direct", "df")
TIMEOUT = 600
REPEATS = 3


def save(path, receipt):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stop_process(child):
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


class DeviceMemory:
    """Sample NVML whole-device used bytes, including the CUDA context."""

    class Info(ctypes.Structure):
        _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]

    def __init__(self, pci):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.check(self.library.nvmlInit_v2())
        self.check(
            self.library.nvmlDeviceGetHandleByPciBusId_v2(
                pci.encode("ascii"), ctypes.byref(self.handle)
            )
        )

    def check(self, code):
        if code != 0:
            raise RuntimeError(f"NVML error {code}")

    def used(self):
        info = self.Info()
        self.check(
            self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
        )
        return info.used

    def close(self):
        self.check(self.library.nvmlShutdown())


def environment(config, root, arm):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GENERATIVEQC_")
    }
    env.update(config["environment"], **config["arm_environment"][arm])
    old_source = config["source"]
    env["PYTHONPATH"] = env["PYTHONPATH"].replace(old_source, str(root / "source"))
    if arm == "standard":
        old_library = Path(env["GENERATIVEQC_LIBRARY"])
        library = root / "build/libgenerativeqc.so"
        env["GENERATIVEQC_LIBRARY"] = str(library)
        env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"].replace(
            str(old_library.parent), str(library.parent)
        )
        env["GENERATIVEQC_STATIONARY_CACHE"] = str(root / "runtime-cache")
        env["GENERATIVEQC_JIT_SCCACHE_ROOT"] = str(root / "jit-owners")
    return env


def verify(root, config, envs):
    spec = json.loads((root / "source.json").read_text())
    source = root / "source"
    build_path = root / "build-receipts/build.json"
    build = json.loads(build_path.read_text())
    if (
        build["status"] != "complete"
        or build["source_revision"] != spec["revision"]
        or build["source_archive_sha256"] != spec["source_archive_sha256"]
        or build["native_source_identity"] != spec["native_source_identity"]
    ):
        raise ValueError("build receipt differs from the staged revision")
    identities = {
        "native_library_sha256": digest(Path(envs["standard"]["GENERATIVEQC_LIBRARY"])),
        "paper_runner_sha256": digest(source / "benchmarks/readme_defaults.py"),
        "build_receipt_sha256": digest(build_path),
    }
    if identities["native_library_sha256"] != build["native_library_sha256"]:
        raise ValueError("native library differs from build receipt")
    if identities["paper_runner_sha256"] != spec["paper_runner_sha256"]:
        raise ValueError("source runner differs from the staged revision")
    probe = (
        "import ctypes, json, cupy as cp, os; "
        "p=cp.cuda.runtime.deviceGetPCIBusId(0); "
        "lib=ctypes.CDLL(os.environ['GENERATIVEQC_LIBRARY']); "
        "lib.generativeqc_get_source_identity.restype=ctypes.c_char_p; "
        "print(json.dumps([p.decode() if isinstance(p,bytes) else p, "
        "lib.generativeqc_get_source_identity().decode(), cp.cuda.runtime.getDeviceCount()]))"
    )
    pci, identity, count = json.loads(
        subprocess.check_output(
            [config["interpreters"]["standard"], "-c", probe],
            env=envs["standard"],
            text=True,
            timeout=120,
        )
    )
    if identity != spec["native_source_identity"] or count != 1:
        raise ValueError("unexpected native artifact identity or device count")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            pci,
            "--query-gpu=name,uuid,memory.total,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()
    if "H100" not in gpu:
        raise ValueError(f"expected H100, got {gpu}")
    return pci, {"gpu": gpu, "native_source_identity": identity, **identities}


def classify(code, raw, output, timeout):
    if timeout:
        return "timeout"
    if (
        code == 0
        and raw.get("status") == "complete"
        and len(raw.get("samples", [])) == 1
    ):
        return "complete"
    text = (raw.get("error", "") + "\n" + output).lower()
    if re.search(
        r"out.of.memory|cudaerrormemoryallocation|alloc_failed|cublas_status_alloc_failed|cuda_error_out_of_memory",
        text,
    ):
        return "oom"
    if "did not converge" in text or "not converged" in text:
        return "unconverged"
    return "failed"


def run(root, output, config, env, memory, row, remaining):
    name = row["name"]
    path = output / (name + ".json")
    logfile = output / (name + ".txt")
    argv = [
        config["interpreters"][row["arm"]],
        "-u",
        str(root / "endpoint.py"),
        row["arm"],
        row["method"],
        row["mode"],
        str(row["atoms"]),
        str(path),
        "--root",
        str(root),
    ]
    begin = time.monotonic()
    baseline = memory.used()
    peak = baseline
    samples = 1
    timed_out = False
    with logfile.open("xb") as log:
        child = subprocess.Popen(
            argv,
            cwd=root / "source",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                peak = max(peak, memory.used())
                samples += 1
                if time.monotonic() - begin >= min(TIMEOUT, remaining):
                    timed_out = True
                    stop_process(child)
                    break
                time.sleep(0.05)
        finally:
            if child.poll() is None:
                stop_process(child)
    output_text = logfile.read_text(errors="replace")
    marks = [
        json.loads(line[6:])
        for line in output_text.splitlines()
        if line.startswith("STAGE ")
    ]
    raw = json.loads(path.read_text()) if path.exists() else {}
    row.update(
        status=classify(child.returncode, raw, output_text, timed_out),
        argv=argv,
        subprocess_seconds=time.monotonic() - begin,
        returncode=child.returncode,
        stages=marks,
        last_stage=marks[-1]["stage"] if marks else "startup",
        memory={
            "source": "NVML whole-device, polled 50 ms during child lifetime",
            "baseline_bytes": baseline,
            "peak_used_bytes": peak,
            "samples": samples,
        },
        raw_sha256=digest(path) if raw else None,
        log_sha256=digest(logfile),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=28200)
    parser.add_argument("--sizes", type=int, choices=SIZES, nargs="+", default=SIZES)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "gpu"
    output.mkdir()
    deadline = time.monotonic() + args.deadline_seconds
    receipt = {"status": "preflight", "rows": [], "timeout_seconds": TIMEOUT}
    destination = output / "campaign.json"
    save(destination, receipt)
    config = json.loads((root / "paper-config.json").read_text())
    envs = {arm: environment(config, root, arm) for arm in ARMS}
    pci, identity = verify(root, config, envs)
    memory = DeviceMemory(pci)
    receipt.update(
        status="running",
        hardware=identity,
        job_name=os.environ["GQC_QZ_JOB_NAME"],
        project=os.environ["GQC_QZ_PROJECT"],
    )
    save(destination, receipt)

    def interrupt(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    population = list(itertools.product(args.sizes, METHODS, MODES))
    prewarm_status = {}
    try:
        for repeat in range(REPEATS):
            for index, (atoms, method, mode) in enumerate(population):
                ordered = ARMS if (index + repeat) % 2 == 0 else ARMS[::-1]
                for stage in ("prewarm", "formal") if repeat == 0 else ("formal",):
                    for arm in ordered:
                        key = (atoms, method, mode, arm)
                        name = f"{stage}-{repeat}-{atoms}-{method}-{mode}-{arm}"
                        row = dict(
                            name=name,
                            stage=stage,
                            repeat=repeat,
                            atoms=atoms,
                            method=method,
                            mode=mode,
                            arm=arm,
                            status="running",
                        )
                        receipt["rows"].append(row)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            row["status"] = "unattempted_deadline"
                        elif stage == "formal" and prewarm_status[key] != "complete":
                            row["status"] = "unattempted_failed_prewarm"
                        else:
                            save(destination, receipt)
                            run(root, output, config, envs[arm], memory, row, remaining)
                            print(name, row["status"], row["last_stage"], flush=True)
                        if stage == "prewarm":
                            prewarm_status[key] = row["status"]
                        save(destination, receipt)
        receipt["status"] = "finished"
    except BaseException as error:
        receipt.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        memory.close()
        save(destination, receipt)


if __name__ == "__main__":
    main()
