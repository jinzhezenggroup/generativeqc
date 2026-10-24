"""Bounded full-size, source-matched MD-J cold/warm qualification.

Run inside an existing finite CUDA Slurm allocation (never on a login node):
  srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=03:00:00 \\
    python benchmarks/md_j_engine_benchmark.py \\
      --output .artifacts/md-j-performance.json --timeout 1800 --total-timeout 10500 \\
      --upstream-root /path/to/pinned/gpu4pyscf --require-upstream

Needs a CUDA GENERATIVEQC_LIBRARY, CUDA_PATH, ccache, PySCF and CuPy. No custom
input/harness is needed: defaults are 96-atom water clusters (224/768 spherical
AOs). --dry-run writes the prescription without importing a GPU runtime. Raw
samples, frozen explicit shells/density, matrices, subprocess logs and failures
are retained in a unique sibling directory. A timeout never qualifies a case.

--scope j measures the complete raw-J adapter only. --scope energy-force runs
complete native PBE0 energy/force endpoints, including retained and moved calls;
--scope all (default) requires both. Native reference/incumbent share the exact
public-AO mask. PySCF and optional
pinned GPU4PySCF use positive 1e-30 thresholds: that comparison is explicitly
near-unscreened and numerically gated, NOT proof of identical screening work.
No performance result promotes the default engine or claims production ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "md-j-engine-performance-v1"
UPSTREAM_COMMIT = "9af748a71ca8f4881c604332c2f0cc1632e033ac"
NATIVE_ENGINES = ("reference", "incumbent")
SCREENING_SCOPE = (
    "Native engines use the same public-AO cutoff (zero). PySCF/GPU4PySCF use "
    "positive 1e-30 logarithmic screening: near-unscreened differential, not "
    "identical screening masks or a production-configuration ranking."
)
TIMING_BOUNDARIES = {
    "cold_complete_j_seconds": "fresh raw-J test-adapter worker: scientific imports + input read + bridge/native binary hashing/loading + basis/owner preparation + first density upload + synchronized J + host result publication; excludes receipt writing and teardown; not production-only preparation",
    "process_wall_seconds": "parent subprocess launch through exit, including imports, first J, output serialization, provenance and teardown; cold workers perform exactly one J",
    "warm_complete_j_seconds": "retained owner: density upload + synchronized J + density integrity readback + host J publication; excludes preparation and matrix file writing",
    "resident_device_seconds": "CUDA event interval around J on its execution stream; excludes density upload, output publication and preparation; not a complete endpoint",
}


@dataclass(frozen=True)
class MolecularCase:
    """Fixed full-size molecular prescription; coordinates are in bohr."""

    name: str
    basis: str
    expected_nao: int
    seed: int
    waters: int = 32

    def atoms(self) -> list[tuple[str, tuple[float, float, float]]]:
        """A deterministic 4x4x2 water lattice with alternating orientations."""
        values = []
        for index in range(self.waters):
            origin = (6.0 * (index % 4), 6.0 * ((index // 4) % 4), 6.0 * (index // 16))
            sign = 1.0 if index % 2 == 0 else -1.0
            for symbol, offset in (
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, sign * 1.10715266)),
                ("H", (0.0, 1.43233673, sign * 1.10715266)),
            ):
                values.append(
                    (symbol, tuple(a + b for a, b in zip(origin, offset, strict=True)))
                )
        return values

    def prescription(self) -> dict[str, Any]:
        """Return all molecular and density choices before executing any engine."""
        return {
            "name": self.name,
            "atoms": self.atoms(),
            "natoms": self.waters * 3,
            "basis": self.basis,
            "representation": "spherical",
            "expected_nao": self.expected_nao,
            "charge": 0,
            "multiplicity": 1,
            "seed": self.seed,
            "density": "PCG64 seeded symmetric positive-semidefinite X@X.T / nao, rank 32; synthetic shared raw-J input, not an SCF density",
            "precision": "fp64",
            "native_screening_tolerance": 0.0,
            "reference_screening_tolerance": 1e-30,
        }


CASES = {
    case.name: case
    for case in (
        MolecularCase("water32-sto-3g", "sto-3g", 224, 90731),
        MolecularCase("water32-def2-svp", "def2-svp", 768, 90732),
    )
}


def sha256(path: Path) -> str:
    """Hash large libraries without materializing another copy."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically retain strict JSON so interruption leaves the prior receipt."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate bounded public options without importing CUDA/native modules."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--scope", choices=("j", "energy-force", "all"), default="all")
    parser.add_argument("--engine", choices=NATIVE_ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--instrumented", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="maximum seconds per subprocess, including preparation/oracle",
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=2100,
        help="maximum seconds for the complete campaign",
    )
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--require-upstream", action="store_true")
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        default=1.0,
        help="required incumbent/reference median ratio for BOTH complete cold and warm J",
    )
    parser.add_argument("--atol", type=float, default=3e-9)
    parser.add_argument("--rtol", type=float, default=3e-10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write a not-run prescription without native/CUDA imports",
    )
    parser.add_argument(
        "--worker",
        choices=("prepare", "oracle", *NATIVE_ENGINES, "upstream", "endpoint"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase",
        choices=("cold", "warm", "diagnostic"),
        default="cold",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name, minimum, maximum in (
        ("cold_repeats", 3, 20),
        ("warm_repeats", 5, 100),
        ("warmup", 1, 10),
        ("timeout", 1, 3600),
        ("total_timeout", 1, 86400),
    ):
        if not minimum <= getattr(args, name) <= maximum:
            parser.error(
                f"--{name.replace('_', '-')} must be between {minimum} and {maximum}"
            )
    for name in ("atol", "rtol", "minimum_speedup"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.minimum_speedup < 1:
        parser.error("--minimum-speedup must be at least 1")
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases must not contain duplicates")
    if args.require_upstream and not args.upstream_root:
        parser.error("--require-upstream requires --upstream-root")
    if args.worker and (not args.run_id or len(args.cases) != 1):
        parser.error("internal workers require a run identity and exactly one case")
    if args.worker and args.worker != "prepare" and not args.input:
        parser.error("internal evaluation workers require --input")
    if args.worker == "endpoint" and args.engine is None:
        parser.error("endpoint worker requires --engine")
    if args.scope == "energy-force" and args.require_upstream:
        parser.error("--require-upstream applies to raw J; select --scope j or all")
    return args


def source_identity() -> dict[str, Any]:
    """Record actual checkout identity and scientific/adapter source contents."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()

    paths = (
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_md_j.cu",
        "src/scf/cuda/md_j_engine.cuh",
        "src/scf/cuda/md_j_engine_contract.hpp",
        "tests/native/md_j_resident_probe.cpp",
        "tests/python/md_j_resident_probe.py",
        "benchmarks/md_j_engine_benchmark.py",
        "benchmarks/_md_j_engine_endpoints.py",
        "python/generativeqc_compiler/integral/md_j_engine_cuda.py",
        "benchmarks/_md_j_reference.py",
        "tools/generate_validation_references.py",
        "upstream/manifest.json",
    )
    from tools.generate_build_identity import _inventory, _source_identity

    native_source = _source_identity(
        ROOT, _inventory(ROOT, ROOT / "cmake/GenerativeQCSourceIdentity.json")
    )
    return {
        "native_source_identity": native_source,
        "git_head": git("rev-parse", "HEAD"),
        "declared_build_revision": os.environ.get("GENERATIVEQC_BENCHMARK_SOURCE"),
        "git_status": git("status", "--porcelain", "--untracked-files=normal"),
        "files_sha256": {path: sha256(ROOT / path) for path in paths},
        "source_binary_binding": "generativeqc_get_source_identity must equal the repository build-identity manifest hash; actual loaded library bytes also pinned",
    }


def native_identity(expected: str, *, qualify_device: bool = False) -> dict[str, Any]:
    """Bind the actually loaded library to the build manifest without GPU priming."""
    from generativeqc import _native

    library = _native.load_library()
    actual = library.generativeqc_get_source_identity().decode()
    if actual != expected:
        raise ValueError(
            f"native source identity mismatch: library {actual}, checkout {expected}"
        )
    loaded = Path(library._name).resolve()
    requested = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    if loaded != requested:
        raise ValueError(
            "runtime loaded a different native library from GENERATIVEQC_LIBRARY"
        )
    result = {
        "source_identity": actual,
        "loaded_library": str(loaded),
        "loaded_library_sha256": sha256(loaded),
    }
    if qualify_device:
        from generativeqc.profiles import probe_device

        probe = probe_device(library)
        if (
            probe["device"].get("release_build") != 1
            or probe["device"].get("fast_compile") != 0
        ):
            raise ValueError(
                "performance evidence requires a release build without fast compilation"
            )
        result["device_build"] = probe
    return result


def environment_provenance(environment: dict[str, str]) -> dict[str, Any]:
    """Retain known nonsecret controls, never arbitrary credential-like values."""
    safe = {
        "GENERATIVEQC_LIBRARY",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_PATH",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MODULE_LOADING",
        "CUDA_CACHE_PATH",
        "CUDA_CACHE_DISABLE",
        "CUDA_FORCE_PTX_JIT",
        "CUDA_DISABLE_PTX_JIT",
        "CUPY_CACHE_DIR",
        "CUPY_CACHE_IN_MEMORY",
        "CCACHE_DIR",
        "CCACHE_BASEDIR",
        "SCCACHE_DIR",
        "SCCACHE_CACHE_SIZE",
    }
    relevant = {
        key
        for key in environment
        if key.startswith(("GENERATIVEQC_", "CUDA_", "CUPY_", "CCACHE_", "SCCACHE_"))
    }
    return {
        "ambient_runtime_controls": {
            key: environment[key] for key in sorted(relevant & safe)
        },
        "unrecorded_ambient_control_names": sorted(relevant - safe),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Require a finite allocation and verified native/build prerequisites."""
    from benchmarks.md_j_engine_acceptance import _cuda_preflight, verify_upstream

    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        raise RuntimeError(
            "qualified performance timing requires CUDA_LAUNCH_BLOCKING unset or 0"
        )
    result = _cuda_preflight()
    if args.upstream_root:
        result["upstream_commit"] = verify_upstream(args.upstream_root.resolve())
    end = result.get("end_time")
    if end and int(end) - time.time() < args.total_timeout + 30:
        raise RuntimeError(
            "Slurm allocation has insufficient remaining time for --total-timeout plus 30s cleanup"
        )
    result["source"] = source_identity()
    result["native_identity"] = native_identity(
        result["source"]["native_source_identity"]
    )
    result["python"] = sys.version
    result["upstream_pin"] = UPSTREAM_COMMIT
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    result["gpu_inventory"] = query.stdout.splitlines()
    result.update(environment_provenance(dict(os.environ)))
    result["effective_native_controls"] = {
        key: value
        for key, value in _worker_environment(args.upstream_root).items()
        if key.startswith("GENERATIVEQC_")
    }
    result["removed_native_control_names"] = sorted(
        key
        for key in os.environ
        if key.startswith("GENERATIVEQC_") and key != "GENERATIVEQC_LIBRARY"
    )
    result["native_profile_policy"] = (
        "GENERATIVEQC_PROFILE=off in every worker; all CUDA Calculators execute the pinned base library rather than a locally selected profile binary"
    )
    result["thread_policy"] = {
        key: "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    result["upstream_build_binding"] = (
        "clean pinned source/import path and stable extension hash are verified; upstream has no native build-source identity ABI, so extension build/source correspondence remains external provenance"
    )
    return result


def _worker_environment(upstream: Path | None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GENERATIVEQC_") or key == "GENERATIVEQC_LIBRARY"
    }
    # A local profile can replace the base CDLL for CUDA only. Pin every
    # Calculator and the bridge to the same measured/hashed base library.
    environment["GENERATIVEQC_PROFILE"] = "off"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            *([str(upstream.resolve())] if upstream else []),
            str(ROOT / "python"),
            str(ROOT / "tests/python"),
            str(ROOT),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment.pop("SOURCE", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    return environment


def _sample_gpu_memory(
    pid: int, stopped: threading.Event, observations: list[int], errors: list[str]
) -> None:
    """NVML process observations via nvidia-smi; sampling is NOT an exact peak."""
    while not stopped.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            sizes = []
            for line in result.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
                    sizes.append(int(fields[1]) * 1024 * 1024)
            if sizes:
                observations.append(sum(sizes))
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(str(error))
            return
        stopped.wait(0.2)


def run_process(
    command: list[str],
    output: Path,
    environment: dict[str, str],
    timeout: float,
    *,
    measure_gpu: bool = True,
) -> dict[str, Any]:
    """Bound and retain a whole process group, including compiler descendants."""
    stdout, stderr = (
        output.with_suffix(".stdout.txt"),
        output.with_suffix(".stderr.txt"),
    )
    started = time.perf_counter()
    observations: list[int] = []
    errors: list[str] = []
    stopped = threading.Event()
    timed_out = False
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        monitor = threading.Thread(
            target=_sample_gpu_memory,
            args=(process.pid, stopped, observations, errors),
            daemon=True,
        )
        if measure_gpu:
            monitor.start()
        try:
            process.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise
        finally:
            wall = time.perf_counter() - started
            stopped.set()
            if measure_gpu:
                monitor.join(timeout=5)
    return {
        "status": "timeout"
        if timed_out
        else "passed"
        if process.returncode == 0
        else "failed",
        "returncode": process.returncode,
        "pid": process.pid,
        "process_wall_seconds": wall,
        "command": command,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "sampled_peak_process_gpu_bytes": max(observations, default=None),
        "gpu_memory_samples": observations,
        "gpu_memory_errors": errors,
        "gpu_memory_scope": "sum of this worker PID's NVML used_memory across GPUs, sampled every >=200ms; lower bound on peak, includes context/pools; not exact allocation peak",
    }


def load_receipt(
    path: Path, *, run_id: str, case: str, engine: str, phase: str
) -> dict[str, Any]:
    """Reject missing, stale, truncated, nonfinite or wrong-worker receipts."""

    def invalid_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON value: {value}")

    receipt = json.loads(path.read_text(), parse_constant=invalid_constant)
    expected = {
        "schema": SCHEMA,
        "run_id": run_id,
        "case": case,
        "engine": engine,
        "phase": phase,
        "status": "passed",
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("worker receipt identity/status mismatch")
    for key in ("matrix_file", "input_file"):
        file = Path(receipt[key])
        # Workers may only refer to artifacts in this fresh campaign directory.
        if (
            file.resolve().parent != path.resolve().parent
            or sha256(file) != receipt[key + "_sha256"]
        ):
            raise ValueError(
                f"worker {key} is outside the campaign or its checksum differs"
            )
    return receipt


def _public_basis(payload: dict[str, Any]) -> Any:
    from generativeqc import Atom, Primitive, Shell

    return SimpleNamespace(
        atoms=tuple(Atom(atom[0], tuple(atom[1])) for atom in payload["atoms"]),
        shells=tuple(
            Shell(
                shell["atom_index"],
                shell["angular_momentum"],
                tuple(Primitive(*primitive) for primitive in shell["primitives"]),
            )
            for shell in payload["shells"]
        ),
        charge=payload["charge"],
        multiplicity=payload["multiplicity"],
        representation="real_spherical",
        nao=payload["nao"],
    )


def _prepare(
    case: MolecularCase, output: Path, run_id: str, *, prepare_bridge: bool = True
) -> None:
    import numpy as np
    from generativeqc_compiler.dft import NativeAO

    payload = case.prescription()
    with NativeAO(case.atoms(), case.basis, representation="spherical") as basis:
        if basis.nao != case.expected_nao:
            raise ValueError(
                f"basis AO count {basis.nao} differs from pinned prescription {case.expected_nao}"
            )
        payload["nao"] = basis.nao
        payload["shells"] = [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            }
            for shell in basis.shells
        ]
    rng = np.random.Generator(np.random.PCG64(case.seed))
    factors = rng.normal(size=(case.expected_nao, 32)) * 0.2
    density = np.ascontiguousarray(factors @ factors.T / case.expected_nao)
    density_file = output.with_suffix(".density.npy")
    np.save(density_file, density, allow_pickle=False)
    payload.update(
        density_file=str(density_file),
        density_sha256=sha256(density_file),
        run_id=run_id,
    )
    # Prepare the test adapter before all cold samples. This process is never a
    # timing sample and initializes no CUDA context. Each later worker still
    # starts with a fresh process/context; the on-disk compiler/JIT caches persist.
    if prepare_bridge:
        from md_j_resident_probe import _probe_library

        bridge = Path(_probe_library()._name).resolve()
        payload["bridge"] = {
            "path": str(bridge),
            "sha256": sha256(bridge),
            "cache_policy": "prebuilt shared adapter; persistent compiler/artifact caches retained, never cleared; per-worker library hashing/loading remains inside adapter cold timing",
        }
    write_json(output, payload)


def direct_pyscf_j(molecule: Any, density: Any) -> Any:
    """Explicit direct contractor: never dispatch RHF's in-core ERI shortcut."""
    from pyscf.scf import hf

    mean_field = molecule.RHF()
    mean_field.direct_scf_tol = 1e-30
    optimizer = mean_field.init_direct_scf(molecule)
    matrix, _ = hf.get_jk(
        molecule, density, hermi=0, vhfopt=optimizer, with_j=True, with_k=False
    )
    return matrix


def _evaluate_worker(args: argparse.Namespace) -> None:
    # Starts before scientific imports, all first GPU context creation and input read.
    started = time.perf_counter()
    if args.instrumented:
        os.environ["GENERATIVEQC_MD_J_WORK_COUNTS"] = "1"
        os.environ["GENERATIVEQC_MD_J_COUNTS"] = "1"
    import numpy as np

    payload = json.loads(args.input.read_text())
    case = CASES[args.cases[0]]
    if (
        payload["run_id"] != args.run_id
        or payload["name"] != case.name
        or payload["nao"] != case.expected_nao
    ):
        raise ValueError("frozen case identity mismatch")
    density_file = Path(payload["density_file"])
    if (
        density_file.resolve().parent != args.input.resolve().parent
        or sha256(density_file) != payload["density_sha256"]
    ):
        raise ValueError("density artifact identity mismatch")
    density = np.load(density_file, allow_pickle=False)
    if (
        density.shape != (case.expected_nao, case.expected_nao)
        or density.dtype != np.float64
        or not np.isfinite(density).all()
    ):
        raise ValueError("invalid FP64 density")
    basis = _public_basis(payload)
    if sha256(Path(payload["bridge"]["path"])) != payload["bridge"]["sha256"]:
        raise ValueError("prebuilt test bridge identity changed")
    metadata: dict[str, Any] = {}
    if args.worker == "oracle":
        import pyscf

        from benchmarks._md_j_reference import reference_molecule

        molecule, scales = reference_molecule(basis)
        # get_j is direct here: no kernel()/SCF or full four-index ERI allocation.
        transformed = density * scales[:, None] * scales[None, :]
        matrix = (
            direct_pyscf_j(molecule, transformed) * scales[:, None] * scales[None, :]
        )
        metadata.update(
            pyscf_version=pyscf.__version__,
            oracle="PySCF direct Coulomb contraction of explicit public-AO shells, no SCF",
            screening=1e-30,
        )
        timings: dict[str, Any] = {"oracle_seconds": time.perf_counter() - started}
    else:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no visible CUDA device")
        if args.worker in NATIVE_ENGINES:
            from dataclasses import replace

            from generativeqc.fock import FockBuildSpec, FockTerm
            from generativeqc_compiler.dft import NativeAO
            from md_j_resident_probe import ResidentMdJProbe

            os.environ["GENERATIVEQC_MD_J_ENGINE"] = args.worker
            ao = NativeAO(payload["atoms"], basis.shells, representation="spherical")
            spec = replace(
                FockBuildSpec.hf("restricted", derivative_order=0),
                exchange=FockTerm(False),
            )
            owner = ResidentMdJProbe(ao, spec, screening_tolerance=0.0)
            schedule = owner.diagnostics["direct_schedule"]
            if not schedule.startswith(
                "md-j-reference/" if args.worker == "reference" else "md-j-hermite/"
            ):
                raise RuntimeError(f"requested engine did not execute: {schedule}")
            metadata.update(
                schedule=schedule,
                plan_device_bytes=owner.diagnostics["device_bytes"],
                screening=0.0,
            )

            def evaluate() -> tuple[Any, float]:
                return owner.evaluate(density).coulomb, owner.last_device_seconds

            def close() -> None:
                owner.close()
                ao.close()
        else:
            import gpu4pyscf
            from gpu4pyscf.scf import j_engine

            from benchmarks._md_j_reference import reference_molecule
            from benchmarks.md_j_engine_acceptance import verify_upstream

            root = args.upstream_root.resolve()
            verify_upstream(root)
            if not Path(gpu4pyscf.__file__).resolve().is_relative_to(root):
                raise RuntimeError("imported GPU4PySCF is not the pinned checkout")
            molecule, scales = reference_molecule(basis)
            owner = j_engine._VHFOpt(molecule, direct_scf_tol=1e-30).build()
            device_density = cp.empty_like(cp.asarray(density))
            start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
            binary = Path(j_engine.libvhf_md._name).resolve()
            metadata.update(
                upstream_commit=UPSTREAM_COMMIT,
                upstream_library=str(binary),
                upstream_library_sha256=sha256(binary),
                screening=1e-30,
            )

            def evaluate() -> tuple[Any, float]:
                transformed = density * scales[:, None] * scales[None, :]
                device_density.set(transformed)
                cp.cuda.Stream.null.synchronize()
                start_event.record()
                result = j_engine.get_j(molecule, device_density, hermi=0, vhfopt=owner)
                end_event.record()
                end_event.synchronize()
                seconds = cp.cuda.get_elapsed_time(start_event, end_event) / 1000
                np.testing.assert_array_equal(cp.asnumpy(device_density), transformed)
                return cp.asnumpy(result) * scales[:, None] * scales[None, :], seconds

            def close() -> None:
                cp.cuda.Stream.null.synchronize()

        preparation = time.perf_counter() - started
        warm_wall, resident = [], []
        try:
            matrix, first_device = evaluate()
            first_complete = time.perf_counter() - started
            first_matrix = matrix.copy()
            max_replay_error = 0.0
            if args.phase == "warm":
                for _ in range(args.warmup):
                    evaluate()
                for _ in range(args.warm_repeats):
                    tick = time.perf_counter()
                    matrix, seconds = evaluate()
                    warm_wall.append(time.perf_counter() - tick)
                    resident.append(seconds)
                    max_replay_error = max(
                        max_replay_error, float(np.max(np.abs(matrix - first_matrix)))
                    )
                    if not np.allclose(
                        matrix, first_matrix, atol=args.atol, rtol=args.rtol
                    ):
                        raise RuntimeError(
                            "warm J does not reproduce the first published matrix"
                        )
        finally:
            close()
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        metadata.update(
            cupy_version=cp.__version__,
            device_name=str(properties["name"]),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            max_replay_error=max_replay_error,
        )
        timings = {
            "preparation_seconds": preparation,
            "cold_complete_j_seconds": first_complete,
            "first_resident_device_seconds": first_device,
            "warm_complete_j_seconds": warm_wall,
            "resident_device_seconds": resident,
            "warmup_calls": args.warmup if args.phase == "warm" else 0,
            "j_calls": 1
            + (args.warmup + args.warm_repeats if args.phase == "warm" else 0),
        }
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite J output")
    matrix_file = args.output.with_suffix(".matrix.npy")
    np.save(matrix_file, np.ascontiguousarray(matrix), allow_pickle=False)
    write_json(
        args.output,
        {
            "schema": SCHEMA,
            "status": "passed",
            "run_id": args.run_id,
            "case": case.name,
            "engine": args.worker,
            "phase": args.phase,
            "pid": os.getpid(),
            "backend": "cpu-oracle" if args.worker == "oracle" else "cuda",
            "instrumented": args.instrumented,
            "input_file": str(args.input),
            "input_file_sha256": sha256(args.input),
            "matrix_file": str(matrix_file),
            "matrix_file_sha256": sha256(matrix_file),
            "library_sha256": sha256(Path(os.environ["GENERATIVEQC_LIBRARY"])),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "timings": timings,
            "metadata": metadata,
            "native_identity": native_identity(
                os.environ["MDJ_EXPECTED_SOURCE_IDENTITY"],
                qualify_device=args.worker != "oracle",
            ),
        },
    )


def _endpoint_worker(args: argparse.Namespace) -> None:
    from benchmarks._md_j_engine_endpoints import run_endpoint_worker

    payload = json.loads(args.input.read_text())
    if payload["run_id"] != args.run_id or payload["name"] != args.cases[0]:
        raise ValueError("endpoint frozen case identity mismatch")
    result = run_endpoint_worker(
        payload,
        engine=args.engine,
        instrumented=args.instrumented,
        warm_repeats=args.warm_repeats,
    )
    write_json(
        args.output,
        {
            "schema": SCHEMA,
            "status": "passed",
            "run_id": args.run_id,
            "case": args.cases[0],
            "engine": args.engine,
            "scope": "energy-force",
            "pid": os.getpid(),
            "input_file": str(args.input),
            "input_file_sha256": sha256(args.input),
            "library_sha256": sha256(Path(os.environ["GENERATIVEQC_LIBRARY"])),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "endpoint_result": result,
            "native_identity": native_identity(
                os.environ["MDJ_EXPECTED_SOURCE_IDENTITY"], qualify_device=True
            ),
        },
    )


def load_endpoint_receipt(
    path: Path, *, run_id: str, case: str, engine: str, input_sha256: str
) -> dict[str, Any]:
    """Validate a fresh complete-endpoint receipt before cohort qualification."""

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite endpoint receipt: {value}")

    result = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    expected = {
        "schema": SCHEMA,
        "status": "passed",
        "run_id": run_id,
        "case": case,
        "engine": engine,
        "scope": "energy-force",
        "input_file_sha256": input_sha256,
    }
    if not isinstance(result, dict) or any(
        result.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("endpoint receipt identity/status mismatch")
    source = Path(result["input_file"])
    if (
        source.resolve().parent != path.resolve().parent
        or sha256(source) != input_sha256
    ):
        raise ValueError(
            "endpoint input artifact changed or escaped campaign directory"
        )
    if not isinstance(result.get("endpoint_result"), dict):
        raise TypeError("missing complete endpoint observables")
    return result


def numerical_comparison(
    actual: Any, expected: Any, atol: float, rtol: float
) -> dict[str, Any]:
    """Use a per-element declared tolerance, not a checksum as a science gate."""
    import numpy as np

    left, right = np.asarray(actual), np.asarray(expected)
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        return {"passed": False, "reason": "shape mismatch or nonfinite matrix"}
    error = np.abs(left - right)
    bound = atol + rtol * np.abs(right)
    return {
        "passed": bool(np.all(error <= bound)),
        "max_abs_error": float(error.max()),
        "max_tolerance_ratio": float((error / bound).max()),
        "atol": atol,
        "rtol": rtol,
    }


def timing_summary(values: list[float]) -> dict[str, Any]:
    """Retain every sample with median and dispersion; never select a best run."""
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("missing or invalid positive timing samples")
    median = statistics.median(values)
    return {
        "raw_seconds": values,
        "count": len(values),
        "median_seconds": median,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "median_absolute_deviation_seconds": statistics.median(
            abs(value - median) for value in values
        ),
    }


def summarize_case(
    rows: list[dict[str, Any]],
    *,
    cold_repeats: int,
    warm_repeats: int,
    engines: list[str],
    minimum_speedup: float,
) -> dict[str, Any]:
    """Fail closed on incomplete processes, unequal inputs or incomplete outputs."""
    failures, summary = [], {}
    for engine in engines:
        chosen = [row for row in rows if row.get("engine") == engine]
        cold = [row for row in chosen if row.get("phase") == "cold"]
        warm = [row for row in chosen if row.get("phase") == "warm"]
        if len(cold) != cold_repeats or len(warm) != 1:
            failures.append(f"{engine}: missing required cold/warm cohorts")
            continue
        try:
            if engine in NATIVE_ENGINES:
                from benchmarks._md_j_engine_endpoints import _selection

                diagnostic = [row for row in chosen if row.get("phase") == "diagnostic"]
                if len(diagnostic) != 1 or not diagnostic[0].get("instrumented"):
                    raise ValueError("missing separate executing-source diagnostic")
                _selection(diagnostic[0]["execution"], engine)
                if any(
                    row.get("instrumented") or row.get("execution", {}).get("work_rows")
                    for row in cold + warm
                ):
                    raise ValueError(
                        "instrumented observations cannot qualify clean timing"
                    )
            else:
                for field in (
                    "upstream_commit",
                    "upstream_library",
                    "upstream_library_sha256",
                ):
                    values = {row.get("metadata", {}).get(field) for row in chosen}
                    if len(values) != 1 or not next(iter(values)):
                        raise ValueError(f"unmatched or missing {field}")
                if chosen[0]["metadata"]["upstream_commit"] != UPSTREAM_COMMIT:
                    raise ValueError(
                        "upstream commit differs from the pinned reference"
                    )
            for row in chosen:
                if (
                    row.get("status") != "passed"
                    or row.get("backend") != "cuda"
                    or not row.get("comparison", {}).get("passed")
                ):
                    raise ValueError(
                        "failed process, backend or independent numerical gate"
                    )
                if (
                    row.get("sampled_peak_process_gpu_bytes") is None
                    or row.get("peak_host_rss_bytes", 0) <= 0
                ):
                    raise ValueError("missing process memory evidence")
            if any(row["timings"]["j_calls"] != 1 for row in cold):
                raise ValueError("cold worker performed more than one J call")
            if (
                len(warm[0]["timings"]["warm_complete_j_seconds"]) != warm_repeats
                or len(warm[0]["timings"]["resident_device_seconds"]) != warm_repeats
            ):
                raise ValueError("missing warm complete/event timing samples")
            summary[engine] = {
                "cold_complete_j": timing_summary(
                    [row["timings"]["cold_complete_j_seconds"] for row in cold]
                ),
                "cold_process_wall": timing_summary(
                    [row["process_wall_seconds"] for row in cold]
                ),
                "warm_complete_j": timing_summary(
                    warm[0]["timings"]["warm_complete_j_seconds"]
                ),
                "resident_device": timing_summary(
                    warm[0]["timings"]["resident_device_seconds"]
                ),
            }
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{engine}: {error}")
    for key in (
        "run_id",
        "case",
        "input_file_sha256",
        "library_sha256",
        "slurm_job_id",
        "cuda_visible_devices",
    ):
        if (
            not rows
            or len({row.get(key) for row in rows}) != 1
            or rows[0].get(key) is None
        ):
            failures.append(f"unmatched or missing {key}")
    pids = [row.get("pid") for row in rows]
    if None in pids or len(pids) != len(set(pids)):
        failures.append("workers must be distinct fresh processes")
    ratios: dict[str, Any] = {}
    if all(engine in summary for engine in NATIVE_ENGINES):
        for endpoint in (
            "cold_complete_j",
            "warm_complete_j",
            "cold_process_wall",
            "resident_device",
        ):
            ratios[endpoint] = (
                summary["incumbent"][endpoint]["median_seconds"]
                / summary["reference"][endpoint]["median_seconds"]
            )
        for endpoint in ("cold_complete_j", "warm_complete_j"):
            if ratios[endpoint] < minimum_speedup:
                failures.append(
                    f"{endpoint}: reference does not meet {minimum_speedup:g}x incumbent/reference gate"
                )
    return {
        "status": "passed" if not failures else "failed",
        "performance_acceptance_passed": not failures,
        "gate_failures": failures,
        "engines": summary,
        "incumbent_over_reference_ratio": ratios,
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run predeclared cohorts; persist partial failure receipts after every step."""
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(
            "--output already exists; choose a fresh receipt path to preserve evidence"
        )
    run_id = uuid.uuid4().hex
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": "not-run",
        "performance_acceptance_passed": False,
        "default_promotion_qualified": False,
        "scope": args.scope,
        "scope_description": "raw J adapter measurements and/or complete native energy-force endpoints, as selected; no default promotion",
        "screening_scope": SCREENING_SCOPE,
        "timing_boundaries": TIMING_BOUNDARIES,
        "prescriptions": [CASES[name].prescription() for name in args.cases],
        "parameters": {
            name: getattr(args, name)
            for name in (
                "scope",
                "cold_repeats",
                "warm_repeats",
                "warmup",
                "timeout",
                "total_timeout",
                "atol",
                "rtol",
                "minimum_speedup",
                "require_upstream",
            )
        },
        "cases": {},
        "samples": [],
        "gate_failures": [],
        "upstream": "requested" if args.upstream_root else "not-run (not supplied)",
    }
    write_json(output, report)
    if args.dry_run:
        return report
    started = time.monotonic()
    directory = Path(tempfile.mkdtemp(prefix=output.stem + "-", dir=output.parent))
    report["artifact_directory"] = str(directory)
    environment = _worker_environment(args.upstream_root)
    try:
        report["provenance"] = preflight(args)
        environment["MDJ_EXPECTED_SOURCE_IDENTITY"] = report["provenance"]["source"][
            "native_source_identity"
        ]
        report["status"] = "running"
        write_json(output, report)
        engines = [*NATIVE_ENGINES, *(["upstream"] if args.upstream_root else [])]
        for case in args.cases:
            input_file = directory / f"{case}-input.json"
            oracle = None
            case_rows: list[dict[str, Any]] = []
            jobs = [("prepare", "cold", 0)]
            if args.scope in ("j", "all"):
                jobs.append(("oracle", "cold", 0))
                for repeat in range(args.cold_repeats):
                    order = engines if repeat % 2 == 0 else list(reversed(engines))
                    jobs.extend((engine, "cold", repeat) for engine in order)
                jobs.extend((engine, "warm", 0) for engine in engines)
                jobs.extend((engine, "diagnostic", 0) for engine in NATIVE_ENGINES)
            if args.scope in ("energy-force", "all"):
                for repeat in range(args.cold_repeats):
                    order = NATIVE_ENGINES if repeat % 2 == 0 else NATIVE_ENGINES[::-1]
                    jobs.extend(
                        ("endpoint-" + engine, "cold", repeat) for engine in order
                    )
                jobs.extend(
                    ("endpoint-" + engine, "warm", 0) for engine in NATIVE_ENGINES
                )
            endpoint_rows: list[dict[str, Any]] = []
            input_hash = None
            for engine, phase, repetition in jobs:
                remaining = args.total_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("campaign exceeded --total-timeout")
                destination = (
                    input_file
                    if engine == "prepare"
                    else directory / f"{case}-{engine}-{phase}-{repetition}.json"
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(destination),
                    "--cases",
                    case,
                    "--scope",
                    args.scope,
                    "--worker",
                    "endpoint" if engine.startswith("endpoint-") else engine,
                    "--phase",
                    phase,
                    "--run-id",
                    run_id,
                    "--input",
                    str(input_file),
                    "--warm-repeats",
                    str(args.warm_repeats),
                    "--warmup",
                    str(args.warmup),
                    "--atol",
                    str(args.atol),
                    "--rtol",
                    str(args.rtol),
                ]
                if phase == "diagnostic":
                    command.append("--instrumented")
                if engine.startswith("endpoint-"):
                    command.extend(("--engine", engine.removeprefix("endpoint-")))
                    if phase == "warm":
                        command.append("--instrumented")
                if args.upstream_root:
                    command.extend(
                        ("--upstream-root", str(args.upstream_root.resolve()))
                    )
                execution = run_process(
                    command,
                    destination,
                    environment,
                    min(args.timeout, remaining),
                    measure_gpu=engine not in ("prepare", "oracle"),
                )
                row = dict(
                    execution,
                    engine=engine,
                    phase=phase,
                    case=case,
                    repetition=repetition,
                    receipt=str(destination),
                )
                report["samples"].append(row)
                write_json(output, report)
                if execution["status"] != "passed":
                    raise RuntimeError(
                        f"{case}/{engine}/{phase}: {execution['status']}; see {execution['stderr']}"
                    )
                if engine == "prepare":
                    input_hash = sha256(input_file)
                    row["input_file_sha256"] = input_hash
                    continue
                if engine.startswith("endpoint-"):
                    receipt = load_endpoint_receipt(
                        destination,
                        run_id=run_id,
                        case=case,
                        engine=engine.removeprefix("endpoint-"),
                        input_sha256=input_hash,
                    )
                else:
                    receipt = load_receipt(
                        destination,
                        run_id=run_id,
                        case=case,
                        engine=engine,
                        phase=phase,
                    )
                if receipt["input_file_sha256"] != input_hash:
                    raise ValueError("frozen input changed after preparation")
                if (
                    receipt["pid"] != execution["pid"]
                    or receipt["library_sha256"]
                    != report["provenance"]["library_sha256"]
                ):
                    raise ValueError(
                        "worker PID/library identity differs from preflight"
                    )
                if (
                    receipt["native_identity"]["source_identity"]
                    != environment["MDJ_EXPECTED_SOURCE_IDENTITY"]
                    or receipt["native_identity"]["loaded_library_sha256"]
                    != report["provenance"]["library_sha256"]
                ):
                    raise ValueError(
                        "actual loaded native library/source differs from preflight"
                    )
                row.update(receipt)
                if engine.startswith("endpoint-"):
                    endpoint_rows.append(
                        {
                            **receipt["endpoint_result"],
                            "repetition": repetition,
                            "pid": receipt["pid"],
                            "library_sha256": receipt["library_sha256"],
                            "source_identity": report["provenance"]["source"],
                            "input_sha256": input_hash,
                            "run_id": run_id,
                            "process_status": execution["status"],
                            "process_wall_seconds": execution["process_wall_seconds"],
                            "sampled_peak_process_gpu_bytes": execution[
                                "sampled_peak_process_gpu_bytes"
                            ],
                            "peak_host_rss_bytes": receipt["peak_host_rss_bytes"],
                        }
                    )
                    write_json(output, report)
                    continue
                import numpy as np

                matrix = np.load(receipt["matrix_file"], allow_pickle=False)
                if engine == "oracle":
                    oracle = matrix
                else:
                    if engine in NATIVE_ENGINES:
                        from benchmarks._md_j_engine_endpoints import _work_receipt

                        row["execution"] = _work_receipt(
                            Path(execution["stderr"]).read_text(), engine
                        )
                    row["comparison"] = numerical_comparison(
                        matrix, oracle, args.atol, args.rtol
                    )
                    case_rows.append(row)
                write_json(output, report)
            case_summary = {}
            if args.scope in ("j", "all"):
                case_summary["j"] = summarize_case(
                    case_rows,
                    cold_repeats=args.cold_repeats,
                    warm_repeats=args.warm_repeats,
                    engines=engines,
                    minimum_speedup=args.minimum_speedup,
                )
            if args.scope in ("energy-force", "all"):
                from benchmarks._md_j_engine_endpoints import validate_endpoint_receipts

                endpoint_summary = validate_endpoint_receipts(
                    endpoint_rows, minimum_samples=args.cold_repeats
                )
                failures = endpoint_summary.setdefault("failures", [])
                for phase_name, ratio in endpoint_summary.get(
                    "incumbent_over_reference", {}
                ).items():
                    if not math.isfinite(ratio) or ratio < args.minimum_speedup:
                        failures.append(
                            f"{phase_name}: reference does not meet {args.minimum_speedup:g}x incumbent/reference gate"
                        )
                if not endpoint_summary.get("incumbent_over_reference"):
                    failures.append("complete endpoint timing comparison is missing")
                if any(
                    row.get("sampled_peak_process_gpu_bytes") is None
                    or row.get("peak_host_rss_bytes", 0) <= 0
                    for row in endpoint_rows
                ):
                    failures.append("complete endpoint memory evidence is missing")
                endpoint_summary["performance_acceptance_passed"] = (
                    endpoint_summary.get("status") == "passed" and not failures
                )
                case_summary["energy-force"] = endpoint_summary
            case_summary["performance_acceptance_passed"] = all(
                part["performance_acceptance_passed"] for part in case_summary.values()
            )
            report["cases"][case] = case_summary
            write_json(output, report)
        report["provenance_after"] = source_identity()
        if (
            report["provenance_after"] != report["provenance"]["source"]
            or sha256(Path(report["provenance"]["library"]))
            != report["provenance"]["library_sha256"]
        ):
            raise ValueError("source or native binary changed during the campaign")
        report["performance_acceptance_passed"] = len(report["cases"]) == len(
            args.cases
        ) and all(
            case["performance_acceptance_passed"] for case in report["cases"].values()
        )
        report["status"] = (
            "passed" if report["performance_acceptance_passed"] else "failed"
        )
    except (
        OSError,
        ValueError,
        ImportError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        report["status"] = "failed"
        report["gate_failures"].append(f"{type(error).__name__}: {error}")
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the explicit campaign, or its isolated one-purpose subprocess."""
    args = parse_args(argv)
    if args.worker:
        # A worker cannot bypass the allocation prerequisite via internal flags.
        from benchmarks.md_j_engine_acceptance import _finite_slurm

        _finite_slurm()
        if args.worker == "prepare":
            _prepare(
                CASES[args.cases[0]],
                args.output,
                args.run_id,
                prepare_bridge=args.scope != "energy-force",
            )
        elif args.worker == "endpoint":
            _endpoint_worker(args)
        else:
            _evaluate_worker(args)
        return
    try:
        report = run_campaign(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "performance_acceptance_passed", "gate_failures")
            }
        )
    )
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    # Direct-script invocation must resolve this checkout's benchmark helpers.
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT))
    main()
