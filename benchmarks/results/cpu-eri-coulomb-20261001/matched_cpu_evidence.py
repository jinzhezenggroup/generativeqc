#!/usr/bin/env python3
"""Run and audit the unchanged 2026-10-01 CPU ERI endpoint protocol.

Evidence-only helper. No production files are written.
Run --help. The run command requires completed builds and a quiet-host attestation.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fnmatch
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "cpu-eri-20261001"
HASHES = {
    "benchmark.py": "91211fa69c5e7ede5a0f5e2c9a0849cd7a80251553712af172708f8925698ff3",
    "run.py": "cabe7fddf62dfbd198e34d22f182f96874af04575775adc1290b816234175b7d",
    "cases.json": "44e5cfd759a0a9da50840e8ef8fb89dfea7489a1c69f25b1a1a4b6ed51ff1f86",
}
CASES = ["h2-sto3g", "water-sto3g", "water-svp", "formaldehyde-svp"]
PHASES = ["cold", "warm", "warm", "changed_geometry"]
ENGINES = ["baseline", "candidate", "pyscf-auto"]
THREAD_ENV = dict(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONHASHSEED="0",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "args": [str(a) for a in args],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def environment(source=None, library=None):
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    # Reject hidden GQC scientific overrides; defaults remain the frozen protocol's.
    unexpected = {
        k: v
        for k, v in env.items()
        if k.startswith(("GENERATIVEQC_", "GQC_"))
        and k not in {"GENERATIVEQC_LIBRARY", "GQC_SOURCE_ROOT", "GENERATIVEQC_PROFILE"}
    }
    if unexpected:
        raise RuntimeError(f"Unset unexpected scientific overrides: {unexpected}")
    env.update(THREAD_ENV)
    env["GENERATIVEQC_PROFILE"] = "off"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if source is not None:
        env["GQC_SOURCE_ROOT"] = str(Path(source).resolve())
        env["PYTHONPATH"] = str(Path(source).resolve() / "python")
    if library is not None:
        env["GENERATIVEQC_LIBRARY"] = str(Path(library).resolve())
    return env


def frozen_check(directory):
    found = {name: sha(directory / name) for name in HASHES}
    if found != HASHES:
        raise RuntimeError(f"Frozen inputs changed: expected {HASHES}; found {found}")
    if list(json.loads((directory / "cases.json").read_text())) != CASES:
        raise RuntimeError("Original four-case inventory changed")
    return found


def source_inventory(root):
    """Independently expand and hash the declared manifest, without importing GQC."""
    root = Path(root).resolve()
    relative_manifest = Path("cmake/GenerativeQCSourceIdentity.json")
    data = json.loads((root / relative_manifest).read_text())
    if data.get("schema_version") != 1:
        raise RuntimeError("Unsupported source identity manifest schema")
    paths = {relative_manifest}
    for group in data["recursive_groups"]:
        relative = Path(group["root"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe source inventory root")
        directory = root / relative
        if not directory.is_dir() or not group["patterns"]:
            raise RuntimeError(f"Incomplete source snapshot: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and any(
                fnmatch.fnmatch(path.relative_to(directory).as_posix(), p)
                for p in group["patterns"]
            ):
                paths.add(path.resolve().relative_to(root))
    for name in data["files"]:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (root / relative).is_file()
        ):
            raise RuntimeError(f"Missing/unsafe source inventory file: {name}")
        paths.add(relative)
    entries = [
        {"path": p.as_posix(), "sha256": sha(root / p)}
        for p in sorted(paths, key=lambda p: p.as_posix())
    ]
    payload = "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()
    return {"identity": hashlib.sha256(payload).hexdigest(), "files": entries}


def host():
    cpuinfo = (
        Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    )
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    return {
        "utc": utc(),
        "node": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_model": model,
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_average": list(os.getloadavg()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").exists()
        else None,
    }


def competing_work():
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            name = Path(args[0]).name
            build = name in {
                "ninja",
                "make",
                "gmake",
                "cc1",
                "cc1plus",
                "g++",
                "c++",
                "gcc",
                "clang",
                "clang++",
                "nvcc",
            }
            build |= name == "cmake" and "--build" in args
            tests = name.startswith("pytest") or (
                name.startswith("python") and "pytest" in args
            )
            endpoint = name.startswith("python") and any(
                Path(a).name == "benchmark.py" for a in args[1:]
            )
            if build or tests or endpoint:
                found.append({"pid": int(proc.name), "args": args})
        except (OSError, IndexError):
            pass
    return found


def ensure_quiet():
    # /proc can be namespaced per tool execution. This is a best-effort check,
    # never a replacement for the parent coordinating all terminal states.
    work = competing_work()
    if work:
        raise RuntimeError(
            f"Build/test/endpoint processes still active; finish them first: {work}"
        )


def identity_worker(source, library):
    # Executed in its own process before timing: cannot populate timed-process caches.
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if not type(f).__module__.startswith("_editable_skbc_generativeqc")
    ]
    sys.path.insert(0, str(Path(source).resolve() / "python"))
    import generativeqc
    from generativeqc import _native
    from threadpoolctl import threadpool_info

    lib = _native.load_library()
    loaded = Path(lib._name).resolve()
    package = Path(generativeqc.__file__).resolve()
    if loaded != Path(library).resolve():
        raise RuntimeError(f"Wrong library loaded: {loaded}")
    if package != Path(source).resolve() / "python/generativeqc/__init__.py":
        raise RuntimeError(f"Wrong Python package loaded: {package}")
    pools = threadpool_info()
    if not pools or any(p["num_threads"] != 1 for p in pools):
        raise RuntimeError(f"Expected one-thread numerical pools: {pools}")
    return {
        "package_file": str(package),
        "native_library": str(loaded),
        "native_sha256": sha(loaded),
        "native_source_identity": lib.generativeqc_get_source_identity().decode(),
        "threadpool_info": pools,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
    }


def inspect(source, library):
    source, library = Path(source).resolve(), Path(library).resolve()
    if not library.is_file():
        raise RuntimeError(f"Native library does not exist: {library}")
    inventory = source_inventory(source)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_identity",
            "--source",
            str(source),
            "--library",
            str(library),
        ],
        env=environment(source, library),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"Identity subprocess failed:\n{result.stdout}\n{result.stderr}"
        )
    loaded = json.loads(result.stdout)
    if loaded["native_source_identity"] != inventory["identity"]:
        raise RuntimeError(
            f"Source/library mismatch: {source} inventory {inventory['identity']} "
            f"versus {library} embedded {loaded['native_source_identity']}"
        )
    cache = library.parent / "CMakeCache.txt"
    config = cache.read_text() if cache.is_file() else None
    if config is not None and "GENERATIVEQC_ENABLE_CUDA:BOOL=OFF" not in config:
        raise RuntimeError("Expected the explicit CPU-only build")
    return {
        "source_root": str(source),
        "library": str(library),
        "library_bytes": library.stat().st_size,
        "loaded": loaded,
        "source_inventory": inventory,
        "commit": command(["git", "rev-parse", "HEAD"], source),
        "tree": command(["git", "rev-parse", "HEAD^{tree}"], source),
        "git_status": command(["git", "status", "--porcelain=v1"], source),
        "tracked_diff": command(["git", "diff", "HEAD", "--"], source),
        "cmake_cache": config,
        "cmake_cache_sha256": sha(cache) if cache.is_file() else None,
        "dynamic_dependencies": command(["ldd", str(library)]),
    }


def sample_gate(payload, engine, case, provenance):
    rows, meta = payload["rows"], payload["metadata"]
    if [r["phase"] for r in rows] != PHASES or [r["index"] for r in rows] != list(
        range(4)
    ):
        raise RuntimeError("Endpoint phases or count differ from frozen protocol")
    if meta["case"] != case or any(r["case"] != case for r in rows):
        raise RuntimeError("Wrong case")
    pools = meta["threadpool_info"]
    if not pools or any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"Not single threaded: {pools}")
    if meta["thread_env"] != THREAD_ENV:
        raise RuntimeError(f"Unexpected thread environment: {meta['thread_env']}")
    for row in rows:
        if not row["converged"] or not math.isfinite(row["energy"]):
            raise RuntimeError("Unconverged or nonfinite endpoint")
        if abs(row["energy_change"]) >= 1e-10 or row["density_rms"] >= 1e-8:
            raise RuntimeError(f"Endpoint convergence gate failed: {row}")
        if engine == "pyscf-auto" and not row["eri_incore"]:
            raise RuntimeError("PySCF did not materialize in-core ERIs")
        if engine != "pyscf-auto":
            precision = row["precision"]
            if (
                precision.get("requested_mode") != "fp64"
                or precision.get("effective_bits") != 64
                or row["backend"] != "cpu_reference"
            ):
                raise RuntimeError(f"Unexpected native backend/precision: {row}")
            if row["incremental_direct_jk"]["active"]:
                raise RuntimeError("Unexpected incremental-direct J/K")
            if row["basis_metadata"]["orbital"]["representation"] != "spherical":
                raise RuntimeError("Expected spherical AO representation")
    if engine != "pyscf-auto":
        expected = provenance[engine]["loaded"]
        for field in [
            "package_file",
            "native_library",
            "native_sha256",
            "native_source_identity",
        ]:
            if meta[field] != expected[field]:
                raise RuntimeError(f"Loaded identity drifted: {field}")


def distribution(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def summarize(directory):
    directory = Path(directory).resolve()
    status = json.loads((directory / "run-status.json").read_text())
    grouped = collections.defaultdict(list)
    observations, incomplete = [], []
    for process in status:
        path = directory / process["json_file"]
        if not path.is_file():
            # Failed subprocesses may only have flushed row JSON into their log.
            partial = []
            for line in (directory / process["log_file"]).read_text().splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "phase" in row:
                        partial.append(row)
                except json.JSONDecodeError:
                    pass
            incomplete.append({"process": process, "partial_rows": partial})
            continue
        payload = json.loads(path.read_text())
        observations.append({"process": process, **payload})
        for row in payload["rows"]:
            grouped[(process["case"], row["phase"], process["engine"])].append(
                {"repeat": process["repeat"], **row}
            )
    cases = {}
    errors = []
    for case in CASES:
        cases[case] = {}
        for phase in dict.fromkeys(PHASES):
            entry = {}
            for engine in ENGINES:
                rows = grouped[case, phase, engine]
                if not rows:
                    continue
                entry[engine] = {
                    field: distribution([r[field] for r in rows])
                    for field in [
                        "wall_seconds",
                        "cpu_seconds",
                        "setup_wall_seconds",
                        "scf_wall_seconds",
                    ]
                }
                entry[engine].update(
                    rows=rows,
                    all_converged=all(r["converged"] for r in rows),
                    iterations=[r["iterations"] for r in rows],
                    energies=[r["energy"] for r in rows],
                    fock_builds=[
                        r.get("native_fock_builds_source_derived", r.get("jk_calls"))
                        for r in rows
                    ],
                )
            for reference in ["baseline", "pyscf-auto"]:
                for candidate, ref in itertools.product(
                    grouped[case, phase, "candidate"], grouped[case, phase, reference]
                ):
                    errors.append(
                        {
                            "case": case,
                            "phase": phase,
                            "reference_engine": reference,
                            "candidate_repeat": candidate["repeat"],
                            "candidate_index": candidate["index"],
                            "reference_repeat": ref["repeat"],
                            "reference_index": ref["index"],
                            "candidate_energy": candidate["energy"],
                            "reference_energy": ref["energy"],
                            "absolute_energy_error_hartree": abs(
                                candidate["energy"] - ref["energy"]
                            ),
                            "force_error": None,
                            "same_iterations": candidate["iterations"]
                            == ref["iterations"],
                        }
                    )
            if all(e in entry for e in ENGINES):
                medians = {e: entry[e]["wall_seconds"]["median"] for e in ENGINES}
                entry["baseline_over_candidate_median"] = (
                    medians["baseline"] / medians["candidate"]
                )
                entry["candidate_over_pyscf_median"] = (
                    medians["candidate"] / medians["pyscf-auto"]
                )
                pairs = []
                for repeat in sorted(
                    {r["repeat"] for r in grouped[case, phase, "baseline"]}
                ):
                    b = [
                        r["wall_seconds"]
                        for r in grouped[case, phase, "baseline"]
                        if r["repeat"] == repeat
                    ]
                    c = [
                        r["wall_seconds"]
                        for r in grouped[case, phase, "candidate"]
                        if r["repeat"] == repeat
                    ]
                    if b and c:
                        pairs.append(
                            {
                                "repeat": repeat,
                                "baseline": statistics.median(b),
                                "candidate": statistics.median(c),
                                "candidate_over_baseline": statistics.median(c)
                                / statistics.median(b),
                            }
                        )
                entry["paired_process_phase_medians"] = pairs
            cases[case][phase] = entry
    manifest = json.loads((directory / "campaign.json").read_text())
    expected_processes = manifest["repeats"] * len(CASES) * len(ENGINES)
    max_errors = {
        engine: max(
            (
                e["absolute_energy_error_hartree"]
                for e in errors
                if e["reference_engine"] == engine
            ),
            default=None,
        )
        for engine in ["baseline", "pyscf-auto"]
    }
    all_rows = [r for o in observations for r in o["rows"]]
    numerical_pass = bool(errors) and all(
        e["absolute_energy_error_hartree"] <= manifest["energy_gate_hartree"]
        for e in errors
    )
    complete = (
        len(observations) == expected_processes
        and len(all_rows) == 4 * expected_processes
    )
    summary = {
        "schema_version": 1,
        "generated_utc": utc(),
        "complete": complete,
        "expected_processes": expected_processes,
        "completed_processes": len(observations),
        "endpoint_count": len(all_rows),
        "all_converged": bool(all_rows) and all(r["converged"] for r in all_rows),
        "all_processes_passed": complete
        and all(s.get("validated", False) for s in status),
        "provenance_stable": manifest.get("provenance_stable", False),
        "campaign_error": manifest.get("error"),
        "numerical_gate_hartree": manifest["energy_gate_hartree"],
        "all_repeat_energy_gate_passed": numerical_pass,
        "maximum_errors": max_errors,
        "native_iteration_counts_unchanged": bool(errors)
        and all(
            e["same_iterations"] for e in errors if e["reference_engine"] == "baseline"
        ),
        "outliers_removed": 0,
        "cases": cases,
        "all_repeat_energy_errors": errors,
        "incomplete_processes": incomplete,
        "limits": [
            "Shared host; all observations and outliers retained; no speedup claim follows from medians alone.",
            "Warm calls in one process are correlated; paired_process_phase_medians preserve the process unit.",
            "Process-cold does not mean disk-cache-cold; backend import is retained separately.",
            "In-core energy-only HF; no DFT/force/GPU/scaling performance claim.",
            "Native fock counts are source-derived; PySCF J/K calls are measured. Screening work equality is not claimed.",
        ],
    }
    write_json(directory / "observations.json", observations)
    write_json(directory / "summary.json", summary)
    lines = ["case phase baseline_s candidate_s pyscf_s baseline/candidate"]
    for case, phases in cases.items():
        for phase, entry in phases.items():
            if all(e in entry for e in ENGINES):
                values = " ".join(
                    f"{entry[e]['wall_seconds']['median']:.9f}" for e in ENGINES
                )
                lines.append(
                    f"{case} {phase} {values} {entry['baseline_over_candidate_median']:.4f}"
                )
    lines.extend(
        [
            f"All-repeat energy maxima (Ha): {max_errors}",
            f"Complete: {complete}; numerical gate passed: {numerical_pass}",
            "Every outlier retained. No confidence interval or robust gain claim implied.",
        ]
    )
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    return summary


def run(args):
    if not args.quiet_host_confirmed:
        raise RuntimeError(
            "Wait for all builds/tests/other timings, then supply --quiet-host-confirmed"
        )
    ensure_quiet()
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            "Use a new empty output directory; campaigns are never overwritten/resumed"
        )
    directory.mkdir(parents=True, exist_ok=True)
    hashes = frozen_check(args.frozen_directory)
    frozen = directory / "frozen"
    frozen.mkdir()
    for name in HASHES:
        shutil.copyfile(args.frozen_directory / name, frozen / name)
    provenance = {
        engine: inspect(
            getattr(args, f"{engine}_source"), getattr(args, f"{engine}_library")
        )
        for engine in ["baseline", "candidate"]
    }
    b, c = (provenance[e]["loaded"] for e in ["baseline", "candidate"])
    providers = lambda d: sorted(
        (p["internal_api"], p.get("prefix"), p["filepath"])
        for p in d["threadpool_info"]
    )
    if providers(b) != providers(c):
        raise RuntimeError("Baseline/candidate loaded different numerical providers")
    campaign = {
        "schema_version": 1,
        "start": host(),
        "repeats": args.repeats,
        "energy_gate_hartree": args.energy_gate,
        "frozen_hashes": hashes,
        "runner_sha256": sha(__file__),
        "runner_path": str(Path(__file__).resolve()),
        "thread_environment": THREAD_ENV,
        "provenance": provenance,
        "quiet_host_attested": True,
        "timeout_seconds": args.timeout,
        "quiet_check_limit": "Process namespaces can hide other exec activity; parent coordination and explicit attestation are authoritative.",
        "engine_order": "baseline,candidate,pyscf-auto; reversed on alternate rounds",
        "expected_endpoints": args.repeats * 4 * 3 * 4,
    }
    write_json(directory / "campaign.json", campaign)
    status = []
    write_json(directory / "run-status.json", status)
    error = None
    try:
        for repeat in range(args.repeats):
            for case in CASES:
                for engine in ENGINES[:: (-1 if repeat % 2 else 1)]:
                    ensure_quiet()
                    native = "candidate" if engine == "candidate" else "baseline"
                    selected = provenance[native]
                    stem = f"{case}-{engine}-{repeat + 1}"
                    row = {
                        "case": case,
                        "engine": engine,
                        "repeat": repeat + 1,
                        "json_file": f"{stem}.json",
                        "log_file": f"{stem}.log",
                        "start_utc": utc(),
                    }
                    args_run = [
                        sys.executable,
                        str(frozen / "benchmark.py"),
                        "pyscf-auto" if engine == "pyscf-auto" else "gqc",
                        case,
                        str(directory / row["json_file"]),
                    ]
                    row["command"] = args_run
                    start = time.perf_counter()
                    with (directory / row["log_file"]).open("w") as log:
                        try:
                            result = subprocess.run(
                                args_run,
                                env=environment(
                                    selected["source_root"], selected["library"]
                                ),
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout,
                                check=False,
                            )
                            row["returncode"] = result.returncode
                        except subprocess.TimeoutExpired:
                            row["returncode"] = None
                            row["error"] = "subprocess timeout"
                    row["process_wall_seconds"] = time.perf_counter() - start
                    row["end_utc"] = utc()
                    row["validated"] = False
                    if row["returncode"] == 0:
                        try:
                            sample_gate(
                                json.loads((directory / row["json_file"]).read_text()),
                                engine,
                                case,
                                provenance,
                            )
                            row["validated"] = True
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                    status.append(row)
                    write_json(directory / "run-status.json", status)
                    print(json.dumps(row), flush=True)
                    if not row["validated"]:
                        raise RuntimeError(
                            f"Endpoint process failed qualification: {stem}; retained log/rows"
                        )
        for engine in ["baseline", "candidate"]:
            old = provenance[engine]
            new = inspect(old["source_root"], old["library"])
            if (
                new["source_inventory"] != old["source_inventory"]
                or new["loaded"] != old["loaded"]
            ):
                raise RuntimeError(
                    f"Source/library identity drifted during campaign: {engine}"
                )
        frozen_check(frozen)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        campaign["end"] = host()
        campaign["error"] = error
        campaign["provenance_stable"] = error is None
        write_json(directory / "campaign.json", campaign)
        summary = summarize(directory)
    if not summary["all_repeat_energy_gate_passed"]:
        raise RuntimeError(
            "All-repeat numerical energy gate failed; all errors retained"
        )
    print((directory / "summary.txt").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser(
        "run", help="Run independent interleaved processes; unchanged frozen endpoint"
    )
    for engine in ["baseline", "candidate"]:
        p.add_argument(f"--{engine}-source", type=Path, required=True)
        p.add_argument(f"--{engine}-library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frozen-directory", type=Path, default=FROZEN)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--energy-gate", type=float, default=1e-9)
    p.add_argument("--quiet-host-confirmed", action="store_true")
    for mode in ["preflight", "_identity"]:
        p = sub.add_parser(mode)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--library", type=Path, required=True)
        if mode == "preflight":
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        if args.repeats < 1 or args.timeout <= 0 or args.energy_gate <= 0:
            parser.error("repeats, timeout and energy-gate must be positive")
        run(args)
    elif args.mode == "preflight":
        write_json(args.output, {"host": host(), **inspect(args.source, args.library)})
        print(args.output)
    elif args.mode == "_identity":
        print(json.dumps(identity_worker(args.source, args.library)))
    else:
        summarize(args.output)
        print((args.output / "summary.txt").read_text())


if __name__ == "__main__":
    main()
