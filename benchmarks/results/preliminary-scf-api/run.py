"""Prepared public-API comparison. No calculation occurs without --run.

One newly launched process constructs Calculator, prepares a one-item batch and
executes exactly once. Density export happens after endpoint timing. Never
reuses prototype timings or invokes the prototype's native bridges.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
CASES = ("water-631g", "water-tetramer-631g")
MODES = ("direct", "hf", "lda")


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_payload(
    value: object, path: str = "", nonfinite: list | None = None
) -> object:
    """Keep failed numerical outputs without writing non-standard NaN JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite is not None:
            nonfinite.append({"path": path, "value": repr(value)})
        return None
    if isinstance(value, dict):
        return {
            key: finite_payload(item, path + "/" + str(key), nonfinite)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            finite_payload(item, path + "/" + str(i), nonfinite)
            for i, item in enumerate(value)
        ]
    return value


def export_final_density(batch: object, expected_nao: int) -> dict:
    # Existing read-only ABI, implemented for DFT batches as well as HF. This
    # does not construct or claim an HF checkpoint for a PBE0 physical model.
    from generativeqc import _native

    state = _native.HfWarmState(
        struct_size=ctypes.sizeof(_native.HfWarmState), abi_version=_native.ABI_VERSION
    )
    query = batch._library.generativeqc_batch_get_hf_warm_state
    _native.check(batch._library, query(batch._batch, 0, ctypes.byref(state)))
    if not state.present:
        raise RuntimeError(
            "successful fresh target did not expose a retained final density"
        )
    if state.density_count != expected_nao**2:
        raise RuntimeError(f"unexpected total AO density size: {state.density_count}")
    density = (ctypes.c_double * state.density_count)()
    coordinates = (ctypes.c_double * state.coordinate_count)()
    state.density, state.coordinates = density, coordinates
    _native.check(batch._library, query(batch._batch, 0, ctypes.byref(state)))
    values = list(density)
    return {
        "method": "generativeqc_batch_get_hf_warm_state read-only buffer export from DFT batch",
        "density": values,
        "density_sha256": canonical(values),
        "density_shape": [expected_nao, expected_nao],
        "density_convention": "restricted total AO density, C row-major, Cartesian normalized shells",
        "coordinates": list(coordinates),
        "energy": state.energy,
        "energy_change": state.energy_change,
        "density_rms": state.density_rms,
        "iterations": state.iterations,
    }


def accepted(row: dict) -> bool:
    checks = row.get("checks", {})
    return (
        row.get("status") == "completed"
        and bool(checks)
        and all(value is True for value in checks.values())
        and not row.get("nonfinite_fields")
        and not row.get("cleanup_error")
        and row.get("returncode", 0) == 0
    )


def worker(args: argparse.Namespace) -> int:
    entered = time.perf_counter()
    row = {
        "case": args.case,
        "mode": args.mode,
        "repeat": args.repeat,
        "thermal": "fresh-process-first-api-endpoint",
        "worker_enter_perf": entered,
        "launch_to_worker_s": entered - args.launch_start_perf,
        "status": "failed",
        "phase": "api_import",
    }
    sys.path[:0] = [str(args.source_root / "python"), str(args.source_root)]
    batch = None
    endpoint_start = None
    try:
        imported = time.perf_counter()
        from generativeqc import (
            Calculator,
            GridSpec,
            InitialGuessSpec,
            KsOptions,
            Primitive,
            Shell,
        )

        row["api_import_s"] = time.perf_counter() - imported
        # Import is separately reported. All argument/input materialization,
        # Calculator construction, native preparation and execution are timed.
        endpoint_start = time.perf_counter()
        row["phase"] = "input_materialization"
        fixture = json.loads((HERE / "inputs.json").read_text())
        case = fixture["cases"][args.case]
        atoms = [(z, tuple(xyz)) for z, xyz in case["atoms"]]
        shells = [
            Shell(
                i,
                shell["angular_momentum"],
                tuple(Primitive(*p) for p in shell["primitives"]),
            )
            for i, (z, _) in enumerate(atoms)
            for shell in fixture["basis_per_element"][str(z)]
        ]
        row["input_materialization_s"] = time.perf_counter() - endpoint_start
        row["phase"] = "constructor"
        phase_start = time.perf_counter()
        initial_guess = None if args.mode == "direct" else InitialGuessSpec(args.mode)
        calc = Calculator(
            "pbe0-rks",
            basis=shells,
            device="cpu",
            basis_representation="cartesian",
            density_fitting="none",
            max_iterations=150,
            diis_history=8,
            energy_tolerance=1e-10,
            density_tolerance=1e-8,
            screening_tolerance=1e-12,
            precision="fp64",
            ks_options=KsOptions(grid=GridSpec(24, 12, 24)),
            initial_guess=initial_guess,
        )
        row["constructor_s"] = time.perf_counter() - phase_start
        row["phase"] = "prepare"
        phase_start = time.perf_counter()
        # Equal final-state retention permits export. Every process starts a
        # new batch, imports no state, and executes once, so no warm seed exists.
        batch = calc.prepare_batch(
            [atoms], charges=[0], multiplicities=[1], warm_start=True
        )
        row["prepare_s"] = time.perf_counter() - phase_start
        row["phase"] = "execute"
        phase_start = time.perf_counter()
        result = batch.execute(properties=("energy",), strict=False).items[0]
        row["execute_s"] = time.perf_counter() - phase_start
        row["endpoint_total_s"] = time.perf_counter() - endpoint_start
        row["phase"] = "post_endpoint_diagnostics"
        row["result"] = asdict(result)
        row["policy"] = None if initial_guess is None else initial_guess.to_payload()
        row["resource_diagnostics"] = batch.resource_diagnostics
        row["configured_target_grid"] = asdict(GridSpec(24, 12, 24))
        row["nao_expected"] = case["expected_nao"]
        row["grid_points_expected"] = case["expected_grid_points"]
        orbital_metadata = result.basis_metadata.get("orbital", {})
        row["basis_counts"] = {
            "shell_count": orbital_metadata.get("shell_count"),
            "primitive_count": orbital_metadata.get("primitive_count"),
            "input_cartesian_ao_count": sum(
                (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2 for s in shells
            ),
        }
        row["checks"] = {
            "shell_count_matches": orbital_metadata.get("shell_count")
            == case["expected_shell_count"],
            "primitive_count_matches": orbital_metadata.get("primitive_count")
            == case["expected_primitive_count"],
            "input_ao_count_matches": row["basis_counts"]["input_cartesian_ao_count"]
            == case["expected_nao"],
            "direct_policy_disabled": args.mode != "direct"
            or result.initial_guess is None,
            "reported_residual_passes": result.ks_diagnostic is not None
            and result.ks_diagnostic.physical_residual_max <= 1e-8,
            "target_succeeded": result.succeeded,
            "target_converged": result.converged,
            "fresh_state": not result.warm_start_used
            and result.restart_origin == "cold",
            "cpu_backend": result.executed_backend == "cpu_reference",
            "correct_grid": result.ks_diagnostic is not None
            and result.ks_diagnostic.grid_points == case["expected_grid_points"],
            "initial_guess_used_when_requested": args.mode == "direct"
            or (
                result.initial_guess is not None
                and result.initial_guess["outcome"] == "used"
            ),
        }
        row["status"] = "completed" if result.succeeded else "failed"
        if result.succeeded:
            phase_start = time.perf_counter()
            try:
                row["final_state"] = export_final_density(batch, case["expected_nao"])
                row["checks"]["density_available"] = True
                row["checks"]["export_ao_count_matches"] = (
                    row["final_state"]["density_shape"] == [case["expected_nao"]] * 2
                )
                row["checks"]["export_energy_matches"] = (
                    row["final_state"]["energy"] == result.energy
                )
                row["checks"]["export_coordinates_match"] = row["final_state"][
                    "coordinates"
                ] == [x for _, xyz in atoms for x in xyz]
            except Exception as error:  # noqa: BLE001 - preserve every failed attempt
                row["density_export_error"] = f"{type(error).__name__}: {error}"
                row["checks"]["density_available"] = False
            row["density_export_s"] = time.perf_counter() - phase_start
        row["phase"] = "finished"
    except Exception as error:  # noqa: BLE001 - preserve every failed attempt
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()
        if endpoint_start is not None and "endpoint_total_s" not in row:
            row["endpoint_total_s"] = time.perf_counter() - endpoint_start
    finally:
        if batch is not None:
            cleanup = time.perf_counter()
            try:
                batch.close()
            except Exception as error:  # noqa: BLE001 - preserve every failed attempt
                row["cleanup_error"] = repr(error)
            row["cleanup_s"] = time.perf_counter() - cleanup
    row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row["peak_rss_scope"] = (
        "Linux process high-water including imports, endpoint and export, not exclusive endpoint allocations"
    )
    row["worker_elapsed_s"] = time.perf_counter() - entered
    nonfinite = []
    row = finite_payload(row, nonfinite=nonfinite)
    row["nonfinite_fields"] = nonfinite
    row["accepted"] = accepted(row)
    dump(args.record, row)
    return 0 if row["accepted"] else 1


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def source_snapshot(root: Path) -> dict:
    paths = ("src", "include", "python", "cmake", "CMakeLists.txt")
    tracked = git(root, "ls-files", "-z", "--", *paths).decode().split("\0")
    untracked = (
        git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *paths)
        .decode()
        .split("\0")
    )
    hashes = {
        path: sha(root / path)
        for path in sorted(set(tracked + untracked))
        if path and (root / path).is_file()
    }
    return {
        "revision": git(root, "rev-parse", "HEAD").decode().strip(),
        "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "dirty": bool(git(root, "status", "--porcelain")),
        "source_files": hashes,
        "source_hash": canonical(hashes),
        "untracked_source_files": [p for p in untracked if p],
    }


def finish_evidence(
    output: Path, rows: list[dict], provenance: dict, fixture: dict
) -> bool:
    from statistics import median

    from generativeqc_compiler.common.evidence import (
        block_error,
        new_evidence,
        outcome,
        write_evidence,
    )

    evidence = new_evidence(
        tier="endpoint",
        subject="Actual preliminary-SCF Calculator API: fresh single-item prepared batches",
        inputs_hash=canonical(fixture),
    )
    evidence["revision"] = provenance["source"]["revision"]
    evidence["hashes"]["source"] = provenance["source"]["source_hash"]
    evidence["hash_reasons"] = {
        "ir": "No generated IR separately retained",
        "equation": "Exact target definitions retained in inputs.json",
        "schedule": "Exact process order retained in provenance.json",
    }
    evidence["device"] = provenance["device"]
    evidence["backend_selected"] = "cpu"
    evidence["toolchain"] = {
        "library_sha256": provenance["library_sha256"],
        "python": sys.version,
    }
    evidence["settings"] = {
        "fixture": fixture,
        "comparison_kind": "solver",
        "scope": "New API only; unrelated to frozen prototype timing records",
        "endpoint_boundary": provenance["endpoint_boundary"],
        "repeats": 3,
        "residual_limit": 1e-8,
    }
    evidence["hardware"] = outcome(
        "not-run",
        "CPU identity/affinity captured; no exclusive CPU allocation or load isolation",
    )
    evidence["performance"] = outcome(
        "not-run",
        "Three repeats below the five-pair promotion gate; descriptive fresh-process API comparison only; no independent full-matrix oracle or measured compilation cost",
    )
    evidence["stages"]["endpoint"] = outcome(
        "pass" if len(rows) == 18 and all(accepted(r) for r in rows) else "fail",
        "18 requested API endpoint attempts; failures preserved",
    )
    evidence["stages"]["source"] = outcome(
        "pass"
        if provenance["source_unchanged"] and provenance["library_unchanged"]
        else "fail",
        "Source snapshot and library hash compared before/after campaign",
    )
    evidence["stages"]["production"] = outcome(
        "not-run", "No default or broad performance qualification"
    )
    evidence["solver_trace_reason"] = (
        "Final target KS iteration history retained; preliminary trajectories not exported; failed attempts have their native diagnostic coverage"
    )
    summaries = []
    for sample_index, row in enumerate(rows):
        if row.get("endpoint_total_s", 0) <= 0:
            continue
        diagnostic = {k: v for k, v in row.items() if k != "final_state"}
        if "final_state" in row:
            diagnostic["final_density_sha256"] = row["final_state"]["density_sha256"]
        evidence["timings"].append(
            {
                "selection": "baseline" if row["mode"] == "direct" else "candidate",
                "seconds": row["endpoint_total_s"],
                "inputs_hash": canonical(
                    {
                        "case": fixture["cases"][row["case"]],
                        "basis": fixture["basis_per_element"],
                        "target": fixture["target"],
                    }
                ),
                "workload": "cold-start",
                "synchronized": True,
                "diagnostics": diagnostic,
            }
        )
        ks = row.get("result", {}).get("ks_diagnostic")
        if ks:
            evidence["residuals"][str(sample_index)] = {
                "value": ks["physical_residual_max"],
                "independently_evaluated": False,
            }
            for iteration in ks["history"]:
                evidence["solver_iterations"].append(
                    {
                        "sample_index": sample_index,
                        "iteration": iteration["iteration"],
                        "energy": sum(iteration["components"].values()),
                        "residual": iteration["physical_residual_max"],
                    }
                )
    complete_pairs = 0
    for case in CASES:
        for mode in MODES:
            selected = [r for r in rows if r["case"] == case and r["mode"] == mode]
            successful = [r for r in selected if r["status"] == "completed"]
            summary = {
                "case": case,
                "mode": mode,
                "attempts": len(selected),
                "completed": len(successful),
                "endpoint_total_samples_s": [
                    r.get("endpoint_total_s") for r in selected
                ],
            }
            if successful:
                summary["median_endpoint_total_s"] = median(
                    r["endpoint_total_s"] for r in successful
                )
                summary["target_iterations"] = [
                    r["result"]["iterations"] for r in successful
                ]
                summary["initial_guess_diagnostics"] = [
                    r["result"]["initial_guess"] for r in successful
                ]
            for row in successful:
                if mode == "direct":
                    continue
                baseline = next(
                    (
                        b
                        for b in rows
                        if b["case"] == case
                        and b["mode"] == "direct"
                        and b["repeat"] == row["repeat"]
                        and b["status"] == "completed"
                    ),
                    None,
                )
                if (
                    baseline is None
                    or "final_state" not in row
                    or "final_state" not in baseline
                ):
                    continue
                key = f"{case}/{mode}/{row['repeat']}"
                evidence["block_errors"][key + "/energy"] = block_error(
                    [row["result"]["energy"]],
                    [baseline["result"]["energy"]],
                    atol=1e-8,
                    rtol=0.0,
                )
                evidence["block_errors"][key + "/density"] = block_error(
                    row["final_state"]["density"],
                    baseline["final_state"]["density"],
                    atol=1e-6,
                    rtol=0.0,
                )
                complete_pairs += 1
            summaries.append(summary)
    passed = (
        complete_pairs == 12
        and all(accepted(r) for r in rows)
        and provenance["source_unchanged"]
        and provenance["library_unchanged"]
        and all(x["passed"] for x in evidence["block_errors"].values())
        and all(
            (r.get("result", {}).get("ks_diagnostic") or {}).get(
                "physical_residual_max", math.inf
            )
            <= 1e-8
            for r in rows
        )
    )
    evidence["stages"]["numerical"] = outcome(
        "pass" if passed else "not-run",
        "Native paired energy/density and reported-residual check only; missing/failed pairs prevent passage; no independent whole-matrix oracle",
    )
    dump(output / "summary.json", summaries)
    write_evidence(output / "evidence.json", evidence)
    return passed


def orchestrate(args: argparse.Namespace) -> int:
    if not args.run:
        print(
            "Prepared only. Add --run after the parent confirms the new library build is ready. No library loaded or calculation performed."
        )
        return 0
    args.source_root = args.source_root.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    sys.path[:0] = [str(args.source_root / "python"), str(args.source_root)]
    from benchmarks._retention import raw_output_path

    output = raw_output_path(args.output, repository_root=args.source_root).resolve()
    if output.exists():
        raise FileExistsError(
            "Use a new output directory so no attempts are overwritten"
        )
    output.mkdir(parents=True)
    (output / "attempts").mkdir()
    fixture = json.loads((HERE / "inputs.json").read_text())
    dump(output / "inputs.json", fixture)
    source = source_snapshot(args.source_root)
    if (
        args.expected_source_revision
        and source["revision"] != args.expected_source_revision
    ):
        raise RuntimeError("source revision changed; review before running")
    if (
        args.expected_library_sha256
        and sha(args.library) != args.expected_library_sha256
    ):
        raise RuntimeError(
            "library checksum differs from the coordinator-approved build"
        )
    paths = ("src", "include", "python", "cmake", "CMakeLists.txt")
    build_revision = (
        git(args.source_root, "rev-parse", args.build_source_revision).decode().strip()
    )
    build_tree = (
        git(args.source_root, "rev-parse", build_revision + "^{tree}").decode().strip()
    )
    if (
        git(args.source_root, "diff", build_revision, "--", *paths)
        or source["untracked_source_files"]
    ):
        raise RuntimeError(
            "production sources differ from the declared library build source"
        )
    (output / "source-tracked.patch").write_bytes(
        git(args.source_root, "diff", "HEAD", "--binary", "--", *paths)
    )
    for name in source["untracked_source_files"]:
        target = output / "source-untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((args.source_root / name).read_bytes())
    env = os.environ | {
        "GENERATIVEQC_LIBRARY": str(args.library),
        "PYTHONPATH": str(args.source_root / "python"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    schedule = []
    for repeat in range(3):
        cases = CASES if repeat % 2 == 0 else tuple(reversed(CASES))
        for case in cases:
            schedule.extend(
                (repeat, case, mode) for mode in MODES[repeat:] + MODES[:repeat]
            )
    if args.smoke:
        schedule = [(0, "water-631g", args.smoke_mode)]
    cpu_models = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        }
    )
    provenance = {
        "source": source,
        "build_source_revision": build_revision,
        "build_source_tree": build_tree,
        "production_sources_match_build": True,
        "source_mapping_scope": "Measured checkout may differ in tests/notes from the build source; src/include/python/cmake/CMakeLists.txt equality verified before run",
        "library_path": str(args.library),
        "library_sha256": sha(args.library),
        "harness_sha256": sha(Path(__file__)),
        "fixture_sha256": sha(HERE / "inputs.json"),
        "schedule": schedule,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {
            "kind": "cpu",
            "models": cpu_models,
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "affinity_isolated": False,
        },
        "runtime_environment": {
            k: v
            for k, v in env.items()
            if k.startswith(("GENERATIVEQC_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))
        },
        "endpoint_boundary": "Input argument materialization + Calculator constructor + prepare_batch + execute(returned diagnostics); library loading in constructor included. API imports, final-state export, close and process launch excluded and recorded separately. Fresh warm-enabled batch retains final density but has no imported/reused initial density and executes exactly once.",
        "checkpoint_boundary": "HF/MP2-only Python checkpoint model cannot label PBE0; existing read-only native DFT warm-state buffer export is used instead",
        "status": "running",
        "scope": "smoke-only, excluded from campaign"
        if args.smoke
        else "18-endpoint actual-API campaign",
    }
    dump(output / "provenance.json", provenance)
    rows = []
    with (output / "records.jsonl").open("w") as stream:
        for index, (repeat, case, mode) in enumerate(schedule):
            record = output / "attempts" / f"{index:02d}-{case}-{mode}.json"
            launch_start = time.perf_counter()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--source-root",
                str(args.source_root),
                "--case",
                case,
                "--mode",
                mode,
                "--repeat",
                str(repeat),
                "--record",
                str(record),
                "--launch-start-perf",
                repr(launch_start),
            ]
            try:
                process = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                row = (
                    json.loads(record.read_text())
                    if record.exists()
                    else {
                        "case": case,
                        "mode": mode,
                        "repeat": repeat,
                        "status": "failed",
                        "error": "worker returned no record",
                    }
                )
                row["returncode"] = process.returncode
                row["stdout"] = process.stdout
                row["stderr"] = process.stderr
            except subprocess.TimeoutExpired as error:
                row = {
                    "case": case,
                    "mode": mode,
                    "repeat": repeat,
                    "status": "failed",
                    "error": "process timeout",
                    "timeout_s": args.timeout,
                    "stdout": (error.stdout or b"").decode(errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout,
                    "stderr": (error.stderr or b"").decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                }
            row["process_wall_s"] = time.perf_counter() - launch_start
            if "endpoint_total_s" in row:
                row["process_nonendpoint_s"] = (
                    row["process_wall_s"] - row["endpoint_total_s"]
                )
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                case,
                mode,
                repeat,
                row["status"],
                row.get("endpoint_total_s"),
                flush=True,
            )
    provenance.update(
        status="completed",
        source_unchanged=source_snapshot(args.source_root)["source_hash"]
        == source["source_hash"],
        library_unchanged=sha(args.library) == provenance["library_sha256"],
    )
    dump(output / "provenance.json", provenance)
    numerical_passed = (
        True if args.smoke else finish_evidence(output, rows, provenance, fixture)
    )
    return int(
        not numerical_passed
        or any(not accepted(row) for row in rows)
        or not provenance["source_unchanged"]
        or not provenance["library_unchanged"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One water-only smoke, excluded from matrix evidence",
    )
    parser.add_argument("--smoke-mode", choices=MODES, default="hf")
    parser.add_argument("--expected-library-sha256")
    parser.add_argument("--expected-source-revision")
    parser.add_argument(
        "--build-source-revision", default="6bf930be60b688fe5917d8fb03464cb44c37be58"
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--library",
        type=Path,
        default=SOURCE_ROOT / "build/cpu-review/libgenerativeqc.so",
    )
    parser.add_argument("--output", type=Path, default=HERE / "run-1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--launch-start-perf", type=float)
    args = parser.parse_args()
    return worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
