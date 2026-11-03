"""Evidence-gated search over existing native CUDA XC execution tiles.

Only scheduling changes: every trial keeps the functional, quadrature, FP64
policy and complete energy/analytic-force endpoint fixed. No host-XC candidate
or reference implementation enters production. Native preparation owns memory
admission; its observed ledger, not an invented kernel-resource estimate, gates
these runtime-only changes to an already compiled library.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

DFT_FUNCTIONALS = {
    f"{name}-{spin}": selector
    for name, selector in (
        ("lda", "LDA_X,LDA_C_PW"),
        ("pbe", "GGA_X_PBE,GGA_C_PBE"),
        ("pbe0", "0.25*HF+0.75*GGA_X_PBE,GGA_C_PBE"),
        ("b3lyp", "0.2*HF+0.08*LDA_X+0.72*GGA_X_B88,0.19*LDA_C_VWN_RPA+0.81*GGA_C_LYP"),
        ("r2scan", "MGGA_X_R2SCAN,MGGA_C_R2SCAN"),
    )
    for spin in ("rks", "uks")
}


def candidate_tiles(args: Any, baseline_tile: int) -> list[int]:
    """Bound and deduplicate executable tiles without retuning scientific inputs."""
    requested = args.xc_tiles
    if requested is None:
        requested = (
            (64, 128, 256, 512, 1024, 1536, 2048, 4096)
            if args.full
            else (128, 256, 512, 1024)
        )
    if any(type(tile) is not int or not 1 <= tile <= 4096 for tile in requested):
        raise ValueError("DFT XC tiles must be integers in [1,4096]")
    return [tile for tile in dict.fromkeys(requested) if tile != baseline_tile][
        : args.max_candidates
    ]


def execution_record(
    workload: dict, calculator: Any, prepared: Any, result: Any
) -> dict:
    """Bind observed native scheduling, semantic work and memory to one replay.

    Host energy/force results have completed their CUDA transfers before return.
    The native source inventory and bundle binary hash bind the unchanged
    compiled implementation; tile candidates do not emit or compile new kernels.
    """
    from generativeqc_compiler.dft.xc_schedule import (
        DEVICE_FUSED,
        molecular_grid_xc_workload,
    )

    options = calculator.ks_options
    scientific = molecular_grid_xc_workload(
        architecture=workload["architecture"],
        functional=options.functional,
        atoms=workload["atoms"],
        grid_spec=options.grid,
        charge=workload["charge"],
        multiplicity=workload["multiplicity"],
        source_identity=workload["source_identity"],
    )
    diagnostics = [item.ks_diagnostic for item in result.items]
    if not diagnostics or any(item is None for item in diagnostics):
        raise ValueError("DFT autotuning requires native KS work diagnostics")
    tiles = {item.tile_points for item in diagnostics}
    if len(tiles) != 1:
        raise ValueError("DFT batch items resolved different XC schedules")
    schedule = DEVICE_FUSED.resolved(tiles.pop())
    if workload.get("tile_points") not in (None, schedule.point_tile):
        raise ValueError("native XC tile differs from the requested candidate")
    resources = prepared.resource_diagnostics
    if resources is None:
        raise ValueError("DFT autotuning requires native resource observations")
    observation = resources[resources["phase"]]
    return {
        "endpoint_kind": "energy_force",
        "synchronized": True,
        "scientific_identity": scientific.identity,
        "scientific_workload": scientific.to_payload(),
        "schedule_identity": schedule.identity,
        "schedule": schedule.to_payload(),
        "source_hash": workload["source_identity"],
        "dft_work": [item.to_payload() for item in diagnostics],
        "resources": copy.deepcopy(observation["device_ledger"]),
    }


def reference_workload(workload: dict) -> dict:
    """Compute an independent PySCF/libxc E+F oracle, outside timed GPU samples.

    The pinned Treutler/NWChem grid family and requested level match the native
    scientific model. Grid response is essential for molecular analytic forces.
    Explicit component selectors avoid user-configurable LDA/B3LYP aliases.
    The public workload currently repeats one geometry, so one independent solve
    suffices for every batch item. CPU reference work is tuning-only.
    """
    import pyscf
    from pyscf import dft, gto, lib

    if workload.get("kind") != "dft" or workload["method"] not in DFT_FUNCTIONALS:
        raise ValueError("independent DFT reference requires a supported KS workload")
    lib.num_threads(1)
    molecule = gto.M(
        atom=workload["atoms"],
        basis=workload["basis"],
        unit="Bohr",
        charge=workload["charge"],
        spin=workload["multiplicity"] - 1,
        cart=workload["representation"] == "cartesian",
        verbose=0,
    )
    solver = (dft.UKS if workload["method"].endswith("-uks") else dft.RKS)(molecule)
    solver.xc = DFT_FUNCTIONALS[workload["method"]]
    solver.grids.level = workload["grid_level"]
    solver.small_rho_cutoff = 0.0
    solver.conv_tol, solver.conv_tol_grad = 1e-12, 1e-10
    solver.direct_scf_tol = 1e-14
    solver.max_cycle = 150
    solver.kernel()
    if not solver.converged:
        solver = solver.newton()
        solver.max_cycle = 150
        solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent PySCF DFT endpoint did not converge")
    gradient = solver.nuc_grad_method()
    gradient.grid_response = True
    forces = -gradient.kernel()
    return {
        "independent_reference": "PySCF/libxc energy and analytic gradient with grid response",
        "pyscf_version": pyscf.__version__,
        "energies": [float(solver.e_tot)] * workload["batch"],
        "forces": [forces.tolist() for _ in range(workload["batch"])],
    }


def numerical_gate(samples: list[dict], reference: dict) -> dict:
    """Apply the existing native KS independent-oracle floors to every sample.

    These discretization-aware oracle bounds complement, never replace, the
    much tighter baseline/candidate schedule-parity gate (1e-9 E / 1e-7 F).
    """
    import numpy as np

    energies = np.asarray(reference["energies"], dtype=float)
    forces = np.asarray(reference["forces"], dtype=float)
    if not reference.get("independent_reference") or not samples:
        raise ValueError("DFT numerical gate requires independent reference evidence")
    if (
        energies.ndim != 1
        or energies.size == 0
        or forces.ndim != 3
        or forces.shape[0] != energies.size
        or forces.shape[1] == 0
        or forces.shape[2] != 3
        or not np.isfinite(energies).all()
        or not np.isfinite(forces).all()
    ):
        raise ValueError(
            "independent DFT reference has invalid energy/force layout or values"
        )
    energy_error, force_error, force_rms = 0.0, 0.0, 0.0
    for sample in samples:
        energy = np.asarray(sample["energies"], dtype=float)
        force = np.asarray(sample["forces"], dtype=float)
        if (
            energy.shape != energies.shape
            or force.shape != forces.shape
            or not np.isfinite(energy).all()
            or not np.isfinite(force).all()
        ):
            raise ValueError(
                "DFT sample does not match the independent reference layout"
            )
        energy_error = max(
            energy_error, float(np.abs(energy - energies).max()) / forces.shape[1]
        )
        force_error = max(force_error, float(np.abs(force - forces).max()))
        force_rms = max(force_rms, float(np.sqrt(np.mean((force - forces) ** 2))))
    return {
        "passed": energy_error <= 1e-5 and force_error <= 1e-4 and force_rms <= 3e-5,
        "independent_reference": reference["independent_reference"],
        "maximum_energy_error_per_atom": energy_error,
        "maximum_force_error": force_error,
        "maximum_force_rms": force_rms,
        "tolerances": {"energy_per_atom": 1e-5, "force_max": 1e-4, "force_rms": 3e-5},
    }


def resource_gate(samples: list[dict], *, schedule_hash: str, source_hash: str) -> dict:
    """Require measured, bounded native allocations for every candidate replay."""
    passed = bool(samples)
    peaks = []
    for sample in samples:
        ledger = sample.get("resources", {})
        peak = ledger.get("peak_bytes")
        admission = ledger.get("admission", {})
        limit = admission.get("limit_bytes")
        combined = admission.get("combined_peak_bytes")
        passed = passed and (
            sample.get("schedule_identity") == schedule_hash
            and sample.get("source_hash") == source_hash
            and type(peak) is int
            and peak > 0
            and type(limit) is int
            and limit >= peak
            and type(combined) is int
            and peak <= combined <= limit
            and ledger.get("rejected_allocations") == 0
        )
        peaks.append(peak)
    return {
        "passed": passed,
        "schedule_hash": schedule_hash,
        "source_hash": source_hash,
        "device_peak_bytes": peaks,
        "scope": "observed native KS capacity ledger; existing compiled kernels, no new PTXAS claim",
    }


def compare_schedules(
    library: Path,
    baseline: Path,
    candidate: Path,
    directory: Path,
    *,
    repeats: int,
    remaining: Any,
    generic: bool = False,
) -> dict:
    """Interleave fresh complete endpoints, preserving raw paired timing evidence."""
    from .autotune import _worker, dft_endpoint_gate

    samples: dict[str, list] = {"baseline": [], "candidate": []}
    directory.mkdir()
    for pair_id in range(repeats):
        order = (
            ("baseline", "candidate") if pair_id % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            sample = _worker(
                library,
                baseline if side == "baseline" else candidate,
                directory / f"endpoint-{pair_id}-{side}.json",
                generic=generic and side == "baseline",
                timeout=remaining(),
            )
            sample.update(pair_id=pair_id, interleaved=True)
            samples[side].append(sample)
    return dft_endpoint_gate(samples["baseline"], samples["candidate"])


def run_dft(
    args: Any, *, base_library: Any, probe: dict, tools: dict, target: Any
) -> dict:
    """Search native XC tiles and atomically extend a compatible local bundle.

    Retaining the incumbent library and evidence prevents a DFT-only search from
    discarding previously accepted HF kernels or other exact DFT workloads.
    An absent, refused, noisy or slower winner never updates the active index.
    """
    import json

    from generativeqc_compiler.dft.xc_schedule import molecular_grid_xc_workload

    from .autotune import _worker, read_xyz
    from .ks import GridSpec, KsOptions, resolve_ks_options
    from .profiles import (
        POLICY,
        PROFILE_SCHEMA,
        atomic_json,
        cache_root,
        compatibility_identity,
        dft_selection_context,
        export_bundle,
        file_hash,
        install_bundle,
        select_dft_schedule,
        select_library,
    )

    if args.method not in DFT_FUNCTIONALS or args.repeats < 5:
        raise ValueError(
            "DFT tuning requires a supported method and at least five repeats"
        )
    options = resolve_ks_options(
        args.method, KsOptions(grid=GridSpec.treutler(args.grid_level))
    )
    workload = {
        "kind": "dft",
        "atoms": read_xyz(args.input, units=args.units),
        "basis": args.basis,
        "method": args.method,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "representation": args.representation,
        "device_id": args.device_id,
        "batch": args.batch,
        "grid_level": args.grid_level,
        "architecture": target.architecture,
        "source_identity": probe["source_identity"],
    }
    scientific = molecular_grid_xc_workload(
        architecture=target.architecture,
        functional=options.functional,
        atoms=workload["atoms"],
        grid_spec=options.grid,
        charge=args.charge,
        multiplicity=args.multiplicity,
        source_identity=probe["source_identity"],
    )
    root = cache_root()
    selection_context = dft_selection_context(
        basis=args.basis,
        representation=args.representation,
        batch=args.batch,
        method_identity=options.method_ir.identity,
        controls=(100, 8, 1e-12, 1e-10, 1e-14),
    )
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="dft-run-", dir=root)).resolve()
    report = {
        "schema": "generativeqc.local_autotune",
        "schema_version": PROFILE_SCHEMA,
        "identity": compatibility_identity(probe),
        "toolchain": tools,
        "workload": workload,
        "mode": "full" if args.full else "quick",
        "directory": str(directory),
        "candidates": [],
        "installed": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "search_scope": "native device-fused XC point tiles; no functional/grid/precision search",
        "revision": None,
        "dirty": None,
    }
    deadline = time.monotonic() + args.budget_seconds

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError(
                "DFT autotuning budget exhausted; active profile is unchanged"
            )
        return seconds

    try:
        if shutil.which("git"):
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=args.source_dir,
                capture_output=True,
                text=True,
                timeout=min(30, remaining()),
                check=False,
            )
            if revision.returncode == 0:
                report["revision"] = revision.stdout.strip()
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=args.source_dir,
                    capture_output=True,
                    text=True,
                    timeout=min(30, remaining()),
                    check=True,
                )
                report["dirty"] = bool(status.stdout.strip())
        incumbent, diagnostics = select_library(base_library, args.device_id)
        library = Path(incumbent._name).resolve()
        report["incumbent_profile"] = diagnostics
        active = select_dft_schedule(
            diagnostics, scientific.to_payload(), selection_context=selection_context
        )
        if active is not None:
            if active["name"] != "device_fused":
                report["reason"] = (
                    "resident-only DFT tuning does not replace an explicit host-XC incumbent"
                )
                return report
            if type(active["point_tile"]) is not int or active["point_tile"] < 1:
                raise ValueError("active DFT schedule must resolve its point tile")
            workload["tile_points"] = active["point_tile"]
            if not args.retune:
                report["reason"] = (
                    "an exact DFT schedule is already active; use --retune to search again"
                )
                return report
        baseline_path = directory / "baseline.json"
        atomic_json(baseline_path, workload)
        baseline = _worker(
            library, baseline_path, directory / "work.json", timeout=remaining()
        )
        if baseline["scientific_identity"] != scientific.identity:
            raise ValueError(
                "native DFT execution differs from the requested scientific workload"
            )
        report["work_profile"] = baseline
        tiles = candidate_tiles(args, baseline["schedule"]["point_tile"])
        report["candidate_tiles"] = tiles
        if not tiles:
            report["reason"] = "no distinct DFT XC tile candidates"
            return report
        reference = _worker(
            library,
            baseline_path,
            directory / "reference.json",
            reference=True,
            timeout=remaining(),
        )
        report["reference"] = reference
        report["baseline_numerical"] = numerical_gate([baseline], reference)
        if not report["baseline_numerical"]["passed"]:
            raise ValueError(
                "baseline DFT endpoint failed the independent PySCF energy/force gate"
            )
        best, best_path = None, baseline_path
        for tile in tiles:
            remaining()
            trial = directory / f"tile-{tile}"
            trial.mkdir()
            candidate_path = trial / "workload.json"
            atomic_json(candidate_path, {**workload, "tile_points": tile})
            row = {
                "tile_points": tile,
                "accepted": False,
                "selection_context": selection_context,
            }
            report["candidates"].append(row)
            try:
                endpoint = compare_schedules(
                    library,
                    best_path,
                    candidate_path,
                    trial / "paired",
                    repeats=args.repeats,
                    remaining=remaining,
                    generic=args.portable_baseline and best is None,
                )
                schedule_hash = endpoint["candidate_schedule_hash"]
                source_hash = endpoint["candidate_source_hash"]
                samples = endpoint["baseline"] + endpoint["candidate"]
                numerical = numerical_gate(samples, reference)
                numerical.update(schedule_hash=schedule_hash, source_hash=source_hash)
                resources = resource_gate(
                    endpoint["candidate"],
                    schedule_hash=schedule_hash,
                    source_hash=source_hash,
                )
                row.update(
                    workload_hash=scientific.identity,
                    schedule_hash=schedule_hash,
                    endpoint=endpoint,
                    numerical=numerical,
                    resources=resources,
                    legality={
                        "legal": True,
                        "schedule_hash": schedule_hash,
                        "scope": "successful native KS preparation and complete execution",
                    },
                )
                if endpoint["passed"] and numerical["passed"] and resources["passed"]:
                    if best is not None:
                        best["accepted"] = False
                    row["accepted"] = True
                    best, best_path = row, candidate_path
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                row["failure"] = str(error)
        if best is None:
            report["reason"] = (
                "no DFT XC schedule passed independent numerical, memory and complete-endpoint gates"
            )
            return report
        if args.portable_baseline:
            endpoint = compare_schedules(
                library,
                baseline_path,
                best_path,
                directory / "incumbent",
                repeats=args.repeats,
                remaining=remaining,
            )
            report["incumbent_endpoint"] = endpoint
            numerical = numerical_gate(
                endpoint["baseline"] + endpoint["candidate"], reference
            )
            numerical.update(
                schedule_hash=best["schedule_hash"],
                source_hash=best["endpoint"]["candidate_source_hash"],
            )
            resources = resource_gate(
                endpoint["candidate"],
                schedule_hash=best["schedule_hash"],
                source_hash=best["endpoint"]["candidate_source_hash"],
            )
            if (
                not endpoint["passed"]
                or not numerical["passed"]
                or not resources["passed"]
            ):
                best["accepted"] = False
                report["reason"] = "the actual incumbent remains faster or within noise"
                return report
            best["endpoint"] = endpoint
            best.update(numerical=numerical, resources=resources)
        remaining()
        winner = {
            "selection_context": selection_context,
            "workload": scientific.to_payload(),
            "workload_hash": scientific.identity,
            "schedule": best["endpoint"]["candidate"][0]["schedule"],
            "schedule_hash": best["schedule_hash"],
            "source_hash": best["endpoint"]["candidate_source_hash"],
            "gates": {
                name: "pass"
                for name in (
                    "legality",
                    "resources",
                    "numerical",
                    "performance",
                    "endpoint",
                )
            },
        }
        previous_profile, evidence = {}, {}
        if diagnostics.get("source") == "local":
            previous = Path(diagnostics["directory"])
            previous_profile = json.loads((previous / "profile.json").read_text())
            evidence = json.loads((previous / "evidence.json").read_text())
            manifest = json.loads((previous / "manifest.json").read_text())
        else:
            manifest = json.loads(
                (
                    args.source_dir
                    / "python/generativeqc_compiler/integral/production_shell_classes.json"
                ).read_text()
            )
        retained = [
            item
            for item in previous_profile.get("dft_schedules", [])
            if item["workload_hash"] != scientific.identity
        ]
        records = [
            item
            for item in evidence.get("dft_schedules", [])
            if item.get("workload_hash") != scientific.identity
        ]
        evidence.update(dft_schedules=[*records, best], dft_autotune=report)
        bundle = directory / "accepted"
        bundle.mkdir()
        shutil.copyfile(library, bundle / "libgenerativeqc.so")
        atomic_json(bundle / "manifest.json", manifest)
        atomic_json(bundle / "evidence.json", evidence)
        profile = {
            "schema": "generativeqc.local_profile",
            "schema_version": PROFILE_SCHEMA,
            "identity": compatibility_identity(probe),
            "toolchain": tools,
            "generator_abi": target.generator_abi,
            "policy": POLICY,
            "kernels": previous_profile.get("kernels", []),
            "dft_schedules": [*retained, winner],
            "artifacts": {
                name: file_hash(bundle / name)
                for name in ("libgenerativeqc.so", "manifest.json", "evidence.json")
            },
        }
        atomic_json(bundle / "profile.json", profile)
        installed = install_bundle(bundle, probe, tools)
        report["installed"] = str(installed)
        if args.export:
            export_bundle(installed, args.export)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["failure"] = str(error)
    finally:
        atomic_json(directory / "report.json", report)
    return report
