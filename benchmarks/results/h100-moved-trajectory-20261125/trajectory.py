"""Acquire the pinned water96 trajectory, or compute separate tighter references."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.direct_phase_moved import displace, geometry, identity
from benchmarks.readme_defaults import (
    COMMON_GATES,
    METHODS,
    _finite_endpoint,
    allocation_metadata,
    digest,
    grid_settings,
    native_calculator,
    native_force_tile,
    native_settings,
    save,
)
from benchmarks.readme_hf_scaling import scaling_cases

if TYPE_CHECKING:
    from collections.abc import Callable

REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"
FUNCTIONALS = ("wb97m-v", "pbe", "r2scan", "pbe0")
PROFILES = ("standard", "gpu4pyscf")
SAMPLE_STEPS = (1, 5, 10)
NSTEPS = 10
NATOMS = 96


def plan() -> dict:
    base = geometry(scaling_cases()["water-96"].atoms)
    steps = [
        {
            "step": 0,
            "geometry_bohr": base,
            "geometry_sha256": identity(base),
            "actual_rms_bohr": 0.0,
        }
    ]
    for step in range(1, NSTEPS + 1):
        steps.append({"step": step, **displace(base, step * 0.005, seed=3169)})
    return {
        "schema": "generativeqc.issue3169.trajectory.v1",
        "source_revision": REVISION,
        "fixture": "scaling_cases()['water-96'].atoms",
        "orbital_basis": "spherical def2-SVP",
        "charge": 0,
        "spin": 0,
        "methods": list(FUNCTIONALS),
        "modes": ["direct", "df"],
        "oracle_steps": list(SAMPLE_STEPS),
        "oracle_xc_grid_levels": [4, 5],
        "oracle_nlc_grid_levels": [2, 3],
        "gates": COMMON_GATES,
        "oracle_grid_stability_fraction": 0.25,
        "steps": steps,
    }


class NativeTrajectory:
    def __init__(self, args: argparse.Namespace, atoms: list) -> None:
        self.calculator = native_calculator(
            args.method, "standard", args.mode, args.inputs
        )
        try:
            self.owner = self.calculator.prepare_batch([atoms], warm_start=True)
        except BaseException:
            self.calculator.clear_cache()
            raise

    def binary(self) -> dict:
        from benchmarks.readme_method_endpoints import binary_identity

        build = binary_identity(self.owner._calculator)
        expected = json.loads(
            (Path(__file__).resolve().parent / "build-receipts/build.json").read_text()
        )
        if (
            expected["status"] != "complete"
            or expected["source_revision"] != REVISION
            or build["source_identity"] != expected["native_source_identity"]
            or build["library_sha256"] != expected["library_sha256"]
        ):
            raise ValueError(f"native binary differs from the staged build: {build}")
        return build

    def evaluate(self, atoms: list, step: int) -> tuple:
        coords = [[xyz for _, xyz in atoms]] if step else None
        batch = self.owner.execute(
            coords, properties=("energy", "forces"), strict=False
        )
        return batch, batch.items[0]

    def record(self, result: tuple) -> dict:
        batch, item = result
        return {
            "energy": float(item.energy) if item.converged and item.succeeded else None,
            "forces": item.forces,
            "converged": item.converged,
            "status": item.status,
            "status_message": item.status_message,
            "executed_backend": item.executed_backend,
            "iterations": item.iterations,
            "fock_builds": item.fock_builds,
            "incremental_direct_jk": item.incremental_direct_jk,
            "initial_guess": item.initial_guess,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "energy_change": item.energy_change,
            "density_rms": item.density_rms,
            "physical_residual_rms": item.physical_residual_rms,
            "precision": item.precision,
            "precision_validation": item.precision_validation.to_dict()
            if item.precision_validation
            else None,
            "batch_precision_validation": batch.precision_validation,
            "ks_diagnostic": item.ks_diagnostic.to_payload()
            if item.ks_diagnostic
            else None,
            "settings": {
                **native_settings(self.owner._calculator, item),
                "resolved_force_tile_points": native_force_tile(self.owner._owner),
            },
        }

    def close(self) -> None:
        try:
            self.owner.close()
        finally:
            self.calculator.clear_cache()


class ReferenceTrajectory:
    def __init__(self, args: argparse.Namespace, atoms: list) -> None:
        import cupy as cp
        from gpu4pyscf.dft import rks
        from pyscf import gto

        self.cp = cp
        self.mol = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", verbose=0)
        self.engine = rks.RKS(self.mol)
        self.engine.xc = METHODS[args.method][1]
        self.method = args.method
        self.seed = None
        self.auxiliary = None
        if args.mode == "df":
            self.engine = self.engine.density_fit()
            self.auxiliary = json.loads(
                (args.inputs / f"{args.method}-reference.json").read_text()
            )
        original = self.engine.get_veff
        self.builds = 0

        def counted(*values: Any, **kwargs: Any) -> Any:
            self.builds += 1
            return original(*values, **kwargs)

        self.engine.get_veff = counted

    def evaluate(self, atoms: list, step: int) -> tuple:
        if step:
            # Use the scanner's geometry-reset machinery without letting a
            # scanner choose, transform, or silently replace the AO seed.
            self.mol = self.mol.set_geom_(atoms, unit="Bohr", inplace=False)
            self.engine.reset(self.mol)
        self.builds = 0
        seeded = self.seed is not None
        energy = float(self.engine.kernel(dm0=self.seed))
        if not self.engine.converged:
            return energy, None, seeded, None
        gradient = self.engine.nuc_grad_method()
        gradient.grid_response = True
        forces = -self.cp.asnumpy(gradient.kernel())
        # The next endpoint needs this geometry's density. Charge preparation
        # and any GPU work it entails to the endpoint that produced it.
        if step < NSTEPS:
            self.seed = self.engine.make_rdm1().copy()
        return energy, forces, seeded, gradient

    def record(self, result: tuple) -> dict:
        energy, forces, seeded, gradient = result
        engine = self.engine
        settings = {
            "energy_tolerance": float(engine.conv_tol),
            "orbital_gradient_tolerance": engine.conv_tol_grad,
            "max_iterations": int(engine.max_cycle),
            "diis_history": int(engine.diis_space),
            "initial_guess": str(engine.init_guess),
            "direct_scf": bool(engine.direct_scf),
            "direct_scf_tolerance": float(engine.direct_scf_tol),
            "small_rho_cutoff": float(engine.small_rho_cutoff),
            "xc_grid": grid_settings(engine.grids),
            "nonlocal_grid": grid_settings(engine.nlcgrids)
            if self.method == "wb97m-v"
            else None,
            "force_grid_response": getattr(gradient, "grid_response", None),
            "auxbasis_response": getattr(gradient, "auxbasis_response", None),
        }
        if self.method == "wb97m-v":
            from gpu4pyscf.dft import numint
            from gpu4pyscf.grad import rks

            settings["nonlocal_density_thresholds"] = {
                "scf": float(numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD),
                "forces": float(rks.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD),
            }
        if self.auxiliary and engine.converged:
            auxmol = engine.with_df.auxmol
            if auxmol._basis != self.auxiliary["basis"]:
                raise ValueError("GPU4PySCF DF basis differs from frozen native input")
            settings["auxiliary_basis_sha256"] = self.auxiliary["basis_sha256"]
            settings["auxiliary_aos"] = auxmol.nao_nr()
        return {
            "energy": energy,
            "forces": forces,
            "converged": bool(engine.converged),
            "executed_backend": "cuda",
            "iterations": int(engine.cycles),
            "effective_potential_builds": self.builds,
            "warm_start_used": seeded,
            "settings": settings,
        }


def clean_sample(sample: dict, seconds: float) -> None:
    forces = sample["forces"]
    if sample["converged"]:
        sample["forces"] = _finite_endpoint(sample["energy"], forces, NATOMS, seconds)
    elif forces is not None:
        sample["forces"] = np.asarray(forces).tolist()
    for key in ("energy", "density_rms", "energy_change", "physical_residual_rms"):
        if sample.get(key) is not None and not np.isfinite(sample[key]):
            sample[key] = None


def timed_trajectory(
    args: argparse.Namespace, record: dict, sync: Callable[[], None]
) -> None:
    trajectory = None
    try:
        for frame in record["trajectory"]["steps"]:
            step = frame["step"]
            atoms = frame["geometry_bohr"]
            record["active_step"] = step
            save(args.output, record)
            sync()
            start = perf_counter()
            if step == 0:
                if args.arm == "standard":
                    trajectory = NativeTrajectory(args, atoms)
                else:
                    trajectory = ReferenceTrajectory(args, atoms)
            result = trajectory.evaluate(atoms, step)
            sync()
            seconds = perf_counter() - start
            sample = trajectory.record(result)
            clean_sample(sample, seconds)
            if args.arm == "standard" and step == 0:
                record["native_build"] = trajectory.binary()
            sample.update(
                step=step,
                geometry_sha256=frame["geometry_sha256"],
                seconds=seconds,
                force_evaluations=int(sample["forces"] is not None),
            )
            record["samples"].append(sample)
            save(args.output, record)
            print(
                f"STEP {args.method} {args.mode} {args.arm} {step} "
                f"{seconds:.3f}s cycles={sample['iterations']} "
                f"seed={sample['warm_start_used']} converged={sample['converged']}",
                flush=True,
            )
            if not sample["converged"] or sample["executed_backend"] != "cuda":
                raise RuntimeError(
                    f"{args.arm} step {step} failed the CUDA SCF endpoint"
                )
            if args.arm == "standard" and (
                sample["precision_validation"] is None
                or sample["precision_validation"]["profile"] != args.arm
            ):
                raise RuntimeError(
                    "native endpoint lacks Standard precision validation"
                )
    finally:
        if isinstance(trajectory, NativeTrajectory):
            trajectory.close()


def independent_oracle(args: argparse.Namespace, record: dict) -> None:
    from benchmarks.readme_defaults_accuracy import solve

    for step in SAMPLE_STEPS:
        frame = record["trajectory"]["steps"][step]
        for level in record["trajectory"]["oracle_xc_grid_levels"]:
            record.update(active_step=step, active_level=level)
            save(args.output, record)
            try:
                sample = solve(
                    args.method, frame["geometry_bohr"], backend="cuda", level=level
                )
            except Exception as error:  # noqa: BLE001 - preserve each independent attempt
                record["failures"].append(
                    {
                        "step": step,
                        "level": level,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                save(args.output, record)
                continue
            sample.update(step=step, geometry_sha256=frame["geometry_sha256"])
            record["samples"].append(sample)
            save(args.output, record)
            print(
                f"ORACLE {args.method} step={step} level={level} {sample['seconds']:.3f}s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=(*PROFILES, "oracle", "plan"))
    parser.add_argument("--method", choices=FUNCTIONALS)
    parser.add_argument("--mode", choices=("direct", "df"), default="direct")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", required=True, type=raw_output_path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.arm == "plan":
        save(args.output, plan())
        return
    if args.method is None or args.inputs is None or args.trajectory is None:
        parser.error("acquisition requires method, inputs and a frozen trajectory")
    if args.arm == "oracle" and args.mode != "direct":
        parser.error("the independent target is Direct for both timing modes")

    trajectory = json.loads(args.trajectory.read_text())
    if trajectory != plan():
        raise ValueError("frozen trajectory differs from the pinned fixture/protocol")
    record = {
        "schema": "generativeqc.issue3169.acquisition.v1",
        "status": "running",
        "arm": args.arm,
        "method": args.method,
        "mode": args.mode,
        "source_revision": REVISION,
        "driver_sha256": digest(Path(__file__)),
        "plan_sha256": digest(args.trajectory),
        "trajectory": trajectory,
        "allocation": allocation_metadata(os.environ.get("GQC_QZ_JOB_NAME")),
        "samples": [],
        "failures": [],
    }
    save(args.output, record)
    try:
        import cupy as cp

        record["environment"] = environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
                "cutensor": ("cutensor-cu12",),
                "numpy": ("numpy",),
            }
        )
        record["threads"] = {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        record["accelerator"] = cuda_accelerator_metadata(
            cp, sampling_point="before acquisition"
        )
        device = record["accelerator"]
        if cp.cuda.runtime.getDeviceCount() != 1 or "H100" not in device["name"]:
            raise ValueError("expected a single scheduler-visible H100")
        if args.arm != "standard" and version("gpu4pyscf-cuda12x") != "1.9.0":
            raise ValueError("reference must use GPU4PySCF 1.9.0")
        if args.mode == "df":
            record["df_inputs"] = {
                path.name: digest(path)
                for path in (
                    args.inputs / f"{args.method}-auxiliary.json",
                    args.inputs / f"{args.method}-reference.json",
                )
            }
        if args.arm == "oracle":
            independent_oracle(args, record)
        else:
            timed_trajectory(args, record, cp.cuda.runtime.deviceSynchronize)
        status = "failed" if record["failures"] else "complete"
        record.update(status=status, active_step=None, active_level=None)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, record)
    if record["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
