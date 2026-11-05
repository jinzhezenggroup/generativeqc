"""Compare complete occupied-XC endpoints across functionals, not just kernels.

Generate an independent PySCF/GPU4PySCF oracle with ``oracle``, run each selector in a
fresh finite Slurm allocation/process with ``run``, then use ``replay`` offline.
Raw execution outputs belong outside benchmarks/results. Trace runs establish
admission and strict handoff separately and are never timing evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from time import perf_counter

from benchmarks._retention import raw_output_path
from benchmarks.occupied_xc_proposals import check_vectors, load, save

METHODS = {
    "lda": "LDA_X,PW",
    "m06-2x": "M06-2X",
    "mn15": "MN15",
    "scan": "SCAN",
    "rscan": "RSCAN",
    "r2scan": "R2SCAN",
    "pbe": "PBE",
    "pbe0": "PBE0",
    "b3lyp": "B3LYP",
}
PHASES = ("cold", "warm", "moved", "moved_warm")
SCHEMA = "generativeqc.occupied-functional-endpoints.v1"


def geometries(atoms: int) -> list[list]:
    """Reuse the established water series and its single-H displacement in Bohr."""
    from benchmarks.readme_hf_scaling import scaling_cases

    original = scaling_cases()[f"water-{atoms}"].atoms
    moved = [(symbol, list(position)) for symbol, position in original]
    moved[1][1][2] += 0.02
    return [original, moved]


def oracle(args: argparse.Namespace) -> None:
    """Converge an independent original/displaced analytic moving-grid oracle."""
    from pyscf import dft, gto, lib

    from benchmarks.readme_defaults import grid_settings
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    if args.oracle_device == "cuda" and (
        not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise RuntimeError("GPU4PySCF reference requires a finite Slurm GPU allocation")
    lib.num_threads(args.threads)
    rows = []
    charge = int(args.spin == "uks")
    for geometry in geometries(args.atoms):
        molecule = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            cart=False,
            charge=charge,
            spin=charge,
            verbose=0,
        )
        reference = dft.UKS(molecule) if charge else dft.RKS(molecule)
        reference.xc = METHODS[args.method]
        reference.conv_tol = 1e-12 if args.atoms == 3 else 1e-11
        reference.conv_tol_grad = 1e-9 if args.atoms == 3 else 1e-7
        reference.direct_scf_tol = 1e-14
        reference.grids.level = (
            (3 if args.atoms == 3 else 5)
            if args.grid_level is None
            else args.grid_level
        )
        reference.max_cycle = 200
        if args.oracle_device == "cuda":
            reference = reference.to_gpu()
        configure_reference_full_fock(reference)
        if args.oracle_device == "cpu":
            reference.direct_scf = True
        energy = reference.kernel()
        if not reference.converged:
            raise RuntimeError("independent PySCF oracle did not converge")
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        forces = -gradient.kernel()
        if args.oracle_device == "cuda" and hasattr(forces, "get"):
            forces = forces.get()
        rows.append({"energy": float(energy), "forces": forces.tolist()})
    import pyscf

    save(
        args.output,
        {
            "method": args.method,
            "spin": args.spin,
            "atoms": args.atoms,
            "basis": "def2-svp",
            "geometries": geometries(args.atoms),
            "pyscf_version": pyscf.__version__,
            "device": args.oracle_device,
            "xc": reference.xc,
            "grid": grid_settings(reference.grids),
            "energy_tolerance": reference.conv_tol,
            "gradient_tolerance": reference.conv_tol_grad,
            "direct_scf_tolerance": reference.direct_scf_tol,
            "full_fock": True,
            "observations": rows,
        },
    )


def run(args: argparse.Namespace) -> None:
    """Time native construction, preparation, SCF, analytic forces and export."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("native execution requires a finite Slurm GPU allocation")
    if os.environ.get("GENERATIVEQC_DF_TRACE") and not args.trace:
        raise RuntimeError("timing cannot run with tracing enabled")
    import numpy as np
    from generativeqc import Calculator, _native
    from test_dft_complete_cuda import no_cpu_derivatives

    from benchmarks.readme_defaults import digest, native_settings

    reference = load(args.oracle)
    for key in ("method", "spin", "atoms"):
        if reference[key] != getattr(args, key):
            raise ValueError(f"independent oracle {key} mismatch")
    if reference["geometries"] != json.loads(json.dumps(geometries(args.atoms))):
        raise ValueError("independent oracle geometry mismatch")
    control = os.environ.get("GENERATIVEQC_CUDA_KS_OCCUPIED_XC")
    if control not in (None, "0"):
        raise ValueError("compare actual unset defaults with exact matrix control 0")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    library_sha256 = digest(library)
    identity = _native.load_library().generativeqc_get_source_identity().decode()
    options = {
        "method": f"{args.method}-{args.spin}",
        "device": "cuda",
        "basis": "def2-svp",
        "basis_representation": "spherical",
    }
    if args.strict_solver:
        options.update(
            energy_tolerance=1e-12, density_tolerance=1e-10, max_iterations=200
        )
    record = {
        "schema": SCHEMA,
        "completed": False,
        "method": args.method,
        "spin": args.spin,
        "atoms": args.atoms,
        "basis": "def2-svp",
        "source_commit": args.source_commit,
        "source_identity": identity,
        "library_sha256": library_sha256,
        "occupied_xc_env": control,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "strict_solver": args.strict_solver,
        "traced": args.trace,
        "oracle": reference,
        "observations": [],
    }
    charge = int(args.spin == "uks")
    original, moved = geometries(args.atoms)
    started = perf_counter()
    calculator = Calculator(**options)
    try:
        with calculator.prepare_batch(
            [original],
            charges=[charge],
            multiplicities=[charge + 1],
            warm_start=True,
        ) as owner:
            preparation = perf_counter() - started
            for index, phase in enumerate(PHASES):
                with no_cpu_derivatives():
                    started = perf_counter()
                    result = owner.execute(
                        [np.asarray([position for _, position in moved])]
                        if phase.startswith("moved")
                        else None,
                        properties=("energy", "forces"),
                        strict=True,
                    ).items[0]
                    seconds = (
                        perf_counter() - started + (preparation if index == 0 else 0)
                    )
                if (
                    not result.succeeded
                    or not result.converged
                    or result.executed_backend != "cuda"
                ):
                    raise RuntimeError("complete native CUDA endpoint did not converge")
                record["observations"].append(
                    {
                        "phase": phase,
                        "seconds": seconds,
                        "energy": result.energy,
                        "forces": np.asarray(result.forces).tolist(),
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "settings": native_settings(calculator, result),
                        "diagnostic": result.ks_diagnostic.to_payload(),
                    }
                )
                save(args.output, record)
    finally:
        calculator.clear_cache()
    record["completed"] = True
    save(args.output, record)
    print(
        json.dumps(
            {
                phase: row["seconds"]
                for phase, row in zip(PHASES, record["observations"], strict=True)
            }
        ),
        flush=True,
    )


def independent_gate(actual: dict, expected: dict, atoms: int) -> tuple[float, float]:
    """Keep strict small-system and established size-normalized large gates."""
    energy_error = abs(actual["energy"] - expected["energy"])
    if len(actual["forces"]) != atoms or len(expected["forces"]) != atoms:
        raise ValueError("incomplete independent force vectors")
    errors = []
    for actual_row, expected_row in zip(
        actual["forces"], expected["forces"], strict=True
    ):
        if len(actual_row) != 3 or len(expected_row) != 3:
            raise ValueError("invalid independent force shape")
        errors.extend(
            abs(left - right)
            for left, right in zip(actual_row, expected_row, strict=True)
        )
    energy_gate, force_gate = (2e-9, 1e-7) if atoms == 3 else (1e-5 * atoms, 1e-4)
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    if not math.isfinite(energy_error) or energy_error > energy_gate:
        raise ValueError(f"independent energy gate failed: {energy_error}")
    if any(not math.isfinite(error) or error > force_gate for error in errors):
        raise ValueError(f"independent force gate failed: {max(errors)}")
    if atoms != 3 and rms > 3e-5:
        raise ValueError(f"independent RMS force gate failed: {rms}")
    return energy_error, max(errors)


def replay(paths: list[Path]) -> dict:
    """Recheck complete vectors, identities, semantic work and a predeclared floor."""
    return replay_records([load(path) for path in paths])


def replay_records(records: list[dict]) -> dict:
    """Validate retained in-memory populations without materializing raw files."""
    if not records:
        raise ValueError("empty endpoint population")
    populations = {control: [] for control in ("0", None)}
    anchor = records[0]
    identity_keys = (
        "method",
        "spin",
        "atoms",
        "basis",
        "source_commit",
        "source_identity",
        "library_sha256",
        "strict_solver",
        "oracle",
    )
    maxima = [0.0, 0.0]
    for record in records:
        if record["schema"] != SCHEMA or not record["completed"] or record["traced"]:
            raise ValueError("incomplete, unsupported or traced timing receipt")
        if any(record[key] != anchor[key] for key in identity_keys):
            raise ValueError("paired scientific or source identity changed")
        for key, size in (
            ("source_commit", 40),
            ("source_identity", 64),
            ("library_sha256", 64),
        ):
            value = record[key]
            if len(value) != size or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid {key}")
        if (
            record["method"] not in METHODS
            or record["spin"] not in ("rks", "uks")
            or record["basis"] != "def2-svp"
            or record["atoms"] not in (3, 12, 48, 96)
        ):
            raise ValueError("unsupported endpoint identity")
        if any(
            record["oracle"][key] != record[key]
            for key in ("method", "spin", "atoms", "basis")
        ):
            raise ValueError("independent oracle identity changed")
        if record["occupied_xc_env"] not in populations:
            raise ValueError("invalid occupied-XC selector")
        if [row["phase"] for row in record["observations"]] != list(PHASES):
            raise ValueError("incomplete endpoint phases")
        for row in record["observations"]:
            if (
                not math.isfinite(row["seconds"])
                or row["seconds"] <= 0
                or row["iterations"] < 1
                or row["fock_builds"] < row["iterations"]
            ):
                raise ValueError("invalid endpoint time or semantic work")
            expected = record["oracle"]["observations"][
                int(row["phase"].startswith("moved"))
            ]
            errors = independent_gate(row, expected, record["atoms"])
            maxima = [max(old, new) for old, new in zip(maxima, errors, strict=True)]
        populations[record["occupied_xc_env"]].append(record)
    if len(populations["0"]) != len(populations[None]) or len(populations["0"]) < 3:
        raise ValueError("require at least three complete matrix/default pairs")
    summary = {
        "method": anchor["method"],
        "atoms": anchor["atoms"],
        "spin": anchor["spin"],
        "independent_max_errors": maxima,
        "phases": {},
    }
    for index, phase in enumerate(PHASES):
        values, work = {}, {}
        for control, population in populations.items():
            label = "matrix" if control == "0" else "default"
            values[label] = [
                record["observations"][index]["seconds"] for record in population
            ]
            work[label] = [
                [
                    record["observations"][index][key]
                    for key in ("iterations", "fock_builds")
                ]
                for record in population
            ]
        for matrix, default in zip(populations["0"], populations[None], strict=True):
            if (
                matrix["observations"][index]["settings"]
                != default["observations"][index]["settings"]
            ):
                raise ValueError("paired grid or solver settings changed")
            check_vectors(default["observations"][index], matrix["observations"][index])
        medians = {
            label: statistics.median(samples) for label, samples in values.items()
        }
        deviations = {
            label: statistics.median(abs(value - medians[label]) for value in samples)
            / medians[label]
            for label, samples in values.items()
        }
        floor = max(0.02, 2 * sum(deviations.values()))
        gain = 1 - medians["default"] / medians["matrix"]
        summary["phases"][phase] = {
            "seconds": values,
            "medians": medians,
            "improvement_fraction": gain,
            "variability_floor": floor,
            "profitable": gain > floor,
            "regresses": -gain > floor,
            "iterations_fock_builds": work,
        }
    return summary


def main() -> None:
    """Keep offline replay free of native, NumPy and PySCF imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("oracle", "run"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--method", choices=METHODS, required=True)
        subparser.add_argument("--spin", choices=("rks", "uks"), default="rks")
        subparser.add_argument("--atoms", type=int, choices=(3, 12, 48, 96), default=3)
        subparser.add_argument("--output", type=raw_output_path, required=True)
        if command == "oracle":
            subparser.add_argument("--threads", type=int, default=1)
            subparser.add_argument("--grid-level", type=int, choices=range(10))
            subparser.add_argument(
                "--oracle-device", choices=("cpu", "cuda"), default="cpu"
            )
        else:
            subparser.add_argument("--oracle", type=Path, required=True)
            subparser.add_argument("--source-commit", required=True)
            subparser.add_argument("--strict-solver", action="store_true")
            subparser.add_argument("--trace", action="store_true")
    commands.add_parser("replay").add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "replay":
        print(json.dumps(replay(args.receipts), indent=2))
    else:
        if args.output.exists():
            parser.error("output must be fresh")
        if args.command == "oracle":
            oracle(args)
        else:
            if len(args.source_commit) != 40 or any(
                character not in "0123456789abcdef" for character in args.source_commit
            ):
                parser.error("source commit must be a full hexadecimal Git SHA")
            run(args)


if __name__ == "__main__":
    main()
