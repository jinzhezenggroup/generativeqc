"""Screen cold PBE0 guesses without adding reference work to production.

SAP and the mod-huckel arm are explicit PySCF experiments, not native providers.
Their molecular/atomic preparation, seed admission and any target retry belong
to the complete energy/force endpoint. MINAO and huckel use native providers.
Every sample owns a fresh process/density; persistent compilation caches remain
enabled. Numerical agreement and a strict >5% median wall reduction are separate
gates, neither of which qualifies an oracle-backed seed for automatic selection.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import socket
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks._support import raw_output_path

SCHEMA = "generativeqc.pbe0-initial-guesses.v1"
GUESSES = ("core", "sap", "mod-huckel", "minao")
MINIMUM_REPEATS = 3
WALL_REDUCTION_GATE = 0.05
DENSITY_GATE = 1e-8


def digest(path: Path) -> str:
    """Bind raw evidence to file bytes rather than caller-supplied labels."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_seed(density: np.ndarray, overlap: np.ndarray, electrons: int) -> dict:
    """Check the same-AO seed without repairing or changing the candidate.

    Occupations are eigenvalues of S^(1/2) D S^(1/2), not of D. Native import
    independently repeats its strict source-metric admission before mutation.
    This CPU reference check is charged experiment work, never a production path.
    """
    if (
        density.ndim != 2
        or density.shape != overlap.shape
        or density.shape[0] != density.shape[1]
        or not np.isfinite(density).all()
        or not np.isfinite(overlap).all()
        or np.max(np.abs(density - density.T)) > 1e-10
        or np.max(np.abs(overlap - overlap.T)) > 1e-10
    ):
        raise ValueError("invalid same-AO density or overlap")
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if eigenvalues[0] <= 1e-12:
        raise ValueError("seed overlap is not positive definite")
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    occupations = np.linalg.eigvalsh(root @ density @ root)
    electron_error = abs(float(np.einsum("ij,ji->", density, overlap)) - electrons)
    if electron_error > 1e-8 or occupations[0] < -1e-8 or occupations[-1] > 2 + 1e-8:
        raise ValueError("seed fails electron count or metric occupation admission")
    return {
        "electron_error": electron_error,
        "minimum_occupation": float(occupations[0]),
        "maximum_occupation": float(occupations[-1]),
    }


def prepare_reference_seed(
    owner: Any, atoms: Any, basis: Any, guess: str, expected_aos: int
) -> dict:
    """Construct only an explicit experimental density, with atomic work counted.

    Real spherical ordering/normalization matches the shared comparison basis.
    No target energy, converged reference density, Fock or DIIS state is imported.
    A refused candidate retains Hcore and stays ineligible in the report.
    """
    from generativeqc import _native
    from generativeqc.checkpoint import _descriptor, _pointer
    from pyscf import gto, scf
    from pyscf.scf import atom_hf

    if guess not in ("sap", "mod-huckel"):
        raise ValueError("not an experimental reference seed")
    record: dict[str, Any] = {
        "provider": "pyscf-cpu-experiment",
        "selected": False,
        "production_qualified": False,
        "molecular_fock_builds": 0,
        "atomic_veff_calls": 0,
        "atomic_scf_solves": 0,
        "work_counters_complete": False,
        "atomic_source_sha256": digest(Path(atom_hf.__file__)),
    }
    original_veff = atom_hf.AtomSphAverageRHF.get_veff
    original_scf = atom_hf.AtomSphAverageRHF.kernel

    def counted_veff(atomic: Any, *arguments: Any, **keywords: Any) -> Any:
        record["atomic_veff_calls"] += 1
        return original_veff(atomic, *arguments, **keywords)

    def counted_scf(atomic: Any, *arguments: Any, **keywords: Any) -> Any:
        record["atomic_scf_solves"] += 1
        result = original_scf(atomic, *arguments, **keywords)
        if not atomic.converged:
            raise ValueError("Modified Huckel atomic RHF did not converge")
        return result

    try:
        molecule = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
        if molecule.nao_nr() != expected_aos or not np.array_equal(
            molecule.atom_coords(),
            np.asarray([atom.position for atom in owner._systems[0]]),
        ):
            raise ValueError("reference/target AO count or ordered coordinates differ")
        preparation = scf.RHF(molecule)
        if guess == "sap":
            record["sap_basis"] = preparation.sap_basis
            density = preparation.init_guess_by_sap()
        else:
            atom_hf.AtomSphAverageRHF.get_veff = counted_veff
            atom_hf.AtomSphAverageRHF.kernel = counted_scf
            density = preparation.init_guess_by_mod_huckel()
        density = np.ascontiguousarray(density, dtype=np.float64)
        record["metric_validation"] = validate_seed(
            density, molecule.intor_symmetric("int1e_ovlp"), molecule.nelectron
        )
        coordinates = np.ascontiguousarray([position for _, position in atoms])
        state = _descriptor()
        query = owner._library.generativeqc_batch_get_hf_warm_state
        _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
        if state.present:
            raise ValueError("candidate cannot replace an existing target density")
        state.density, state.coordinates = _pointer(density), _pointer(coordinates)
        state.density_count, state.coordinate_count = density.size, coordinates.size
        state.present = 1
        _native.check(
            owner._library,
            owner._library.generativeqc_batch_restore_hf_warm_states(
                owner._batch, ctypes.byref(state), 1
            ),
            context=owner._context,
        )
        record.update(selected=True, work_counters_complete=True)
    except (ValueError, RuntimeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        atom_hf.AtomSphAverageRHF.get_veff = original_veff
        atom_hf.AtomSphAverageRHF.kernel = original_scf
    return record


def export_density(owner: Any, path: Path) -> dict:
    """Export after timing, solely for paired final-state gates."""
    from generativeqc import _native
    from generativeqc.checkpoint import _descriptor, _pointer

    state = _descriptor()
    query = owner._library.generativeqc_batch_get_hf_warm_state
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    if not state.present or not 0 < state.density_count <= 768**2:
        raise RuntimeError("missing or unbounded final density")
    density = np.empty(state.density_count, dtype=np.float64)
    coordinates = np.empty(state.coordinate_count, dtype=np.float64)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    np.save(path, density)
    return {"path": path.name, "sha256": digest(path), "count": density.size}


def numerical_gate(record: dict, oracle: dict) -> dict:
    """Keep the existing finite, converged 1e-8 Eh / 1e-7 Eh/Bohr gates."""
    from benchmarks.compare_df_direct_endpoint import check_endpoint

    checked = check_endpoint(
        SimpleNamespace(
            energy=record["energy"],
            forces=np.asarray(record["forces"]),
            succeeded=record["status_code"] == 0,
            converged=record["converged"],
        ),
        oracle,
    )
    residual = record["physical_residual_rms"]
    checked["physical_residual_gate"] = bool(
        np.isfinite(residual) and 0 <= residual <= 1e-10
    )
    checked["gate"] = checked["gate"] and checked["physical_residual_gate"]
    return checked


def measure(args: argparse.Namespace) -> None:
    """Run one cold endpoint under Slurm; retain failures rather than dropping them."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("GPU work requires finite srun and its assigned visibility")
    import cupy as cp
    from generativeqc import Calculator, GridSpec, InitialGuessSpec, KsOptions

    from benchmarks import readme_omol25
    from benchmarks.compare_df_direct_endpoint import reference_work_counter
    from benchmarks.compare_gpu4pyscf_batch import (
        load_comparison_basis,
        native_build_metadata,
        require_tuned_native_build,
    )
    from benchmarks.ks_preliminary_density import read_ao_work
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_omol25 import protocol, source_hashes
    from benchmarks.readme_pbe0 import PBE0
    from benchmarks.readme_wb97mv import reference_engine

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(48, 16, 32)
    science = protocol(args.atoms, basis, grid, MINIMUM_REPEATS, benchmark=PBE0)
    production_root = Path(readme_omol25.__file__).parents[1]
    source = source_hashes()
    for name in (
        "python/generativeqc/initial_guess.py",
        "src/scf/initial_guess/minao.cpp",
        "src/scf/initial_guess/minao_data.hpp",
        "src/scf/solver/proposal_control.cpp",
        "src/scf/solver/eigen_frame.cpp",
        "src/scf/cuda/df_eigensystem.cpp",
    ):
        source[name] = digest(production_root / name)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": args.engine,
        "guess": args.guess if args.engine == "native" else "independent-reference",
        "repeat": args.repeat,
        "status": "running",
        "protocol": science,
        "harness_sha256": digest(Path(__file__)),
        "source_file_sha256": source,
        "source_label": os.environ.get("GENERATIVEQC_BENCHMARK_SOURCE", "unverified"),
        "basis_sha256": digest(args.basis_file),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "versions": {
            name: importlib.metadata.version(name) for name in ("pyscf", "numpy")
        },
        "environment": {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("GENERATIVEQC_", "OMP_", "MKL_", "OPENBLAS_"))
        },
        "cold_definition": "fresh process and owner; compiler/runtime caches reused",
        "endpoint_boundary": "prepare + seed construction/admission + synchronized energy/forces",
        "seed_seconds": 0.0,
    }
    oracle = json.loads(args.reference.read_text()) if args.reference else None
    if args.engine == "native":
        if (
            not oracle
            or oracle["status"] != "measured"
            or oracle["protocol"] != science
        ):
            raise ValueError("missing or incompatible independent reference")
        record["reference_sha256"] = digest(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        record["gpu"] = {
            "name": gpu["name"].decode(),
            "major": gpu["major"],
            "minor": gpu["minor"],
            "pci_bus_id": cp.cuda.Device().pci_bus_id,
        }
        if args.engine == "native":
            calculator = Calculator(
                "pbe0-rks",
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                ks_options=KsOptions(grid=grid),
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=100,
                initial_guess=InitialGuessSpec(args.guess)
                if args.guess in ("minao", "huckel")
                else None,
            )
            record["native_build"] = native_build_metadata(calculator)
            require_tuned_native_build(record["native_build"])
            started = perf_counter()
            owner = calculator.prepare_batch([case.atoms], warm_start=True)
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - started
            if args.guess in ("sap", "mod-huckel"):
                started = perf_counter()
                record["seed"] = prepare_reference_seed(
                    owner, case.atoms, reference_basis, args.guess, science["aos"]
                )
                cp.cuda.runtime.deviceSynchronize()
                record["seed_seconds"] = perf_counter() - started
            force_times = []
            force_work = []
            original_force = owner._public_dft_cuda_force

            def observed_force(*arguments: Any, **keywords: Any) -> Any:
                """Observe the ordinary force call without extra work or fences."""
                force_started = perf_counter()
                result = original_force(*arguments, **keywords)
                force_times.append(perf_counter() - force_started)
                force_work.append(result[1])
                return result

            owner._public_dft_cuda_force = observed_force
            started = perf_counter()
            item = owner.execute(strict=False, properties=("energy", "forces")).items[0]
            cp.cuda.runtime.deviceSynchronize()
            record["execute_seconds"] = perf_counter() - started
            record["force_seconds"] = sum(force_times)
            record["force_work"] = force_work
            record.update(
                energy=float(item.energy),
                forces=item.forces.tolist() if item.forces is not None else None,
                converged=bool(item.converged),
                status_code=int(item.status),
                iterations=int(item.iterations),
                fock_builds=item.fock_builds,
                density_rms=float(item.density_rms),
                physical_residual_rms=float(item.physical_residual_rms),
                warm_start_used=bool(item.warm_start_used),
                warm_start_fallback=bool(item.warm_start_fallback),
                initial_guess=item.initial_guess,
                ao_work=read_ao_work(owner),
            )
            record["acceptance"] = numerical_gate(record, oracle)
            record["final_density"] = export_density(
                owner, args.output.with_suffix(".density.npy")
            )
            if (
                digest(Path(record["native_build"]["library_path"]))
                != record["native_build"]["library_sha256"]
            ):
                raise RuntimeError("native library changed during endpoint")
        else:
            started = perf_counter()
            engine = reference_engine(case.atoms, reference_basis, grid, xc="PBE0")
            engine.conv_tol, engine.conv_tol_grad = 1e-12, 1e-10
            engine.direct_scf_tol, engine.max_cycle = 1e-14, 100
            from benchmarks.compare_gpu4pyscf_batch import _configure_reference_scf

            _configure_reference_scf(
                engine,
                energy_tolerance=1e-12,
                gradient_tolerance=1e-10,
                max_iterations=100,
                full_fock=True,
            )
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - started
            with reference_work_counter(engine) as work:
                started = perf_counter()
                energy = engine.kernel()
                gradient = engine.nuc_grad_method()
                gradient.grid_response = True
                forces = cp.asnumpy(-gradient.kernel())
                cp.cuda.runtime.deviceSynchronize()
                record["execute_seconds"] = perf_counter() - started
            record.update(
                energy=float(energy),
                forces=forces.tolist(),
                converged=bool(engine.converged),
                status_code=0 if engine.converged else 4,
                fock_builds=work["scf_jk_builds"],
            )
            if not engine.converged:
                raise RuntimeError("independent reference did not converge")
        record["complete_seconds"] = sum(
            record[name]
            for name in ("prepare_seconds", "seed_seconds", "execute_seconds")
        )
        record["status"] = "measured"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        if owner is not None:
            owner.close()


def summarize(directory: Path, *, arms: tuple[str, ...] = GUESSES) -> dict:
    """Fail closed on incomplete/mixed evidence and retain every negative arm.

    Pair final densities by repeat, rather than comparing just one favorable
    sample. Report both Hcore and the existing native MINAO default as baselines.
    A reference-backed experiment is never an automatic-production candidate.
    """
    if (
        not {"core", "minao"}.issubset(arms)
        or len(set(arms)) != len(arms)
        or any(arm not in (*GUESSES, "huckel") for arm in arms)
    ):
        raise ValueError(
            "screening requires unique admitted arms including core and minao"
        )
    samples = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") == SCHEMA:
            samples.append((path, record))
    result: dict[str, Any] = {
        "schema": SCHEMA + ".summary",
        "minimum_repeats": MINIMUM_REPEATS,
        "wall_reduction_gate": WALL_REDUCTION_GATE,
        "density_gate": DENSITY_GATE,
        "arms": arms,
        "cases": {},
    }
    for atoms in (48, 96):
        rows = [
            (path, row) for path, row in samples if row["protocol"]["atoms"] == atoms
        ]
        references = [(path, row) for path, row in rows if row["engine"] == "reference"]
        native = [(path, row) for path, row in rows if row["engine"] == "native"]
        if len(references) != 1 or not native:
            raise ValueError(
                f"water-{atoms}: need one independent reference and all native arms"
            )
        reference_path, oracle = references[0]
        if oracle["status"] != "measured" or not oracle["converged"]:
            raise ValueError("failed independent reference")
        groups: dict[str, dict[int, dict]] = {guess: {} for guess in arms}
        identities = set()
        for path, row in native:
            guess, repeat = row["guess"], row["repeat"]
            if (
                guess not in groups
                or type(repeat) is not int
                or repeat < 0
                or repeat in groups[guess]
            ):
                raise ValueError("invalid or duplicate guess/repeat")
            if row["status"] != "measured" or row["protocol"] != oracle["protocol"]:
                raise ValueError("failed endpoint or changed scientific protocol")
            if row["reference_sha256"] != digest(reference_path):
                raise ValueError("reference evidence changed")
            seconds = row["complete_seconds"]
            phases = [
                row[name]
                for name in ("prepare_seconds", "seed_seconds", "execute_seconds")
            ]
            if (
                not np.isfinite(seconds)
                or seconds <= 0
                or any(not np.isfinite(value) or value < 0 for value in phases)
                or not np.isclose(seconds, sum(phases), rtol=1e-12, atol=0)
            ):
                raise ValueError("invalid complete endpoint time")
            density = row["final_density"]
            density_path = path.parent / density["path"]
            if digest(density_path) != density["sha256"]:
                raise ValueError("final-density evidence changed")
            final = np.load(density_path, allow_pickle=False)
            if (
                final.shape != (row["protocol"]["aos"] ** 2,)
                or not np.isfinite(final).all()
            ):
                raise ValueError("invalid final density")
            checked = dict(
                row,
                density_array=final,
                numerical_gate=numerical_gate(row, oracle)["gate"],
            )
            groups[guess][repeat] = checked
            identities.add(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "native_build",
                            "harness_sha256",
                            "source_file_sha256",
                            "host",
                            "gpu",
                        )
                    },
                    sort_keys=True,
                )
            )
        repeats = set(groups["core"])
        if (
            len(identities) != 1
            or len(repeats) < MINIMUM_REPEATS
            or any(set(group) != repeats for group in groups.values())
        ):
            raise ValueError("mixed source/library/device or incomplete paired repeats")
        medians = {
            guess: median(row["complete_seconds"] for row in group.values())
            for guess, group in groups.items()
        }
        case = {}
        for guess, group in groups.items():
            density_error = max(
                float(
                    np.max(
                        np.abs(
                            row["density_array"]
                            - groups["core"][repeat]["density_array"]
                        )
                    )
                )
                for repeat, row in group.items()
            )
            seeds_used = all(
                row.get("seed", {}).get("selected", False)
                and row["seed"]["work_counters_complete"]
                and row["warm_start_used"]
                if guess in ("sap", "mod-huckel")
                else not row["warm_start_used"]
                for row in group.values()
            )
            native_seed_used = guess not in ("minao", "huckel") or all(
                row["initial_guess"]
                and row["initial_guess"].get("kind") == guess
                and row["initial_guess"]["outcome"] == "used"
                and row["initial_guess"]["work_counters_complete"]
                and (
                    row["initial_guess"]["preliminary_fock_builds"] == 0
                    if guess == "minao"
                    else row["initial_guess"]["preliminary_fock_builds"] > 0
                )
                for row in group.values()
            )
            science_passed = (
                all(row["numerical_gate"] for row in group.values())
                and density_error <= DENSITY_GATE
            )
            no_retry = all(
                not row["warm_start_fallback"]
                and (row["initial_guess"] or {}).get("target_attempts", 1) == 1
                for row in group.values()
            )
            comparisons = {}
            for baseline in ("core", "minao"):
                reduction = 1 - medians[guess] / medians[baseline]
                performance_gate = (
                    medians[guess] < (1 - WALL_REDUCTION_GATE) * medians[baseline]
                )
                baseline_passed = all(
                    row["numerical_gate"] for row in groups[baseline].values()
                )
                comparisons[baseline] = {
                    "wall_reduction": reduction,
                    "performance_gate": performance_gate,
                    "screening_gate": science_passed
                    and baseline_passed
                    and seeds_used
                    and native_seed_used
                    and no_retry
                    and performance_gate,
                }
            case[guess] = {
                "complete_seconds": [
                    group[repeat]["complete_seconds"] for repeat in sorted(repeats)
                ],
                "median_complete_seconds": medians[guess],
                "numerical_gate": science_passed,
                "maximum_paired_density_error": density_error,
                "candidate_used": seeds_used and native_seed_used,
                "comparisons": comparisons,
                "oracle_backed_seed": guess in ("sap", "mod-huckel"),
                "native_admission_gate": guess in ("minao", "huckel")
                and comparisons["core" if guess == "minao" else "minao"][
                    "screening_gate"
                ],
                "automatic_policy_changed": False,
            }
        result["cases"][str(atoms)] = case
    result["raw_sha256"] = {path.name: digest(path) for path, _ in samples}
    return result


def main() -> None:
    """Expose one-process samples and a host-only evidence checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "reference", "summarize"))
    parser.add_argument("--atoms", type=int, choices=(48, 96))
    parser.add_argument("--guess", choices=(*GUESSES, "huckel"), default="core")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--basis-file",
        type=Path,
        default=Path("benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"),
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument(
        "--native-huckel-only",
        action="store_true",
        help="summarize fresh core/minao/native-Huckel controls without oracle-backed arms",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.engine == "summarize":
        if not args.directory:
            parser.error("summarize requires --directory")
        report = summarize(
            args.directory,
            arms=("core", "minao", "huckel") if args.native_huckel_only else GUESSES,
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    else:
        if not args.atoms or args.repeat < 0:
            parser.error("samples require --atoms and a nonnegative --repeat")
        measure(args)


if __name__ == "__main__":
    main()
