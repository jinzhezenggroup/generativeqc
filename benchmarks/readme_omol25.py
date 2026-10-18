"""Benchmark the OMol25 functional/basis on the README's nested water clusters.

Separate GPU4PySCF and native processes use one offline spherical basis and
moving quadrature. This matches the HF endpoint/frozen-density protocol, not
OMol25's ORCA implementation, molecular distribution, or integration grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc.elements import atomic_number
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)

from benchmarks._gpu4pyscf_grid import reference_grid_order
from benchmarks._readme_grid import grid_work
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks._support import environment_metadata, raw_output_path
from benchmarks.compare_df_direct_endpoint import check_endpoint, reference_work_counter
from benchmarks.compare_gpu4pyscf_batch import (
    gpu_convergence_payload,
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.readme_accuracy_oracle import (
    merge_geometry_oracles,
    read_geometry_oracle,
)
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_wb97mv import (
    configure_reference_full_fock,
    configure_reference_mixed_rsh_df,
    reference_engine,
    reference_vv10_domain,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SIZES = (3, 6, 12, 24, 48, 96)
SCHEMA = "generativeqc.readme-omol25.v3"


@dataclass(frozen=True)
class EndpointSpec:
    """Scientific identity for runners sharing the HF complete-endpoint protocol."""

    method: str
    basis: str
    schema: str
    reference_full_fock: bool = False
    density_fitting: bool = False

    @property
    def native_method(self) -> str:
        """Use the registered RKS identifier, not a bare XC functional alias."""
        return "rhf" if self.method == "hf" else f"{self.method}-rks"

    @property
    def has_vv10(self) -> bool:
        """Only the range-separated OMol25 composition needs nonlocal screening."""
        return self.method == "wb97m-v"

    @property
    def is_r2scan3c(self) -> bool:
        """The 3c endpoint includes its fixed basis, D4 and gCP corrections."""
        return self.method == "r2scan-3c"

    @property
    def reference_xc(self) -> str:
        """Select only the electronic functional; 3c corrections run separately."""
        return "R2SCAN" if self.is_r2scan3c else self.method.upper().replace("-", "_")


OMOL25 = EndpointSpec("wb97m-v", "def2-TZVPD", SCHEMA)


def source_hashes() -> dict[str, str]:
    """Identify dirty consumers and the native schedule behind the binary hash."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "python/generativeqc/calculator.py",
        "python/generativeqc/ks.py",
        "python/generativeqc/_ks_snapshot.py",
        "python/generativeqc/_snapshot_grid_cache.py",
        "python/generativeqc/resources_ks.py",
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/_force_active_ao.py",
        "python/generativeqc/batch.py",
        "python/generativeqc/precision_policy.py",
        "python/generativeqc/_precision_batch.py",
        "python/generativeqc/_precision_work.py",
        "python/generativeqc/_df_precision.py",
        "python/generativeqc/_composite_precision.py",
        "python/generativeqc_compiler/dft/ao_cuda.py",
        "python/generativeqc_compiler/dft/xc_contraction_cuda.py",
        "python/generativeqc_compiler/method/stationary_cuda.py",
        "python/generativeqc_compiler/method/stationary_resources.py",
        "python/generativeqc_compiler/method/stationary_composite_resources.py",
        "python/generativeqc/_stationary_composite_cuda.py",
        "python/generativeqc/_geometry_response_cache.py",
        "benchmarks/readme_omol25.py",
        "benchmarks/readme_accuracy_oracle.py",
        "benchmarks/ks_preliminary_density.py",
        "benchmarks/readme_pbe0.py",
        "benchmarks/readme_dft.py",
        "benchmarks/_readme_grid.py",
        "benchmarks/_gpu4pyscf_grid.py",
        "tools/generativeqc_gcp/reference.py",
        "tools/parameters/r2scan3c_gcp.json",
        "benchmarks/readme_wb97mv.py",
        "benchmarks/compare_df_direct_endpoint.py",
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_jk_kernels.cu",
        "src/scf/cuda/direct_jk_kernels.hpp",
        "src/scf/cuda/direct_jk_plan.hpp",
        # The prepared DFT provider has its own value/derivative routes. Retain
        # their compiler owners as well as the runtime that selects them, so a
        # dirty recurrence experiment is identifiable beside the binary hash.
        "src/scf/cuda/direct_angular_fock.cu",
        "src/scf/cuda/direct_bounded_dddd.cu",
        "src/scf/cuda/direct_bounded_fallback.cu",
        "src/scf/cuda/direct_coulomb.cpp",
        "src/scf/cuda/direct_force_class_domains.cu",
        "src/scf/cuda/direct_force_class_domains.hpp",
        "src/scf/cuda/direct_force_class_pages.cuh",
        "src/scf/cuda/direct_force_diagnostic.hpp",
        "src/scf/cuda/direct_spd_force_schedule.hpp",
        "src/dft/cuda_ks.cpp",
        "src/dft/cuda_ks_final_validation_policy.hpp",
        "python/generativeqc/resources_hf.py",
        "src/scf/cuda/packed_basis.hpp",
        "src/scf/cuda/rhf_policy.cpp",
        "src/scf/cuda/rhf_policy.hpp",
        "src/scf/rhf.cpp",
        "src/scf/application_precision.hpp",
        "src/scf/cuda/df_force_response.cpp",
        "src/scf/cuda/df_gradient_bridge.cu",
        "src/scf/cuda/df_shell_kernel.cuh",
        "src/scf/cuda/df_shell_launch.cuh",
        "src/scf/cuda/df_shell_dispatch.cuh",
        "src/scf/cuda/df_shell_derivatives.cu",
        "python/generativeqc_compiler/integral/df_shell_derivatives.py",
        "python/generativeqc_compiler/integral/df_rys_shell.py",
        "python/generativeqc_compiler/integral/df_shell_units.py",
        "tools/generate_df_kernels.py",
        "python/generativeqc_compiler/integral/direct_pair_materialized_cuda.py",
        "python/generativeqc_compiler/integral/direct_pair_materialized_gradient_cuda.py",
        "python/generativeqc_compiler/integral/direct_cooperative_gradient_cuda.py",
        "python/generativeqc_compiler/integral/direct_pair_support_cuda.py",
        "python/generativeqc_compiler/integral/direct_cartesian_contraction_cuda.py",
        "python/generativeqc_compiler/integral/direct_recurrence_cuda.py",
        "python/generativeqc_compiler/integral/direct_order2_shell_cuda.py",
        "src/scf/cuda/one_electron_gradient_bridge.cu",
        "src/scf/cuda/basis_transform_kernels.cu",
        "src/scf/cuda/basis_transform_kernels.hpp",
        "python/generativeqc_compiler/integral/direct_source_contraction_cuda.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def protocol(
    atoms: int,
    basis: Any,
    spec: Any,
    repeats: int,
    *,
    benchmark: EndpointSpec = OMOL25,
    auxiliary_basis: Any = None,
    density_fitting_memory_budget_bytes: int = 0,
) -> dict[str, Any]:
    """Describe inputs without loading a native library or touching the GPU.

    Retain workspace hints so distinct resource/work populations are not pooled,
    even though the hint changes neither the Hamiltonian nor its accuracy gates.
    """
    if (
        type(density_fitting_memory_budget_bytes) is not int
        or not 0 <= density_fitting_memory_budget_bytes <= 1 << 40
        or (density_fitting_memory_budget_bytes and not benchmark.density_fitting)
    ):
        raise ValueError(
            "DF workspace allowance requires DF and an integer in [0,2**40]"
        )
    if benchmark.density_fitting and auxiliary_basis is None:
        raise ValueError("DF protocol requires an explicit auxiliary basis identity")
    original = scaling_cases()[f"water-{atoms}"].atoms
    moved = [
        (
            number,
            tuple(np.asarray(coords) + ((0, 0, 0.001) if index == 1 else (0, 0, 0))),
        )
        for index, (number, coords) in enumerate(original)
    ]
    aos = sum(
        sum(
            (2 * shell.angular_momentum + 1) * len(shell.coefficients)
            for shell in basis.by_element[atomic_number(number)].shells
        )
        for number, _ in original
    )
    return json.loads(
        json.dumps(
            {
                "case": f"water-{atoms}",
                "atoms": atoms,
                "aos": aos,
                "method": "RHF"
                if benchmark.method == "hf"
                else f"{benchmark.method.upper()}/RKS",
                "basis": benchmark.basis,
                "basis_identity": basis.identity,
                "representation": "spherical",
                "geometries_bohr": [original, moved],
                "grid": None if benchmark.method == "hf" else asdict(spec),
                "properties": ["energy", "forces"],
                "force_return": "host_array_in_timed_endpoint",
                "density_fitting": benchmark.density_fitting,
                **(
                    {
                        "auxiliary_basis": auxiliary_basis.name,
                        "auxiliary_basis_identity": auxiliary_basis.identity,
                        "density_fitting_relative_threshold": 1e-10,
                        "density_fitting_memory_budget_bytes": density_fitting_memory_budget_bytes,
                        "reference_auxbasis_response": True,
                    }
                    if benchmark.density_fitting
                    else {}
                ),
                **(
                    {
                        "reference_rsh_exchange": {
                            "primary_j": "density-fitted-full-range",
                            "primary_k": "density-fitted-full-range",
                            "long_range_correction_k": "exact-direct",
                            "long_range_correction_gradient": "exact-direct-analytic",
                        },
                    }
                    if benchmark.density_fitting and benchmark.has_vv10
                    else {}
                ),
                **(
                    {
                        "reference_corrections": "CPU dftd4 + independent CPU gCP; included in every endpoint",
                        "d4_parameters": {
                            "s6": 1.0,
                            "s8": 0.0,
                            "s9": 2.0,
                            "a1": 0.42,
                            "a2": 5.65,
                            "ga": 2.0,
                            "gc": 1.0,
                        },
                    }
                    if benchmark.is_r2scan3c
                    else {}
                ),
                **(
                    {
                        "nonlocal_density_policy": MOLECULAR_VV10_DENSITY_POLICY,
                        "nonlocal_density_threshold": float(
                            MOLECULAR_VV10_DENSITY_THRESHOLD
                        ),
                        "reference_nonlocal_weight_threshold": 1e-14,
                    }
                    if benchmark.has_vv10
                    else {}
                ),
                "repeats": repeats,
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "reference_gradient_tolerance": 1e-10,
                "screening_tolerance": 1e-12,
                "reference_direct_scf_tolerance": 1e-14,
                "reference_fock_policy": (
                    "full-density-rebuild"
                    if benchmark.reference_full_fock
                    else "incremental"
                ),
                "max_iterations": 100,
                "energy_gate": 1e-8,
                "force_gate": 1e-7,
                "cold_definition": "calculator construction, preparation and first synchronized energy/force endpoint; process startup and provenance collection excluded; existing compilation caches retained",
                "warm_definition": f"{repeats} fixed engine-local post-cold/post-move density replays",
                "threads": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            }
        )
    )


def native_solver_controls(policy: str) -> dict[str, float | int]:
    """Preserve FP64 controls while letting Fast own its budget-selected solver.

    Passing the FP64 benchmark's tight tolerances as explicit user ceilings
    suppresses Fast's default stationarity selection. The independent reference
    still uses the tight target controls in ``protocol``; candidate controls and
    per-call refinement/fallback work are recorded separately.
    """
    if policy == "fp64":
        return {
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-12,
            "max_iterations": 100,
        }
    if policy == "fast":
        return {"max_iterations": 100}
    raise ValueError("native benchmark policy must be fp64 or fast")


def check_record(
    row: dict[str, Any], oracle: dict[str, Any], *, budget: dict | None = None
) -> dict[str, Any]:
    """Reuse HF's finite-value, shape, convergence and complete-force gates."""
    checked = check_endpoint(
        SimpleNamespace(
            energy=row["energy"],
            forces=np.asarray(row["forces"]),
            converged=row["converged"],
            succeeded=row["status"] == 0,
        ),
        oracle,
    )
    if budget is not None:
        valid = checked["energy_error"] is not None
        rms_error = (
            float(
                np.sqrt(
                    np.mean(
                        (np.asarray(row["forces"]) - np.asarray(oracle["forces"])) ** 2
                    )
                )
            )
            if valid
            else None
        )
        checked.update(
            fp64_gate=checked["gate"],
            force_rms_error=rms_error,
            gate=bool(
                valid
                and checked["energy_error"] <= budget["energy_abs"]
                and checked["force_error"] <= budget["force_max_abs"]
                and (budget["force_rms"] is None or rms_error <= budget["force_rms"])
            ),
            acceptance_scope="explicit-application-budget-not-fp64-parity",
        )
    return checked


def complete_execution(row: dict[str, Any], atoms: int) -> bool:
    """Check observable E+F completion without claiming independent accuracy.

    In particular, a converged energy with absent or malformed host forces is
    never a completed endpoint, even when no independent oracle is available.
    """
    try:
        forces = np.asarray(row["forces"], dtype=float)
        return bool(
            row["status"] == 0
            and row["converged"]
            and forces.shape == (atoms, 3)
            and np.isfinite(forces).all()
            and np.isfinite(float(row["energy"]))
            and np.isfinite(float(row["complete_seconds"]))
            and row["complete_seconds"] > 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def reference_baselines(
    oracle: dict[str, Any] | None,
    scientific: dict[str, Any],
    *,
    allow_missing: bool = False,
) -> dict[int, dict[str, Any]]:
    """Use completed geometry oracles even if later reference calls timed out.

    Execution independence never permits a different Hamiltonian, geometry,
    basis or grid. Missing geometry baselines stay absent; they are not replaced
    by native results or by a reference from the other geometry.
    """
    if oracle is None:
        if allow_missing:
            return {}
        raise ValueError("independent reference is unavailable")
    if oracle["protocol"] != scientific:
        raise ValueError("independent reference protocol differs")
    baselines = {}
    for row in oracle["records"]:
        if row["phase"] not in ("cold", "moved"):
            continue
        geometry = 0 if row["phase"] == "cold" else 1
        if row["geometry"] != geometry or geometry in baselines:
            raise ValueError("duplicate or mislabeled independent reference")
        if complete_execution(row, scientific["atoms"]) and row.get("gate") is True:
            baselines[geometry] = row
    if not allow_missing and (oracle["status"] != "measured" or len(baselines) != 2):
        raise ValueError("independent reference is incomplete")
    return baselines


def require_force_endpoint(capabilities: dict[str, Any]) -> None:
    """Honor the candidate's public capability, without hard-coding an SPD gate.

    Generic integral or internal geometry admission does not qualify the public
    force endpoint. Qualified f-shell candidates must not be rejected here.
    """
    if "forces" not in capabilities["supported_properties"]:
        raise NotImplementedError(
            "the native Calculator does not advertise analytic forces for this basis; "
            "no HF-equivalent energy-plus-force endpoint is available"
        )


def reference_xc_backend(engine: Any, *, spin: int = 0) -> dict[str, Any]:
    """Read the same cached XCfun flags used by GPU4PySCF after the timer.

    GPU4PySCF 1.8.1 falls back for the whole semilocal expression if any
    component lacks CUDA LibXC support; a mixed list is not a mixed execution
    backend. Missing flags remain unknown, never evidence of CPU fallback.
    This describes semilocal evaluation, not exact exchange or VV10 kernels.
    """
    if not hasattr(engine, "xc"):
        return {"backend": "no-semilocal-xc", "components": []}
    record = {
        "xc_code": engine.xc,
        "spin": spin,
        "backend": "unknown",
        "components": [],
    }
    try:
        functions = engine._numint._init_xcfuns(engine.xc, spin)
        for function, coefficient in functions:
            on_gpu = getattr(function, "on_gpu", None)
            record["components"].append(
                {
                    "functional_id": int(function.func_id),
                    "coefficient": float(coefficient),
                    "on_gpu": on_gpu if type(on_gpu) is bool else None,
                }
            )
    except (AttributeError, TypeError, ValueError, NotImplementedError) as error:
        record["unavailable_reason"] = f"{type(error).__name__}: {error}"
        return record
    flags = [component["on_gpu"] for component in record["components"]]
    if not flags:
        record["backend"] = "no-semilocal-xc"
    elif any(flag is False for flag in flags):
        record["backend"] = "pyscf-cpu-libxc"
    elif all(flag is True for flag in flags):
        record["backend"] = "cuda-libxc"
    return record


def reference_endpoint_seconds(
    *,
    started: float,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = perf_counter,
) -> float:
    """Stop the GPU endpoint clock only after outstanding device work completes."""
    synchronize()
    return clock() - started


def native_endpoint_record(
    item: Any,
    *,
    phase: str,
    geometry_index: int,
    repeat: int,
    seconds: float,
    prepare: float,
    owner: Any,
    force_work: Any,
    has_xc: bool = True,
) -> dict[str, Any]:
    """Serialize the public endpoint, preserving absent forces and diagnostics."""
    from benchmarks.dft_force_components import normalize_force_work
    from benchmarks.ks_preliminary_density import read_ao_work

    return {
        "phase": phase,
        "geometry": geometry_index,
        "repeat": max(0, repeat - 1),
        "energy": item.energy if np.isfinite(item.energy) else None,
        "forces": item.forces.tolist() if item.forces is not None else None,
        "converged": item.converged,
        "status": item.status,
        "detail": item.status_message,
        "iterations": item.iterations,
        "fock_builds": item.fock_builds,
        "energy_change": item.energy_change,
        "density_rms": item.density_rms,
        "physical_residual_rms": item.physical_residual_rms,
        # Public execution already copied this physical history.
        # Preserve it after timing so trajectory changes remain
        # auditable; a warm retry describes its final attempt,
        # while item.fock_builds retains the complete work count.
        "native_ks_diagnostic": (
            item.ks_diagnostic.to_payload() if item.ks_diagnostic is not None else None
        ),
        "warm_start_used": item.warm_start_used,
        "warm_start_fallback": item.warm_start_fallback,
        "seconds": seconds,
        "prepare_seconds": prepare,
        "complete_seconds": seconds + prepare,
        "native_scf_ao_work": (
            read_ao_work(owner)
            if owner is not None and has_xc and item.status == 0 and item.converged
            else None
        ),
        "native_force_components": (
            normalize_force_work(force_work) if force_work is not None else None
        ),
        **(
            {
                "corrections": {
                    "backend": item.dispersion.backend,
                    "d4_energy": item.dispersion.d4.energy,
                    "gcp_energy": item.dispersion.gcp.energy,
                }
            }
            if getattr(item, "dispersion", None) is not None
            and item.dispersion.gcp is not None
            else {}
        ),
    }


def retain_endpoint_record(
    record: dict[str, Any],
    row: dict[str, Any],
    reference: dict[str, Any] | None,
    save: Callable[[str], None],
) -> None:
    """Persist completion separately from accuracy; known gate failures still fail."""
    if reference is None:
        completed = complete_execution(row, record["protocol"]["atoms"])
        row.update(
            execution_gate=completed,
            energy_error=None,
            force_error=None,
            gate=None if completed else False,
            accuracy_status="reference_unavailable",
        )
    else:
        row.update(
            check_record(row, reference, budget=record.get("application_budget"))
        )
    record["records"].append(row)
    save(record["stage"])
    if row["gate"] is False:
        raise RuntimeError(f"independent energy/force gate failed: {row['phase']}")


def reference_energy_forces(
    engine: Any,
    seed: Any,
    *,
    export: Callable[[Any], Any],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = perf_counter,
    correction: Callable[[], tuple[float, Any]] | None = None,
) -> tuple[float, Any, float]:
    """Time SCF, seed copy, analytic gradient and host force export together."""
    synchronize()
    started = clock()
    energy = float(engine.kernel(dm0=None if seed is None else seed.copy()))
    if not engine.converged:
        raise RuntimeError("GPU4PySCF SCF did not converge")
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    if hasattr(engine, "with_df"):
        # Nuclear derivatives include motion of the fitting basis and metric.
        gradient.auxbasis_response = True
    with reference_grid_order() if hasattr(engine, "xc") else nullcontext():
        forces = export(-gradient.kernel())
    if correction is not None:
        correction_energy, correction_gradient = correction()
        energy += correction_energy
        forces -= correction_gradient
    seconds = reference_endpoint_seconds(
        started=started, synchronize=synchronize, clock=clock
    )
    return energy, forces, seconds


def force_execution_options(
    *,
    active_ao: bool,
    max_device_bytes: int | None = None,
    max_host_bytes: int | None = None,
) -> dict[str, Any]:
    """Explicit benchmark-only force allowances, additional to the SCF owner.

    Larger AO domains can exceed the production consumer's default allowance.
    An override still goes through complete composite admission and is retained
    in the report; it neither changes the science nor promotes a public default.
    Omitted limits preserve the consumer's defaults exactly.
    """
    options: dict[str, Any] = {}
    for name, value in (
        ("max_device_bytes", max_device_bytes),
        ("max_host_bytes", max_host_bytes),
    ):
        if value is not None:
            if type(value) is not int or not 0 < value <= 1 << 40:
                raise ValueError(f"force {name} must be an integer in [1,2**40]")
            options[name] = value
    if active_ao:
        options.update(active_ao_cutoff=1e-16, active_ao_cache_bytes=64 << 20)
    return options


def reference_correction(atoms: Any) -> Callable[[], tuple[float, Any]]:
    """Create an independent 3c oracle; all geometry-dependent work stays timed.

    GPU4PySCF supplies the electronic endpoint. The reference's D4 and gCP
    corrections execute on CPU and must be labeled as such in comparisons.
    The native endpoint never calls these reference implementations.
    """
    from dftd4.interface import DampingParam, DispersionModel

    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp

    numbers = np.asarray([atomic_number(symbol) for symbol, _ in atoms], dtype=np.int32)
    coordinates = np.asarray([xyz for _, xyz in atoms])

    def evaluate() -> tuple[float, Any]:
        damping = DampingParam(s6=1.0, s8=0.0, s9=2.0, a1=0.42, a2=5.65)
        d4 = DispersionModel(
            numbers, coordinates, charge=0.0, ga=2.0, gc=1.0
        ).get_dispersion(damping, grad=True)
        gcp = evaluate_r2scan3c_gcp(tuple(numbers), coordinates)
        return float(d4["energy"]) + gcp.energy, d4["gradient"] + gcp.gradient

    return evaluate


def main(benchmark: EndpointSpec = OMOL25, argv: list[str] | None = None) -> None:
    """Journal complete independent endpoints, including experimental cold seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("reference", "native"))
    parser.add_argument(
        "--policy",
        choices=("fp64", "fast"),
        default="fp64",
        help="Native application arithmetic; fp64 remains the ordinary optimized Calculator",
    )
    parser.add_argument(
        "--fast-budget",
        type=float,
        nargs=3,
        metavar=("ENERGY", "MAX_FORCE", "RMS_FORCE"),
        help="Fast allowance in Eh, Eh/bohr, Eh/bohr; every independent norm must pass",
    )
    parser.add_argument("--atoms", type=int, choices=SIZES, required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--density-fitting", choices=("none", "cuda"), default="none")
    parser.add_argument(
        "--df-memory-budget-bytes",
        type=int,
        default=0,
        help="Explicit native DF workspace hint; zero preserves automatic selection",
    )
    parser.add_argument("--auxiliary-basis-file", type=Path)
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=(48, 16, 32),
        metavar=("RADIAL", "POLAR", "AZIMUTH"),
        help="Explicit shared XC/VV10 grid; the historical default is not a convergence recommendation",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--accuracy-reference",
        type=Path,
        action="append",
        default=[],
        help="Accuracy-only independent geometry JSON beside the output; no timing sample",
    )
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Measure native completion independently; missing accuracy gates remain unchecked",
    )
    parser.add_argument("--reference-full-fock", action="store_true")
    parser.add_argument(
        "--preliminary-provider",
        choices=("none", "lda16", "pbe16"),
        default="none",
        help="Benchmark-only same-basis GPU density; its entire lifecycle counts as cold",
    )
    parser.add_argument(
        "--force-active-ao",
        action="store_true",
        help="Opt into the separately qualified force AO maps (cutoff 1e-16)",
    )
    parser.add_argument(
        "--force-max-device-bytes",
        type=int,
        help="Explicit additional composite-force device allowance; excludes SCF storage",
    )
    parser.add_argument(
        "--force-max-host-bytes",
        type=int,
        help="Explicit additional composite-force host allowance; excludes SCF storage",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args(argv)
    if args.engine == "reference" and args.policy != "fp64":
        parser.error("the independent reference must remain FP64")
    if args.fast_budget is not None and args.policy != "fast":
        parser.error("--fast-budget requires --policy fast")
    if args.policy == "fast" and (
        args.preliminary_provider != "none"
        or args.force_active_ao
        or args.force_max_device_bytes is not None
        or args.force_max_host_bytes is not None
    ):
        parser.error(
            "Fast qualification cannot be combined with private seed/force experiments"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU work requires finite srun on main with --gres=gpu:5090:1")
    if args.repeats < 1 or min(args.grid) < 1:
        parser.error("repeats and grid dimensions must be positive")
    if (
        args.engine == "native"
        and args.reference is None
        and not args.allow_missing_reference
    ):
        parser.error("native execution requires the independent --reference JSON")
    if (args.density_fitting == "cuda") != (args.auxiliary_basis_file is not None):
        parser.error("DF requires an explicit matching --auxiliary-basis-file")
    if not 0 <= args.df_memory_budget_bytes <= 1 << 40 or (
        args.df_memory_budget_bytes
        and (args.engine != "native" or args.density_fitting != "cuda")
    ):
        parser.error(
            "DF workspace allowance requires native DF and a value in [0,2**40]"
        )
    benchmark = replace(benchmark, density_fitting=args.density_fitting == "cuda")
    try:
        force_options = force_execution_options(
            active_ao=args.force_active_ao,
            max_device_bytes=args.force_max_device_bytes,
            max_host_bytes=args.force_max_host_bytes,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.engine == "reference" and (
        args.preliminary_provider != "none" or force_options
    ):
        parser.error("seed and force AO experiments apply only to the native engine")
    if not benchmark.has_vv10 and (
        args.preliminary_provider != "none" or force_options
    ):
        parser.error("these experimental policies are qualified only for WB97M-V")
    if args.reference_full_fock:
        benchmark = replace(benchmark, reference_full_fock=True)

    import cupy as cp
    from generativeqc import (
        Calculator,
        GridSpec,
        KsOptions,
        PrecisionCalculator,
        PrecisionPolicy,
        TargetErrorBudget,
    )

    try:
        precision_policy = (
            PrecisionPolicy(
                "fast",
                budget=TargetErrorBudget(*args.fast_budget)
                if args.fast_budget
                else None,
            )
            if args.policy == "fast"
            else None
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    from benchmarks._support import cuda_accelerator_metadata
    from benchmarks.dft_force_components import normalize_force_work

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    auxiliary_basis = reference_auxiliary = None
    if benchmark.density_fitting:
        auxiliary_basis, reference_auxiliary = load_retained_comparison_basis(
            args.auxiliary_basis_file, case, role="auxiliary", compute_forces=True
        )
    if basis.name.lower() != benchmark.basis.lower():
        raise ValueError(
            f"this benchmark requires the complete {benchmark.basis} basis"
        )
    spec = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    scientific = protocol(
        args.atoms,
        basis,
        spec,
        args.repeats,
        benchmark=benchmark,
        auxiliary_basis=auxiliary_basis,
        density_fitting_memory_budget_bytes=args.df_memory_budget_bytes,
    )
    oracle = None
    if args.engine == "native" and args.reference and args.reference.exists():
        oracle = json.loads(args.reference.read_text())
    record: dict[str, Any] = {
        "schema": benchmark.schema
        if precision_policy is None
        else f"{benchmark.schema}.fast.v1",
        "engine": args.engine,
        "status": "running",
        "protocol": scientific,
        "grid_work": None
        if benchmark.method == "hf"
        else grid_work(args.atoms, spec, has_vv10=benchmark.has_vv10),
        "records": [],
        "source_file_sha256": source_hashes(),
        "application_budget": None
        if precision_policy is None
        else asdict(precision_policy.budget),
        "native_experiment": {
            "precision_policy": args.policy,
            "solver_control_policy": "application-policy-selected"
            if args.policy == "fast"
            else "fixed-fp64-target",
            "requested_solver_controls": native_solver_controls(args.policy),
            "preliminary_provider": args.preliminary_provider,
            "force_active_ao_cutoff": 1e-16 if args.force_active_ao else None,
            "force_active_ao_cache_bytes": 64 << 20 if args.force_active_ao else None,
            "force_max_device_bytes": args.force_max_device_bytes,
            "force_max_host_bytes": args.force_max_host_bytes,
            # Without a benchmark seed override, Calculator owns initialization
            # through its public default (currently automatic Modified Huckel).
            "public_initialization_policy": args.preliminary_provider == "none",
        },
        "native_schedule_settings": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("GENERATIVEQC_") and name != "GENERATIVEQC_LIBRARY"
        },
        "native_schedule_policy": "automatic-generated-SPD/canonical-through-f"
        if benchmark.has_vv10
        else "automatic-generated-SPD",
        "basis_file_sha256": hashlib.sha256(args.basis_file.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()
        if args.reference and args.reference.exists()
        else None,
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
                "cutensor": ("cutensor-cu12",),
                "numpy": ("numpy",),
                "dftd4": ("dftd4",),
            }
        ),
        "scheduler": {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_GPUS_ON_NODE")
        },
    }
    record["environment"]["accelerator"] = cuda_accelerator_metadata(cp)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def retain(row: dict[str, Any], reference: dict[str, Any] | None) -> None:
        retain_endpoint_record(record, row, reference, save)

    accuracy_receipts = []
    if args.accuracy_reference:
        if args.engine != "native":
            parser.error("accuracy-only references apply to native validation")
        record["accuracy_reference_files"] = []
        for path in args.accuracy_reference:
            receipt, binding = read_geometry_oracle(path, directory=args.output.parent)
            accuracy_receipts.append(receipt)
            record["accuracy_reference_files"].append(binding)

    owner = None
    restore_force = None
    try:
        save("setup")
        geometries = scientific["geometries_bohr"]
        if args.engine == "reference":
            density = None
            for geometry_index, geometry in enumerate(geometries):
                phase = "cold" if geometry_index == 0 else "moved"
                save(f"{phase}/prepare")
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                if benchmark.method == "hf":
                    from gpu4pyscf.scf import hf
                    from pyscf import gto

                    engine = hf.RHF(
                        gto.M(
                            atom=geometry,
                            basis=reference_basis,
                            unit="Bohr",
                            cart=False,
                            verbose=0,
                        )
                    )
                else:
                    engine = reference_engine(
                        geometry,
                        reference_basis,
                        spec,
                        xc=benchmark.reference_xc,
                    )
                if benchmark.density_fitting:
                    engine = engine.density_fit(auxbasis=reference_auxiliary)
                # Wrap the final engine: binding get_veff before density_fit()
                # would keep a bound method on the original direct owner.
                if benchmark.reference_full_fock:
                    configure_reference_full_fock(engine)
                if benchmark.density_fitting and benchmark.has_vv10:
                    # The default GPU4PySCF DF-RSH object also fits K_LR,
                    # unlike native's independently prepared exact LR owner.
                    # Match both K_LR in SCF and its analytic gradient.
                    configure_reference_mixed_rsh_df(engine)
                from gpu4pyscf.dft import libxc
                from gpu4pyscf.lib import cutensor

                record["reference_tensor_engine"] = (
                    "cuTENSOR" if cutensor.cutensor is not None else "CuPy einsum"
                )
                record["reference_libxc"] = libxc.__version__
                engine.conv_tol = 1e-12
                engine.conv_tol_grad = 1e-10
                engine.direct_scf_tol = 1e-14
                engine.max_cycle = 100
                correction = (
                    reference_correction(geometry) if benchmark.is_r2scan3c else None
                )
                cp.cuda.Stream.null.synchronize()
                prepare_seconds = perf_counter() - started
                if engine.mol.nao_nr() != scientific["aos"]:
                    raise ValueError("native and reference basis dimensions differ")

                def execute_reference(
                    phase: str,
                    repeat: int,
                    seed: Any,
                    prepare: float = 0,
                    *,
                    engine: Any = engine,
                    geometry_index: int = geometry_index,
                    correction: Any = correction,
                ) -> dict[str, Any]:
                    """Bind the geometry owner; density copies remain inside the timer."""
                    save(f"{phase}/{repeat}")
                    with (
                        reference_work_counter(engine) as work,
                        (
                            reference_vv10_domain(
                                scientific["nonlocal_density_threshold"]
                            )
                            if benchmark.has_vv10
                            else nullcontext()
                        ) as domain,
                    ):
                        energy, forces, seconds = reference_energy_forces(
                            engine,
                            seed,
                            export=cp.asnumpy,
                            synchronize=cp.cuda.Stream.null.synchronize,
                            correction=correction,
                        )
                        convergence = gpu_convergence_payload(
                            [engine], [work["tracker"]]
                        )[0]
                    return {
                        "phase": phase,
                        "geometry": geometry_index,
                        "repeat": repeat,
                        "energy": energy,
                        "forces": forces.tolist(),
                        "converged": bool(engine.converged),
                        "status": 0,
                        "iterations": convergence["iterations"],
                        "scf_jk_builds": work["scf_jk_builds"],
                        "scf_final_residuals": convergence["final_residuals"],
                        "warm_start_used": seed is not None,
                        "reference_xc_backend": reference_xc_backend(engine),
                        **({"reference_vv10_domain": domain} if domain else {}),
                        "seconds": seconds,
                        "prepare_seconds": prepare,
                        "complete_seconds": seconds + prepare,
                    }

                baseline = execute_reference(phase, 0, density, prepare_seconds)
                retain(baseline, baseline)
                density = engine.make_rdm1().copy()
                if benchmark.method != "hf":
                    record.setdefault("grid_points", []).append(
                        {
                            "semilocal": len(engine.grids.weights),
                            **(
                                {"vv10": len(engine.nlcgrids.weights)}
                                if benchmark.has_vv10
                                else {}
                            ),
                        }
                    )
                for repeat in range(args.repeats):
                    retain(
                        execute_reference(
                            "warm" if geometry_index == 0 else "moved-warm",
                            repeat,
                            density,
                        ),
                        baseline,
                    )
        else:
            from benchmarks.ks_preliminary_density import prepare_seed

            baselines = reference_baselines(
                oracle, scientific, allow_missing=args.allow_missing_reference
            )
            baselines = merge_geometry_oracles(
                baselines,
                accuracy_receipts,
                scientific,
                basis_sha256=record["basis_file_sha256"],
            )
            if force_options:
                from generativeqc._stationary_composite_cuda import (
                    PreparedCompositeStationaryCudaGradient,
                )

                # Scoped benchmark opt-in; the production planner keeps its
                # full-AO fallback and no default is changed by this experiment.
                restore_force = PreparedCompositeStationaryCudaGradient.execute

                def active_force(self: Any, *values: Any, **kwargs: Any) -> Any:
                    kwargs.update(force_options)
                    return restore_force(self, *values, **kwargs)

                PreparedCompositeStationaryCudaGradient.execute = active_force
            save("native/calculator")
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            calculator_type = (
                Calculator if precision_policy is None else PrecisionCalculator
            )
            precision_options = (
                {} if precision_policy is None else {"policy": precision_policy}
            )
            calc = calculator_type(
                method=benchmark.native_method,
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                ks_options=None if benchmark.method == "hf" else KsOptions(grid=spec),
                density_fitting=args.density_fitting,
                auxiliary_basis=auxiliary_basis,
                density_fitting_relative_threshold=1e-10,
                density_fitting_memory_budget_bytes=args.df_memory_budget_bytes,
                **native_solver_controls(args.policy),
                **precision_options,
            )
            cp.cuda.Stream.null.synchronize()
            calculator_seconds = perf_counter() - started
            record["native_calculator_seconds"] = calculator_seconds
            selected_calculator = calc if precision_policy is None else calc._candidate
            record["native_experiment"]["initial_candidate_solver_controls"] = {
                "energy_tolerance": selected_calculator._energy_tolerance,
                "density_tolerance": selected_calculator._density_tolerance,
                "screening_tolerance": selected_calculator._screening_tolerance,
                "max_iterations": selected_calculator._max_iterations,
            }
            record["native_build"] = native_build_metadata(
                calc if precision_policy is None else calc._candidate
            )
            require_tuned_native_build(record["native_build"])
            record["native_capabilities"] = {
                "supported_properties": sorted(calc.capabilities.supported_properties),
                "maximum_basis_angular_momentum": max(
                    shell.angular_momentum
                    for element in basis.elements
                    for shell in element.shells
                ),
            }
            save("native/capability")
            require_force_endpoint(record["native_capabilities"])
            save("cold/prepare")
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            owner = calc.prepare_batch([geometries[0]], warm_start=True)
            force_work = None
            original_force = (
                owner._public_dft_cuda_force if precision_policy is None else None
            )

            def observed_force(*values: Any) -> Any:
                """Retain already-produced work; default owners have no resource ledger."""
                nonlocal force_work
                result = original_force(*values)
                force_work = result[1]
                return result

            if original_force is not None:
                owner._public_dft_cuda_force = observed_force
            cp.cuda.Stream.null.synchronize()
            prepare_seconds = calculator_seconds + perf_counter() - started
            save("cold/preliminary-density")
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            record["preliminary_density"] = prepare_seed(
                owner if precision_policy is None else owner._owner,
                geometries[0],
                args.preliminary_provider,
            )
            cp.cuda.Stream.null.synchronize()
            seed_seconds = perf_counter() - started
            record["target_prepare_seconds"] = prepare_seconds
            record["preliminary_wrapper_seconds"] = seed_seconds
            prepare_seconds += seed_seconds
            for geometry_index in range(2):
                baseline = baselines.get(geometry_index)
                coords = (
                    None
                    if geometry_index == 0
                    else [np.asarray([xyz for _, xyz in geometries[1]])]
                )
                owner.set_warm_start_updates(True)
                for repeat in range(args.repeats + 1):
                    phase = (
                        ("cold" if geometry_index == 0 else "moved")
                        if repeat == 0
                        else ("warm" if geometry_index == 0 else "moved-warm")
                    )
                    save(f"{phase}/{max(0, repeat - 1)}")
                    cp.cuda.Stream.null.synchronize()
                    started = perf_counter()
                    force_work = None
                    batch_result = owner.execute(
                        coords, strict=False, properties=("energy", "forces")
                    )
                    item = batch_result.items[0]
                    cp.cuda.Stream.null.synchronize()
                    seconds = perf_counter() - started
                    prepare = prepare_seconds if phase == "cold" else 0
                    row = native_endpoint_record(
                        item,
                        phase=phase,
                        geometry_index=geometry_index,
                        repeat=repeat,
                        seconds=seconds,
                        prepare=prepare,
                        owner=owner if precision_policy is None else owner._owner,
                        force_work=force_work,
                        has_xc=benchmark.method != "hf",
                    )
                    if precision_policy is not None:
                        # Never dereference a suspended candidate or relabel its
                        # counters as work performed by a returned FP64 auditor.
                        row.update(
                            precision=item.precision,
                            precision_validation=item.precision_validation.to_dict()
                            if item.precision_validation is not None
                            else None,
                            batch_precision_validation=batch_result.precision_validation,
                            native_diagnostics_scope="candidate"
                            if owner._owner is not None
                            else "unavailable-after-audit-suspension",
                        )
                    if benchmark.density_fitting:
                        diagnostic_owner = (
                            owner if precision_policy is None else owner._owner
                        )
                        row["density_fitting_metric"] = (
                            None
                            if diagnostic_owner is None
                            else [
                                diagnostic.to_dict()
                                for diagnostic in diagnostic_owner.last_density_fitting_metric_diagnostics()
                            ]
                        )
                    retain(row, baseline)
                    if repeat == 0:
                        owner.set_warm_start_updates(False)
            if original_force is not None:
                owner._public_dft_cuda_force = original_force
            record["native_force_components"] = (
                normalize_force_work(force_work) if force_work is not None else None
            )
        record["status"] = (
            "completed_unchecked"
            if any(row["gate"] is None for row in record["records"])
            else "measured"
        )
        save("complete")
    except NotImplementedError as error:
        record.update(status="unsupported", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    finally:
        if restore_force is not None:
            PreparedCompositeStationaryCudaGradient.execute = restore_force
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
