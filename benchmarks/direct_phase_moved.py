"""Host protocol and transparent API observer for the #2678 Direct campaign.

The plan is preparation, not a measurement receipt. API intervals are inclusive
host wall observations, never pure kernel time or additive component totals.
Device execution is held until a frozen full SDK and owned allocation exist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SCHEMA = "generativeqc.direct-phase-moved.plan.v1"
GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}
PHASES = (
    "scf_j",
    "scf_full_range_k",
    "scf_sr_k",
    "scf_lr_k",
    "scf_semilocal_xc",
    "scf_vv10",
    "force_one_electron",
    "force_jk",
    "force_xc_grid_partition",
    "force_vv10",
)


def geometry(value: Any) -> list:
    """Validate exact ordered atom identities and finite Bohr coordinates."""
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("missing ordered Bohr geometry")
    result = []
    for atom in value:
        if not isinstance(atom, (list, tuple)) or len(atom) != 2:
            raise ValueError("malformed atom")
        symbol, xyz = atom
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("missing element symbol")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError("coordinates must have three components")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in xyz):
            raise ValueError("coordinates must be finite real numbers")
        result.append([symbol, list(xyz)])
    return result


def identity(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def displace(atoms: Any, rms_bohr: float, *, seed: int = 2678) -> dict:
    """Seeded PCG64 displacements; RMS is the per-atom vector norm RMS.

    Both amplitudes use the same direction field, each relative to the cold
    geometry. Record the actual coordinates/deltas rather than relying on RNG
    version reproducibility. No reference density enters this operation.
    """
    cold = geometry(atoms)
    if (
        type(rms_bohr) not in (float, int)
        or not math.isfinite(rms_bohr)
        or rms_bohr <= 0
    ):
        raise ValueError("displacement RMS must be finite and positive")
    delta = np.random.Generator(np.random.PCG64(seed)).normal(size=(len(cold), 3))
    delta *= rms_bohr / float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    xyz = np.array([x[1] for x in cold], dtype=float) + delta
    moved = [[x[0], point.tolist()] for x, point in zip(cold, xyz, strict=True)]
    if not np.isfinite(xyz).all() or any(
        a[1] == b[1] for a, b in zip(cold, moved, strict=True)
    ):
        raise ValueError("every atom must have a finite representable displacement")
    actual = xyz - np.array([x[1] for x in cold], dtype=float)
    return {
        "geometry_bohr": moved,
        "geometry_sha256": identity(moved),
        "cold_geometry_sha256": identity(cold),
        "seed": seed,
        "rng": "PCG64",
        "rms_definition": "sqrt(mean_atom(sum_xyz(delta**2)))",
        "requested_rms_bohr": rms_bohr,
        "actual_rms_bohr": float(np.sqrt(np.mean(np.sum(actual * actual, axis=1)))),
        "displacements_bohr": actual.tolist(),
    }


@dataclass(frozen=True)
class Sample:
    engine: str
    population: str
    phase: str
    repeat: int
    geometry_sha256: str
    seed_policy: str


def plan(atoms: Any, *, method: str = "pbe0", seed: int = 2678) -> dict:
    """Declare one complete cohort without executing/importing either engine."""
    cold = geometry(atoms)
    if method not in ("pbe0", "wb97m-v") or len(cold) not in (48, 96):
        raise ValueError("campaign requires PBE0/WB97M-V and 48/96 atoms")
    moved = [displace(cold, rms, seed=seed) for rms in (0.01, 0.05)]
    rows = []
    for engine in ("standard", "gpu4pyscf"):
        for population in ("clean", "instrumented"):
            for repeat in range(4):
                rows.append(
                    Sample(
                        engine,
                        population,
                        "cold" if repeat == 0 else "warm",
                        max(0, repeat - 1),
                        identity(cold),
                        "stock_guess" if repeat == 0 else "immutable_own_cold_density",
                    )
                )
        # Each displacement owner needs its own cold setup outside the moved
        # timer; both amplitudes start from cold, never from the other step.
        for step in moved:
            for repeat in range(3):
                rows.append(
                    Sample(
                        engine,
                        "moved_clean",
                        "moved",
                        repeat,
                        step["geometry_sha256"],
                        "immutable_own_cold_density",
                    )
                )
    return {
        "schema": SCHEMA,
        "status": "PREPARED",
        "scientific_status": "INCOMPLETE",
        "issue": 2678,
        "method": method,
        "atoms": len(cold),
        "mode": "direct",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "geometry_bohr": cold,
        "geometry_sha256": identity(cold),
        "moved": moved,
        "samples": [asdict(row) for row in rows],
        "common_observable_gates": GATES,
        "grid_stability_gate_fraction": 0.25,
        "oracle_geometries": [
            identity(cold),
            *(row["geometry_sha256"] for row in moved),
        ],
        "oracle_contract": {
            "producer": "independent PySCF/GPU4PySCF; untimed fresh direct solves",
            "xc_levels": [4, 5],
            "cpu_level": 5,
            "energy_tolerance": 1e-11,
            "orbital_gradient_tolerance": 1e-7,
            "screening_tolerance": 1e-14,
            "full_density_fock": True,
            "force_grid_response": True,
            "nonlocal_stability": "separate VV10 levels for WB97M-V",
            "failed_stability": "retain failure and measure denser adjacent grids",
        },
        "prerequisites": {
            "allocation": "owned real finite Slurm allocation; preserve assigned visibility",
            "gpu": "same physical GPU UUID for both engines and every population",
            "sdk": "builder handoff: source/Python/native/DSO/AOT/modes/hash manifest",
            "observer": "clean/instrumented E+F and semantic work parity before timing claims",
            "density": "freeze each owner's cold density; disallow warm fallback",
            "publication": "lossless raw receipts, failures, manifest and source/binary identities",
        },
        "remaining_issue_scope": [
            f"direct/{other}/{size}"
            for other in ("pbe0", "wb97m-v")
            for size in (48, 96)
            if (other, size) != (method, len(cold))
        ],
    }


class ApiObserver:
    """Observe one process's owned instances, without sync or operand copies.

    Nested inclusive intervals retain parent IDs. No subtraction or aggregation
    into scientific phases is automatic. Opaque phases remain UNAVAILABLE.
    Observer bookkeeping errors must not mask scientific return/exception values.
    """

    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        self.clock = clock
        self.events: list[dict] = []
        self.errors: list[str] = []
        self._active: list[int] = []
        self._next = 0

    def wrap(self, function: Callable, label: str) -> Callable:
        @wraps(function)
        def observed(*args: Any, **kwargs: Any) -> Any:
            serial = self._next
            self._next += 1
            parent = self._active[-1] if self._active else None
            started = None
            try:
                started = self.clock()
            except Exception as error:  # noqa: BLE001 - observation cannot replace scientific behavior
                self.errors.append(type(error).__name__)
            self._active.append(serial)
            status = "exception"
            try:
                result = function(*args, **kwargs)
                status = "returned"
                return result
            finally:
                self._active.pop()
                try:
                    end = self.clock()
                    elapsed = None if started is None else end - started
                    if elapsed is not None and (
                        not math.isfinite(elapsed) or elapsed < 0
                    ):
                        raise ValueError("invalid observer clock")
                    self.events.append(
                        {
                            "id": serial,
                            "parent": parent,
                            "label": label,
                            "status": status,
                            "seconds": elapsed,
                            "boundary": "inclusive_api_host_wall",
                            "pure_kernel": False,
                            "additive": False,
                        }
                    )
                except Exception as error:  # noqa: BLE001 - preserve the original result/exception
                    self.errors.append(type(error).__name__)

        return observed

    @contextmanager
    def installed(self, owner: Any, labels: dict[str, str]) -> Iterator[None]:
        """Restore original instance attributes/descriptors even after failure."""
        undo = []
        try:
            for name, label in labels.items():
                original = getattr(owner, name)
                if not callable(original):
                    raise TypeError(f"{name} must be callable")
                present = name in vars(owner)
                value = vars(owner).get(name)
                setattr(owner, name, self.wrap(original, label))
                undo.append((name, present, value))
            yield
        finally:
            for name, present, value in reversed(undo):
                if present:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)

    def receipt(self) -> dict:
        return {
            "events": self.events,
            "observer_errors": self.errors,
            "phases": {
                phase: {"status": "UNAVAILABLE", "seconds": None} for phase in PHASES
            },
            "aggregation": "none; inclusive API walls may overlap and are not kernel durations",
        }


def endpoint(sample: dict, natoms: int) -> tuple[float, np.ndarray]:
    """Reject absent, unconverged or broadcastable-but-incomplete endpoints."""
    energy = sample.get("energy")
    forces = np.asarray(sample.get("forces"), dtype=float)
    if (
        sample.get("converged") is not True
        or type(energy) not in (float, int)
        or not math.isfinite(energy)
        or forces.shape != (natoms, 3)
        or not np.isfinite(forces).all()
    ):
        raise ValueError("incomplete/nonfinite E+F endpoint")
    return energy, forces


def observable_errors(
    sample: dict, reference: dict, natoms: int, *, fraction: float = 1
) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("gate fraction must be in (0, 1]")
    energy, forces = endpoint(sample, natoms)
    target, reference_forces = endpoint(reference, natoms)
    delta = forces - reference_forces
    errors = {
        "energy_abs_per_atom_hartree": abs(energy - target) / natoms,
        "force_max_abs_hartree_per_bohr": float(np.max(np.abs(delta))),
        "force_rms_hartree_per_bohr": float(np.sqrt(np.mean(delta * delta))),
    }
    return {
        **errors,
        "passed": all(v <= fraction * GATES[k] for k, v in errors.items()),
    }


def observer_parity(clean: dict, instrumented: dict, natoms: int) -> dict:
    """Require exact returned E+F and declared semantic work, independently of timing.

    This preflight does not qualify scientific accuracy; both populations still
    require independent references. Missing work cannot establish neutrality.
    """
    for key in (
        "geometry_sha256",
        "engine",
        "sdk_identity",
        "gpu_uuid",
        "seed_identity",
    ):
        if not clean.get(key) or clean[key] != instrumented.get(key):
            raise ValueError(f"observer parity identity differs: {key}")
    count = (
        "effective_potential_builds"
        if clean["engine"] == "gpu4pyscf"
        else "fock_builds"
    )
    required = ("iterations", count, "warm_start_used", "warm_start_fallback")
    for sample in (clean, instrumented):
        work = sample.get("work", {})
        if any(work.get(key) is None for key in required):
            raise ValueError("missing observer semantic work")
    a, af = endpoint(clean, natoms)
    b, bf = endpoint(instrumented, natoms)
    return {
        "passed": a == b
        and np.array_equal(af, bf)
        and clean["work"] == instrumented["work"],
        "scientific_acceptance": "INCOMPLETE",
    }


def _admission_keys(value: Any, keys: tuple[str, ...], label: str) -> dict:
    if not isinstance(value, dict) or any(key not in value for key in keys):
        raise ValueError(f"{label}: missing required fields or malformed object")
    return value


def _admission_finite(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _admission_finite(child)
    elif isinstance(value, list):
        for child in value:
            _admission_finite(child)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite receipt value")
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("receipt must contain JSON data only")


def _admission_equal(first: Any, second: Any) -> bool:
    """Do not let Python's True == 1 conceal malformed counter/settings data."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _admission_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _admission_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def _admission_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("required counter must be a reported nonnegative integer")
    return value


def _admission_physical(row: dict) -> None:
    _admission_keys(
        row,
        ("components", "density_change_max", "physical_residual_max", "electrons"),
        "physical diagnostic",
    )
    components = _admission_keys(
        row["components"],
        ("nuclear", "one_electron", "hartree", "xc"),
        "energy components",
    )
    if set(components) != {"nuclear", "one_electron", "hartree", "xc"} or any(
        type(value) not in (int, float) for value in components.values()
    ):
        raise ValueError("missing physical energy diagnostics")
    if any(
        type(row[key]) not in (int, float) or row[key] < 0
        for key in ("density_change_max", "physical_residual_max")
    ):
        raise ValueError("invalid physical convergence diagnostics")
    if (
        not isinstance(row["electrons"], list)
        or len(row["electrons"]) != 2
        or any(
            type(value) not in (int, float) or value < 0 for value in row["electrons"]
        )
    ):
        raise ValueError("invalid physical electron diagnostics")


def _admission_endpoint(sample: dict) -> None:
    _admission_keys(sample, ("energy", "forces", "converged"), "complete E+F")
    forces = sample["forces"]
    if not isinstance(forces, list) or any(
        not isinstance(row, list)
        or any(type(value) not in (int, float) for value in row)
        for row in forces
    ):
        raise ValueError(
            "forces must be numerical JSON arrays, not coercible strings/booleans"
        )
    endpoint(sample, 48)


def _observer_settings(settings: dict, engine: str) -> None:
    """Validate values, not just identical key absence or identical nulls."""
    for key in ("energy_tolerance", "screening_tolerance"):
        if type(settings[key]) not in (int, float) or settings[key] <= 0:
            raise ValueError("required tolerance is unavailable or invalid")
    for key in ("max_iterations", "diis_history"):
        if _admission_count(settings[key]) == 0:
            raise ValueError("invalid solver control")
    if (
        not isinstance(settings["initial_guess"], str)
        or not settings["initial_guess"]
        or settings["force_grid_response"] is not True
    ):
        raise ValueError("required guess/force-response setting is unavailable")
    grid = settings["xc_grid"]
    if engine == "standard":
        if (
            type(settings["density_tolerance"]) not in (int, float)
            or settings["density_tolerance"] <= 0
            or _admission_count(settings["xc_grid_points"]) == 0
            or settings["nonlocal_grid"] is not None
            or settings["nonlocal_grid_inherits_xc"] is not True
            or settings["nonlocal_density_policy"] is not None
        ):
            raise ValueError("invalid bounded PBE0 settings")
        _admission_keys(
            grid,
            ("radial_points", "angular_polar", "angular_azimuth", "version", "units"),
            "native grid",
        )
        for key in ("radial_points", "angular_polar", "angular_azimuth", "version"):
            _admission_count(grid[key])
        if grid["radial_points"] == 0 or grid["units"] != "Bohr":
            raise ValueError("missing native quadrature/units")
    else:
        value = settings["orbital_gradient_tolerance"]
        resolved = settings["orbital_gradient_tolerance_resolved"]
        if (
            (value is not None and (type(value) not in (int, float) or value <= 0))
            or type(resolved) not in (int, float)
            or resolved <= 0
            or type(settings["direct_scf"]) is not bool
            or type(settings["small_rho_cutoff"]) not in (int, float)
            or settings["small_rho_cutoff"] < 0
        ):
            raise ValueError("invalid GPU4PySCF settings")
        _admission_keys(
            grid,
            ("level", "points", "radial", "angular_pruning", "radii_adjust", "becke"),
            "GPU4PySCF grid",
        )
        _admission_count(grid["level"])
        if _admission_count(grid["points"]) == 0 or any(
            not isinstance(grid[key], str) or not grid[key]
            for key in ("radial", "angular_pruning", "radii_adjust", "becke")
        ):
            raise ValueError("missing actual stock quadrature settings")


def _observer_semantic_view(sample: dict, engine: str) -> dict:
    """Validate documented available fields; unknown coverage is not a count.

    Schemas: KsDiagnostic/KsIteration, read_initial_guess_diagnostic and the
    versioned precision-work decoder. Full numerical histories stay in raw
    receipts; their floating-point values and preparation clocks are not work.
    """
    work = _admission_keys(
        sample["work"],
        (
            "iterations",
            "fock_builds",
            "warm_start_used",
            "warm_start_fallback",
        ),
        "work",
    )
    _admission_count(work["iterations"])
    if (
        type(work["warm_start_used"]) is not bool
        or type(work["warm_start_fallback"]) is not bool
    ):
        raise ValueError("missing warm-start branch identity")
    if engine == "gpu4pyscf":
        _admission_keys(
            work,
            (
                "effective_potential_builds",
                "effective_potential_build_semantics",
                "vv10_reuse",
                "vv10_applicability",
            ),
            "GPU4PySCF work",
        )
        _admission_count(work["effective_potential_builds"])
        if (
            work["fock_builds"] is not None
            or not work["effective_potential_build_semantics"]
        ):
            raise ValueError(
                "GPU4PySCF invocation counts cannot masquerade as native Fock counts"
            )
        if (
            work["vv10_reuse"] is not None
            or work["vv10_applicability"] != "not applicable to PBE0"
        ):
            raise ValueError("unexpected nonlocal scope")
        if set(work) != {
            "iterations",
            "fock_builds",
            "warm_start_used",
            "warm_start_fallback",
            "effective_potential_builds",
            "effective_potential_build_semantics",
            "vv10_reuse",
            "vv10_applicability",
        }:
            raise ValueError("unclassified GPU4PySCF work field")
        return {"work": work}
    _admission_count(work["fock_builds"])
    _admission_keys(work, ("ks_diagnostic",), "native work")
    ks_keys = (
        "occupations",
        "grid_points",
        "tile_points",
        "ao_order",
        "scf_domain",
        "initial_density_used",
        "fock_builds",
        "history",
        "coulomb_exchange_work",
        "nonlocal_work",
        "nonlocal_history_scopes",
    )
    ks = _admission_keys(work["ks_diagnostic"], ks_keys, "KS diagnostic")
    if set(work) != {
        "iterations",
        "fock_builds",
        "warm_start_used",
        "warm_start_fallback",
        "ks_diagnostic",
    }:
        raise ValueError("unclassified native work field")
    if set(ks) != set(ks_keys) | {
        "components",
        "electrons",
        "density_change_max",
        "physical_residual_max",
    }:
        raise ValueError("unknown KS diagnostic schema field")
    _admission_physical(ks)
    for key in ("grid_points", "tile_points", "ao_order", "fock_builds"):
        _admission_count(ks[key])
    if (
        ks["grid_points"] == 0
        or ks["tile_points"] == 0
        or not isinstance(ks["scf_domain"], str)
        or not ks["scf_domain"]
        or type(ks["initial_density_used"]) is not bool
        or not isinstance(ks["occupations"], list)
        or len(ks["occupations"]) != 2
    ):
        raise ValueError("missing KS domain/occupation/grid identity")
    for count in ks["occupations"]:
        _admission_count(count)
    if ks["nonlocal_work"] is not None or ks["nonlocal_history_scopes"] is not None:
        raise ValueError("nonlocal work is outside the PBE0 cohort")
    census = ks["coulomb_exchange_work"]
    if census is not None:
        _admission_keys(
            census,
            ("complete", "coulomb_applications", "exchange_applications"),
            "operator census",
        )
        if type(census["complete"]) is not bool:
            raise ValueError("invalid coverage flag")
        for key in ("coulomb_applications", "exchange_applications"):
            if census["complete"]:
                _admission_count(census[key])
            elif census[key] is not None:
                raise ValueError(
                    "incomplete ABI census must remain null, never zero-filled"
                )
    history = ks["history"]
    if not isinstance(history, list) or len(history) != work["iterations"]:
        raise ValueError("missing or inconsistent complete iteration history")
    branches = []
    for row in history:
        _admission_keys(
            row,
            (
                "iteration",
                "occupation_stabilized",
                "components",
                "energy_change",
                "density_change_max",
                "physical_residual_max",
                "electrons",
            ),
            "history",
        )
        _admission_count(row["iteration"])
        _admission_physical(row)
        if set(row) - {
            "iteration",
            "occupation_stabilized",
            "components",
            "energy_change",
            "density_change_max",
            "physical_residual_max",
            "electrons",
            "operator_scope",
        }:
            raise ValueError("unknown iteration schema field")
        if row["energy_change"] is not None and type(row["energy_change"]) not in (
            int,
            float,
        ):
            raise ValueError("invalid optional physical energy-change baseline")
        if type(row["occupation_stabilized"]) is not bool:
            raise ValueError("invalid iteration branch")
        branch = {key: row[key] for key in ("iteration", "occupation_stabilized")}
        if (
            "operator_scope" in row
        ):  # Explicitly conditional in KsDiagnostic.to_payload.
            if not isinstance(row["operator_scope"], str) or not row["operator_scope"]:
                raise ValueError("invalid optional operator scope")
            branch["operator_scope"] = row["operator_scope"]
        branches.append(branch)
    guess_keys = (
        "kind",
        "outcome",
        "work_counters_complete",
        "preliminary_iterations",
        "preliminary_fock_builds",
        "target_attempts",
        "discarded_target_iterations",
        "discarded_target_fock_builds",
        "preparation_numeric_capacity",
        "preparation_seconds",
    )
    _admission_keys(sample, ("initial_guess_diagnostic", "precision"), "native sample")
    guess = _admission_keys(
        sample["initial_guess_diagnostic"], guess_keys, "initial guess"
    )
    if (
        guess["work_counters_complete"] is not True
        or guess["kind"] not in ("hf", "lda", "minao", "huckel")
        or guess["outcome"]
        not in (
            "disabled",
            "existing_density",
            "used",
            "preparation_failed",
            "budget_skipped",
            "target_retried",
        )
    ):
        raise ValueError("required initial-guess work/branch coverage is unavailable")
    for key in guess_keys[3:-1]:
        _admission_count(guess[key])
    if (
        type(guess["preparation_seconds"]) not in (int, float)
        or guess["preparation_seconds"] < 0
    ):
        raise ValueError("invalid reported preparation clock")
    p = _admission_keys(
        sample["precision"],
        (
            "policy_version",
            "detail_version",
            "requested_mode",
            "effective_bits",
            "complete",
            "operator_inventory_complete",
            "operator_work_counters_valid",
            "native_provenance",
            "returned_state_identity",
            "scf_fock_timeline",
            "operators",
            "conversion_count",
            "fallback_count",
        ),
        "precision-work",
    )
    if (
        p["policy_version"] != 1
        or p["detail_version"] != 1
        or p["requested_mode"] != "fp64"
        or p["effective_bits"] != 64
        or any(
            p[key] is not True
            for key in (
                "complete",
                "operator_inventory_complete",
                "operator_work_counters_valid",
            )
        )
    ):
        raise ValueError("unsupported or incomplete precision-work schema")
    for key in ("policy_version", "detail_version", "effective_bits"):
        _admission_count(p[key])
    provenance_keys = (
        "policy_version",
        "requested_mode",
        "effective_bits",
        "mixed_precision_fock_threshold",
        "strict_refinement_applied",
        "mixed_precision_reserved_error",
        "refinement_iterations",
        "mixed_stage_fock_builds",
        "strict_stage_fock_builds",
        "post_scf_fock_builds",
        "execution_retries",
        "mixed_admission_census",
        "final_residual_audits",
        "skipped_final_fock_builds",
        "operator_work_counters_valid",
    )
    provenance = _admission_keys(
        p["native_provenance"], provenance_keys, "native precision provenance"
    )
    _admission_keys(p, provenance_keys, "aggregate precision provenance")
    if not all(_admission_equal(p[key], provenance[key]) for key in provenance_keys):
        raise ValueError("precision detail/aggregate provenance differs")
    if type(provenance["strict_refinement_applied"]) is not bool or any(
        type(provenance[key]) not in (int, float) or provenance[key] < 0
        for key in ("mixed_precision_fock_threshold", "mixed_precision_reserved_error")
    ):
        raise ValueError("missing precision branch/control values")
    for key in provenance_keys[6:-1]:
        _admission_count(provenance[key])
    _admission_count(p["conversion_count"])
    _admission_count(p["fallback_count"])
    if not p["returned_state_identity"] or not isinstance(p["scf_fock_timeline"], list):
        raise ValueError("missing precision call/state identity")
    bound = set()
    for row in p["scf_fock_timeline"]:
        _admission_keys(
            row,
            ("kind", "sequence", "iteration", "state", "phase", "count"),
            "precision event",
        )
        for key in ("sequence", "iteration", "count"):
            _admission_count(row[key])
        if (
            not row["state"]
            or not row["phase"]
            or row["kind"]
            not in (
                "strict_fock",
                "final_audit",
                "post_scf_fock",
                "retry",
                "fallback",
                "conversion",
            )
        ):
            raise ValueError("unknown precision event/branch")
        if row["state"] == p["returned_state_identity"] and row["count"] > 0:
            bound.add(row["kind"])
    if not {"strict_fock", "final_audit"} <= bound:
        raise ValueError("physical work/audit is not bound to returned state")
    if not isinstance(p["operators"], list) or not p["operators"]:
        raise ValueError("missing reported operator inventory")
    names = set()
    for operator in p["operators"]:
        _admission_keys(
            operator,
            (
                "name",
                "storage",
                "compute",
                "accumulation",
                "reduction",
                "arithmetic_mode",
                "count",
            ),
            "precision operator",
        )
        _admission_count(operator["count"])
        name = operator["name"]
        if (
            not isinstance(name, str)
            or name
            not in {
                "coulomb_j",
                "exchange_k",
                "xc",
                "fock_assembly",
                "physical_residual",
                "eigensolver",
                "density_build",
                "diis",
                "matrix_product",
                "diagnostics",
                "occupation_stabilization",
                "coulomb_recurrence",
                "exchange_recurrence",
                "nonlocal_correlation",
            }
            or name in names
            or operator["arithmetic_mode"] != "strict"
            or any(
                operator[key] != "fp64"
                for key in ("storage", "compute", "accumulation", "reduction")
            )
        ):
            raise ValueError("invalid/unknown operator identity or precision")
        names.add(name)
    if (
        not {"coulomb_j", "exchange_k", "xc", "fock_assembly", "physical_residual"}
        <= names
    ):
        raise ValueError("required hybrid operator inventory is missing")
    return {
        "work": {
            key: work[key]
            for key in (
                "iterations",
                "fock_builds",
                "warm_start_used",
                "warm_start_fallback",
            )
        },
        "ks": {key: ks[key] for key in ks_keys if key != "history"},
        "history_branches": branches,
        "initial_guess": {
            key: guess[key] for key in guess_keys if key != "preparation_seconds"
        },
        "precision_work": p,
    }


def admit_observer_pair(
    clean: dict, instrumented: dict, reference: dict, *, expected: dict
) -> dict:
    """Read-only, fail-closed admission of public-observable observer stability.

    The caller supplies independently authenticated frozen provenance/settings;
    this function is not a source audit or a proof of unobserved primitive work.
    The old bitwise predicate remains unchanged and is reported separately.
    Quarter limits reuse the campaign's pre-existing stability margin, not a
    fitted rtol/atol. Original independent E/F gates are applied to every sample.
    """
    context = (
        "engine",
        "method",
        "mode",
        "basis",
        "basis_representation",
        "charge",
        "spin",
        "source_revision",
        "driver_sha256",
        "sdk_identity",
        "native_source_identity",
        "preflight_sha256",
    )
    _admission_keys(
        expected, (*context, "gpu", "geometry_bohr", "settings"), "expected contract"
    )
    _admission_finite(expected)
    if (
        expected["engine"] not in ("standard", "gpu4pyscf")
        or expected["method"] != "pbe0"
        or expected["mode"] != "direct"
        or expected["basis"] != "def2-svp"
        or expected["basis_representation"] != "spherical"
        or type(expected["charge"]) is not int
        or type(expected["spin"]) is not int
        or expected["charge"] != 0
        or expected["spin"] != 0
    ):
        raise ValueError("unsupported observer cohort")
    for key in (
        "source_revision",
        "driver_sha256",
        "sdk_identity",
        "native_source_identity",
        "preflight_sha256",
    ):
        if not isinstance(expected[key], str) or not re.fullmatch(
            r"[0-9a-f]{40}" if key == "source_revision" else r"[0-9a-f]{64}",
            expected[key],
        ):
            raise ValueError("missing authenticated source/SDK/driver identity")
    gpu_keys = (
        "uuid",
        "pci_bus_id",
        "name",
        "memory_mib",
        "driver_version",
        "compute_capability",
        "assigned_visibility",
    )
    gpu = _admission_keys(expected["gpu"], gpu_keys, "physical GPU")
    if (
        any(
            not isinstance(gpu[key], str) or not gpu[key]
            for key in (
                "uuid",
                "pci_bus_id",
                "name",
                "driver_version",
                "assigned_visibility",
            )
        )
        or not gpu["uuid"].startswith("GPU-")
        or not _admission_equal(gpu["compute_capability"], [12, 0])
        or type(gpu["memory_mib"]) is not int
        or gpu["memory_mib"] <= 0
    ):
        raise ValueError("missing physical GPU/visibility identity")
    atoms = geometry(expected["geometry_bohr"])
    if len(atoms) != 48:
        raise ValueError("observer cohort requires 48 atoms")
    settings_keys = (
        "energy_tolerance",
        "screening_tolerance",
        "max_iterations",
        "diis_history",
        "initial_guess",
        "xc_grid",
        "force_grid_response",
    )
    settings_keys += (
        (
            "density_tolerance",
            "xc_grid_points",
            "nonlocal_grid",
            "nonlocal_grid_inherits_xc",
            "nonlocal_density_policy",
        )
        if expected["engine"] == "standard"
        else (
            "orbital_gradient_tolerance",
            "orbital_gradient_tolerance_resolved",
            "direct_scf",
            "small_rho_cutoff",
        )
    )
    _admission_keys(expected["settings"], settings_keys, "frozen settings")
    if (
        not isinstance(expected["settings"]["xc_grid"], dict)
        or not expected["settings"]["xc_grid"]
    ):
        raise ValueError("missing effective grid settings")
    _observer_settings(expected["settings"], expected["engine"])
    _admission_keys(
        reference,
        (
            "schema",
            "status",
            "method",
            "mode",
            "basis",
            "basis_representation",
            "charge",
            "spin",
            "geometry_bohr",
            "geometry_sha256",
            "producer",
            "backend",
            "samples",
            "source_revision",
            "driver_sha256",
            "reference_helper_sha256",
            "allocation",
            "package_versions",
        ),
        "independent reference",
    )
    _admission_finite(reference)
    if (
        reference["schema"] != "generativeqc.direct-phase-moved.oracle.v1"
        or reference["status"] != "complete"
        or reference["producer"] != "pyscf"
        or reference["backend"] != "cpu"
        or not _admission_equal(geometry(reference["geometry_bohr"]), atoms)
        or reference["geometry_sha256"] != identity(atoms)
        or not reference["allocation"]
        or not isinstance(reference["samples"], list)
        or not reference["samples"]
        or any(
            not _admission_equal(reference[key], expected[key])
            for key in (
                "method",
                "mode",
                "basis",
                "basis_representation",
                "charge",
                "spin",
            )
        )
    ):
        raise ValueError("missing or incompatible independent CPU reference")
    target = reference["samples"][-1]
    for key in ("source_revision", "driver_sha256", "reference_helper_sha256"):
        if not isinstance(reference[key], str) or not re.fullmatch(
            r"[0-9a-f]{40}" if key == "source_revision" else r"[0-9a-f]{64}",
            reference[key],
        ):
            raise ValueError("missing independent oracle source/driver identity")
    _admission_keys(
        reference["package_versions"], ("pyscf", "numpy", "libxc"), "oracle packages"
    )
    if not all(
        isinstance(value, str) and value
        for value in reference["package_versions"].values()
    ):
        raise ValueError("missing oracle package identity")
    _admission_keys(target, ("controls", "level", "xc_grid"), "oracle sample")
    if (
        not _admission_equal(
            target["controls"],
            {
                "energy": 1e-11,
                "orbital_gradient": 1e-7,
                "screening": 1e-14,
                "full_density_fock": True,
                "force_grid_response": True,
            },
        )
        or type(target["level"]) is not int
        or target["level"] != 5
    ):
        raise ValueError("independent reference controls/finest grid differ")
    _admission_keys(target["xc_grid"], ("level", "points"), "actual oracle grid")
    if (
        target["xc_grid"]["level"] != 5
        or _admission_count(target["xc_grid"]["points"]) == 0
    ):
        raise ValueError("missing actual dense reference grid")
    _admission_endpoint(target)
    for record, population in ((clean, "clean"), (instrumented, "instrumented")):
        _admission_keys(
            record,
            (
                *context,
                "schema",
                "status",
                "population",
                "gpu",
                "geometry_bohr",
                "geometry_sha256",
                "cold_geometry_bohr",
                "timing_role",
                "samples",
            ),
            "population",
        )
        _admission_finite(record)
        if (
            record["schema"] != "generativeqc.direct-phase-moved.gpu.v1"
            or record["status"] != "complete"
            or record["population"] != population
            or not _admission_equal(record["gpu"], gpu)
            or any(not _admission_equal(record[key], expected[key]) for key in context)
            or not _admission_equal(geometry(record["geometry_bohr"]), atoms)
            or not _admission_equal(geometry(record["cold_geometry_bohr"]), atoms)
            or record["geometry_sha256"] != identity(atoms)
        ):
            raise ValueError("population schema/provenance/geometry mismatch")
    pilot = clean["timing_role"] == instrumented["timing_role"] == "qualification_pilot"
    if not pilot and (
        clean["timing_role"] != "clean" or instrumented["timing_role"] != "instrumented"
    ):
        raise ValueError("unknown population timing role")
    count = 1 if pilot else 4
    if any(
        not isinstance(record["samples"], list) or len(record["samples"]) != count
        for record in (clean, instrumented)
    ):
        raise ValueError("incomplete observer sample population")
    checks = []
    for index, (first, last) in enumerate(
        zip(clean["samples"], instrumented["samples"], strict=True)
    ):
        for sample in (first, last):
            _admission_keys(
                sample,
                (
                    "status",
                    "phase",
                    "repeat",
                    "seconds",
                    "engine",
                    "sdk_identity",
                    "gpu_uuid",
                    "seed_identity",
                    "seed_identity_kind",
                    "seed_policy",
                    "geometry_bohr",
                    "geometry_sha256",
                    "settings",
                    "work",
                ),
                "sample",
            )
            if (
                sample["status"] != "measured"
                or sample["phase"] != ("cold" if index == 0 else "warm")
                or type(sample["repeat"]) is not int
                or sample["repeat"] != max(0, index - 1)
                or sample["engine"] != expected["engine"]
                or sample["sdk_identity"] != expected["sdk_identity"]
                or sample["gpu_uuid"] != gpu["uuid"]
                or not _admission_equal(geometry(sample["geometry_bohr"]), atoms)
                or sample["geometry_sha256"] != identity(atoms)
                or sample["seed_identity"]
                != identity({"cold_geometry": atoms, "policy": "immutable own cold"})
                or sample["seed_identity_kind"]
                != "declared policy and cold geometry; no cross-engine AO density identity asserted"
                or sample["seed_policy"]
                != ("stock_guess" if index == 0 else "immutable_own_cold_density")
                or type(sample["seconds"]) not in (int, float)
                or sample["seconds"] <= 0
            ):
                raise ValueError("missing or inconsistent sample/seed identity")
            _admission_keys(sample["settings"], settings_keys, "sample settings")
            _observer_settings(sample["settings"], expected["engine"])
            _admission_endpoint(sample)
            _admission_keys(
                sample["work"],
                ("warm_start_used", "warm_start_fallback"),
                "warm branch",
            )
            if (
                sample["work"]["warm_start_used"] is not (index > 0)
                or sample["work"]["warm_start_fallback"] is not False
            ):
                raise ValueError("unexpected cold/warm/fallback branch")
        a = _observer_semantic_view(first, expected["engine"])
        b = _observer_semantic_view(last, expected["engine"])
        checks.append(
            {
                "settings_equal": _admission_equal(first["settings"], last["settings"])
                and _admission_equal(first["settings"], expected["settings"]),
                "available_semantics_equal": _admission_equal(a, b),
                "clean_independent_errors": observable_errors(first, target, 48),
                "instrumented_independent_errors": observable_errors(last, target, 48),
                "pair_quarter_errors": observable_errors(
                    first, last, 48, fraction=0.25
                ),
                "original_bitwise_diagnostic": observer_parity(first, last, 48),
            }
        )
    return {
        "schema": "generativeqc.direct-phase-moved.observer-admission.v1",
        "admitted": all(
            row["settings_equal"]
            and row["available_semantics_equal"]
            and all(
                row[key]["passed"]
                for key in (
                    "clean_independent_errors",
                    "instrumented_independent_errors",
                    "pair_quarter_errors",
                )
            )
            for row in checks
        ),
        "scope": "authenticated public-observable semantics + original independent science gates + pre-existing quarter stability budget; caller must also audit source pass-through",
        "samples": checks,
        "full_primitive_work_neutrality": "UNAVAILABLE",
        "actual_D0_byte_identity": "UNAVAILABLE",
        "overall_cohort_completion": "NOT_EVALUATED",
    }


def measure_population(
    factory: Callable[[], Any],
    atoms: Any,
    *,
    moved: Any = None,
    synchronize: Callable[[], None],
    retain: Callable[[dict], None],
    clock: Callable[[], float] = perf_counter,
    warm_repeats: int = 3,
) -> None:
    """Execute one owned four-call population, freezing only its cold seed.

    Adapters provide execute(coordinates), freeze_cold(), record(result), close().
    record() is untimed serialization; execute() must return complete host E+F.
    Cold includes constructor/preparation. Moved populations retain the initial
    cold call as setup and time three identical displaced calls from its frozen
    density. Every amplitude gets a fresh owner via a separate invocation.

    This controller supplies no resource authorization. Its caller must verify
    the actual finite allocation and frozen SDK before constructing an adapter.
    It retains failures before propagating the original scientific exception.
    """
    if type(warm_repeats) is not int or not 0 <= warm_repeats <= 3:
        raise ValueError(
            "bounded pilot/formal population requires zero to three warm repeats"
        )
    cold = geometry(atoms)
    target = cold if moved is None else geometry(moved)
    if [row[0] for row in target] != [row[0] for row in cold]:
        raise ValueError("moved geometry must preserve ordered atom identities")
    owner = None
    failure = None
    sample_index = 0
    try:
        synchronize()
        started = clock()
        owner = factory()
        for index in range(warm_repeats + 1):
            sample_index = index
            if index:
                synchronize()
                started = clock()
            coordinates = (
                None if not index or moved is None else [row[1] for row in target]
            )
            result = owner.execute(coordinates)
            synchronize()
            seconds = clock() - started
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid complete-endpoint duration")
            # Force export belongs to execute(), while record() must not run
            # additional scientific work or return mutable runtime dictionaries.
            row = copy.deepcopy(owner.record(result))
            endpoint(row, len(cold))
            phase = (
                ("cold_setup" if moved is not None else "cold")
                if not index
                else ("moved" if moved is not None else "warm")
            )
            row.update(
                status="measured",
                phase=phase,
                repeat=max(0, index - 1),
                seconds=seconds,
                geometry_bohr=cold if not index else target,
                geometry_sha256=identity(cold if not index else target),
                seed_policy="stock_guess"
                if not index
                else "immutable_own_cold_density",
            )
            if index:
                work = row.get("work", {})
                if (
                    work.get("warm_start_used") is not True
                    or work.get("warm_start_fallback") is not False
                ):
                    row["status"] = "failed"
                    row["error"] = "warm endpoint did not use immutable cold density"
                    retain(row)
                    raise RuntimeError(row["error"])
            retain(row)
            if not index:
                owner.freeze_cold()
    except BaseException as error:
        failure = error
        retain(
            {
                "status": "failed",
                "call_index": sample_index,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                retain(
                    {
                        "status": "cleanup_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if failure is None:
                    raise


class NativeEndpoint:
    """Standard Direct PBE0 owner using the README's public production calls.

    Construction/preparation belongs to the cold timer; scientific result
    extraction and binary hashing happen after that timer. No reference engine
    is imported, and changed coordinates use PreparedBatch's public replay API.
    Call only after allocation and complete source-matched SDK qualification.
    """

    def __init__(self, atoms: Any) -> None:
        from generativeqc import Calculator

        self.atoms = geometry(atoms)
        self.calculator = Calculator(
            method="pbe0-rks",
            device="cuda",
            basis="def2-svp",
            basis_representation="spherical",
        )
        try:
            self.owner = self.calculator.prepare_batch([self.atoms], warm_start=True)
        except BaseException:
            self.calculator.clear_cache()
            raise
        self._binary = None

    def execute(self, coordinates: Any) -> Any:
        return self.owner.execute(
            None if coordinates is None else [coordinates],
            strict=True,
            properties=("energy", "forces"),
        ).items[0]

    def freeze_cold(self) -> None:
        self.owner.set_warm_start_updates(False)

    def record(self, result: Any) -> dict:
        from benchmarks.readme_defaults import native_settings
        from benchmarks.readme_method_endpoints import binary_identity

        if result.executed_backend != "cuda":
            raise RuntimeError("native endpoint did not execute CUDA")
        if self._binary is None:
            self._binary = binary_identity(self.calculator)
        return {
            "engine": "standard",
            "energy": float(result.energy),
            "forces": np.asarray(result.forces).tolist()
            if result.forces is not None
            else None,
            "converged": bool(result.converged),
            "precision": result.precision,
            "initial_guess_diagnostic": result.initial_guess,
            "native_build": self._binary,
            "settings": native_settings(self.calculator, result),
            "work": {
                "iterations": result.iterations,
                "fock_builds": result.fock_builds,
                "warm_start_used": result.warm_start_used,
                "warm_start_fallback": result.warm_start_fallback,
                "ks_diagnostic": result.ks_diagnostic.to_payload()
                if result.ks_diagnostic
                else None,
            },
            "force_resources": copy.deepcopy(self.owner.resource_diagnostics),
        }

    def close(self) -> None:
        try:
            self.owner.close()
        finally:
            self.calculator.clear_cache()


class ReferenceEndpoint:
    """Stock GPU4PySCF Direct PBE0 with complete moving-grid forces.

    The immutable seed comes solely from this owner's cold solve. Geometry
    reset is inside the moved endpoint, only when coordinates actually change.
    get_veff invocations retain their own counter meaning; they are not renamed
    as native Fock builds or counted as primitive J/K work.
    """

    def __init__(self, atoms: Any, *, instrumented: bool = False) -> None:
        import cupy as cp
        from gpu4pyscf.dft import rks
        from pyscf import gto

        self.atoms = geometry(atoms)
        self.cp = cp
        self.gto = gto
        self.mol = gto.M(
            atom=self.atoms,
            unit="Bohr",
            basis="def2-svp",
            cart=False,
            charge=0,
            spin=0,
            verbose=0,
        )
        self.engine = rks.RKS(self.mol)
        self.engine.xc = "PBE0"
        self.current = [row[1] for row in self.atoms]
        self.seed = None
        self.builds = 0
        self.observer = ApiObserver() if instrumented else None
        original = self.engine.get_veff
        self._original_veff = original

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.builds += 1
            return original(*args, **kwargs)

        self.engine.get_veff = counted

    def execute(self, coordinates: Any) -> dict:
        if coordinates is not None and not np.array_equal(self.current, coordinates):
            self.mol = self.gto.M(
                atom=[
                    [row[0], xyz]
                    for row, xyz in zip(self.atoms, coordinates, strict=True)
                ],
                unit="Bohr",
                basis="def2-svp",
                cart=False,
                charge=0,
                spin=0,
                verbose=0,
            )
            self.engine.reset(self.mol)
            self.current = copy.deepcopy(coordinates)
        self.builds = 0
        offset = len(self.observer.events) if self.observer else 0
        if self.observer:
            with ExitStack() as stack:
                stack.enter_context(
                    self.observer.installed(
                        self.engine, {"get_jk": "scf.get_jk", "get_k": "scf.get_k"}
                    )
                )
                stack.enter_context(
                    self.observer.installed(
                        self.engine._numint, {"nr_rks": "scf.semilocal_xc"}
                    )
                )
                energy = float(
                    self.engine.kernel(
                        dm0=None if self.seed is None else self.seed.copy()
                    )
                )
        else:
            energy = float(
                self.engine.kernel(dm0=None if self.seed is None else self.seed.copy())
            )
        if not self.engine.converged:
            raise RuntimeError("GPU4PySCF SCF did not converge")
        gradient = self.engine.nuc_grad_method()
        gradient.grid_response = True
        if self.observer:
            from gpu4pyscf.grad import rks as gradient_rks

            with ExitStack() as stack:
                stack.enter_context(
                    self.observer.installed(
                        gradient,
                        {
                            "_hcore_energy": "force.one_electron_and_pulay",
                            "jk_energy_per_atom": "force.jk",
                        },
                    )
                )
                stack.enter_context(
                    self.observer.installed(
                        gradient_rks,
                        {
                            "get_exc_full_response": "force.xc_grid_partition",
                        },
                    )
                )
                forces = -self.cp.asnumpy(gradient.kernel())
        else:
            forces = -self.cp.asnumpy(gradient.kernel())
        self.gradient = gradient
        return {
            "energy": energy,
            "forces": forces,
            "converged": True,
            "iterations": int(self.engine.cycles),
            "effective_potential_builds": self.builds,
            "events": copy.deepcopy(self.observer.events[offset:])
            if self.observer
            else None,
        }

    def freeze_cold(self) -> None:
        self.seed = self.engine.make_rdm1().copy()

    def record(self, result: dict) -> dict:
        from benchmarks.readme_defaults import grid_settings

        return {
            "engine": "gpu4pyscf",
            "energy": result["energy"],
            "forces": np.asarray(result["forces"]).tolist(),
            "converged": result["converged"],
            "settings": {
                "energy_tolerance": float(self.engine.conv_tol),
                "orbital_gradient_tolerance": self.engine.conv_tol_grad,
                "orbital_gradient_tolerance_resolved": float(self.engine.conv_tol_grad)
                if self.engine.conv_tol_grad is not None
                else math.sqrt(self.engine.conv_tol),
                "direct_scf": bool(self.engine.direct_scf),
                "screening_tolerance": float(self.engine.direct_scf_tol),
                "max_iterations": int(self.engine.max_cycle),
                "initial_guess": str(self.engine.init_guess),
                "diis_history": int(self.engine.diis_space),
                "small_rho_cutoff": float(self.engine.small_rho_cutoff),
                "xc_grid": grid_settings(self.engine.grids),
                "force_grid_response": bool(self.gradient.grid_response),
            },
            "work": {
                "iterations": result["iterations"],
                "fock_builds": None,
                "effective_potential_builds": result["effective_potential_builds"],
                "effective_potential_build_semantics": "executed get_veff invocations, including initialization/audits",
                "warm_start_used": self.seed is not None,
                "warm_start_fallback": False,
                "vv10_reuse": None,
                "vv10_applicability": "not applicable to PBE0",
            },
            "api_observations": result["events"],
            "observer_errors": list(self.observer.errors) if self.observer else None,
        }

    def close(self) -> None:
        if self.engine is not None:
            self.engine.get_veff = self._original_veff
        self._original_veff = None
        self.seed = None
        self.gradient = None
        self.engine = None
        self.mol = None


def qualify_geometry(samples: list[dict], gpu: dict, cpu: dict, contract: dict) -> dict:
    """Gate every sample against geometry-bound independent tighter E+F.

    Stored pass flags are ignored. Two adjacent GPU quadratures and an
    independent CPU PySCF endpoint must satisfy quarter gates first. This
    bounded first cohort handles PBE0; VV10 needs an additional independent
    nonlocal-grid stability population before WB97M-V can be admitted.
    """
    if contract.get("method") != "pbe0":
        raise ValueError("WB97M-V nonlocal-grid qualification is not yet implemented")
    expected = {
        key: contract[key]
        for key in ("method", "mode", "basis", "basis_representation", "charge", "spin")
    }
    if expected != {
        "method": "pbe0",
        "mode": "direct",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
    }:
        raise ValueError("scientific contract differs from declared cohort")
    atoms = geometry(contract["geometry_bohr"])
    natoms = len(atoms)
    if natoms != 48:
        raise ValueError("first qualified cohort requires 48 atoms")

    def check_context(record: dict) -> None:
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("scientific input identity differs")
        if geometry(record.get("geometry_bohr")) != atoms:
            raise ValueError("oracle/sample geometry differs from displaced target")

    for oracle, backend, producer in (
        (gpu, "cuda", "gpu4pyscf"),
        (cpu, "cpu", "pyscf"),
    ):
        check_context(oracle)
        if (
            oracle.get("status") != "complete"
            or oracle.get("backend") != backend
            or oracle.get("producer") != producer
            or not oracle.get("allocation")
            or not oracle.get("package_versions", {}).get(producer)
            or not re.fullmatch(r"[0-9a-f]{40}", oracle.get("source_revision", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", oracle.get("driver_sha256", ""))
        ):
            raise ValueError("incomplete independent oracle provenance")
        if oracle.get("controls") != {
            "energy_tolerance": 1e-11,
            "orbital_gradient_tolerance": 1e-7,
            "screening_tolerance": 1e-14,
            "full_density_fock": True,
            "force_grid_response": True,
        }:
            raise ValueError("unqualified oracle controls")
        for row in oracle.get("samples", []):
            endpoint(row, natoms)
            if type(row.get("level")) is not int or not 4 <= row["level"] <= 9:
                raise ValueError("unqualified oracle grid")
            if (row.get("xc_grid") or {}).get("level") != row["level"]:
                raise ValueError("actual oracle grid differs from declared level")
    gr = gpu.get("samples", [])
    cr = cpu.get("samples", [])
    if len(gr) != 2 or gr[1]["level"] != gr[0]["level"] + 1:
        raise ValueError("two adjacent tighter GPU grids required")
    if len(cr) not in (1, 2) or cr[-1]["level"] != gr[-1]["level"]:
        raise ValueError("independent CPU endpoint at finest grid required")
    if len(cr) == 2 and cr[0]["level"] != cr[1]["level"] - 1:
        raise ValueError("CPU stability pair requires adjacent grids")
    if not samples:
        raise ValueError("empty timing population")
    for sample in samples:
        check_context(sample)
    stability = observable_errors(gr[0], gr[1], natoms, fraction=0.25)
    agreement = observable_errors(cr[-1], gr[-1], natoms, fraction=0.25)
    cpu_stability = (
        observable_errors(cr[0], cr[1], natoms, fraction=0.25) if len(cr) == 2 else None
    )
    checks = [observable_errors(sample, gr[-1], natoms) for sample in samples]
    passed = (
        stability["passed"]
        and agreement["passed"]
        and (cpu_stability is None or cpu_stability["passed"])
        and all(row["passed"] for row in checks)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "geometry_sha256": identity(atoms),
        "grid_stability": stability,
        "cpu_gpu_agreement": agreement,
        "cpu_grid_stability": cpu_stability,
        "samples": checks,
        "oracle_receipt_sha256": {"gpu": identity(gpu), "cpu": identity(cpu)},
        "scope": "one geometry; does not establish complete campaign population or device provenance",
    }


def run_cpu_oracle(
    atoms: Any, output: Path, *, source_revision: str, levels: tuple[int, ...] = (4, 5)
) -> None:
    """Run untimed independent PySCF E+F in an already owned CPU allocation.

    Uses the audited README reference solve, including full-density Focks and
    moving-grid response. This entry point never constructs a native calculator
    or supplies a reference density to production. Slurm's finite time limit
    remains the outer computation timeout; each completed grid is checkpointed.
    """
    from benchmarks._retention import raw_output_path

    output = raw_output_path(output)
    atoms = geometry(atoms)
    if len(atoms) != 48 or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError(
            "first CPU oracle requires 48 atoms and a full source revision"
        )
    if levels not in ((5,), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)):
        raise ValueError(
            "CPU reference requires a finest grid or adjacent tighter pair"
        )
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal():
        raise RuntimeError(
            "independent CPU solve requires an owned finite Slurm allocation"
        )
    allocation = subprocess.run(
        ["/usr/bin/scontrol", "show", "job", job, "--oneliner"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(\S+)", allocation))
    if (
        fields.get("JobId") != job
        or fields.get("JobState") != "RUNNING"
        or fields.get("TimeLimit") in (None, "UNLIMITED", "Partition_Limit")
        or fields.get("UserId", "").split("(")[-1].rstrip(")") != str(os.getuid())
    ):
        raise RuntimeError(
            "live allocation does not establish ownership and finite runtime"
        )
    if output.exists():
        raise FileExistsError(output)
    import pyscf
    from pyscf.dft import libxc

    from benchmarks import readme_defaults_accuracy as reference

    expected = {
        "energy": 1e-11,
        "orbital_gradient": 1e-7,
        "screening": 1e-14,
        "full_density_fock": True,
        "force_grid_response": True,
    }
    record = {
        "schema": "generativeqc.direct-phase-moved.oracle.v1",
        "status": "running",
        "method": "pbe0",
        "mode": "direct",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "geometry_bohr": atoms,
        "geometry_sha256": identity(atoms),
        "backend": "cpu",
        "producer": "pyscf",
        "source_revision": source_revision,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_helper_sha256": hashlib.sha256(
            Path(reference.__file__).read_bytes()
        ).hexdigest(),
        "allocation": allocation,
        "python": sys.version,
        "executable": sys.executable,
        "package_versions": {
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "libxc": libxc.libxc_version(),
        },
        "thread_controls": {
            name: os.environ.get(name)
            for name in (
                "SLURM_CPUS_PER_TASK",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "controls": {
            "energy_tolerance": 1e-11,
            "orbital_gradient_tolerance": 1e-7,
            "screening_tolerance": 1e-14,
            "full_density_fock": True,
            "force_grid_response": True,
        },
        "samples": [],
        "scientific_status": "INCOMPLETE",
    }
    # Exclusive first publication prevents duplicate workers from sharing output.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def checkpoint() -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=output.name,
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary = Path(stream.name)
        temporary.replace(output)

    try:
        for level in levels:
            sample = reference.solve("pbe0", atoms, backend="cpu", level=level)
            endpoint(sample, len(atoms))
            if sample.get("controls") != expected or sample.get("level") != level:
                raise ValueError(
                    "executed reference controls differ from declared protocol"
                )
            record["samples"].append(sample)
            checkpoint()
        if len(record["samples"]) == 2:
            record["cpu_grid_stability"] = observable_errors(
                *record["samples"], len(atoms), fraction=0.25
            )
        record["status"] = "complete"
        # CPU computation completion does not establish GPU agreement, accuracy
        # of timed endpoints, or completion of the dual-engine campaign.
    except BaseException as error:
        record.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        checkpoint()


def gpu_device() -> dict:
    """Identify only the assigned logical device, using its physical PCI ID."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("cohort requires exactly one scheduler-visible GPU")
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    pci = device.pci_bus_id
    result = (
        subprocess.run(
            [
                "nvidia-smi",
                "-i",
                pci,
                "--query-gpu=uuid,pci.bus_id,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(result) != 1:
        raise RuntimeError("physical GPU identity is ambiguous")
    uid, bus, name, memory, driver = [value.strip() for value in result[0].split(",")]
    if (props["major"], props["minor"]) != (12, 0):
        raise RuntimeError("frozen SM120 SDK requires a real compatible device")
    return {
        "uuid": uid,
        "pci_bus_id": bus,
        "name": name,
        "memory_mib": int(memory),
        "driver_version": driver,
        "compute_capability": [props["major"], props["minor"]],
        "assigned_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_gpu_worker(
    atoms: Any,
    output: Path,
    *,
    preflight: Path,
    engine: str,
    kind: str,
    reference_path: Path,
    moved: Any = None,
    warm_repeats: int = 3,
    pilot: bool = False,
    cold_reference_path: Path | None = None,
) -> None:
    """One fresh-process population inside a separately verified GPU lease.

    Qualification and clean/instrumented receipts are separate files. Timed
    production samples use only their own cold density; references are read
    solely for post-endpoint numerical checks. No SDK/source/cache is modified.
    """
    from benchmarks._retention import raw_output_path

    atoms = geometry(atoms)
    target = atoms if moved is None else geometry(moved)
    if (
        len(atoms) != 48
        or engine not in ("standard", "gpu4pyscf")
        or kind not in ("oracle", "clean", "instrumented", "moved")
    ):
        raise ValueError("unsupported bounded GPU cohort")
    if (kind == "moved") != (moved is not None) or (
        kind == "oracle" and engine != "gpu4pyscf"
    ):
        raise ValueError("inconsistent GPU population")
    output = raw_output_path(output)
    if output.exists():
        raise FileExistsError(output)
    proof = json.loads(preflight.read_text(encoding="utf-8"))
    job = os.environ.get("SLURM_JOB_ID")
    if proof.get("status") != "verified" or not job or proof.get("job_id") != job:
        raise RuntimeError("this allocation has no verified SDK/device preflight")
    allocation = subprocess.run(
        ["/usr/bin/scontrol", "show", "job", job, "--oneliner"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    if "JobState=RUNNING" not in allocation or "gres/gpu" not in allocation:
        raise RuntimeError("live GPU allocation required")
    config_path = Path(proof["config_path"])
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != proof["config_sha256"]:
        raise RuntimeError("frozen SDK config changed")
    config = json.loads(config_bytes)
    if (
        config["production_source_revision"]
        != "ececb48cd56a00a56fbb6223dc53bc61dfe6e11d"
        or Path(os.environ.get("GENERATIVEQC_LIBRARY", "")).resolve()
        != Path(config["library"]).resolve()
        or hashlib.sha256(Path(config["library"]).read_bytes()).hexdigest()
        != config["library_sha256"]
    ):
        raise RuntimeError("production library/source differs from frozen SDK")
    device = gpu_device()
    if device["uuid"] != proof["gpu"]["uuid"]:
        raise RuntimeError("cohort physical GPU changed")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference.get("status") != "complete"
        or geometry(reference["geometry_bohr"]) != target
    ):
        raise ValueError("independent CPU reference does not match the exact target")
    reference_sample = reference["samples"][-1]
    endpoint(reference_sample, 48)
    cold_sample = reference_sample
    if kind == "moved":
        if cold_reference_path is None:
            raise ValueError(
                "moved population also requires independent cold-setup reference"
            )
        cold_reference = json.loads(cold_reference_path.read_text(encoding="utf-8"))
        if (
            cold_reference.get("status") != "complete"
            or geometry(cold_reference["geometry_bohr"]) != atoms
        ):
            raise ValueError("cold-setup reference identity differs")
        cold_sample = cold_reference["samples"][-1]
        endpoint(cold_sample, 48)
    import cupy as cp

    record = {
        "schema": "generativeqc.direct-phase-moved.gpu.v1",
        "status": "running",
        "scientific_status": "INCOMPLETE",
        "engine": engine,
        "population": kind,
        "timing_role": "qualification_pilot"
        if pilot
        else ("untimed_independent_reference" if kind == "oracle" else kind),
        "method": "pbe0",
        "mode": "direct",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "geometry_bohr": target,
        "cold_geometry_bohr": atoms,
        "geometry_sha256": identity(target),
        "source_revision": config["production_source_revision"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sdk_identity": proof["config_sha256"],
        "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
        "native_source_identity": config["native_source_identity"],
        "gpu": device,
        "allocation": allocation,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "samples": [],
        "phase_policy": "inclusive API host wall or explicitly CUDA-event intervals; never pure kernel time or additive totals",
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")

    def checkpoint() -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=output.name,
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(output)

    trace = output.with_suffix(".cuda.jsonl")
    for name in (
        "GENERATIVEQC_DF_TRACE",
        "GENERATIVEQC_DF_HOST_TRACE",
        "GENERATIVEQC_DF_PROGRESS_TRACE",
    ):
        if os.environ.get(name):
            raise RuntimeError("population must not inherit diagnostic tracing")
    if kind == "instrumented" and engine == "standard":
        if trace.exists():
            raise FileExistsError(trace)
        os.environ["GENERATIVEQC_DF_TRACE"] = str(trace)
    trace_count = 0

    def retain(row: dict) -> None:
        nonlocal trace_count
        if row.get("status") == "measured":
            if engine == "standard":
                build = row["native_build"]
                if (
                    build["library_sha256"] != config["library_sha256"]
                    or build["source_identity"] != config["native_source_identity"]
                ):
                    raise RuntimeError(
                        "actually loaded native library differs from qualified SDK"
                    )
            row.update(
                engine=engine,
                sdk_identity=proof["config_sha256"],
                gpu_uuid=device["uuid"],
                seed_identity=identity(
                    {"cold_geometry": atoms, "policy": "immutable own cold"}
                ),
                seed_identity_kind="declared policy and cold geometry; no cross-engine AO density identity asserted",
            )
            row["independent_cpu_errors"] = observable_errors(
                row,
                cold_sample if row["phase"] == "cold_setup" else reference_sample,
                48,
            )
            if kind == "instrumented" and engine == "standard":
                from benchmarks.df_component_ledger import read_trace

                records = read_trace(trace)
                row["cuda_trace_record_range"] = [trace_count, len(records)]
                row["cuda_event_observations"] = [
                    {
                        "operation": rec["operation"],
                        "execution": rec["execution"],
                        "region": region["name"],
                        "parent": region["parent"],
                        "host_ms": region["host_ms"],
                        "cuda_event_inclusive_ms": region["gpu_ms"],
                    }
                    for rec in records[trace_count:]
                    for region in rec["regions"]
                    if region["name"]
                    in (
                        "ks_jk",
                        "ks_xc",
                        "direct_j",
                        "direct_jk_force",
                        "one_electron_response",
                    )
                ]
                trace_count = len(records)
        record["samples"].append(row)
        checkpoint()
        if row.get("independent_cpu_errors", {}).get("passed") is False:
            raise RuntimeError(
                "complete endpoint failed independent CPU observable gates"
            )

    try:
        if kind == "oracle":
            from importlib.metadata import version

            import pyscf
            from pyscf.dft import libxc

            from benchmarks.readme_defaults_accuracy import solve

            record.update(
                producer="gpu4pyscf",
                backend="cuda",
                package_versions={
                    "pyscf": pyscf.__version__,
                    "gpu4pyscf": version("gpu4pyscf-cuda12x"),
                    "cupy": cp.__version__,
                    "numpy": np.__version__,
                    "pyscf_libxc": libxc.libxc_version(),
                },
            )
            for level in (4, 5):
                row = solve("pbe0", atoms, backend="cuda", level=level)
                endpoint(row, 48)
                record["samples"].append(row)
                checkpoint()
                if row.get("level") != level or row.get("controls") != {
                    "energy": 1e-11,
                    "orbital_gradient": 1e-7,
                    "screening": 1e-14,
                    "full_density_fock": True,
                    "force_grid_response": True,
                }:
                    raise ValueError("executed independent GPU oracle controls differ")
            record["controls"] = {
                "energy_tolerance": 1e-11,
                "orbital_gradient_tolerance": 1e-7,
                "screening_tolerance": 1e-14,
                "full_density_fock": True,
                "force_grid_response": True,
            }
            record["gpu_grid_stability"] = observable_errors(
                *record["samples"], 48, fraction=0.25
            )
            record["cpu_gpu_agreement"] = observable_errors(
                record["samples"][-1], reference_sample, 48, fraction=0.25
            )
            if (
                not record["gpu_grid_stability"]["passed"]
                or not record["cpu_gpu_agreement"]["passed"]
            ):
                raise RuntimeError("independent GPU grid/CPU quarter gates failed")
        else:
            factory = (
                (lambda: NativeEndpoint(atoms))
                if engine == "standard"
                else (
                    lambda: ReferenceEndpoint(
                        atoms, instrumented=kind == "instrumented"
                    )
                )
            )
            measure_population(
                factory,
                atoms,
                moved=moved,
                synchronize=cp.cuda.runtime.deviceSynchronize,
                retain=retain,
                warm_repeats=warm_repeats,
            )
        record["status"] = "complete"
    except BaseException as error:
        record.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        checkpoint()
        if kind == "instrumented" and engine == "standard":
            os.environ.pop("GENERATIVEQC_DF_TRACE", None)


def main(argv: list[str] | None = None) -> None:
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks._retention import raw_output_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        type=Path,
        required=True,
        help="JSON ordered [symbol, xyz] Bohr atoms",
    )
    parser.add_argument("--method", choices=("pbe0", "wb97m-v"), default="pbe0")
    parser.add_argument("--seed", type=int, default=2678)
    parser.add_argument("--cpu-oracle", action="store_true")
    parser.add_argument(
        "--gpu-kind", choices=("oracle", "clean", "instrumented", "moved")
    )
    parser.add_argument(
        "--engine", choices=("standard", "gpu4pyscf"), default="standard"
    )
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cold-reference", type=Path)
    parser.add_argument("--moved-geometry", type=Path)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args(argv)
    atoms = json.loads(args.geometry.read_text(encoding="utf-8"))
    if args.gpu_kind:
        if (
            args.cpu_oracle
            or args.method != "pbe0"
            or not args.preflight
            or not args.reference
        ):
            parser.error(
                "bounded GPU cohort requires PBE0, preflight and independent reference"
            )
        run_gpu_worker(
            atoms,
            args.output,
            preflight=args.preflight,
            engine=args.engine,
            kind=args.gpu_kind,
            reference_path=args.reference,
            moved=json.loads(args.moved_geometry.read_text())
            if args.moved_geometry
            else None,
            warm_repeats=args.warm_repeats,
            pilot=args.pilot,
            cold_reference_path=args.cold_reference,
        )
        return
    if args.cpu_oracle:
        if args.method != "pbe0" or args.source_revision is None:
            parser.error("first CPU cohort requires PBE0 and --source-revision")
        run_cpu_oracle(
            atoms,
            args.output,
            source_revision=args.source_revision,
            levels=tuple(args.levels),
        )
        return
    value = plan(
        atoms,
        method=args.method,
        seed=args.seed,
    )
    # A plan never overwrites prior output or creates a scientific PASS receipt.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
