"""Independent, fail-closed qualification of the unchanged CUDA matrix adapter.

The primitive timer excludes transfers and owner preparation. It is deliberately
not a complete KS endpoint or provider promotion receipt. Raw failed rows survive.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = -314159.25


def endpoint_model(spin: str, moved: bool = False) -> dict:
    """Fixed asymmetric water dimer, Cartesian def2-TZVP (96 AO), in Bohr."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.43, 0.0, 1.11],
            [-1.43, 0.0, 1.11],
            [0.3, 0.2, 5.4],
            [1.66, 0.3, 6.62],
            [-1.18, 0.1, 6.43],
        ]
    )
    if moved:
        positions[4] += [0.012, -0.006, 0.004]
    return {
        "spin": spin,
        "charge": 0 if spin == "rks" else 1,
        "multiplicity": 1 if spin == "rks" else 2,
        "atomic_numbers": [8, 1, 1, 8, 1, 1],
        "coordinates": positions.tolist(),
        "basis": "def2-tzvp",
        "basis_representation": "cartesian",
        "method": f"pbe-{spin}",
        "precision": "fp64",
        "density_fitting": "none",
        "grid": {
            "version": 1,
            "radial_points": 48,
            "angular_polar": 16,
            "angular_azimuth": 32,
        },
        "energy_tolerance": 1e-11,
        "density_tolerance": 1e-9,
        "max_iterations": 200,
    }


def endpoint_inputs(model: dict) -> tuple[dict, tuple, object]:
    """Reuse basis/quadrature data only; the reference owns every SCF equation."""
    from generativeqc import Atom
    from generativeqc.calculator import _named_basis_shells
    from generativeqc_compiler.dft import GridSpec, MolecularGrid

    atoms = tuple(
        Atom(z, tuple(xyz))
        for z, xyz in zip(model["atomic_numbers"], model["coordinates"], strict=True)
    )
    shells = _named_basis_shells(model["basis"], atoms)
    inputs = {
        **model,
        "shells": [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [(p.exponent, p.coefficient) for p in shell.primitives],
            }
            for shell in shells
        ],
    }
    grid = MolecularGrid(
        atoms,
        GridSpec(**model["grid"]),
        charge=model["charge"],
        multiplicity=model["multiplicity"],
    ).explicit()
    return inputs, atoms, grid


def reference_scf(model: dict) -> dict:
    """Independent PySCF/Libxc, exactly matched shell data and explicit quadrature."""
    import pyscf
    from pyscf import dft

    from tools.generate_validation_references import pyscf_molecule

    if pyscf.__version__ != "2.14.0" or dft.libxc.__version__ != "7.0.0":
        raise ValueError("endpoint oracle requires PySCF 2.14.0 / Libxc 7.0.0")
    inputs, _, grid = endpoint_inputs(model)
    molecule, scale, _ = pyscf_molecule(inputs)
    if molecule.nao_nr() < 96:
        raise ValueError("pilot does not reach the 96-AO acceptance domain")
    mean_field = (dft.RKS if model["spin"] == "rks" else dft.UKS)(molecule)
    mean_field.xc = "GGA_X_PBE,GGA_C_PBE"
    mean_field.grids.coords = np.asarray(grid.points)
    mean_field.grids.weights = np.asarray(grid.weights)
    mean_field.grids.non0tab = None
    mean_field.small_rho_cutoff = 0.0
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 200
    mean_field.diis_space = 8
    mean_field.direct_scf_tol = 1e-14
    history = []
    mean_field.callback = lambda env: history.append(
        {
            "cycle": int(env["cycle"]),
            "energy": float(env["e_tot"]),
            "density_change_norm": float(env["norm_ddm"]),
            "orbital_gradient_norm": float(env["norm_gorb"]),
        }
    )
    energy = float(mean_field.kernel())
    density = np.asarray(mean_field.make_rdm1()).reshape(
        -1, molecule.nao_nr(), molecule.nao_nr()
    )
    potential = np.asarray(
        mean_field.get_veff(molecule, density[0] if model["spin"] == "rks" else density)
    ).reshape(density.shape)
    fock = np.asarray(mean_field.get_hcore())[None] + potential
    overlap = np.asarray(mean_field.get_ovlp())
    residual = fock @ density @ overlap - overlap @ density @ fock
    normalized_density = density / (scale[:, None] * scale[None, :])
    return {
        "energy": energy,
        "density": normalized_density.tolist(),
        "converged": bool(mean_field.converged),
        "history": history,
        "physical_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "ao_count": int(molecule.nao_nr()),
        "pyscf": pyscf.__version__,
        "libxc": dft.libxc.__version__,
        "points_sha256": hashlib.sha256(np.asarray(grid.points).tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(
            np.asarray(grid.weights).tobytes()
        ).hexdigest(),
    }


def reference_endpoint(folder: Path) -> dict:
    """Reconverge displaced reference energies, including moving-grid response."""
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for spin, moved in itertools.product(("rks", "uks"), (False, True)):
        model = endpoint_model(spin, moved)
        baseline = reference_scf(model)
        forces = []
        displaced = []
        for step in (1e-3, 5e-4):
            force = np.empty((6, 3))
            for atom, axis in itertools.product(range(6), range(3)):
                energy = []
                for sign in (-1, 1):
                    variation = json.loads(json.dumps(model))
                    variation["coordinates"][atom][axis] += sign * step
                    result = reference_scf(variation)
                    displaced.append(
                        {
                            "step": step,
                            "atom": atom,
                            "axis": axis,
                            "sign": sign,
                            **result,
                        }
                    )
                    energy.append(result["energy"])
                    write_json(
                        folder / f"{spin}-{int(moved)}-progress.json",
                        {"baseline": baseline, "displaced": displaced},
                    )
                force[atom, axis] = -(energy[1] - energy[0]) / (2 * step)
            forces.append(force)
        stability = float(np.max(np.abs(forces[0] - forces[1])))
        accepted = (
            baseline["converged"]
            and baseline["physical_residual_rms"] < 1e-8
            and all(
                r["converged"] and r["physical_residual_rms"] < 1e-8 for r in displaced
            )
            and stability < 2e-6
        )
        row = {
            "model": model,
            "moved": moved,
            **baseline,
            "forces": forces[1].tolist(),
            "force_step_stability": stability,
            "oracle_accepted": bool(accepted),
            "displaced": displaced,
        }
        write_json(folder / f"{spin}-{int(moved)}.json", row)
        rows.append({"spin": spin, "moved": moved, "accepted": bool(accepted)})
    result = {
        "rows": rows,
        "status": "PASS" if all(row["accepted"] for row in rows) else "FAIL",
        "purpose": "independent oracle only; never production state",
    }
    write_json(folder / "summary.json", result)
    return result


def endpoint_pilot(
    folder: Path, library: Path, reference: Path, spin: str, graph: bool
) -> dict:
    """One fresh-process production E+F trajectory; missing pair/phase gates stay open."""
    os.environ["GENERATIVEQC_LIBRARY"] = str(library.resolve())
    os.environ["GENERATIVEQC_CUDA_KS_CHUNK"] = "2" if graph else "1"
    os.environ["GENERATIVEQC_CUDA_KS_REPLAY"] = "1" if graph else "0"
    from generativeqc import Calculator
    from generativeqc._ks_snapshot import NativeKsSnapshot
    from generativeqc.ks import KsOptions
    from generativeqc_compiler.dft import GridSpec

    folder.mkdir(parents=True, exist_ok=False)
    model = endpoint_model(spin)
    _, atoms, _ = endpoint_inputs(model)
    records = []
    library_hash = sha256(library)
    started = time.perf_counter()
    try:
        calculator = Calculator(
            method=model["method"],
            basis=model["basis"],
            device="cuda",
            basis_representation="cartesian",
            precision="fp64",
            density_fitting="none",
            ks_options=KsOptions(grid=GridSpec(**model["grid"])),
            max_iterations=model["max_iterations"],
            energy_tolerance=model["energy_tolerance"],
            density_tolerance=model["density_tolerance"],
            screening_tolerance=1e-14,
        )
        with calculator.prepare_batch(
            [atoms], charges=[model["charge"]], multiplicities=[model["multiplicity"]]
        ) as plan:
            preparation_seconds = time.perf_counter() - started
            for index, phase in enumerate(("cold", "warm", "warm", "moved", "moved")):
                current = endpoint_model(spin, phase == "moved")
                coordinates = [current["coordinates"]] if phase == "moved" else None
                begin = time.perf_counter()
                result = plan.execute(
                    coordinates, properties=("energy", "forces"), strict=False
                ).items[0]
                elapsed = time.perf_counter() - begin
                row = {
                    "index": index,
                    "phase": phase,
                    "endpoint_seconds": elapsed,
                    "preparation_seconds": preparation_seconds
                    if phase == "cold"
                    else 0.0,
                    "cold_total_seconds": preparation_seconds + elapsed
                    if phase == "cold"
                    else None,
                    "status": result.status,
                    "message": result.status_message,
                    "converged": result.converged,
                    "backend": result.executed_backend,
                    "energy": result.energy if np.isfinite(result.energy) else None,
                    "forces": None if result.forces is None else result.forces.tolist(),
                    "iterations": result.iterations,
                    "precision": result.precision,
                    "history_and_final_diagnostic": None
                    if result.ks_diagnostic is None
                    else result.ks_diagnostic.to_payload(),
                    "resource_diagnostics": plan.resource_diagnostics,
                    "matrix_phase_seconds": None,
                    "actual_matrix_semantic_calls": None,
                    "provider_selection": "unobserved-auto",
                    "graph_requested": graph,
                    "actual_graph_replay_count": None,
                    "accepted": False,
                }
                if result.succeeded:
                    snapshot = NativeKsSnapshot(plan, 0)
                    try:
                        snapshot.check_current()
                        n, spins, natoms = map(int, snapshot.metadata[1:4])
                        offset = 2 + natoms * 4
                        density = snapshot.values[
                            offset : offset + spins * n * n
                        ].reshape(spins, n, n)
                        np.savez(
                            folder / f"{index}-final-state.npz",
                            metadata=snapshot.metadata,
                            values=snapshot.values,
                        )
                        oracle_result = json.loads(
                            (
                                reference / f"{spin}-{int(phase == 'moved')}.json"
                            ).read_text()
                        )
                        row["energy_error"] = abs(
                            result.energy - oracle_result["energy"]
                        )
                        row["force_error"] = (
                            None
                            if result.forces is None
                            else float(
                                np.max(
                                    np.abs(
                                        result.forces
                                        - np.asarray(oracle_result["forces"])
                                    )
                                )
                            )
                        )
                        row["density_error"] = float(
                            np.max(
                                np.abs(density - np.asarray(oracle_result["density"]))
                            )
                        )
                        row["final_state_metadata"] = snapshot.metadata
                        row["final_state_gates"] = final_state_gates(
                            snapshot.metadata, snapshot.values, current
                        )
                        row["oracle_accepted"] = oracle_result["oracle_accepted"]
                        row["numerical_accepted"] = bool(
                            result.converged
                            and result.executed_backend == "cuda"
                            and oracle_result["oracle_accepted"]
                            and n >= 96
                            and row["energy_error"] < 1e-8
                            and row["force_error"] is not None
                            and row["force_error"] < 3e-5
                            and row["density_error"] < 1e-6
                            and row["final_state_gates"]["accepted"]
                        )
                    finally:
                        snapshot.close()
                records.append(row)
                write_json(
                    folder / "pilot.json",
                    {
                        "status": "INCOMPLETE",
                        "model": model,
                        "library_sha256": library_hash,
                        "rows": records,
                        "blockers": [
                            "No verified native/library ablation pair",
                            "No observed matrix-phase or Graph-replay semantic counters",
                        ],
                    },
                )
    except (OSError, RuntimeError, ValueError, TypeError, ArithmeticError) as error:
        write_json(
            folder / "pilot.json",
            {
                "status": "INCOMPLETE",
                "model": model,
                "rows": records,
                "library_sha256": library_hash,
                "failure_type": type(error).__name__,
                "failure": str(error),
            },
        )
    return json.loads((folder / "pilot.json").read_text())


def final_state_gates(metadata: tuple, values: np.ndarray, model: dict) -> dict:
    """Independent NumPy stationarity, metric orthonormality and density closure."""
    n, spins, natoms = map(int, metadata[1:4])
    offset = 2 + natoms * 4
    blocks = {}
    for name in (
        "density",
        "fock",
        "coefficients",
        "energies",
        "occupations",
        "weighted_density",
    ):
        shape = (spins, n) if name in ("energies", "occupations") else (spins, n, n)
        count = int(np.prod(shape))
        blocks[name] = values[offset : offset + count].reshape(shape)
        offset += count
    overlap = values[offset : offset + n * n].reshape(n, n)
    density, fock, coefficients = (
        blocks[name] for name in ("density", "fock", "coefficients")
    )
    residual = fock @ density @ overlap - overlap @ density @ fock
    metric = coefficients.swapaxes(-1, -2) @ overlap @ coefficients
    reconstructed = (
        coefficients * blocks["occupations"][:, None, :]
    ) @ coefficients.swapaxes(-1, -2)
    expected_electrons = sum(model["atomic_numbers"]) - model["charge"]
    electrons = float(np.trace(density @ overlap, axis1=-2, axis2=-1).sum())
    result = {
        "physical_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "orthonormality_error": float(np.max(np.abs(metric - np.eye(n)))),
        "density_closure_error": float(np.max(np.abs(density - reconstructed))),
        "electron_error": abs(electrons - expected_electrons),
        "density_orbital_generation_match": bool(metadata[10] == metadata[11]),
        "finite": bool(np.isfinite(values).all()),
    }
    result["accepted"] = bool(
        result["finite"]
        and result["density_orbital_generation_match"]
        and result["physical_residual_rms"] < 1e-8
        and result["orthonormality_error"] < 1e-8
        and result["density_closure_error"] < 1e-7
        and result["electron_error"] < 1e-7
    )
    return result


@dataclass(frozen=True)
class Case:
    """One adapter invocation with explicit physical storage and execution mode."""

    n: int
    batch: int
    spins: int
    ordinary: bool
    left_spin: bool
    right_spin: bool
    transpose: bool
    mask: str
    graph: bool
    library: bool
    scale: float = 1.0


def cases() -> tuple[Case, ...]:
    """Fixed Cartesian coverage, including failed-system poison and odd tails."""
    rows = []
    layouts = [(True, 1, False, False)] + [
        (False, spins, left, right)
        for spins, left, right in itertools.product(
            (1, 2), (False, True), (False, True)
        )
    ]
    for n, batch, layout, transpose, mask, graph, library in itertools.product(
        (3, 16, 17, 19, 97),
        (1, 3),
        layouts,
        (False, True),
        ("active", "mixed", "inactive", "failed"),
        (False, True),
        (False, True),
    ):
        ordinary, spins, left, right = layout
        rows.append(
            Case(
                n,
                batch,
                spins,
                ordinary,
                left,
                right,
                transpose,
                mask,
                graph,
                library,
                -0.375 if ordinary else 1.0,
            )
        )
    return tuple(rows)


def inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate independent blocks, preserving identical operands across providers."""
    rng = np.random.default_rng(1873 + 31 * case.n + case.batch)
    left = rng.uniform(
        -0.5, 0.5, (case.batch, case.spins if case.left_spin else 1, case.n, case.n)
    )
    right = rng.uniform(
        -0.5, 0.5, (case.batch, case.spins if case.right_spin else 1, case.n, case.n)
    )
    active = np.ones(case.batch, dtype=np.uint8)
    if case.mask in ("mixed", "failed"):
        active[-1] = 0
    elif case.mask == "inactive":
        active[:] = 0
    elif case.mask != "active":
        raise ValueError("unknown mask")
    if case.mask == "failed":
        left[-1] = np.nan
        right[-1] = np.nan
    return left, right, active


def oracle(
    case: Case, left: np.ndarray, right: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """Use NumPy wide matmul, rather than copied CUDA element/reduction indexing."""
    expected = np.full(
        (case.batch, case.spins, case.n, case.n), SENTINEL, dtype=np.longdouble
    )
    for system in np.flatnonzero(active):
        a = np.broadcast_to(left[system], (case.spins, case.n, case.n)).astype(
            np.longdouble
        )
        b = np.broadcast_to(right[system], (case.spins, case.n, case.n)).astype(
            np.longdouble
        )
        if case.transpose:
            a = a.swapaxes(-1, -2)
        expected[system] = (a @ b) * np.longdouble(case.scale)
    return expected


def encode(array: np.ndarray) -> bytes:
    """System/spin-major blocks with each matrix in column-major FP64 storage."""
    return np.asarray(array, dtype="<f8").swapaxes(-1, -2).tobytes(order="C")


def assess(
    case: Case, actual: np.ndarray, expected: np.ndarray, active: np.ndarray
) -> dict:
    """Gate all outputs; an inactive overwrite cannot pass a numerical tolerance."""
    if actual.shape != expected.shape:
        raise ValueError("output shape mismatch")
    inactive = active == 0
    mask_ok = bool(np.all(actual[inactive] == SENTINEL))
    selected = active != 0
    finite = bool(np.all(np.isfinite(actual[selected])))
    error = float(
        np.max(
            np.abs(actual[selected].astype(np.longdouble) - expected[selected]),
            initial=0,
        )
    )
    # Fixed gate, not fitted to observations. O(n) accumulation of bounded inputs.
    limit = 64 * np.finfo(np.float64).eps * case.n * max(1.0, abs(case.scale))
    return {
        "status": "PASS" if finite and mask_ok and error <= limit else "FAIL",
        "active_finite": finite,
        "inactive_unchanged": mask_ok,
        "max_abs_error": error if np.isfinite(error) else None,
        "absolute_limit": limit,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    def safe(value: object) -> object:
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return {"nonfinite": str(value)}
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(item) for item in value]
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_identity(folder: Path, compiler: Path, cuda: Path, cache: Path) -> dict:
    """Bind compiled dependencies and actually loaded libraries, never environment secrets."""
    dependencies = set()
    for path in folder.glob("*.d"):
        text = path.read_text().replace("\\\n", " ")
        dependencies.update(
            Path(token).resolve() for token in shlex.split(text.split(":", 1)[1])
        )
    tools = {compiler.resolve(), cache.resolve(), (cuda / "bin/nvcc").resolve()}
    for name in ("ptxas", "nvlink", "fatbinary", "cudafe++"):
        path = cuda / "bin" / name
        if path.is_file():
            tools.add(path.resolve())
    for path in (cuda / "nvvm/bin/cicc", cuda / "bin/crt/cicc"):
        if path.is_file():
            tools.add(path.resolve())
    for name in ("cc1plus", "collect2", "as", "ld"):
        result = subprocess.check_output(
            [str(compiler), f"-print-prog-name={name}"], text=True
        ).strip()
        path = Path(result)
        if not path.is_absolute():
            import shutil

            resolved = shutil.which(result)
            if resolved is None:
                raise ValueError(f"missing compiler child {name}")
            path = Path(resolved)
        tools.add(path.resolve())
    for name in (
        "crtbeginS.o",
        "crtendS.o",
        "libstdc++.so",
        "libgcc.a",
        "libgcc_s.so.1",
        "liblto_plugin.so",
    ):
        result = subprocess.check_output(
            [str(compiler), f"-print-file-name={name}"], text=True
        ).strip()
        path = Path(result)
        if not path.is_file():
            raise ValueError(f"missing compiler link input {name}")
        dependencies.add(path.resolve())
    libraries = set()
    for executable in tools | {folder / "matrix-qualification"}:
        process = subprocess.run(
            ["ldd", str(executable)], capture_output=True, text=True, check=False
        )
        for match in re.finditer(r"(?:=>\s+)?(/[^\s]+)\s+\(", process.stdout):
            path = Path(match[1])
            if path.is_file():
                libraries.add(path.resolve())
    maps = folder / "run/loaded-maps.txt"
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                path = Path(fields[5])
                if path.is_file():
                    libraries.add(path.resolve())
    artifacts = {
        path.resolve()
        for pattern in ("*.o", "*.d", "*.txt")
        for path in folder.glob(pattern)
    }
    artifacts.add((folder / "matrix-qualification").resolve())
    files = {
        str(path): sha256(path)
        for path in sorted(dependencies | tools | libraries | artifacts)
    }
    status = (folder / "source-status.txt").read_text().strip()
    payload = {
        "schema": "ks-matrix-build-identity-v1",
        "source_head": (folder / "source-head.txt").read_text().strip(),
        "source_dirty": bool(status),
        "source_status": status,
        "files": files,
        "dependency_count": len(dependencies),
        "compiler_children": [str(path) for path in sorted(tools)],
        "loaded_libraries": [str(path) for path in sorted(libraries)],
        "actual_maps_available": maps.is_file(),
        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(folder / "build-identity.json", payload)
    receipt_path = folder / "run/receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        process = json.loads((folder / "run/process.json").read_text())
        source_matched = bool(
            not status
            and maps.is_file()
            and dependencies
            and process["driver_sha256"] == sha256(folder / "matrix-qualification")
            and process["exit_code"] == 0
        )
        receipt["build_identity_sha256"] = sha256(folder / "build-identity.json")
        receipt["source_matched_identity_available"] = source_matched
        if not source_matched:
            receipt["status"] = "NOT_QUALIFIED"
        write_json(receipt_path, receipt)
    return payload


def prepare(folder: Path) -> dict:
    """Save inputs and immutable expected case identities before a GPU run."""
    folder.mkdir(parents=True, exist_ok=False)
    rows = cases()
    commands = []
    identities = []
    for index, case in enumerate(rows):
        left, right, active = inputs(case)
        prefix = folder / str(index)
        Path(f"{prefix}.input").write_bytes(
            encode(left) + encode(right) + active.tobytes()
        )
        commands.append(
            " ".join(
                str(int(value))
                for value in (
                    index,
                    case.n,
                    case.batch,
                    case.spins,
                    case.ordinary,
                    case.left_spin,
                    case.right_spin,
                    case.transpose,
                    case.graph,
                    case.library,
                )
            )
            + f" {case.scale}\n"
        )
        identities.append(
            {
                "id": index,
                "case": asdict(case),
                "input_sha256": sha256(Path(f"{prefix}.input")),
            }
        )
    (folder / "cases.txt").write_text("".join(commands), encoding="ascii")
    payload = {
        "schema": "ks-matrix-primitive-v1",
        "rows": identities,
        "wide_oracle": bool(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant),
        "oracle_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "complete_endpoint_qualified": False,
    }
    write_json(folder / "manifest.json", payload)
    return payload


def collect(folder: Path) -> dict:
    """Reject duplicate, missing or stale inputs and retain every device failure."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    protocol = [
        {"id": index, "case": asdict(case)} for index, case in enumerate(cases())
    ]
    declared = [{"id": row["id"], "case": row["case"]} for row in manifest["rows"]]
    if declared != protocol:
        raise ValueError("manifest does not cover the fixed protocol")
    raw = [
        json.loads(line) for line in (folder / "device.jsonl").read_text().splitlines()
    ]
    device = [row for row in raw if row.get("kind") == "device"]
    observations = [row for row in raw if row.get("kind") == "case"]
    by_id = {row["id"]: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError("duplicate device row")
    expected_ids = {row["id"] for row in manifest["rows"]}
    if not set(by_id) <= expected_ids:
        raise ValueError("unexpected device row")
    results = []
    for identity in manifest["rows"]:
        index = identity["id"]
        case = Case(**identity["case"])
        if sha256(folder / f"{index}.input") != identity["input_sha256"]:
            raise ValueError("input changed after preparation")
        left, right, active = inputs(case)
        if (folder / f"{index}.input").read_bytes() != encode(left) + encode(
            right
        ) + active.tobytes():
            raise ValueError("input differs from independent oracle inputs")
        observed = by_id.get(index)
        row = {**identity, "device": observed, "status": "INCOMPLETE"}
        if observed is not None and observed.get("status") == 0:
            path = folder / f"{index}.output"
            data = np.frombuffer(path.read_bytes(), dtype="<f8")
            expected = oracle(case, left, right, active)
            # Two executions; both must satisfy the gate, including Graph reset/replay.
            actual = data.reshape((2, case.batch, case.spins, case.n, case.n)).swapaxes(
                -1, -2
            )
            gates = [assess(case, value, expected, active) for value in actual]
            row.update(
                status="PASS" if all(g["status"] == "PASS" for g in gates) else "FAIL",
                gates=gates,
                output_sha256=sha256(path),
            )
        results.append(row)
    probes = [row for row in raw if row.get("kind") == "owner"]
    probe_ids = [row.get("probe") for row in probes]
    expected_probes = {"small", "admit", "allocation", "memory-error", "capture"}
    owners_ok = (
        set(probe_ids) == expected_probes
        and len(probe_ids) == len(expected_probes)
        and all(row.get("pass") is True for row in probes)
    )
    wide = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    passed = (
        wide
        and len(device) == 1
        and owners_ok
        and all(row["status"] == "PASS" for row in results)
    )
    receipt = {
        "schema": manifest["schema"],
        "status": "PASS" if passed else "NOT_QUALIFIED",
        "complete_endpoint_qualified": False,
        "device": device,
        "owner_probes": probes,
        "owner_probes_accepted": owners_ok,
        "wide_oracle": bool(wide),
        "oracle_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "rows": results,
        "raw_sha256": sha256(folder / "device.jsonl"),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "fail_count": sum(row["status"] == "FAIL" for row in results),
        "incomplete_count": sum(row["status"] == "INCOMPLETE" for row in results),
    }
    write_json(folder / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "collect",
            "run",
            "identity",
            "oracle-endpoint",
            "pilot-endpoint",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--spin", choices=("rks", "uks"))
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if args.mode in ("oracle-endpoint", "pilot-endpoint"):
        sys.path[:0] = [str(ROOT), str(ROOT / "python")]
        if args.mode == "oracle-endpoint":
            result = reference_endpoint(args.output)
            return 0 if result["status"] == "PASS" else 1
        if args.library is None or args.reference is None or args.spin is None:
            parser.error("pilot-endpoint requires --library, --reference and --spin")
        endpoint_pilot(args.output, args.library, args.reference, args.spin, args.graph)
        return 1  # The auto-route pilot cannot qualify absent paired-provider gates.
    if args.mode == "identity":
        if args.compiler is None or args.cuda is None or args.cache is None:
            parser.error("identity requires --compiler, --cuda and --cache")
        build_identity(args.output, args.compiler, args.cuda, args.cache)
        return 0
    if args.mode in ("prepare", "run"):
        prepare(args.output)
    if args.mode == "run":
        if args.driver is None:
            parser.error("run requires --driver")
        with (
            (args.output / "device.jsonl").open("w", encoding="utf-8") as stdout,
            (args.output / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.run(
                [str(args.driver.resolve()), str(args.output.resolve())],
                stdout=stdout,
                stderr=stderr,
                timeout=3600,
                check=False,
            )
        write_json(
            args.output / "process.json",
            {"exit_code": process.returncode, "driver_sha256": sha256(args.driver)},
        )
    if args.mode in ("collect", "run"):
        result = collect(args.output)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "pass_count",
                        "fail_count",
                        "incomplete_count",
                    )
                }
            )
        )
        return 0 if result["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
