"""Measure fresh RHF-CCSD(T) endpoints and energy displacements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from contextlib import suppress
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

Record = dict[str, Any]
Atoms = list[tuple[int, list[float]]]

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import ndarray

    Endpoint = tuple[float, ndarray | None, float, Record]

HERE = Path(__file__).resolve().parent
BASIS = "cc-pvdz"
AUX = "cc-pvdz-jkfit"
ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O"}
STEPS = (0.001, 0.0005)
REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"
NATIVE_SOURCE_IDENTITY = (
    "3f4f4a5ff958ee10dd67138eba54369fecc55563f07147fc3b2def229c984eb2"
)
ARMS = ("native-conv", "native-df", "psi4-conv", "psi4-df", "pyscf-conv")


def write(path: Path, value: Record) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def basis_file(fixture: Record, role: str) -> Path:
    expected = fixture[role + "_basis"]
    path = HERE / "basis" / (expected["identity"] + ".json")
    if digest(path) != expected["sha256"]:
        raise ValueError(f"Basis snapshot checksum differs: {path.name}")
    return path


def native(
    atoms: Atoms, fitted: bool, forces: bool, args: argparse.Namespace, fixture: Record
) -> Endpoint:
    from generativeqc import Calculator, load_basis

    orbital = load_basis(basis_file(fixture, "orbital"))
    auxiliary = load_basis(basis_file(fixture, "auxiliary")) if fitted else None
    start = time.perf_counter()
    calc = Calculator(
        method="df-ccsd(t)" if fitted else "ccsd(t)",
        basis=orbital,
        basis_representation="spherical",
        density_fitting="full_cuda" if fitted else "none",
        auxiliary_basis=auxiliary,
        device="cuda",
        precision="fp64",
        ccsd_frozen_core=0,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-10,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-12,
        ccsd_residual_tolerance=1e-10,
        correlation_memory_budget_bytes=args.correlation_budget_gib << 30,
    )
    value = calc.singlepoint(
        atoms,
        charge=0,
        multiplicity=1,
        properties=("energy", "forces") if forces else ("energy",),
    )
    elapsed = time.perf_counter() - start
    if not value.converged or value.correlation is None:
        raise RuntimeError("GenerativeQC SCF or CCSD convergence flag is false")
    return (
        value.energy,
        value.forces,
        elapsed,
        {
            "correlation": asdict(value.correlation),
            "correlation_budget_gib": args.correlation_budget_gib,
            "cc_performance": asdict(value.cc_performance)
            if value.cc_performance
            else None,
            "rhf_energy": value.correlation.reference_energy,
            "hf_cycles": value.iterations,
            "hf_density_rms": value.density_rms,
            "executed_backend": value.executed_backend,
        },
    )


def shells(fixture: Record, role: str) -> dict[str, list[Record]]:
    data = json.loads(basis_file(fixture, role).read_bytes())
    return {ELEMENT[row["atomic_number"]]: row["shells"] for row in data["elements"]}


def psi4_basis(
    psi4: ModuleType, fixture: Record, role: str, name: str, option: str
) -> None:
    lines = ["spherical", "****"]
    for element, records in shells(fixture, role).items():
        lines.append(f"{element} 0")
        for shell in records:
            l = "SPDFGH"[shell["angular_momentum"]]
            for contraction in shell["coefficients"]:
                lines.append(f"{l} {len(contraction)} 1.00")
                lines.extend(
                    f"{Decimal(e):f} {Decimal(c):f}"
                    for e, c in zip(
                        shell["exponents"],
                        contraction,
                        strict=True,
                    )
                )
        lines.append("****")
    psi4.basis_helper("\n".join(lines), name=name, key=option, set_option=False)


def pyscf_basis(fixture: Record) -> dict[str, list[list[int | list[float]]]]:
    return {
        element: [
            [
                shell["angular_momentum"],
                *[
                    [float(e), *map(float, c)]
                    for e, c in zip(
                        shell["exponents"],
                        zip(*shell["coefficients"], strict=True),
                        strict=True,
                    )
                ],
            ]
            for shell in records
        ]
        for element, records in shells(fixture, "orbital").items()
    }


def psi4_endpoint(
    atoms: Atoms, fitted: bool, forces: bool, args: argparse.Namespace, fixture: Record
) -> Endpoint:
    import psi4

    orbital_name, auxiliary_name = "issue3167orb", "issue3167aux"
    psi4_basis(psi4, fixture, "orbital", orbital_name, "BASIS")
    if fitted:
        psi4_basis(psi4, fixture, "auxiliary", auxiliary_name, "DF_BASIS_SCF")
    start = time.perf_counter()
    molecule = psi4.geometry(
        "0 1\n"
        + "\n".join(
            f"{ELEMENT[z]} " + " ".join(format(x, ".17g") for x in xyz)
            for z, xyz in atoms
        )
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n"
    )
    settings = {
        "basis": orbital_name,
        "puream": True,
        "reference": "rhf",
        "freeze_core": False,
        "scf_type": "disk_df" if fitted else "pk",
        "cc_type": "df" if fitted else "conv",
        "qc_module": "occ" if fitted else "ccenergy",
        "e_convergence": 1e-12,
        "d_convergence": 1e-10,
        "maxiter": 200,
        "ints_tolerance": 1e-14,
        "ccenergy__e_convergence": 1e-12,
        "ccenergy__r_convergence": 1e-10,
        "cclambda__r_convergence": 1e-10,
        "dfocc__e_convergence": 1e-12,
        "dfocc__r_convergence": 1e-10,
    }
    if fitted:
        settings.update(
            df_basis_scf=auxiliary_name,
            df_basis_cc=auxiliary_name,
            df_fitting_condition=1e-10,
        )
    psi4.set_options(settings)
    if forces:
        gradient, wfn = psi4.gradient(
            "ccsd(t)", molecule=molecule, dertype=1, return_wfn=True
        )
        force = -gradient.to_array()
        energy = psi4.variable("CCSD(T) TOTAL ENERGY")
    else:
        energy, wfn = psi4.energy("ccsd(t)", molecule=molecule, return_wfn=True)
        force = None
    elapsed = time.perf_counter() - start
    actual_geometry = wfn.molecule().geometry().to_array()
    if any(
        abs(actual_geometry[i, a] - x) > 1e-12
        for i, (_, xyz) in enumerate(atoms)
        for a, x in enumerate(xyz)
    ):
        raise RuntimeError("Psi4 changed the coordinates")
    nbf = wfn.basisset().nbf()
    if nbf != fixture["nao"]:
        raise RuntimeError(f"Unexpected Psi4 orbital count {nbf}")
    auxiliary_counts = {}
    if fitted:
        for key in ("DF_BASIS_SCF", "DF_BASIS_CC"):
            auxiliary_counts[key] = wfn.get_basisset(key).nbf()
        if any(n != fixture["naux"] for n in auxiliary_counts.values()):
            raise RuntimeError(f"Unexpected Psi4 auxiliary counts {auxiliary_counts}")
    return (
        energy,
        force,
        elapsed,
        {
            "settings": settings,
            "nao": nbf,
            "auxiliary_counts": auxiliary_counts,
            "electrons": wfn.nalpha() + wfn.nbeta(),
            "rhf_energy": psi4.variable("SCF TOTAL ENERGY"),
        },
    )


def pyscf_endpoint(
    atoms: Atoms, fitted: bool, forces: bool, args: argparse.Namespace, fixture: Record
) -> Endpoint:
    from pyscf import cc, gto, scf
    from pyscf.cc import ccsd_t_lambda
    from pyscf.grad.ccsd_t import Gradients

    orbitals = pyscf_basis(fixture)
    start = time.perf_counter()
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=orbitals,
        cart=False,
        symmetry=False,
        charge=0,
        spin=0,
        max_memory=args.memory_mib,
        verbose=4,
    )
    if mol.nao_nr() != fixture["nao"]:
        raise RuntimeError(f"Unexpected PySCF orbital count {mol.nao_nr()}")
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    mf.direct_scf_tol, mf.chkfile = 1e-14, None
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    solver = cc.CCSD(mf, frozen=0)
    solver.conv_tol, solver.conv_tol_normt, solver.max_cycle = 1e-12, 1e-10, 150
    eris = solver.ao2mo()
    solver.kernel(eris=eris)
    if not solver.converged:
        raise RuntimeError("PySCF CCSD did not converge")
    triples = solver.ccsd_t(eris=eris)
    energy = solver.e_tot + triples
    force = None
    if forces:
        converged, l1, l2 = ccsd_t_lambda.kernel(
            solver,
            eris=eris,
            max_cycle=150,
            tol=1e-10,
        )
        if not converged:
            raise RuntimeError("PySCF CCSD(T) Lambda did not converge")
        force = -Gradients(solver).kernel(l1=l1, l2=l2, eris=eris)
    elapsed = time.perf_counter() - start
    return (
        energy,
        force,
        elapsed,
        {
            "nao": mol.nao_nr(),
            "electrons": mol.nelectron,
            "rhf_energy": mf.e_tot,
            "ccsd_energy": solver.e_tot,
            "triples_energy": triples,
            "hf_cycles": getattr(mf, "cycles", None),
            "ccsd_cycles": getattr(solver, "cycles", None),
            "ccsdt_lambda_converged": True if forces else None,
        },
    )


def source_revision(source: Path, archive: Path | None) -> str:
    if (source / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--verify", "HEAD"], text=True
        ).strip()
        if subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=False
        ).returncode:
            raise RuntimeError(f"Source checkout has tracked changes: {source}")
    else:
        archive = archive or source.parent / f"source-{REVISION[:5]}.tar.gz"
        with tarfile.open(archive) as staged:
            revision = staged.pax_headers.get("comment")
    if revision != REVISION:
        raise RuntimeError(f"Source revision differs: {revision} != {REVISION}")
    return revision


def native_provenance(source: Path, archive: Path | None) -> Record:
    import ctypes

    revision = source_revision(source, archive)
    path = source / "tools/generate_build_identity.py"
    spec = importlib.util.spec_from_file_location("source_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = module._inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    expected = module._source_identity(source, inputs)
    if expected != NATIVE_SOURCE_IDENTITY:
        raise RuntimeError(f"Source inputs differ from pinned revision {revision}")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    receipt_sha = digest(library)
    identity = ctypes.CDLL(str(library)).generativeqc_get_source_identity
    identity.restype = ctypes.c_char_p
    found = identity().decode()
    if found != expected:
        raise RuntimeError(f"Native source identity differs: {found} != {expected}")
    return {
        "source_revision": revision,
        "source_identity": found,
        "library_sha256": receipt_sha,
        "native_input_count": len(inputs),
    }


def worker(args: argparse.Namespace, label: str) -> None:
    case, trial, *move = label.split(".")
    fixtures = json.loads((HERE / "fixtures.json").read_text())["cases"]
    fixture = fixtures[case]
    atoms = fixture["atoms_bohr"]
    diagnostic = {}
    if move:
        step_index, atom, axis, sign = map(int, move)
        step = STEPS[step_index]
        atoms[atom][1][axis] += sign * step
        diagnostic = {"step_bohr": step, "atom": atom, "axis": axis, "sign": sign}
    output = args.output / (label + ".json")
    engine, model = args.arm.split("-")
    try:
        if engine == "native":
            import generativeqc

            versions = {"generativeqc": generativeqc.__version__}
            endpoint = native
        elif engine == "psi4":
            import psi4

            versions = {"psi4": psi4.__version__}
            if psi4.__version__ != "1.10.2":
                raise ValueError(versions)
            psi4.set_memory(args.memory_mib * (1 << 20))
            psi4.set_num_threads(args.threads)
            psi4.core.IOManager.shared_object().set_default_path(os.environ["TMPDIR"])
            psi4.core.set_output_file(str(args.output / (label + ".psi4.txt")), False)
            endpoint = psi4_endpoint
        else:
            import pyscf
            from pyscf import lib

            versions = {"pyscf": pyscf.__version__}
            if pyscf.__version__ != "2.11.0":
                raise ValueError(versions)
            lib.num_threads(args.threads)
            endpoint = pyscf_endpoint
        energy, force, elapsed, work = endpoint(
            atoms, model == "df", not move, args, fixture
        )
        if not math.isfinite(energy):
            raise RuntimeError("Nonfinite total energy")
        force = None if force is None else force.tolist()
        if not move and (
            force is None
            or len(force) != len(atoms)
            or any(len(row) != 3 or not all(map(math.isfinite, row)) for row in force)
        ):
            raise RuntimeError("Invalid Cartesian force vector")
        record = {
            "status": "completed",
            "energy_hartree": float(energy),
            "force_hartree_per_bohr": force,
            "endpoint_seconds": elapsed,
            "versions": versions,
            "work": work,
        }
    except Exception as exc:  # noqa: BLE001 - retain worker failure details
        traceback.print_exc()
        record = {"status": "failed", "exception": repr(exc)}
    write(
        output,
        {
            **record,
            "case": case,
            "trial": trial,
            "arm": args.arm,
            "displacement": diagnostic or None,
        },
    )


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    raise RuntimeError("Missing CPU model")


def hardware(args: argparse.Namespace) -> Record:
    info = {
        "cpu_model": cpu_model(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "host_cpu_topology": subprocess.check_output(["lscpu", "-J"], text=True),
        "host_memory": Path("/proc/meminfo").read_text().splitlines()[0],
        "cgroup_memory": {
            str(path): path.read_text().strip()
            for path in (
                Path("/sys/fs/cgroup/memory.max"),
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            )
            if path.exists()
        },
        "threads": args.threads,
        "application_memory_mib": args.memory_mib,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "allocation": {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "SLURM_CPUS_PER_TASK",
                "GQC_QZ_JOB_NAME",
                "GQC_QZ_PROJECT",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "scratch_free_before_bytes": shutil.disk_usage(args.scratch).free,
    }
    if args.arm.startswith("native"):
        info["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if len(info["gpu"].splitlines()) != 1 or "H100" not in info["gpu"]:
            raise RuntimeError(f"Expected one allocated H100, got {info['gpu']}")
        info.update(native_provenance(args.source, args.source_archive))
    return info


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True)
    info = hardware(args)
    fixtures = HERE / "fixtures.json"
    info.update(driver_sha256=digest(Path(__file__)), fixture_sha256=digest(fixtures))
    write(args.output / "hardware.json", info)
    if args.expected_cpu and args.expected_cpu.lower() not in info["cpu_model"].lower():
        raise RuntimeError(
            f"Requested CPU {args.expected_cpu}, found {info['cpu_model']}"
        )
    env = {
        **os.environ,
        **{
            key: str(args.threads)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "PYTHONNOUSERSITE": "1",
    }
    cases = args.cases or list(json.loads(fixtures.read_text())["cases"])
    labels = (
        [] if args.fd_only else [f"{c}.{r}" for r in range(args.repeats) for c in cases]
    )
    if args.fd or args.fd_only:
        labels += [
            f"C1.fd.{step}.{atom}.{axis}.{sign}"
            for step in range(len(STEPS))
            for atom in range(5)
            for axis in range(3)
            for sign in (-1, 1)
        ]
    write(
        args.output / "plan.json",
        {"labels": labels, "trial_timeout_seconds": args.timeout},
    )
    failures = 0
    for label in labels:
        print("START", args.arm, label, flush=True)
        start = time.perf_counter()
        record = args.output / (label + ".json")
        argv = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            label,
        ]
        with (
            tempfile.TemporaryDirectory(
                prefix="gqc-3167-", dir=args.scratch
            ) as scratch,
            (args.output / (label + ".txt")).open("x") as log,
        ):
            process = subprocess.Popen(
                argv,
                env={**env, "TMPDIR": scratch, "PYSCF_TMPDIR": scratch},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if timed_out:
                write(
                    record,
                    {
                        "status": "timeout",
                        "label": label,
                        "arm": args.arm,
                        "timeout_seconds": args.timeout,
                    },
                )
        if not record.exists():
            write(
                record,
                {
                    "status": "worker_exit",
                    "label": label,
                    "arm": args.arm,
                    "returncode": process.returncode,
                },
            )
        status = json.loads(record.read_text())["status"]
        failures += status != "completed"
        print(
            "END",
            args.arm,
            label,
            status,
            round(time.perf_counter() - start, 3),
            flush=True,
        )
    write(args.output / "finished.json", {"planned": len(labels), "failed": failures})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threads", required=True, type=int)
    p.add_argument("--memory-mib", type=int, default=76800)
    p.add_argument(
        "--correlation-budget-gib", type=int, choices=range(1, 81), default=32
    )
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--fd", action="store_true")
    p.add_argument("--fd-only", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--source", type=Path)
    p.add_argument("--source-archive", type=Path)
    p.add_argument("--expected-cpu")
    p.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    args = p.parse_args()
    if args.worker:
        worker(args, args.worker)
    else:
        run(args)


if __name__ == "__main__":
    main()
