"""Bounded CPU endpoint/oracle capture for #1791; one provider per process.

Run with an explicit GENERATIVEQC_LIBRARY and source PYTHONPATH. This runner
never invokes reference export, seeds a determinant, or alters SCF controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import subprocess
import time
from pathlib import Path

THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--library-sha", required=True)
    p.add_argument("--provider", choices=("scalar", "openblas"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("refusing to overwrite evidence")
    if any(os.environ.get(k) != "1" for k in THREADS):
        raise ValueError("all BLAS/OpenMP thread settings must explicitly be 1")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve(strict=True)
    if hashlib.sha256(library.read_bytes()).hexdigest() != a.library_sha:
        raise ValueError("native library hash mismatch")
    head = subprocess.check_output(
        ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != a.source_sha:
        raise ValueError("source revision mismatch")

    import numpy as np
    import pyscf
    from generativeqc import Atom, Calculator
    from generativeqc.progressive import _retained_density
    from generativeqc_compiler.common.resources import ResourceBudget
    from pyscf import gto, scf

    from tools.generate_build_identity import _inventory, _source_identity
    from tools.oh_uhf_symmetry import (
        AO_LABELS,
        certify_oh,
        fit_axial_angle,
        raw_comparison,
    )

    capsule = (
        a.source / "benchmarks/results/cpu-bounded-scalar-hf-20261003/capsule.json.xz"
    )
    data = json.loads(lzma.decompress(capsule.read_bytes()))["components"]["molecular"]
    historical = [r for r in data["raw_failures"] if r["case"] == "oh6_exact_uhf"]
    calc = Calculator(
        method="uhf",
        basis="sto-3g",
        basis_representation="spherical",
        device="cpu",
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        resource_budget=ResourceBudget(host_bytes=344712),
    )
    source_identity = _source_identity(
        a.source.resolve(),
        _inventory(
            a.source.resolve(),
            a.source.resolve() / "cmake/GenerativeQCSourceIdentity.json",
        ),
    )
    if calc._library.generativeqc_get_source_identity().decode() != source_identity:
        raise ValueError("native library scientific source identity mismatch")
    xyz = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.834))]
    # Bind the independent oracle to *actual bundled primitives*, not stock PySCF STO-3G.
    shells = calc._shells_for_atoms([Atom.from_value(x) for x in xyz])
    basis = {"O0": [], "H1": []}
    for shell in shells:
        basis[("O0", "H1")[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[[q.exponent, q.coefficient] for q in shell.primitives],
            ]
        )
    frozen = data["independent_oracles"]["oh6_exact_uhf-oracle-1.0.json"]["data"][
        "basis"
    ]
    if basis != frozen:
        raise ValueError("bundled primitives differ from frozen Hamiltonian")
    module = Path(__import__("generativeqc").__file__).resolve()
    if module.parent != a.source.resolve() / "python/generativeqc":
        raise ValueError("Python facade is not from specified source")
    cache = (library.parent / "CMakeCache.txt").read_text()
    if f"GENERATIVEQC_CPU_LINALG_PROVIDER:STRING={a.provider}" not in cache:
        raise ValueError("provider not bound to CMake configuration")
    rows = []
    started = time.perf_counter()
    with calc.prepare_batch([xyz], charges=[0], multiplicities=[2]) as batch:
        for phase, bond in [("original", 1.834), ("moved", 1.85234)]:
            coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]])
            t = time.perf_counter()
            result = batch.execute(
                None if phase == "original" else [coords],
                properties=("energy", "forces"),
                strict=True,
            ).items[0]
            endpoint_seconds = time.perf_counter() - t
            # Getter copies the density retained by THIS energy+force endpoint.
            density = _retained_density(batch).copy()
            if not result.converged:
                raise ValueError("primary endpoint did not converge")
            mol = gto.M(
                atom=[("O0", coords[0]), ("H1", coords[1])],
                basis=basis,
                unit="Bohr",
                charge=0,
                spin=1,
                cart=False,
                verbose=0,
            )
            if mol.ao_labels() != [
                "0 O0 1s    ",
                "0 O0 2s    ",
                "0 O0 2px   ",
                "0 O0 2py   ",
                "0 O0 2pz   ",
                "1 H1 1s    ",
            ]:
                raise ValueError("independent AO ordering mismatch")
            mf = scf.UHF(mol)
            mf.init_guess = "1e"
            mf.max_cycle = 200
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-11
            mf.direct_scf_tol = 1e-14
            mf.kernel()
            if not mf.converged:
                raise ValueError("independent UHF oracle did not converge")
            reference = mf.make_rdm1()
            inputs = {
                "coordinates": coords.tolist(),
                "ao_labels": list(AO_LABELS),
                "overlap": mf.get_ovlp().tolist(),
                "hcore": mf.get_hcore().tolist(),
                "eri": mol.intor("int2e", aosym="s1").tolist(),
                "density": density.tolist(),
                "reference": reference.tolist(),
                "endpoint_energy": float(result.energy),
                "reference_energy": float(mf.e_tot),
                "angle": fit_axial_angle(density, reference),
            }
            diagnostic = certify_oh(**inputs)
            force_error = float(
                np.max(np.abs(result.forces + mf.nuc_grad_method().kernel()))
            )
            rows.append(
                {
                    "provider": a.provider,
                    "phase": phase,
                    "certificate_inputs": inputs,
                    "diagnostic": diagnostic,
                    "observed_raw_status": diagnostic["raw"]["status"],
                    "force_error": force_error,
                    "force_gate": 1e-7,
                    "converged": result.converged,
                    "oracle_converged": mf.converged,
                    "oracle_cycles": mf.cycles,
                    "iterations": result.iterations,
                    "fock_builds": result.fock_builds,
                    "executed_backend": result.executed_backend,
                    "warm_start_used": result.warm_start_used,
                    "forces": result.forces.tolist(),
                    "endpoint_seconds": endpoint_seconds,
                    "endpoint_scope": "execute energy+forces; density getter excluded",
                }
            )
    # Independent nondegenerate molecular fixture: no OH equivalence fallback.
    h2 = gto.M(
        atom=[("H0", (0, 0, -0.7)), ("H1", (0, 0, 0.7))],
        basis={"H0": basis["H1"], "H1": basis["H1"]},
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    h2mf = scf.UHF(h2)
    h2mf.init_guess = "1e"
    h2mf.conv_tol = 1e-13
    h2mf.conv_tol_grad = 1e-11
    h2mf.max_cycle = 200
    h2mf.kernel()
    if not h2mf.converged:
        raise ValueError("nondegenerate H2 oracle failed to converge")
    h2d, h2s = h2mf.make_rdm1(), h2mf.get_ovlp()
    gap = float(min(x[1] - x[0] for x in h2mf.mo_energy))
    if gap <= 1e-3:
        raise ValueError("H2 fixture is not independently nondegenerate")
    linked = []
    for line in subprocess.check_output(["ldd", str(library)], text=True).splitlines():
        for word in line.split():
            if word.startswith("/"):
                dependency = Path(word).resolve(strict=True)
                linked.append(
                    {
                        "path": str(dependency),
                        "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
                    }
                )
                break
    oracle_libraries = {
        str(path.relative_to(Path(pyscf.__file__).parent)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for pattern in ("lib/libcgto.so", "lib/deps/lib/libcint.so*")
        for path in Path(pyscf.__file__).parent.glob(pattern)
        if path.is_file()
    }
    report = {
        "schema": "oh-uhf-symmetry-1791-v1",
        "contract": {
            "hamiltonian": "all-electron nonrelativistic exact Coulomb; no ECP or DF",
            "method": "uhf",
            "basis": "bundled sto-3g; exact primitives compared to frozen capsule",
            "basis_representation": "spherical",
            "charge": 0,
            "multiplicity": 2,
            "spin_electrons": [5, 4],
            "units": "Bohr, Hartree, forces Hartree/Bohr",
            "precision": "fp64",
            "max_iterations": 150,
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-14,
            "host_budget_bytes": 344712,
            "oracle_init_guess": "1e",
            "oracle_max_cycle": 200,
            "oracle_conv_tol": 1e-13,
            "oracle_conv_tol_grad": 1e-11,
            "oracle_direct_scf_tol": 1e-14,
            "density_origin": "primary batch warm-state getter after each energy+force endpoint; no reference export solve",
        },
        "rows": rows,
        "nondegenerate_fixture": {
            "name": "neutral H2/STO-3G, double occupied bonding orbital",
            "coordinates": [[0, 0, -0.7], [0, 0, 0.7]],
            "overlap": h2s.tolist(),
            "density": h2d.tolist(),
            "hcore": h2mf.get_hcore().tolist(),
            "eri": h2.intor("int2e", aosym="s1").tolist(),
            "energy": float(h2mf.e_tot),
            "orbital_gap_hartree": gap,
            "self_raw": raw_comparison(h2d, h2d, h2s),
            "symmetry_certificate": "unsupported; raw comparison required",
        },
        "historical_raw_failures": historical,
        "provenance": {
            "source_revision": head,
            "library": str(library),
            "library_sha256": a.library_sha,
            "cmake_cache_sha256": hashlib.sha256(
                (library.parent / "CMakeCache.txt").read_bytes()
            ).hexdigest(),
            "build_header_sha256": hashlib.sha256(
                (library.parent / "generated/build_identity.hpp").read_bytes()
            ).hexdigest(),
            "native_source_identity": calc._library.generativeqc_get_source_identity().decode(),
            "capsule_sha256": hashlib.sha256(capsule.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "certificate_sha256": hashlib.sha256(
                (Path(__file__).parent / "oh_uhf_symmetry.py").read_bytes()
            ).hexdigest(),
            "pyscf_version": pyscf.__version__,
            "oracle_libraries_sha256": oracle_libraries,
            "native_linked_dependencies": linked,
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "compiler_version": subprocess.check_output(
                ["/usr/bin/c++", "--version"], text=True
            ).splitlines()[0],
            "machine": platform.machine(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "threads": {k: os.environ[k] for k in THREADS},
            "basis": basis,
            "total_seconds": time.perf_counter() - started,
        },
        "limits": [
            "New source/provider execution, not recovery of historical e3/c9 matrices or binaries",
            "raw decisions remain independent; determinant policy unresolved",
            "CPU only; no GPU, performance or complete allocation qualification",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(report, f, allow_nan=False, separators=(",", ":"))
        f.write("\n")
    for row in rows:
        print(
            a.provider,
            row["phase"],
            "raw",
            row["observed_raw_status"],
            "symmetry",
            row["diagnostic"]["symmetry"]["status"],
        )
    if any(
        r["diagnostic"]["symmetry"]["status"] != "PASS" or r["force_error"] > 1e-7
        for r in rows
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
