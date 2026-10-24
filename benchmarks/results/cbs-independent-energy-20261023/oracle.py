"""Reconverge full-DF QM7 energies using only PySCF, never a native result seed.

Input contains geometries and basis provenance, but no native energies. Each
basis is checkpointed independently; failed attempts must use a new output root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carbon", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=32000)
    args = parser.parse_args()
    import numpy as np
    import pyscf
    from pyscf import cc, df, gto, lib, scf
    from pyscf.cc import ccsd_t, dfccsd
    from pyscf.mp import dfmp2
    from scipy.io import loadmat
    from threadpoolctl import threadpool_info

    assert pyscf.__version__ == "2.14.0", pyscf.__version__
    assert args.threads > 0 and args.memory_mb > 0
    lib.num_threads(args.threads)
    inputs = json.loads(args.inputs.read_text())
    assert digest(args.dataset) == inputs["dataset_sha256"]
    protocol = inputs["protocols"][str(args.carbon)]
    data = loadmat(args.dataset)
    index = protocol["qm7_index_zero_based"]
    selected = data["Z"][index] > 0
    atoms = [
        [int(z), r.tolist()]
        for z, r in zip(
            data["Z"][index][selected],
            data["R"][index][selected].astype(float),
            strict=True,
        )
    ]
    assert atoms == protocol["atoms_bohr"]
    assert (
        sorted(a[0] for a in atoms) == [1] * (2 * args.carbon + 2) + [6] * args.carbon
    )
    assert protocol["hf_reference"] == "df" and protocol["frozen_core"] is False
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"C{args.carbon}.json"
    if path.exists():
        raise FileExistsError("Retain earlier attempts; choose a fresh output root")
    record = {
        "schema": "generativeqc.independent-cbs-energy.v1",
        "status": "INCOMPLETE",
        "carbon": args.carbon,
        "protocol": protocol,
        "stages": [],
        "driver_sha256": digest(__file__),
        "inputs_sha256": digest(args.inputs),
        "dataset_sha256": digest(args.dataset),
        "started_unix": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("pyscf", "numpy", "scipy", "h5py", "threadpoolctl")
        },
        "threads": args.threads,
        "max_memory_mb": args.memory_mb,
        "threadpools": threadpool_info(),
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "scheduler": {
            k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK")
        },
        "seed": "PySCF default minao; no native density/orbitals/amplitudes",
        "source_files": {
            str(Path(m.__file__).relative_to(Path(pyscf.__file__).parent)): digest(
                m.__file__
            )
            for m in (scf.hf, df.df, dfmp2, dfccsd, ccsd_t)
        },
    }
    started = time.perf_counter()
    save(path, record)
    try:
        for cardinal, orbital, auxiliary in zip(
            (2, 3), protocol["orbital_bases"], protocol["auxiliary_bases"], strict=True
        ):
            stage = {
                "cardinal": cardinal,
                "orbital": orbital,
                "auxiliary": auxiliary,
                "status": "INCOMPLETE",
                "started_unix": time.time(),
            }
            record["stages"].append(stage)
            basis_records = {}
            for role, name in (("orbital", orbital), ("auxiliary", auxiliary)):
                raw = {z: gto.basis.load(name, gto.mole._symbol(z)) for z in (1, 6)}
                checksum = hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode()
                ).hexdigest()
                expected = inputs["bases"][name]
                assert checksum == expected["pyscf_records_sha256"], (name, checksum)
                basis_records[role] = {
                    "records": raw,
                    "sha256": checksum,
                    "native_identity": expected["identity"],
                }
            stage["basis_records"] = basis_records
            log = args.output / f"C{args.carbon}-{cardinal}.log"
            mol = gto.M(
                atom=atoms,
                unit="Bohr",
                basis=orbital,
                charge=0,
                spin=0,
                cart=False,
                verbose=4,
                output=str(log),
                max_memory=args.memory_mb,
            )
            auxmol = df.addons.make_auxmol(mol, auxiliary)
            assert mol.nao_nr() == protocol["nao"][cardinal - 2]
            assert auxmol.nao_nr() == protocol["naux"][cardinal - 2]
            assert mol.nelectron == protocol["electrons"] and not mol.has_ecp()
            reference = scf.RHF(mol).density_fit(auxbasis=auxiliary)
            reference.chkfile = str(args.output / f"C{args.carbon}-{cardinal}.scf.chk")
            reference.conv_tol = 1e-13
            reference.conv_tol_grad = 1e-10
            reference.max_cycle = 200
            stage_start = time.perf_counter()
            save(path, record)
            reference.kernel()
            stage["rhf"] = {
                "energy": float(reference.e_tot),
                "converged": bool(reference.converged),
                "cycles": reference.cycles,
                "seconds": time.perf_counter() - stage_start,
                "gradient_norm": float(
                    np.linalg.norm(
                        reference.get_grad(reference.mo_coeff, reference.mo_occ)
                    )
                ),
            }
            save(path, record)
            if not reference.converged:
                raise RuntimeError("Independent RI-JK RHF did not converge")
            mp = dfmp2.DFMP2(reference, frozen=0).density_fit(auxbasis=auxiliary)
            mp.with_df = reference.with_df
            tick = time.perf_counter()
            mp.kernel()
            stage["mp2"] = {
                "energy": float(mp.e_tot),
                "correlation": float(mp.e_corr),
                "seconds": time.perf_counter() - tick,
                "frozen": 0,
                "reference_df_shared": mp.with_df is reference.with_df,
            }
            save(path, record)
            if cardinal == 2:
                coupled = cc.CCSD(reference, frozen=0).density_fit(auxbasis=auxiliary)
                coupled.with_df = reference.with_df
                coupled.conv_tol = 1e-12
                coupled.conv_tol_normt = 1e-10
                coupled.max_cycle = 150
                coupled.chkfile = str(args.output / f"C{args.carbon}-{cardinal}.cc.chk")
                tick = time.perf_counter()
                coupled.kernel()
                stage["ccsd"] = {
                    "energy": float(coupled.e_tot),
                    "correlation": float(coupled.e_corr),
                    "converged": bool(coupled.converged),
                    "seconds": time.perf_counter() - tick,
                    "frozen": 0,
                    "reference_df_shared": coupled.with_df is reference.with_df,
                }
                save(path, record)
                if not coupled.converged:
                    raise RuntimeError("Independent DF-CCSD did not converge")
                tick = time.perf_counter()
                correction = float(coupled.ccsd_t())
                stage["triples"] = {
                    "correction": correction,
                    "energy": float(coupled.e_tot + correction),
                    "seconds": time.perf_counter() - tick,
                }
                save(path, record)
            mol.stdout.flush()
            stage.update(
                status="completed",
                seconds=time.perf_counter() - stage_start,
                log_sha256=digest(log),
            )
            mol.stdout.close()
            save(path, record)
        low, high = record["stages"]
        values = {
            "low/rhf": low["rhf"]["energy"],
            "low/mp2": low["mp2"]["energy"],
            "low/df-ccsd(t)": low["triples"]["energy"],
            "high/rhf": high["rhf"]["energy"],
            "high/mp2": high["mp2"]["energy"],
        }
        h = (-1 / math.expm1(1.63), 1 + 1 / math.expm1(1.63))
        m = (-8 / 19, 27 / 19)
        assert list(h) == protocol["hf_weights"] and list(m) == protocol["mp2_weights"]
        values["composite"] = (
            h[0] * values["low/rhf"]
            + h[1] * values["high/rhf"]
            + m[0] * (values["low/mp2"] - values["low/rhf"])
            + m[1] * (values["high/mp2"] - values["high/rhf"])
            + values["low/df-ccsd(t)"]
            - values["low/mp2"]
        )
        assert all(math.isfinite(v) for v in values.values())
        record.update(status="completed", energies=values)
    except Exception as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(
            seconds=time.perf_counter() - started,
            ended_unix=time.time(),
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        save(path, record)


if __name__ == "__main__":
    main()
