"""One fresh-MINAO PySCF reconvergence; no native module or native state input."""

import argparse
import hashlib
import importlib.metadata
import json
import lzma
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    args = parser.parse_args()
    import numpy as np
    import pyscf
    from pyscf import cc, df, gto, lib, scf
    from pyscf.cc import ccsd_t, dfccsd
    from pyscf.mp import dfmp2
    from scipy.io import loadmat
    from threadpoolctl import threadpool_info

    if pyscf.__version__ != "2.14.0" or min(args.threads, args.memory_mb) <= 0:
        raise ValueError("Require pinned PySCF 2.14.0 and positive resource bounds")
    inputs = json.loads(lzma.decompress(args.inputs.read_bytes()))
    task = inputs["tasks"][args.task]
    protocol = inputs["protocols"][str(task["carbon"])]
    if (
        task["carbon"] not in (4, 7)
        or protocol["qm7_index_zero_based"] != {4: 48, 7: 5604}[task["carbon"]]
        or protocol["hf_reference"] != "df"
        or protocol["frozen_core"] is not False
        or protocol["orbital_bases"] != ["cc-pvdz", "cc-pvtz"]
        or protocol["auxiliary_bases"] != ["cc-pvdz-jkfit", "cc-pvtz-jkfit"]
    ):
        raise ValueError("Require the exact full-DF all-electron C4/C7 protocol")
    if sha(args.dataset) != inputs["dataset_sha256"]:
        raise ValueError("Dataset identity mismatch")
    dataset = loadmat(args.dataset)
    index = protocol["qm7_index_zero_based"]
    selected = dataset["Z"][index] > 0
    center = [
        [int(z), r.tolist()]
        for z, r in zip(
            dataset["Z"][index][selected],
            dataset["R"][index][selected].astype(float),
            strict=True,
        )
    ]
    if center != protocol["atoms_bohr"]:
        raise ValueError("Original float32 Bohr geometry mismatch")
    coordinates = np.array([a[1] for a in center])
    direction = np.random.default_rng(709).normal(size=coordinates.shape)
    direction /= np.linalg.norm(direction)
    displaced = [
        [a[0], r.tolist()]
        for a, r in zip(
            center,
            coordinates + task["sign"] * task["step_bohr"] * direction,
            strict=True,
        )
    ]
    if (
        task["sign"] not in (-1, 1)
        or task["step_bohr"] not in (1e-4, 3e-5)
        or direction.tolist() != task["direction"]
        or displaced != task["atoms_bohr"]
    ):
        raise ValueError("Displacement protocol mismatch")
    lib.num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "receipt.json"
    record = {
        "schema": "generativeqc.independent-cbs-displaced-energy.v1",
        "status": "INCOMPLETE",
        "task_id": args.task,
        "task": task,
        "protocol": protocol,
        "driver_sha256": sha(__file__),
        "inputs_sha256": sha(args.inputs),
        "dataset_sha256": sha(args.dataset),
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
        "seed": "fresh MINAO per basis; no native state and no previous geometry state",
        "source_files": {
            str(Path(m.__file__).relative_to(Path(pyscf.__file__).parent)): sha(
                m.__file__
            )
            for m in (scf.hf, df.df, dfmp2, dfccsd, ccsd_t)
        },
        "stages": [],
    }
    start = time.perf_counter()
    write(path, record)
    try:
        for cardinal, orbital, auxiliary in zip(
            (2, 3), protocol["orbital_bases"], protocol["auxiliary_bases"], strict=True
        ):
            stage = {
                "cardinal": cardinal,
                "orbital": orbital,
                "auxiliary": auxiliary,
                "status": "INCOMPLETE",
                "basis_records": {},
            }
            record["stages"].append(stage)
            for role, name in (("orbital", orbital), ("auxiliary", auxiliary)):
                basis = {z: gto.basis.load(name, gto.mole._symbol(z)) for z in (1, 6)}
                digest = hashlib.sha256(
                    json.dumps(basis, sort_keys=True).encode()
                ).hexdigest()
                if digest != inputs["bases"][name]["pyscf_records_sha256"]:
                    raise ValueError("Basis records mismatch")
                stage["basis_records"][role] = {
                    "records": basis,
                    "sha256": digest,
                    "native_identity": inputs["bases"][name]["identity"],
                }
            write(path, record)
            log = args.output / f"basis-{cardinal}.log"
            mol = gto.M(
                atom=displaced,
                unit="Bohr",
                basis=orbital,
                charge=0,
                spin=0,
                cart=False,
                verbose=4,
                output=str(log),
                max_memory=args.memory_mb,
            )
            if (
                mol.nao_nr() != protocol["nao"][cardinal - 2]
                or mol.has_ecp()
                or mol.nelectron != protocol["electrons"]
                or df.addons.make_auxmol(mol, auxiliary).nao_nr()
                != protocol["naux"][cardinal - 2]
            ):
                raise ValueError("Electronic/basis dimensions mismatch")
            reference = scf.RHF(mol).density_fit(auxbasis=auxiliary)
            reference.chkfile = str(args.output / f"basis-{cardinal}.scf.chk")
            reference.init_guess = "minao"
            reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = (
                1e-13,
                1e-10,
                200,
            )
            reference.kernel()
            stage["rhf"] = {
                "energy": float(reference.e_tot),
                "converged": bool(reference.converged),
                "cycles": reference.cycles,
                "gradient_norm": float(
                    np.linalg.norm(
                        reference.get_grad(reference.mo_coeff, reference.mo_occ)
                    )
                ),
            }
            write(path, record)
            if not reference.converged:
                raise RuntimeError("RHF failed to converge")
            mp = dfmp2.DFMP2(reference, frozen=0).density_fit(auxbasis=auxiliary)
            mp.with_df = reference.with_df
            mp.kernel()
            stage["mp2"] = {
                "energy": float(mp.e_tot),
                "correlation": float(mp.e_corr),
                "frozen": 0,
                "reference_df_shared": mp.with_df is reference.with_df,
            }
            write(path, record)
            if cardinal == 2:
                coupled = cc.CCSD(reference, frozen=0).density_fit(auxbasis=auxiliary)
                coupled.with_df = reference.with_df
                coupled.conv_tol, coupled.conv_tol_normt, coupled.max_cycle = (
                    1e-12,
                    1e-10,
                    150,
                )
                coupled.chkfile = str(args.output / "basis-2.cc.chk")
                coupled.kernel()
                stage["ccsd"] = {
                    "energy": float(coupled.e_tot),
                    "converged": bool(coupled.converged),
                    "frozen": 0,
                    "reference_df_shared": coupled.with_df is reference.with_df,
                }
                write(path, record)
                if not coupled.converged:
                    raise RuntimeError("CCSD failed to converge")
                triples = float(coupled.ccsd_t())
                stage["triples"] = {
                    "correction": triples,
                    "energy": float(coupled.e_tot + triples),
                }
            mol.stdout.flush()
            stage.update(status="completed", log_sha256=sha(log))
            mol.stdout.close()
            write(path, record)
        low, high = record["stages"]
        values = {
            "low/rhf": low["rhf"]["energy"],
            "low/mp2": low["mp2"]["energy"],
            "low/df-ccsd(t)": low["triples"]["energy"],
            "high/rhf": high["rhf"]["energy"],
            "high/mp2": high["mp2"]["energy"],
        }
        h, m = (-1 / math.expm1(1.63), 1 + 1 / math.expm1(1.63)), (-8 / 19, 27 / 19)
        if list(h) != protocol["hf_weights"] or list(m) != protocol["mp2_weights"]:
            raise ValueError("CBS coefficients mismatch")
        values["composite"] = (
            h[0] * values["low/rhf"]
            + h[1] * values["high/rhf"]
            + m[0] * (values["low/mp2"] - values["low/rhf"])
            + m[1] * (values["high/mp2"] - values["high/rhf"])
            + values["low/df-ccsd(t)"]
            - values["low/mp2"]
        )
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite energies")
        record.update(status="completed", energies=values)
    except Exception as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(
            ended_unix=time.time(),
            seconds=time.perf_counter() - start,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write(path, record)


if __name__ == "__main__":
    main()
