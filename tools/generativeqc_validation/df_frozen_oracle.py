"""Rebuild an independently checked DF reference from the #1840 frozen package.

No native runtime, GPU, RHF or CC calculation is loaded. This deliberately
separate validation process interposes libcint's root selection, checks its
moments independently, and replaces total-degree <=2 integrals with analytic
Gaussian center derivatives accumulated in host extended precision. All
orbital transformations use extended accumulation. Production remains
entirely native FP64 and never imports this module.

Run under a finite CPU allocation with PySCF 2.14.0, mpmath, a C++20 compiler,
OpenMP and ccache. The large input and output arrays belong outside Git.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash the actual retained bytes, not a regenerated array serialization."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_package(package: Path) -> dict[str, Any]:
    """Reject changed frozen evidence before constructing a replacement oracle."""
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = (package / entry["path"]).resolve()
        if not path.is_relative_to(package) or digest(path) != entry["sha256"]:
            raise ValueError(f"frozen package integrity failure: {entry['path']}")
    metadata = json.loads((package / "basis-and-ordering.json").read_text())
    if metadata["case"] != "ethane230":
        raise ValueError("this frozen-frame recipe qualifies ethane230 only")
    return metadata


def build_helpers(output: Path) -> tuple[Path, Path]:
    """Build validation-only helpers against the installed independent library."""
    import pyscf

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("the frozen reference requires PySCF 2.14.0")
    deps = Path(pyscf.__file__).resolve().parent / "lib/deps"
    if not re.search(
        r'#define\s+CINT_VERSION\s+"6\.1\.3"', (deps / "include/cint.h").read_text()
    ):
        raise RuntimeError("the checked root ABI requires libcint 6.1.3")
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if not cache or not compiler:
        raise RuntimeError("ccache and a host C++ compiler are required")
    subprocess.run([cache, "--version"], check=True)
    library = deps / "lib/libcint.so"
    for name in ("df_reference_roots", "df_precision_oracle"):
        source = HERE / (name + (".c" if name.endswith("roots") else ".cpp"))
        obj = output / (name + ".o")
        # The C interposer uses C11 atomics; compile it with the C driver.
        driver = shutil.which("cc") if name.endswith("roots") else compiler
        if not driver:
            raise RuntimeError("host C compiler is required for the root interposer")
        flags = ["-std=c11"] if name.endswith("roots") else ["-std=c++20", "-fopenmp"]
        subprocess.run(
            [
                cache,
                driver,
                *flags,
                "-O2",
                "-fno-fast-math",
                "-fPIC",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
        )
        link = [driver, "-shared", str(obj), "-o", str(output / (name + ".so"))]
        link += (
            [str(library), "-Wl,-rpath," + str(library.parent)]
            if name.endswith("roots")
            else ["-fopenmp"]
        )
        subprocess.run(link, check=True)
    return output / "df_reference_roots.so", library


def check_moments(bridge: ct.CDLL) -> dict[str, Any]:
    """Check all defining moments against 90-digit incomplete gamma integrals."""
    import mpmath as mp

    bridge.CINTrys_roots.argtypes = [ct.c_int, ct.c_double, ct.c_void_p, ct.c_void_p]
    bridge.CINTrys_roots.restype = ct.c_int
    values = [
        0.0,
        1e-20,
        1e-12,
        1e-8,
        3e-7,
        0.1,
        0.5,
        1.0,
        2.053027104306768,
        5.0,
        10.0,
        np.nextafter(20.0, 0.0),
        20.0,
        np.nextafter(20.0, np.inf),
        40.0,
        60.0,
        100.0,
        1e3,
        1e6,
        *[i * 0.37 for i in range(1, 200)],
    ]
    worst, at = 0.0, None
    with mp.workdps(90):
        for n in range(1, 7):
            for t in values:
                u, w = np.empty(n), np.empty(n)
                if bridge.CINTrys_roots(n, t, u.ctypes.data, w.ctypes.data):
                    raise RuntimeError("independent root solver failed")
                if not np.all(u > 0) or not np.all(w > 0):
                    raise ValueError("invalid quadrature roots/weights")
                for k in range(2 * n):
                    exponent = k + mp.mpf(".5")
                    expected = (
                        mp.gammainc(exponent, 0, mp.mpf(t))
                        / (2 * mp.mpf(t) ** exponent)
                        if t
                        else 1 / mp.mpf(2 * k + 1)
                    )
                    actual = mp.fsum(
                        mp.mpf(ww) * (mp.mpf(uu) / (1 + mp.mpf(uu))) ** k
                        for uu, ww in zip(u, w, strict=True)
                    )
                    error = float(abs(actual / expected - 1))
                    if error > worst:
                        worst, at = error, [n, t, k]
    # libcint's moment recurrence is quad, but its final eigensolver/export is
    # binary64. Qualify that interface explicitly, not an all-quad fiction.
    if worst >= 1e-14:
        raise ValueError(f"independent quadrature moment failure: {worst}, {at}")
    return {
        "max_relative_error": worst,
        "worst_case": at,
        "cases": 6 * len(values),
        "dps": 90,
    }


def basis_arrays(metadata: dict[str, Any], side: str) -> tuple[np.ndarray, ...]:
    """Pack the exact original shell input; no fitting or renormalized inputs."""
    info, coefficients, ao = [], [], []
    for i, shell in enumerate(metadata[side]["shells"]):
        primitives = shell["input_primitives"]
        info.append(
            [
                shell["angular_momentum"],
                shell["atom_index"],
                len(coefficients),
                len(primitives),
            ]
        )
        coefficients.extend(primitives)
        ao.extend([[i, k] for k in range(shell["ao_end"] - shell["ao_begin"])])
    return (
        np.array(info, dtype=np.int32),
        np.array(coefficients),
        np.array(ao, dtype=np.int32),
    )


def verify_basis(mol: Any, metadata: dict[str, Any]) -> None:
    """Check primitive values, general-contraction expansion and public ordering."""
    expanded = []
    for atom in range(mol.natm):
        for shell in mol._basis[mol.atom_pure_symbol(atom)]:
            for contraction in range(len(shell[1]) - 1):
                expanded.append(
                    (atom, shell[0], [[r[0], r[1 + contraction]] for r in shell[1:]])
                )
    if (
        len(expanded) != len(metadata["shells"])
        or [s.strip() for s in mol.ao_labels()] != metadata["ao_labels"]
    ):
        raise ValueError("frozen basis/order differs from the installed reference")
    for frozen, (atom, angular, primitives) in zip(
        metadata["shells"], expanded, strict=True
    ):
        if (
            frozen["atom_index"] != atom
            or frozen["angular_momentum"] != angular
            or not np.array_equal(frozen["input_primitives"], primitives)
        ):
            raise ValueError("frozen primitive/contraction mismatch")


def worker(package: Path, output: Path) -> None:
    """Construct the corrected reference in a fresh interposed process."""
    from pyscf import df, gto, lib

    metadata = audit_package(package)
    if np.finfo(np.longdouble).nmant < 63:
        raise RuntimeError("the analytic reference requires at least a 64-bit mantissa")
    lib.num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    bridge = ct.CDLL(str(output / "df_reference_roots.so"))
    bridge.accurate_root_calls.restype = ct.c_ulong
    report = {
        "schema": "generativeqc.df-frozen-reference.v1",
        "case": "ethane230",
        "moment_check": check_moments(bridge),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "old_reference_sha256": digest(package / "reference.npz"),
        "metadata_sha256": digest(package / "basis-and-ordering.json"),
        "recipe_sha256": digest(Path(__file__)),
        "helper_sha256": {
            name: digest(HERE / name)
            for name in ("df_reference_roots.c", "df_precision_oracle.cpp")
        },
    }
    atoms = [
        (a["atomic_number"], r)
        for a, r in zip(
            metadata["atoms_bohr"], metadata["reference_positions_bohr"], strict=True
        )
    ]
    mol = gto.M(atom=atoms, unit="Bohr", basis="aug-cc-pvtz", verbose=0)
    aux = df.addons.make_auxmol(mol, "aug-cc-pvtz-ri")
    verify_basis(mol, metadata["orbital"])
    verify_basis(aux, metadata["auxiliary"])
    before = bridge.accurate_root_calls()
    metric = aux.intor("int2c2e")
    raw = np.ascontiguousarray(
        df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1"), dtype=np.longdouble
    )
    report["integral_root_calls"] = int(bridge.accurate_root_calls() - before)
    if report["integral_root_calls"] < 1000:
        raise RuntimeError("libcint did not use the qualified root interposer")
    helper = ct.CDLL(str(output / "df_precision_oracle.so"))
    helper.analytic_low_degree.argtypes = [ct.c_int] * 4 + [ct.c_void_p] * 8
    helper.analytic_low_degree.restype = None
    helper.extended_mo_ld.argtypes = [ct.c_size_t, ct.c_size_t] + [ct.c_void_p] * 3
    helper.extended_mo_ld.restype = None
    si, sc, ao = basis_arrays(metadata, "orbital")
    qi, qc, qa = basis_arrays(metadata, "auxiliary")
    xyz = np.array(metadata["reference_positions_bohr"])
    helper.analytic_low_degree(
        230,
        488,
        len(si),
        len(qi),
        *[x.ctypes.data for x in (si, qi, sc, qc, ao, qa, xyz, raw)],
    )
    with np.load(package / "reference.npz") as old:
        c = np.ascontiguousarray(old["C"])
        original_rank = int(
            np.count_nonzero(
                old["metric_eigenvalues"] > 1e-10 * old["metric_eigenvalues"][-1]
            )
        )
    values, vectors = np.linalg.eigh(metric)
    keep = values > 1e-10 * values[-1]
    if int(keep.sum()) != original_rank or original_rank != 488:
        raise ValueError("corrected metric changed the frozen rank")
    root = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    mo = np.empty((488, 230, 230))
    helper.extended_mo_ld(230, 488, raw.ctypes.data, c.ctypes.data, mo.ctypes.data)
    # MO-first extended accumulation isolates cancellation. The final small
    # whitening dot uses binary64; no pair projection is applied to either raw
    # values or Bmo. The independently defined symmetric root is retained.
    bmo = (root.T @ mo.reshape(488, -1)).reshape(488, 230, 230)
    np.savez(
        output / "reference.npz",
        C=c,
        Bmo=bmo,
        raw_three_center=raw,
        metric=metric,
        metric_eigenvalues=values,
        metric_inverse_sqrt=root,
    )
    report.update(
        metric_rank=int(keep.sum()),
        metric_absolute_cutoff=float(1e-10 * values[-1]),
        reference_sha256=digest(output / "reference.npz"),
        coefficient_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
        pair_projection="none",
        benzene264="not qualified",
    )
    (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output.resolve()
    if args.worker:
        worker(package, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    shim, library = build_helpers(output)
    (output / "libcint.json").write_text(
        json.dumps({"path": str(library), "sha256": digest(library)}, indent=2) + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--package",
            str(package),
            "--output",
            str(output),
            "--worker",
        ],
        env={**os.environ, "LD_PRELOAD": str(shim)},
        check=True,
    )


if __name__ == "__main__":
    main()
