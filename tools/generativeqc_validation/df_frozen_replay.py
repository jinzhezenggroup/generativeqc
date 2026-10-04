"""Replay the frozen #1840 CUDA factor source with full precision provenance.

Builds small validation interposers, captures high/low raw tiles and the actual
unprojected GPU factors, and requires the unchanged seven original gates plus
unprojected Bov/Bvv gates. Invoke only within a finite Slurm GPU allocation.
The existing package and corrected reference are inputs; no RHF/CC is run.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite frozen factor/block")
    delta = abs(actual - reference)
    tolerance = 3e-10 + 3e-10 * abs(reference)
    index = np.unravel_index(delta.argmax(), delta.shape)
    return {
        "max_abs": float(delta[index]),
        "index": list(map(int, index)),
        "actual": float(actual[index]),
        "reference": float(reference[index]),
        "failures": int(np.count_nonzero(delta > tolerance)),
        "max_scaled": float(np.max(delta / tolerance)),
    }


def worker(
    package: Path, reference_path: Path, output: Path, library_path: Path
) -> None:
    """Run native publication and audit every captured unprojected element."""
    from generativeqc import Atom, Primitive, Shell

    from tools.generativeqc_posthf.sources import NativeSource

    rows = iter((package / "frozen-input.txt").read_text().splitlines())
    na, ns, qs, budget = map(int, next(rows).split())
    atoms = []
    for _ in range(na):
        r = next(rows).split()
        atoms.append(Atom(int(r[0]), tuple(map(float, r[1:]))))

    def shells(count: int) -> tuple[Shell, ...]:
        result = []
        for _ in range(count):
            atom, l, np_ = map(int, next(rows).split())
            result.append(
                Shell(
                    atom,
                    l,
                    tuple(
                        Primitive(*map(float, next(rows).split())) for _ in range(np_)
                    ),
                )
            )
        return tuple(result)

    orb, aux = shells(ns), shells(qs)
    with np.load(package / "reference.npz") as z:
        c = np.ascontiguousarray(z["C"])
        original = z["Bmo"]
    with np.load(reference_path) as z:
        if not np.array_equal(c, z["C"]):
            raise ValueError("corrected reference changed the frozen orbital frame")
        b = z["Bmo"]
    n, o, v, q = 230, 9, 221, 488
    names = ["Boo", "Bov", "Bvv", "ovov", "ovvo", "oovv", "ovoo", "oooo"]
    shapes = [
        (q, o, o),
        (q, o, v),
        (q, v, v),
        (o, v, o, v),
        (o, v, v, o),
        (o, o, v, v),
        (o, v, o, o),
        (o, o, o, o),
    ]
    sizes = [int(np.prod(x)) for x in shapes]
    out = np.full(sum(sizes), np.nan)
    counts = np.zeros(12, dtype=np.uintp)
    values = np.zeros(7)
    error = ct.create_string_buffer(4096)
    dll = ct.CDLL(str(output / "source_probe.so"))
    call = dll.df_cc_source_probe
    call.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    with NativeSource(
        atoms, orb, auxiliary_basis=aux, representation="spherical"
    ) as source:
        start = time.perf_counter()
        status = call(
            source._handle,
            c.ctypes.data,
            o,
            budget,
            out.ctypes.data,
            out.size,
            counts.ctypes.data,
            values.ctypes.data,
            error,
            len(error),
        )
        seconds = time.perf_counter() - start
        if status:
            raise RuntimeError(error.value.decode())
    native = {}
    offset = 0
    for name, shape, size in zip(names, shapes, sizes):
        native[name] = out[offset : offset + size].reshape(shape)
        offset += size
    capture = ct.CDLL(str(output / "capture.so")).copy_frozen_capture
    capture.argtypes = [ct.c_void_p] * 8
    capture.restype = ct.c_int
    metric = np.empty((q, q))
    root = np.empty((q, q), order="F")
    eigen = np.empty(q)
    raw = np.empty((n, n, q))
    low = np.empty_like(raw)
    braw = np.empty_like(raw)
    ccounts = np.zeros(7, dtype=np.uintp)
    status = capture(
        *[x.ctypes.data for x in (metric, root, eigen, raw, low, braw, ccounts)], error
    )
    if status:
        raise RuntimeError(error.value.decode())
    unprojected = braw.transpose(2, 0, 1)
    # Latest master already projects the complete physical factor before packing.
    # Apply that same published convention explicitly on the reference, and also
    # require the captured pre-projection GPU factor to pass without projection.
    bp = 0.5 * (b + b.transpose(0, 2, 1))

    def blocks(bmo: np.ndarray) -> dict[str, np.ndarray]:
        boo, bov, bvo, bvv = (
            bmo[:, :o, :o],
            bmo[:, :o, o:],
            bmo[:, o:, :o],
            bmo[:, o:, o:],
        )
        result = {"Boo": boo, "Bov": bov, "Bvv": bvv}
        for name, left, right in zip(
            names[3:], (bov, bov, boo, bov, boo), (bov, bvo, bvv, boo, boo)
        ):
            result[name] = np.einsum("Qpq,Qrs->pqrs", left, right, optimize=True)
        return result

    expected = blocks(bp)
    gates = {name: compare(native[name], expected[name]) for name in names}
    ungates = {
        name: compare(unprojected[:, x, y], b[:, x, y])
        for name, x, y in [
            ("Bov", slice(None, o), slice(o, None)),
            ("Bvv", slice(o, None), slice(o, None)),
        ]
    }
    np.savez_compressed(
        output / "candidate_native.npz",
        metric=metric,
        root=root,
        eigenvalues=eigen,
        raw=raw,
        raw_low=low,
        Bmo_unprojected=unprojected,
        **native,
    )
    report = {
        "job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": subprocess.check_output(["hostname", "-s"], text=True).strip(),
        "library_sha256": digest(library_path),
        "frozen_frame_sha256": digest(package / "reference.npz"),
        "corrected_reference_sha256": digest(reference_path),
        "capture_source_sha256": digest(
            ROOT / "tests/native/df_frozen_precision_capture.cpp"
        ),
        "original_atol": 3e-10,
        "original_rtol": 3e-10,
        "gates": gates,
        "unprojected_factor_gates": ungates,
        "original_reference_failures": {
            "Bov": compare(unprojected[:, :o, o:], original[:, :o, o:]),
            "Bvv": compare(unprojected[:, o:, o:], original[:, o:, o:]),
        },
        "metric_rank": int(counts[11]),
        "metric_absolute_cutoff": float(values[0]),
        "budget": budget,
        "counts": counts.tolist(),
        "capture_counts": ccounts.tolist(),
        "instrumented_seconds": seconds,
        "timing_is_not_a_benchmark": True,
        "phase_seconds": values[2:].tolist(),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "accepted": int(counts[11]) == q
        and all(
            x["failures"] == 0 for x in list(gates.values()) + list(ungates.values())
        ),
    }
    (output / "candidate-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["accepted"]:
        raise RuntimeError("frozen source failed the unchanged original factor gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "a finite Slurm GPU allocation with assigned visibility is required"
        )
    package, reference, output, library = (
        p.resolve() for p in (args.package, args.reference, args.output, args.library)
    )
    if args.worker:
        worker(package, reference, output, library)
        return
    from tools.generativeqc_validation.df_frozen_oracle import audit_package

    audit_package(package)
    output.mkdir(parents=True, exist_ok=False)
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if not cache or not compiler:
        raise RuntimeError("ccache and C++ compiler required")
    subprocess.run([cache, "--version"], check=True)
    for name, source in [
        ("capture", ROOT / "tests/native/df_frozen_precision_capture.cpp"),
        ("source_probe", ROOT / "tests/native/df_cc_source_probe.cpp"),
    ]:
        obj = output / (name + ".o")
        includes = [
            "-I" + str(p)
            for p in (
                ROOT / "src",
                ROOT / "include",
                library.parent / "generated",
                args.cuda_root / "include",
            )
        ]
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O2",
                "-fPIC",
                *includes,
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
        )
        subprocess.run(
            [
                compiler,
                "-shared",
                str(obj),
                "-Wl,--no-as-needed",
                str(library),
                "-Wl,--as-needed",
                "-L" + str(args.cuda_root / "lib64"),
                "-lcudart",
                "-lcublas",
                "-ldl",
                "-Wl,-rpath," + str(library.parent),
                "-Wl,-rpath," + str(args.cuda_root / "lib64"),
                "-o",
                str(output / (name + ".so")),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        env={
            **os.environ,
            "LD_PRELOAD": str(output / "capture.so"),
            "GENERATIVEQC_LIBRARY": str(library),
        },
        check=True,
    )


if __name__ == "__main__":
    main()
