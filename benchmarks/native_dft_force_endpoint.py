"""Source-bound native-only complete CPU/CUDA E/F timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc_compiler.method import resolve_method

from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from generativeqc.batch import BatchItemResult, PreparedBatch


def unavailable_python_force(*arguments: object, **keywords: object) -> NoReturn:
    """Fail even if an ostensibly native context attempts a Python fallback."""
    raise RuntimeError("native endpoint escaped to a Python scientific force owner")


def summarize(item: BatchItemResult) -> dict[str, object]:
    """Convert complete public results inside the measured endpoint."""
    expected_backend = "cuda" if arguments.backend == "cuda" else "cpu_reference"
    if (
        not item.succeeded
        or not item.converged
        or item.forces is None
        or item.executed_backend != expected_backend
    ):
        raise RuntimeError("no converged complete native E/F result")
    diagnostic = item.ks_diagnostic
    return {
        "energy": item.energy,
        "forces": item.forces.tolist(),
        "iterations": item.iterations,
        "fock_builds": item.fock_builds,
        "warm_start_used": item.warm_start_used,
        "executed_backend": item.executed_backend,
        "grid_points": diagnostic.grid_points if diagnostic else None,
        "tile_points": diagnostic.tile_points if diagnostic else None,
        "coulomb_exchange_work": diagnostic.coulomb_exchange_work
        if diagnostic
        else None,
        "nonlocal_work": diagnostic.nonlocal_work if diagnostic else None,
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=11)
parser.add_argument("--warmups", type=int, default=3)
parser.add_argument("--output", type=raw_output_path, required=True)
parser.add_argument("--case", default="all")
parser.add_argument("--profiled", action="store_true")
parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
arguments = parser.parse_args()
if arguments.backend == "cuda":
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
assert not os.environ.get("GENERATIVEQC_DF_TRACE")
water = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]
cases = [
    (
        f"{method}-{spin}-{int(fitted)}",
        method,
        spin,
        fitted,
        water,
        "method-defined" if method == "r2scan-3c" else "sto-3g",
    )
    for method in (
        "lda",
        "pbe",
        "pbe0",
        "b3lyp",
        *(("pbe50",) if arguments.backend == "cpu" else ()),
        "r2scan",
        "wb97m-v",
        *(("m06-2x", "mn15") if arguments.backend == "cuda" else ()),
        "pbe-d3(bj)",
        "pbe0-d4(bj-eeq-atm)",
        "r2scan-3c",
    )
    for spin in (0, 1)
    for fitted in (False, True)
]
for count in (36, 48) if arguments.backend == "cuda" else ():
    cluster = [
        (
            element,
            (point[0] + 7.0 * (index % 4), point[1] + 7.0 * (index // 4), point[2]),
        )
        for index in range(count // 3)
        for element, point in water
    ]
    cases.append((f"pbe-water{count}-df-d", "pbe", 0, True, cluster, "def2-svp"))
for functional in (
    (
        "LDA_C_VWN_4",
        "LDA_C_CHACHIYO",
        "LDA_C_PZ_MOD",
        "GGA_X_APBE",
        "GGA_X_PBE_SOL",
        "GGA_X_RPBE",
        "GGA_X_B86",
        "GGA_C_APBE",
        "MGGA_X_LTA",
        "MGGA_X_GVT4",
        "MGGA_C_TPSS",
    )
    if arguments.backend == "cuda"
    else ()
):
    for spin in (0, 1):
        atoms = [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.1, 0.2, 1.4)),
            ("H", (1.6, 0.4, 0.3)),
            ("H", (1.8, 0.3, 1.8)),
        ]
        cases.append(
            (
                f"libxc-{functional}-{spin}-0",
                f"libxc:{functional}",
                spin,
                False,
                atoms,
                "sto-3g",
            )
        )

library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
metadata = {
    "protocol": "generativeqc-native-dft-force-v2",
    "d3_selector": "MethodIR resolved inside timed construction for both variants",
    "variant": os.environ["R1_VARIANT"],
    "base_revision": os.environ["R1_BASE_REVISION"],
    "source_census_sha256": hashlib.sha256(
        Path(os.environ["R1_SOURCE_MANIFEST"]).read_bytes()
    ).hexdigest(),
    "frontend_census_sha256": hashlib.sha256(
        Path(os.environ["R1_FRONTEND_MANIFEST"]).read_bytes()
    ).hexdigest(),
    "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "samples": arguments.samples,
    "warmups": arguments.warmups,
    "grid": "named product 12/4/8; automatic product 24/8/16 tile37; same independent nonlocal prescription",
    "precision": "fp64",
    "energy_tolerance": {"named": 1e-12, "automatic": 1e-13},
    "density_tolerance": {"named": 1e-10, "automatic": 1e-12},
    "screening_tolerance": {"named": 1e-13, "automatic": 1e-12},
    "df_range_exchange_environment": os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE"),
    "tracing": arguments.profiled,
    "timing": "cold construction/preparation/E/F/public conversion/cleanup; warm/moved execute and public conversion; JSON receipt I/O excluded",
    "artifact_state": "existing caches, explicit warmups; no runtime scientific fallback",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "cases": [
        case[0]
        for case in cases
        if arguments.case == "all" or case[0] in arguments.case.split(",")
    ],
}
with arguments.output.open("w", buffering=1) as output:
    output.write(json.dumps({"metadata": metadata}) + "\n")
    for name, method, spin, fitted, atoms, basis_name in cases:
        if name not in metadata["cases"]:
            continue

        def prepare(
            method: str = method,
            spin: int = spin,
            fitted: bool = fitted,
            atoms: list[tuple[str, tuple[float, float, float]]] = atoms,
            basis_name: str = basis_name,
        ) -> PreparedBatch:
            """Include model resolution, construction and native admission."""
            automatic = method.startswith("libxc:")
            selector = (
                f"libxc-{'uks' if spin else 'rks'}:{method.split(':', 1)[1]}"
                if automatic
                else f"{method}-{'uks' if spin else 'rks'}"
            )
            if method == "pbe-d3(bj)":
                # The parent supports MethodIR, not the newly repaired D3 string.
                # Resolve both variants here so cold timing charges the same API.
                selector = resolve_method(
                    "PBE-D3(BJ)", spin="polarized" if spin else "unpolarized"
                )
            calculator = Calculator(
                method=selector,
                **({"basis": basis_name} if method != "r2scan-3c" else {}),
                basis_representation="spherical",
                device=arguments.backend,
                density_fitting=arguments.backend if fitted else "none",
                ks_options=KsOptions(
                    grid=GridSpec(24, 8, 16) if automatic else GridSpec(12, 4, 8),
                    nonlocal_grid=GridSpec(12, 4, 8) if method == "wb97m-v" else None,
                    tile_points=37 if automatic else None,
                ),
                **({"initial_guess": None} if automatic else {}),
                max_iterations=300 if automatic else 200,
                energy_tolerance=1e-13 if automatic else 1e-12,
                density_tolerance=1e-12 if automatic else 1e-10,
                screening_tolerance=1e-12 if automatic else 1e-13,
            )
            batch = calculator.prepare_batch(
                [atoms], charges=[spin], multiplicities=[spin + 1]
            )
            try:
                if not batch._native_dft_force_eligible():
                    raise RuntimeError(
                        "prepared context does not admit complete native forces"
                    )
                batch._public_dft_cpu_force = unavailable_python_force
                batch._public_dft_cuda_force = unavailable_python_force
                return batch
            except BaseException:
                batch.close()
                raise

        def write_record(
            phase: str,
            sample: int,
            started: float,
            payload: dict[str, object],
            name: str = name,
        ) -> None:
            """Stop timing before serializing the external evidence receipt."""
            elapsed = time.perf_counter() - started
            if sample >= 0:
                output.write(
                    json.dumps(
                        {
                            "case": name,
                            "phase": phase,
                            "sample": sample,
                            "seconds": elapsed,
                            "native": True,
                            "process_peak_rss_kib": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss,
                            **payload,
                        }
                    )
                    + "\n"
                )

        try:
            for sample in range(-arguments.warmups, arguments.samples):
                started = time.perf_counter()
                with prepare() as batch:
                    payload = summarize(
                        batch.execute(
                            strict=True, properties=("energy", "forces")
                        ).items[0]
                    )
                write_record("cold", sample, started, payload)
            positions = np.array([point for _, point in atoms])
            moved = positions.copy()
            if method.startswith("libxc:"):
                moved[1, 2] += 0.07
            else:
                moved[0, 2] += 0.02
            with prepare() as batch:
                batch.execute(strict=True, properties=("energy", "forces"))
                for phase in ("warm", "moved"):
                    for sample in range(-arguments.warmups, arguments.samples):
                        coordinates = (
                            None
                            if phase == "warm"
                            else [moved if sample % 2 else positions]
                        )
                        started = time.perf_counter()
                        payload = summarize(
                            batch.execute(
                                coordinates,
                                strict=True,
                                properties=("energy", "forces"),
                            ).items[0]
                        )
                        write_record(phase, sample, started, payload)
            print("PASS", name, flush=True)
        except Exception as failure:  # noqa: BLE001
            # Preserve every failed endpoint; the auditor refuses partial runs.
            output.write(
                json.dumps(
                    {"case": name, "error": f"{type(failure).__name__}: {failure}"}
                )
                + "\n"
            )
            print("FAIL", name, repr(failure), flush=True)
