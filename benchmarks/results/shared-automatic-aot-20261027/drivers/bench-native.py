"""Complete SCF+E+F timings; receipt hashing and diagnostics stay outside timing."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/python")
import generativeqc.automatic_artifacts as artifact_api
from generativeqc import Calculator, KsOptions
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from generativeqc.ks_diagnostics import read_ks_transport_diagnostic
from generativeqc_compiler.common.provenance import file_hash
from test_libxc_native_cuda_forces import ATOMS, GRID, MOVED, PROPERTIES

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
# This finite benchmark pins one immutable DSO before/after the entire run.
# Each diagnostic still authenticates every callback/descriptor mapping and the
# complete inode/size/mtime/ctime tuple. Reuse SHA work only inside this bracket,
# outside endpoint timers; ordinary runtime diagnostics retain their own policy.
library_path = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
library_sha256 = file_hash(library_path)
receipt_memo = {}
loaded_artifact = artifact_api._loaded_artifact


def frozen_artifact(address, memo=None):
    value = loaded_artifact(address, receipt_memo)
    if value is not None:
        assert value["sha256"] == library_sha256
    return value


artifact_api._loaded_artifact = frozen_artifact
WATER = [("O", (0.0, 0.0, 0.0)), ("H", (0.2, 1.5, 1.1)), ("H", (-0.3, -1.4, 1.0))]
WATER_MOVED = [*WATER[:2], ("H", (-0.27, -1.4, 1.0))]
rows = []
cases = [
    ("h4", method, ATOMS, MOVED, 2 if "uks" in method else 1)
    for method in (
        "pbe-rks",
        "pbe-uks",
        *[
            f"libxc-{spin}:{name}"
            for name in ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA")
            for spin in ("rks", "uks")
        ],
    )
]
cases += [
    ("water", f"libxc-{spin}:{name}", WATER, WATER_MOVED, 1)
    for name in ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_C_TPSS")
    for spin in ("rks", "uks")
]


def calculator(method):
    return Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        initial_guess=None,
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        screening_tolerance=1e-12,
        ks_options=KsOptions(grid=GRID, tile_points=37),
    )


if os.environ.get("PACKAGE_SCOPE") == "full":
    cases = [case for case in cases if case[1].startswith("libxc-")]
# Initialize the device context through a common curated energy endpoint.
# Neither package prewarms an automatic point/force program before its cold case.
with calculator("pbe-rks").prepare_batch([ATOMS]) as batch:
    batch.execute(properties=("energy",), strict=True)
for system, method, atoms, moved, multiplicity in cases:
    for repeat in range(2):
        start = time.perf_counter()
        calc = calculator(method)
        with calc.prepare_batch(
            [atoms],
            charges=[multiplicity - 1],
            multiplicities=[multiplicity],
            warm_start=True,
        ) as batch:
            setup = time.perf_counter() - start
            previous_transport = None
            for phase, geometry in [
                ("cold", atoms),
                ("warm", atoms),
                ("warm", atoms),
                ("warm", atoms),
                ("moved", moved),
            ]:
                start = time.perf_counter()
                result = batch.execute(
                    coordinates=([p for _, p in geometry],),
                    properties=PROPERTIES,
                    strict=True,
                ).items[0]
                seconds = (
                    time.perf_counter() - start + (setup if phase == "cold" else 0)
                )
                assert result.executed_backend == "cuda" and result.converged
                diagnostic = result.ks_diagnostic.to_payload()
                transport = read_ks_transport_diagnostic(
                    batch._library, batch._batch, index=0
                ).to_payload()
                delta = {
                    key: value - (previous_transport[key] if previous_transport else 0)
                    for key, value in transport.items()
                }
                previous_transport = transport
                artifact = (
                    read_automatic_artifact_diagnostic(batch)
                    if method.startswith("libxc-")
                    else None
                )
                if artifact:
                    assert (
                        artifact["force_completed"]
                        and artifact["fallback_reason"] == "none"
                    )
                rows.append(
                    dict(
                        system=system,
                        method=method,
                        repeat=repeat,
                        phase=phase,
                        seconds=seconds,
                        energy=result.energy,
                        forces=result.forces.tolist(),
                        iterations=result.iterations,
                        fock_builds=diagnostic["fock_builds"],
                        coulomb_exchange_work=diagnostic["coulomb_exchange_work"],
                        transport_delta=delta,
                        transport_cumulative=transport,
                        artifact=artifact,
                        warm_used=result.warm_start_used,
                        warm_fallback=result.warm_start_fallback,
                        precision=result.precision,
                    )
                )
        Path(sys.argv[1]).write_text(
            json.dumps(
                dict(
                    library=os.environ["GENERATIVEQC_LIBRARY"],
                    library_sha256=library_sha256,
                    complete=False,
                    slurm_job_id=os.environ["SLURM_JOB_ID"],
                    rows=rows,
                ),
                indent=2,
            )
            + "\n"
        )
    print(system, method, "completed", flush=True)

assert file_hash(library_path) == library_sha256
artifact_api._loaded_artifact = loaded_artifact
final = json.loads(Path(sys.argv[1]).read_text())
final["complete"] = True
final["receipt_policy"] = (
    "per-query mapping/stat authentication; finite immutable-DSO digest bracket; all diagnostics excluded from timers"
)
Path(sys.argv[1]).write_text(json.dumps(final, indent=2) + "\n")
