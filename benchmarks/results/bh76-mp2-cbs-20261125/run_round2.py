"""Retry the frozen first-round failures with the production CUDA build."""

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path

import run_gpu as native

ENDPOINT_SECONDS = 7200
NUMERIC_BYTES = 64 << 30
SCALAR_WORK = 50_000_000_000_000
WORK_ENV = "GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK"
ANGULAR = "angular_momentum_refusal"


def point(system: dict, task: dict) -> dict:
    from generativeqc import Calculator, CBSBasis, CCSDTCBSCalculator
    from pyscf.gto import charge

    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot

    recipe = task["recipe"]
    low, high = native.RECIPES[recipe]
    geometry = native.bohr_geometry(system)
    elements = [charge(symbol) for symbol, _ in system["angstrom"]]
    names = {n: native.basis_name(system, n) for n in (2, low, high)}
    aux_names = {n: name + "-jkfit" for n, name in names.items()}
    orbital = {n: basis_snapshot(name, elements) for n, name in names.items()}
    aux = {n: basis_snapshot(name, elements) for n, name in aux_names.items()}
    molecular = {"charge": system["charge"], "multiplicity": system["multiplicity"]}
    settings = {
        "orbital_bases": names,
        "auxiliary_bases": aux_names,
        "initial_guess": "auto",
        "scf_damping_override": None,
        "scf_stability_search": False,
    }
    diagnosis = task["round1_diagnosis"]
    gaps: list[str] = []
    if recipe == "TQ_D" and system["multiplicity"] != 1:
        # The QZ energy-only component has the same owner and basis operators
        # as this UMP2 capability report. It does not execute integrals or SCF.
        calculator = Calculator(
            method="ump2",
            basis=orbital[high],
            auxiliary_basis=aux[high],
            density_fitting="full_cuda",
            device="cuda",
            precision="fp64",
        )
        report = calculator.basis_capability_report(geometry, compute_forces=False)
        settings["high_basis_preflight"] = {
            "basis": names[high],
            "method": "ump2",
            "eligible": report["eligible"],
            "reasons": report["reasons"],
        }
        if any("supports only l<=" in reason.lower() for reason in report["reasons"]):
            gaps.append(ANGULAR)
    if diagnosis == ANGULAR and ANGULAR not in gaps:
        raise ValueError(
            f"first-round angular refusal not reproduced by preflight: {settings}"
        )
    if gaps:
        return {
            **settings,
            "status": "capability_gap",
            "native_attempted": False,
            "gaps": gaps,
        }

    start = time.perf_counter()
    try:
        owner = CCSDTCBSCalculator(
            (CBSBasis(orbital[low], low), CBSBasis(orbital[high], high)),
            correction_basis=orbital[2],
            auxiliary_bases=(aux[low], aux[high]),
            correction_auxiliary_basis=aux[2],
            hf_reference="df",
            device="cuda",
            policy="fp64",
            preconvergence="fp64",
            calculator_options={"correlation_memory_budget_bytes": NUMERIC_BYTES},
        )
        result = owner.singlepoint(geometry, **molecular, properties=("energy",))
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        return {
            **settings,
            "native_attempted": True,
            "status": "failed",
            "wall_seconds": time.perf_counter() - start,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    return {
        **settings,
        "native_attempted": True,
        "status": "ok",
        "wall_seconds": elapsed,
        "result": asdict(result),
    }


def rank(output: Path, visible: str, tasks: Queue, plan_sha: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["GENERATIVEQC_STATIONARY_CACHE"] = str(output / "runtime-cache")
    os.environ["GENERATIVEQC_JIT_SCCACHE_ROOT"] = str(output / "jit-owners")
    os.environ[WORK_ENV] = str(SCALAR_WORK)
    output.mkdir()
    receipt = {
        **native.provenance(),
        "round2_driver_sha256": native.sha(__file__),
        "plan_sha256": plan_sha,
        "max_scalar_work": SCALAR_WORK,
        "numeric_memory_bytes": NUMERIC_BYTES,
        "endpoint_limit_seconds": ENDPOINT_SECONDS,
        "build_variant": "paper_h100_production",
    }
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (
        (output / "records.jsonl").open("x") as journal,
        (output / "process.txt").open("x") as log,
        tempfile.TemporaryDirectory(dir=output) as temporary,
    ):
        path = Path(temporary) / "point.json"
        for task in iter(tasks.get, None):
            path.unlink(missing_ok=True)
            native.append(
                journal, {**task, "status": "started", "utc_unix": time.time()}
            )
            label = f"{task['recipe']} {task['system']}"
            log.write(f"\n--- {label} ---\n")
            log.flush()
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "point", json.dumps(task), str(path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=ENDPOINT_SECONDS,
                    check=False,
                )
                if proc.returncode == 0 and path.is_file():
                    result = json.loads(path.read_text())
                    path.unlink()
                else:
                    result = {"status": "process_failed", "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "limit_seconds": ENDPOINT_SECONDS}
            native.append(
                journal,
                {**task, **result, "process_wall_seconds": time.perf_counter() - start},
            )
            print(output.name, label, result["status"], flush=True)


def launch(output: Path, plan_path: Path, cards: int) -> None:
    from cupy.cuda import runtime

    plan = json.loads(plan_path.read_text())
    assert plan["inputs_sha256"] == native.sha(native.ROOT / "inputs.json")
    assert len({(p["system"], p["recipe"]) for p in plan["points"]}) == len(
        plan["points"]
    )
    assert runtime.getDeviceCount() == cards
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else list(map(str, range(cards)))
    assert len(set(devices)) == cards
    output.mkdir()
    context = get_context("spawn")
    tasks = context.Queue()
    for task in [*plan["points"], *([None] * cards)]:
        tasks.put(task)
    workers = [
        context.Process(
            target=rank, args=(output / f"rank-{i}", gpu, tasks, native.sha(plan_path))
        )
        for i, gpu in enumerate(devices)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        tasks.cancel_join_thread()
        tasks.close()
        raise RuntimeError("one or more GPU workers failed, inspect their journals")
    tasks.close()
    tasks.join_thread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("campaign")
    campaign.add_argument("output", type=Path)
    campaign.add_argument("plan", type=Path)
    campaign.add_argument("cards", type=int)
    single = commands.add_parser("point")
    single.add_argument("task")
    single.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "point":
        assert os.environ[WORK_ENV] == str(SCALAR_WORK)
        work = json.loads(args.task)
        systems = json.loads((native.ROOT / "inputs.json").read_text())["systems"]
        faulthandler.dump_traceback_later(300, repeat=True)
        try:
            result = point(systems[work["system"]], work)
        finally:
            faulthandler.cancel_dump_traceback_later()
        args.output.write_text(json.dumps(native.record_json(result), allow_nan=False))
    else:
        launch(args.output, args.plan, args.cards)
