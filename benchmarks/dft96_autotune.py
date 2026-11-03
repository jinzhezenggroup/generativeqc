"""Qualify resident-session DFT schedules without activating runtime profiles.

Run the controller inside a finite Slurm GPU allocation. Each subprocess owns
one Calculator and one frozen converged density. Two resident owners execute
strictly serial, alternating-order complete energy/analytic-force samples.
This amortizes cold convergence, unlike the CLI's fresh process per sample;
the final holdout uses two new owners. Scientific inputs and admission gates
are unchanged, and no experiment installs a profile or native-policy recipe.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from generativeqc.profiles import atomic_json, canonical_hash, file_hash
from typing_extensions import Self

from benchmarks._retention import raw_output_path


def configurations(tile: int | None, policy: dict) -> dict:
    """Represent only execution choices; all scientific controls live elsewhere."""
    return {"tile_points": tile, "environment": policy}


def reference_gate(reference: dict) -> dict:
    """Check the actual full-Fock residual, not a solver's configured tolerance."""
    residual = reference.get("orbital_gradient_norm")
    valid = (
        type(residual) in (int, float)
        and math.isfinite(residual)
        and 0 <= residual <= 1e-10
    )
    return {
        "passed": reference.get("converged") is True and valid,
        "orbital_gradient_norm": residual,
        "maximum_orbital_gradient_norm": 1e-10,
        "scope": "independent CPU reference convergence and recomputed full-Fock orbital gradient",
    }


def worker(workload: dict, configuration: dict, control: socket.socket) -> None:
    """Keep one native owner alive, freezing density before any timed request."""
    from generativeqc._dft_autotune import execution_record
    from generativeqc.calculator import Calculator
    from generativeqc.ks import GridSpec, KsOptions
    from generativeqc.profiles import compatibility_identity, probe_device
    from generativeqc_compiler.common.resources import ResourceBudget

    stream = control.makefile("rw")
    calculator = Calculator(
        method=workload["method"],
        basis=workload["basis"],
        device="cuda",
        device_id=workload["device_id"],
        basis_representation=workload["representation"],
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        ks_options=KsOptions(
            grid=GridSpec.treutler(workload["grid_level"]),
            tile_points=configuration["tile_points"],
        ),
        resource_budget=ResourceBudget(),
    )
    native_workload = {**workload, "tile_points": configuration["tile_points"]}
    probe = probe_device(calculator._library, workload["device_id"])
    if probe["source_identity"] != workload["source_identity"]:
        raise ValueError("native library differs from the requested source identity")
    with calculator.prepare_batch(
        [workload["atoms"]],
        charges=[workload["charge"]],
        multiplicities=[workload["multiplicity"]],
        warm_start=True,
    ) as prepared:
        started = time.perf_counter()
        cold = prepared.execute(properties=("energy", "forces"), strict=True)
        cold_seconds = time.perf_counter() - started
        prepared.set_warm_start_updates(False)
        prepared.execute(properties=("energy", "forces"), strict=True)
        stream.write(
            json.dumps({"ready": True, "identity": compatibility_identity(probe)})
            + "\n"
        )
        stream.flush()
        for line in stream:
            request = json.loads(line)
            if request["command"] == "close":
                break
            if request["command"] != "sample":
                raise ValueError("unknown resident-worker command")
            started = time.perf_counter()
            result = prepared.execute(properties=("energy", "forces"), strict=True)
            elapsed = time.perf_counter() - started
            record = execution_record(native_workload, calculator, prepared, result)
            native_schedule = record["schedule"]
            schedule = {
                "xc": native_schedule,
                "environment": configuration["environment"],
            }
            record.update(
                seconds=elapsed,
                cold_seconds=cold_seconds,
                cold_iterations=[item.iterations for item in cold.items],
                iterations=[item.iterations for item in result.items],
                energies=[item.energy for item in result.items],
                forces=[item.forces.tolist() for item in result.items],
                converged=all(item.converged for item in result.items),
                backend=[item.executed_backend for item in result.items],
                native_xc_schedule=native_schedule,
                schedule=schedule,
                schedule_identity=canonical_hash(schedule),
                pair_id=request["pair_id"],
                interleaved=True,
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()


class Resident:
    """Own one isolated worker with finite control and teardown timeouts."""

    def __init__(
        self, library: Path, plan: Path, configuration: dict, directory: Path
    ) -> None:
        from generativeqc.execution_tune import worker_environment

        directory.mkdir(parents=True, exist_ok=True)
        self.socket, child = socket.socketpair()
        self.socket.settimeout(900)
        self.stream = self.socket.makefile("rw")
        self.log = (directory / "worker.log").open("w")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.dft96_autotune",
                str(plan),
                "--worker-fd",
                str(child.fileno()),
                "--configuration",
                json.dumps(configuration),
            ],
            env=worker_environment(library, "jk", configuration["environment"]),
            pass_fds=(child.fileno(),),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        child.close()
        try:
            self.identity = self.receive()["identity"]
        except BaseException:
            self.close()
            raise

    def receive(self) -> dict:
        """Fail closed on a dead child rather than substituting missing samples."""
        line = self.stream.readline()
        if not line:
            raise RuntimeError(f"resident worker exited; see {self.log.name}")
        return json.loads(line)

    def sample(self, pair_id: int) -> dict:
        self.stream.write(json.dumps({"command": "sample", "pair_id": pair_id}) + "\n")
        self.stream.flush()
        return self.receive()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                try:
                    self.stream.write('{"command":"close"}\n')
                    self.stream.flush()
                    self.process.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=15)
        finally:
            self.stream.close()
            self.socket.close()
            self.log.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def compare(
    baseline: Resident,
    candidate: Resident,
    reference: dict,
    *,
    repeats: int,
    same_xc: bool,
) -> dict:
    """Retain every alternating-order pair and apply independent acceptance gates."""
    from generativeqc._dft_autotune import numerical_gate, resource_gate
    from generativeqc.autotune import dft_endpoint_gate

    samples = {"baseline": [], "candidate": []}
    owners = {"baseline": baseline, "candidate": candidate}
    for pair_id in range(repeats):
        order = (
            ("baseline", "candidate") if pair_id % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            sample = owners[side].sample(pair_id)
            samples[side].append(sample)
            print(
                f"pair={pair_id} side={side} seconds={sample['seconds']:.6f}",
                flush=True,
            )
    endpoint = dft_endpoint_gate(samples["baseline"], samples["candidate"])
    all_samples = samples["baseline"] + samples["candidate"]
    work = [
        [
            (item["grid_points"], item["fock_builds"], item["coulomb_exchange_work"])
            for item in sample["dft_work"]
        ]
        for sample in all_samples
    ]
    complete = all(
        item["coulomb_exchange_work"].get("complete") is True
        for sample in all_samples
        for item in sample["dft_work"]
    )
    invariant_work = complete and all(item == work[0] for item in work)
    if same_xc:
        invariant_work = invariant_work and all(
            sample["native_xc_schedule"] == all_samples[0]["native_xc_schedule"]
            for sample in all_samples
        )
    evidence = {
        "endpoint": endpoint,
        "numerical": numerical_gate(all_samples, reference),
        "work": {
            "passed": invariant_work,
            "observed": work,
            "requires_same_xc": same_xc,
        },
        "identity": {"passed": baseline.identity == candidate.identity},
    }
    for side in ("baseline", "candidate"):
        evidence[f"{side}_resources"] = resource_gate(
            samples[side],
            schedule_hash=endpoint[f"{side}_schedule_hash"],
            source_hash=endpoint[f"{side}_source_hash"],
        )
    evidence["passed"] = all(value["passed"] for value in evidence.values())
    return evidence


def run(
    plan_path: Path,
    output: Path,
    *,
    repeats: int,
    requested_configuration: dict | None = None,
) -> dict:
    """Search XC then J/K, requalifying the final composition against defaults."""
    output = raw_output_path(output)
    if not os.environ.get("SLURM_JOB_ID") or repeats < 5:
        raise ValueError(
            "at least five paired repeats inside a Slurm allocation are required"
        )
    plan = json.loads(plan_path.read_text())
    workload = plan["workload"]
    if len(workload["atoms"]) != 96 or workload["batch"] != 1:
        raise ValueError(
            "this qualification requires the explicit 96-atom singleton workload"
        )
    from generativeqc.autotune import source_identity

    canonical_source = source_identity(Path(__file__).resolve().parents[1])
    if canonical_source != workload["source_identity"]:
        raise ValueError("source checkout differs from the requested native build")
    library = Path(plan["library"])
    if file_hash(library) != plan["library_hash"]:
        raise ValueError("native library changed since the plan was prepared")
    reference_path = Path(plan["reference"])
    reference = json.loads(reference_path.read_text())
    reference_qualification = reference_gate(reference)
    if not reference_qualification["passed"]:
        raise ValueError("independent reference failed its strict residual gate")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "generativeqc.dft96-resident-qualification.v1",
        "plan": plan,
        "reference": reference,
        "reference_hash": file_hash(reference_path),
        "reference_qualification": reference_qualification,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "protocol": "serial alternating pairs between isolated resident frozen-density owners; fresh final holdout",
        "repeats": repeats,
        "trials": [],
        "selected": None,
        "canonical_source_identity": canonical_source,
        "configuration_requested": requested_configuration,
    }
    incumbent = requested_configuration or configurations(None, {})
    try:
        for component in () if requested_configuration is not None else ("xc", "jk"):
            if component == "xc":
                candidates = [configurations(tile, {}) for tile in (512, 1024, 2048)]
            else:
                candidates = [
                    configurations(incumbent["tile_points"], policy)
                    for policy in (
                        {
                            "GENERATIVEQC_DISABLE_MD_J": "1",
                            "GENERATIVEQC_DIRECT_J_FOCK_LOWERING": "rys",
                        },
                        {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "fill"},
                        {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "primitive"},
                    )
                ]
            best, best_speedup = incumbent, 1.0
            print(f"stage={component} incumbent={incumbent}", flush=True)
            with Resident(
                library, plan_path, incumbent, output / component / "baseline"
            ) as baseline:
                report["identity"] = baseline.identity
                for index, configuration in enumerate(candidates):
                    trial = {"component": component, "configuration": configuration}
                    report["trials"].append(trial)
                    try:
                        with Resident(
                            library,
                            plan_path,
                            configuration,
                            output / component / f"candidate-{index}",
                        ) as candidate:
                            trial["evidence"] = compare(
                                baseline,
                                candidate,
                                reference,
                                repeats=repeats,
                                same_xc=component == "jk",
                            )
                        speedup = trial["evidence"]["endpoint"]["speedup"]
                        print(
                            f"configuration={configuration} passed={trial['evidence']['passed']} speedup={speedup:.6f}",
                            flush=True,
                        )
                        if trial["evidence"]["passed"] and speedup > best_speedup:
                            best, best_speedup = configuration, speedup
                    except (OSError, ValueError, RuntimeError) as error:
                        trial["failure"] = str(error)
                    atomic_json(output / "report.json", report)
            incumbent = copy.deepcopy(best)
        if incumbent != configurations(None, {}):
            print(f"fresh holdout configuration={incumbent}", flush=True)
            with ExitStack() as stack:
                baseline = stack.enter_context(
                    Resident(
                        library,
                        plan_path,
                        configurations(None, {}),
                        output / "holdout" / "baseline",
                    )
                )
                candidate = stack.enter_context(
                    Resident(
                        library, plan_path, incumbent, output / "holdout" / "candidate"
                    )
                )
                report["identity"] = baseline.identity
                report["holdout"] = compare(
                    baseline, candidate, reference, repeats=repeats, same_xc=False
                )
            if report["holdout"]["passed"]:
                report["selected"] = incumbent
        return report
    finally:
        atomic_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=raw_output_path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker-fd", type=int)
    parser.add_argument("--configuration", type=json.loads)
    args = parser.parse_args()
    if args.worker_fd is not None:
        plan = json.loads(args.plan.read_text())
        worker(
            plan["workload"], args.configuration, socket.socket(fileno=args.worker_fd)
        )
    else:
        if args.output is None:
            parser.error("--output is required")
        run(
            args.plan.resolve(),
            args.output.resolve(),
            repeats=args.repeats,
            requested_configuration=args.configuration,
        )


if __name__ == "__main__":
    main()
