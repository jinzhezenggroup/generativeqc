"""Qualify explicitly budgeted, resident-owner RHF DF96 execution policies.

This benchmark does not install a profile or change Calculator defaults. Every
candidate keeps the orbital/auxiliary bases, metric cutoff, precision and memory
budget fixed, and uses complete native energy/analytic-force endpoints. The
independent CPU reference is a prerequisite, never a production dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import ExitStack
from pathlib import Path

from generativeqc.profiles import atomic_json, canonical_hash, file_hash

from benchmarks._retention import raw_output_path
from benchmarks.dft96_autotune import Resident, reference_gate


def worker(workload: dict, configuration: dict, control: socket.socket) -> None:
    """Freeze one independently converged native density, not the CPU oracle."""
    from generativeqc._df_autotune import execution_record
    from generativeqc.calculator import Calculator
    from generativeqc.execution_tune import validate_policy
    from generativeqc.profiles import compatibility_identity, probe_device
    from generativeqc_compiler.common.resources import ResourceBudget

    policy = configuration["environment"]
    validate_policy("df", policy)
    started = time.perf_counter()
    calculator = Calculator(
        method="rhf",
        basis=workload["basis"],
        device="cuda",
        device_id=workload["device_id"],
        basis_representation=workload["representation"],
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        resource_budget=ResourceBudget(device_bytes=12 << 30),
        density_fitting="cuda",
        auxiliary_basis=workload["auxiliary_basis"],
        density_fitting_relative_threshold=workload["metric_relative_threshold"],
        density_fitting_memory_budget_bytes=workload["df_memory_bytes"],
    )
    identity = compatibility_identity(
        probe_device(calculator._library, workload["device_id"])
    )
    if identity["source_identity"] != workload["source_identity"]:
        raise ValueError("DF worker native source identity differs")
    stream = control.makefile("rw")
    with calculator.prepare_batch(
        [workload["atoms"]],
        charges=[workload["charge"]],
        multiplicities=[workload["multiplicity"]],
        warm_start=True,
    ) as prepared:
        cold = prepared.execute(properties=("energy", "forces"), strict=True)
        cold_seconds = time.perf_counter() - started

        def record(result: object, seconds: float) -> dict:
            observation = execution_record(workload, prepared, result)
            schedule = {"component": "df", "environment": policy}
            observation.update(
                seconds=seconds,
                iterations=[item.iterations for item in result.items],
                energies=[item.energy for item in result.items],
                forces=[item.forces.tolist() for item in result.items],
                converged=all(item.converged for item in result.items),
                backend=[item.executed_backend for item in result.items],
                scientific_identity=canonical_hash(workload),
                scientific_workload=workload,
                schedule=schedule,
                schedule_identity=canonical_hash(schedule),
            )
            return observation

        cold_record = record(cold, cold_seconds)
        prepared.set_warm_start_updates(False)
        prepared.execute(properties=("energy", "forces"), strict=True)
        stream.write(
            json.dumps({"ready": True, "identity": identity, "cold": cold_record})
            + "\n"
        )
        stream.flush()
        for line in stream:
            request = json.loads(line)
            if request["command"] == "close":
                break
            if request["command"] != "sample":
                raise ValueError("unknown DF resident command")
            started = time.perf_counter()
            result = prepared.execute(properties=("energy", "forces"), strict=True)
            observation = record(result, time.perf_counter() - started)
            observation.update(pair_id=request["pair_id"], interleaved=True)
            stream.write(json.dumps(observation) + "\n")
            stream.flush()


def compare(
    baseline: Resident, candidate: Resident, reference: dict, *, repeats: int
) -> dict:
    """Retain alternating pairs; both owners must pass allocation admission."""
    from generativeqc._dft_autotune import resource_gate
    from generativeqc.autotune import dft_endpoint_gate
    from generativeqc.execution_tune import gates, passed

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
    evidence = gates(endpoint, reference)
    evidence["baseline_resources"] = resource_gate(
        samples["baseline"],
        schedule_hash=endpoint["baseline_schedule_hash"],
        source_hash=endpoint["baseline_source_hash"],
    )
    evidence["identity"] = {"passed": baseline.identity == candidate.identity}
    evidence["passed"] = (
        passed(evidence)
        and evidence["baseline_resources"]["passed"]
        and evidence["identity"]["passed"]
    )
    return evidence


def run(plan_path: Path, output: Path, *, repeats: int, policies: list[dict]) -> dict:
    """Search against defaults, then requalify the winner on two fresh owners."""
    output = raw_output_path(output)
    from generativeqc.autotune import source_identity
    from generativeqc.execution_tune import validate_policy

    if not os.environ.get("SLURM_JOB_ID") or repeats < 5:
        raise ValueError("DF96 requires five paired repeats in a Slurm allocation")
    plan = json.loads(plan_path.read_text())
    workload = plan["workload"]
    if (
        workload["kind"] != "df-hf"
        or workload["method"] != "rhf"
        or len(workload["atoms"]) != 96
        or workload["batch"] != 1
    ):
        raise ValueError("DF96 requires the explicit singleton RHF workload")
    canonical_source = source_identity(Path(__file__).resolve().parents[1])
    if canonical_source != workload["source_identity"]:
        raise ValueError("DF96 checkout differs from the native source inventory")
    library = Path(plan["library"])
    if file_hash(library) != plan["library_hash"]:
        raise ValueError("DF96 native library bytes changed")
    reference = json.loads(Path(plan["reference"]).read_text())
    if (
        reference.get("reference_family") != "df-hf"
        or not reference_gate(reference)["passed"]
    ):
        raise ValueError("DF96 requires an independent converged fitted-HF reference")
    for policy in policies:
        validate_policy("df", policy)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "generativeqc.df96-resident-qualification.v1",
        "plan": plan,
        "reference": reference,
        "reference_hash": file_hash(Path(plan["reference"])),
        "canonical_source_identity": canonical_source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "protocol": "serial alternating resident frozen-density owners; fresh holdout; no activation",
        "repeats": repeats,
        "trials": [],
        "selected": None,
    }

    def owner(stack: ExitStack, policy: dict, directory: Path) -> Resident:
        return stack.enter_context(
            Resident(
                library,
                plan_path,
                {"environment": policy},
                directory,
                worker_module="benchmarks.df96_autotune",
                worker_component="df",
            )
        )

    best, best_speedup = None, 1.0
    try:
        with ExitStack() as stack:
            baseline = owner(stack, {}, output / "baseline")
            report.update(
                identity=baseline.identity, baseline_cold=baseline.ready["cold"]
            )
            atomic_json(output / "report.json", report)
            for index, policy in enumerate(policies):
                print(f"policy={policy}", flush=True)
                trial = {"policy": policy}
                report["trials"].append(trial)
                try:
                    with ExitStack() as candidate_stack:
                        candidate = owner(
                            candidate_stack, policy, output / f"candidate-{index}"
                        )
                        trial["cold"] = candidate.ready["cold"]
                        trial["evidence"] = compare(
                            baseline, candidate, reference, repeats=repeats
                        )
                    evidence = trial["evidence"]
                    speedup = evidence["endpoint"]["speedup"]
                    print(
                        f"passed={evidence['passed']} speedup={speedup:.6f}", flush=True
                    )
                    if evidence["passed"] and speedup > best_speedup:
                        best, best_speedup = policy, speedup
                except (OSError, ValueError, RuntimeError) as error:
                    trial["failure"] = str(error)
                atomic_json(output / "report.json", report)
        if best is not None:
            with ExitStack() as stack:
                baseline = owner(stack, {}, output / "holdout/baseline")
                candidate = owner(stack, best, output / "holdout/candidate")
                report["holdout"] = compare(
                    baseline, candidate, reference, repeats=repeats
                )
            if report["holdout"]["passed"]:
                report["selected"] = best
        return report
    finally:
        atomic_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=raw_output_path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--policy", type=json.loads, action="append")
    parser.add_argument("--worker-fd", type=int)
    parser.add_argument("--configuration", type=json.loads)
    args = parser.parse_args()
    if args.worker_fd is not None:
        worker(
            json.loads(args.plan.read_text())["workload"],
            args.configuration,
            socket.socket(fileno=args.worker_fd),
        )
    else:
        if args.output is None or not args.policy:
            parser.error("--output and at least one --policy are required")
        run(
            args.plan.resolve(),
            args.output.resolve(),
            repeats=args.repeats,
            policies=args.policy,
        )


if __name__ == "__main__":
    main()
