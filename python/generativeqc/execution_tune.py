"""Isolated, exact-workload tuning and replay of native J/K execution policies.

Policies are recipes for an explicit subprocess replay, not process-global
Calculator defaults. Native capability/admission fallbacks remain authoritative.
No Python environment mutation can leak a winner into unrelated calculations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .profiles import atomic_json, canonical_hash, file_hash

SCHEMA = "generativeqc.native-execution-recipe.v1"
JK_CONTROLS = {
    "GENERATIVEQC_DISABLE_MD_J": ("0", "1"),
    "GENERATIVEQC_MD_J_ENGINE": ("incumbent", "reference", "primitive"),
    "GENERATIVEQC_MD_J_RECIPROCAL": ("0", "1"),
    "GENERATIVEQC_DIRECT_J_FOCK_LOWERING": ("incumbent", "rys"),
    "GENERATIVEQC_DIRECT_K_FOCK_LOWERING": ("incumbent", "rys", "block", "rys-task"),
    "GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": ("incumbent", "fill", "primitive", "work"),
}


def validate_policy(component: str, policy: dict) -> None:
    """Accept only audited execution controls, never precision or screening."""
    from ._df_autotune import DF_CONTROLS

    controls = {"jk": JK_CONTROLS, "df": DF_CONTROLS}.get(component)
    if controls is None or not isinstance(policy, dict):
        raise ValueError("unsupported execution component or policy")
    if any(
        key not in controls or value not in controls[key]
        for key, value in policy.items()
    ):
        raise ValueError("unsupported native execution policy")


def candidate_policies(component: str, *, hybrid: bool, full: bool) -> list[dict]:
    """Search existing bounded native lowerings without changing the Hamiltonian."""
    if component == "df":
        from ._df_autotune import candidate_policies as df_policies

        return df_policies(full=full)
    policies = [
        {"GENERATIVEQC_MD_J_ENGINE": "incumbent"},
        {
            "GENERATIVEQC_DISABLE_MD_J": "1",
            "GENERATIVEQC_DIRECT_J_FOCK_LOWERING": "rys",
        },
        {"GENERATIVEQC_MD_J_RECIPROCAL": "0"},
    ]
    if hybrid:
        policies = [
            {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "fill"},
            {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "primitive"},
            {"GENERATIVEQC_DIRECT_K_FOCK_LOWERING": "incumbent"},
            *policies,
        ]
        if full:
            policies += [
                {"GENERATIVEQC_DIRECT_K_FOCK_LOWERING": value}
                for value in ("rys", "block", "rys-task")
            ]
    for policy in policies:
        validate_policy(component, policy)
    return policies


def worker_environment(library: Path, component: str, policy: dict) -> dict:
    """Preserve Slurm visibility while excluding ambient scientific/debug knobs."""
    validate_policy(component, policy)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GENERATIVEQC_")
    }
    environment.update(GENERATIVEQC_LIBRARY=str(library), GENERATIVEQC_PROFILE="off")
    environment.update(policy)
    return environment


def execute(workload: dict, component: str, policy: dict) -> dict:
    """Measure the complete native warm E+analytic-F endpoint, not J/K timers."""
    from ._autotune_worker import execute_workload

    validate_policy(component, policy)
    sample = execute_workload(workload)
    if component == "jk":
        work = [row.get("coulomb_exchange_work") for row in sample["dft_work"]]
        if not work or any(not row or row.get("complete") is not True for row in work):
            raise ValueError(
                "J/K tuning requires complete native Coulomb/exchange work observations"
            )
    schedule = {"component": component, "environment": policy}
    sample.update(
        scientific_identity=canonical_hash(workload),
        scientific_workload=copy.deepcopy(workload),
        native_xc_schedule=sample.get("schedule"),
        schedule=schedule,
        schedule_identity=canonical_hash(schedule),
    )
    return sample


def worker(
    library: Path,
    workload: dict,
    component: str,
    policy: dict,
    directory: Path,
    *,
    timeout: float,
) -> dict:
    """Run one fresh native owner in an isolated, finite-timeout process."""
    directory.mkdir(parents=True)
    path, output = directory / "input.json", directory / "output.json"
    atomic_json(path, {"workload": workload, "component": component, "policy": policy})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "generativeqc.execution_tune",
            "--worker",
            str(path),
            str(output),
        ],
        env=worker_environment(library, component, policy),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    (directory / "worker.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"native policy worker failed; see {directory / 'worker.log'}"
        )
    return json.loads(output.read_text())


def compare(
    library: Path,
    workload: dict,
    component: str,
    baseline: dict,
    candidate: dict,
    directory: Path,
    *,
    repeats: int,
    remaining: Any,
) -> dict:
    """Keep every paired sample, including refused or noisy policy trials."""
    from .autotune import dft_endpoint_gate

    samples = {"baseline": [], "candidate": []}
    for pair_id in range(repeats):
        order = (
            ("baseline", "candidate") if pair_id % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            sample = worker(
                library,
                workload,
                component,
                baseline if side == "baseline" else candidate,
                directory / f"pair-{pair_id}-{side}",
                timeout=remaining(),
            )
            sample.update(pair_id=pair_id, interleaved=True)
            samples[side].append(sample)
    return dft_endpoint_gate(samples["baseline"], samples["candidate"])


def gates(endpoint: dict, reference: dict) -> dict:
    """Bind independent numerical and measured memory gates to the same replays."""
    from ._dft_autotune import numerical_gate, resource_gate

    samples = endpoint["baseline"] + endpoint["candidate"]
    if reference.get("reference_family") == "df-hf":
        from ._df_autotune import numerical_gate, work_gate

        observed_work = work_gate(samples, reference)
    else:
        observed_work = jk_work_gate(samples)
    return {
        "endpoint": endpoint,
        "numerical": numerical_gate(samples, reference),
        "resources": resource_gate(
            endpoint["candidate"],
            schedule_hash=endpoint["candidate_schedule_hash"],
            source_hash=endpoint["candidate_source_hash"],
        ),
        "work": observed_work,
    }


def jk_work_gate(samples: list[dict]) -> dict:
    """Keep observed Coulomb/exchange and XC work identical across J/K trials."""
    work = [
        [
            (item["grid_points"], item["fock_builds"], item["coulomb_exchange_work"])
            for item in sample["dft_work"]
        ]
        for sample in samples
    ]
    same_work = bool(work) and all(item == work[0] for item in work)
    same_xc = all(
        sample["native_xc_schedule"] == samples[0]["native_xc_schedule"]
        for sample in samples
    )
    return {
        "passed": same_work and same_xc,
        "scope": "observed grid/Fock/J/K census and resolved XC policy are unchanged",
    }


def passed(evidence: dict) -> bool:
    """Promotion is a conjunction; no failed gate is replaced with timing."""
    return all(
        evidence.get(name, {}).get("passed") is True
        for name in ("endpoint", "numerical", "resources", "work")
    )


def run(args: Any, *, base_library: Any, probe: dict, tools: dict, target: Any) -> dict:
    """Qualify a replay recipe without modifying the user's active profile index."""
    from ._dft_autotune import DFT_FUNCTIONALS, numerical_gate
    from .autotune import _worker, inspect_library, read_xyz
    from .ks import GridSpec, KsOptions, resolve_ks_options
    from .profiles import cache_root, compatibility_identity

    fitted = args.component == "df"
    if args.repeats < 5 or args.method not in (("rhf",) if fitted else DFT_FUNCTIONALS):
        raise ValueError(
            "native policy tuning requires a supported method and five pairs"
        )
    options = (
        None
        if fitted
        else resolve_ks_options(
            args.method, KsOptions(grid=GridSpec.treutler(args.grid_level))
        )
    )
    workload = {
        "kind": "df-hf" if fitted else "dft",
        "atoms": read_xyz(args.input, units=args.units),
        "basis": args.basis,
        "method": args.method,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "representation": args.representation,
        "device_id": args.device_id,
        "batch": args.batch,
        "grid_level": args.grid_level,
        "architecture": target.architecture,
        "source_identity": probe["source_identity"],
    }
    if not fitted:
        workload["reference_engine"] = (
            "reused"
            if getattr(args, "reference_file", None)
            else getattr(args, "reference_engine", "pyscf")
        )
    if fitted:
        from ._df_autotune import numerical_gate

        workload.update(
            auxiliary_basis=args.auxiliary_basis or args.basis,
            metric_relative_threshold=1e-10,
            df_memory_bytes=args.df_memory_mib << 20,
        )
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=f"{args.component}-run-", dir=root)
    ).resolve()
    selection = inspect_library(Path(base_library._name), args.device_id, select=True)
    library = Path(selection["library"])
    report = {
        "schema": SCHEMA,
        "directory": str(directory),
        "installed": None,
        "qualified_recipe": None,
        "component": args.component,
        "workload": workload,
        "identity": compatibility_identity(probe),
        "toolchain": tools,
        "library": str(library),
        "library_hash": file_hash(library),
        "incumbent_profile": selection["diagnostics"],
        "candidates": [],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timing_scope": "frozen-density warm complete energy+analytic forces; excludes prepare/cold/warmup",
        "fallback": "native capability/resource fallbacks unchanged; no process-global activation",
    }
    deadline = time.monotonic() + args.budget_seconds

    def remaining() -> float:
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise TimeoutError("execution tuning budget exhausted; no recipe published")
        return duration

    try:
        workload_path = directory / "workload.json"
        atomic_json(workload_path, workload)
        if getattr(args, "reference_file", None):
            from ._dft_reference import load

            reference = load(args.reference_file, args.reference_sha256, workload)
        else:
            reference = _worker(
                library,
                workload_path,
                directory / "reference.json",
                reference=True,
                timeout=remaining(),
            )
        report["reference"] = reference
        baseline = worker(
            library,
            workload,
            args.component,
            {},
            directory / "baseline",
            timeout=remaining(),
        )
        report["baseline"] = baseline
        if not numerical_gate([baseline], reference)["passed"]:
            raise ValueError("incumbent failed the independent energy/force gate")
        hybrid = fitted or bool(options.execution_plan.exchange)
        best_policy, best = {}, None
        for index, policy in enumerate(
            candidate_policies(args.component, hybrid=hybrid, full=args.full)[
                : args.max_candidates
            ]
        ):
            row = {"policy": policy, "accepted": False}
            report["candidates"].append(row)
            try:
                endpoint = compare(
                    library,
                    workload,
                    args.component,
                    best_policy,
                    policy,
                    directory / f"candidate-{index}",
                    repeats=args.repeats,
                    remaining=remaining,
                )
                row.update(gates(endpoint, reference))
                if passed(row):
                    if best is not None:
                        best["accepted"] = False
                    row["accepted"] = True
                    best_policy, best = policy, row
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                row["failure"] = str(error)
        if best is None:
            report["reason"] = (
                "no native policy passed all endpoint, numerical and resource gates"
            )
            return report
        endpoint = compare(
            library,
            workload,
            args.component,
            {},
            best_policy,
            directory / "holdout",
            repeats=args.repeats,
            remaining=remaining,
        )
        report["holdout"] = gates(endpoint, reference)
        if not passed(report["holdout"]):
            best["accepted"] = False
            report["reason"] = (
                "selected policy failed fresh incumbent holdout qualification"
            )
            return report
        remaining()
        recipe = {
            key: copy.deepcopy(report[key])
            for key in (
                "schema",
                "component",
                "workload",
                "identity",
                "library",
                "library_hash",
                "timing_scope",
            )
        }
        recipe.update(
            policy=best_policy, evidence_hash=canonical_hash(report["holdout"])
        )
        recipe_path = directory / "recipe.json"
        atomic_json(recipe_path, recipe)
        report["qualified_recipe"] = str(recipe_path)
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        report["failure"] = str(error)
    finally:
        atomic_json(directory / "report.json", report)
    return report


def replay(path: Path, output: Path, *, timeout: float = 600) -> dict:
    """Replay only a matching source/device/binary-bound, evidence-qualified workload."""
    from .autotune import inspect_library
    from .profiles import compatibility_identity

    recipe = json.loads(path.read_text())
    if recipe.get("schema") != SCHEMA:
        raise ValueError("unsupported execution recipe schema")
    validate_policy(recipe["component"], recipe["policy"])
    library = Path(recipe["library"])
    if file_hash(library) != recipe["library_hash"]:
        raise ValueError("execution recipe native library bytes changed")
    probe = inspect_library(library, recipe["workload"]["device_id"])["probe"]
    if compatibility_identity(probe) != recipe["identity"]:
        raise ValueError("execution recipe device/source compatibility changed")
    report = json.loads(path.with_name("report.json").read_text())
    if (
        not passed(report.get("holdout", {}))
        or canonical_hash(report["holdout"]) != recipe["evidence_hash"]
    ):
        raise ValueError("execution recipe lacks matching qualified holdout evidence")
    accepted = [row["policy"] for row in report["candidates"] if row.get("accepted")]
    if (
        recipe["workload"] != report["workload"]
        or accepted != [recipe["policy"]]
        or recipe["identity"] != report["identity"]
        or recipe["library_hash"] != report["library_hash"]
        or recipe["component"] != report["component"]
    ):
        raise ValueError("execution recipe differs from the qualified workload/policy")
    from .autotune import dft_endpoint_gate

    endpoint = report["holdout"]["endpoint"]
    checked = gates(
        dft_endpoint_gate(endpoint["baseline"], endpoint["candidate"]),
        report["reference"],
    )
    schedule_hash = canonical_hash(
        {"component": recipe["component"], "environment": recipe["policy"]}
    )
    if (
        not passed(checked)
        or canonical_hash(checked) != recipe["evidence_hash"]
        or checked["endpoint"]["candidate_schedule_hash"] != schedule_hash
        or checked["endpoint"]["scientific_identity"]
        != canonical_hash(recipe["workload"])
        or checked["endpoint"]["candidate_source_hash"]
        != recipe["identity"]["source_identity"]
    ):
        raise ValueError("execution recipe raw endpoint evidence failed revalidation")
    temporary = Path(tempfile.mkdtemp(prefix="execution-replay-"))
    result = worker(
        library,
        recipe["workload"],
        recipe["component"],
        recipe["policy"],
        temporary / "worker",
        timeout=timeout,
    )
    atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """The replay frontend is separate from the Python-free native runtime CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        if args.worker:
            payload = json.loads(args.input.read_text())
            atomic_json(
                args.output,
                execute(payload["workload"], payload["component"], payload["policy"]),
            )
        else:
            replay(args.input, args.output, timeout=args.timeout)
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"execution-tune: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
