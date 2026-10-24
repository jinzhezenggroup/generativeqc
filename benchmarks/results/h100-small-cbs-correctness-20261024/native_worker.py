"""Execute only one pre-enumerated CBS process after an external root release.

Nothing in this module imports a solver before release/input/device admission.
The historical Slurm-only command is never invoked or modified.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from observables import capture
from transport import checked, read, release, retain_return, write_new

sys.dont_write_bytecode = True


def helper(plan: dict) -> Any:
    path = checked(
        Path(plan["allocation_helper"]["path"]), plan["allocation_helper"]["sha256"]
    )
    spec = importlib.util.spec_from_file_location("pinned_qz_allocation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device(plan: dict, proof: dict, output: Path) -> dict:
    adapter = helper(plan)
    observation = adapter.device_observation()
    # Original sidecar survives a later binding rejection.
    write_new(output, observation)
    adapter.bind_allocated_instance(proof, plan["expected_job_name"], observation)
    boot = read(
        checked(
            Path(proof["runtime_preflight"]["path"]),
            proof["runtime_preflight"]["sha256"],
        )
    )["device"]
    for field in (
        "nvidia_smi",
        "cuda_visible_devices",
        "hostname",
        "memory_bytes",
        "compute_capability",
        "visible_device_count",
    ):
        if observation.get(field) != boot.get(field):
            raise ValueError(
                "actual CUDA-0 identity differs from root-bound preflight: " + field
            )
    return observation


def serialized_metadata(result: Any, refs: tuple | list | None, core: dict) -> dict:
    """Enrich only after immutable primary observables have been persisted."""
    record = {
        name: getattr(result, name)
        for name in (
            "hf_energy",
            "mp2_correlation_energy",
            "delta_ccsdt_energy",
            "hf_forces",
            "mp2_correlation_forces",
            "delta_ccsdt_forces",
            "hf_weights",
            "mp2_weights",
            "executed_backend",
            "endpoint_seconds",
            "execution_groups",
            "basis_metadata",
            "convergence",
            "converged",
        )
    }
    record["components"] = []
    if (
        refs is None
        or not isinstance(core["components"], list)
        or len(refs) != len(core["components"])
    ):
        raise ValueError("returned constituent metadata is unavailable")
    for index, component in enumerate(refs):
        r = component.result
        record["components"].append(
            {
                "roles": component.roles,
                "method": component.method,
                "basis_identity": component.basis_identity,
                "coefficient": component.coefficient,
                "elapsed_seconds": component.elapsed_seconds,
                "energy": core["components"][index]["energy"],
                "forces": core["components"][index]["forces"],
                "converged": r.converged,
                "iterations": r.iterations,
                "executed_backend": r.executed_backend,
                "correlation": None if r.correlation is None else asdict(r.correlation),
                "cc_performance": None
                if r.cc_performance is None
                else asdict(r.cc_performance),
            }
        )
    return record


def failure_kind(stage: str, error: BaseException) -> str:
    if not isinstance(error, Exception):
        return "CONTROL_INTERRUPTION"
    if isinstance(error, (OSError, TimeoutError, subprocess.TimeoutExpired)):
        return "TRANSPORT"
    if stage == "SCI_NATIVE" and isinstance(
        error, (RuntimeError, ValueError, ArithmeticError, MemoryError)
    ):
        return "SCIENTIFIC_REQUEST"
    if stage == "SCI_NUMERIC" and isinstance(error, (ValueError, ArithmeticError)):
        return "SCIENTIFIC_NUMERIC"
    return (
        "INTEGRITY"
        if stage in ("INTEGRITY", "POST_DEVICE", "SCI_NUMERIC")
        else "TRANSPORT"
    )


def run(plan_path: Path, release_path: Path, process_id: str) -> None:
    plan, proof = release(plan_path, release_path)
    process = next(p for p in plan["processes"] if p["id"] == process_id)
    calls = [c for c in plan["calls"] if c["process"] == process_id]
    if len(calls) != process["calls"]:
        raise ValueError("process inventory mismatch")
    out = Path(plan["output_root"]) / process_id
    out.mkdir(exist_ok=False)
    outcome = {
        "status": "PREFLIGHT",
        "stage": "PREFLIGHT",
        "calls_attempted": 0,
        "calls_returned": 0,
        "science_acceptance": "INCOMPLETE",
    }
    try:
        if os.name != "posix":
            raise RuntimeError("real H100 execution requires POSIX")
        import resource

        resource.setrlimit(
            resource.RLIMIT_FSIZE, (plan["caps"]["process_log_max_bytes"],) * 2
        )
        source = plan["science_source"]
        checked(Path(source["library"]), source["library_sha256"])
        bundle = read(
            checked(
                plan_path.parent / plan["input_bundle_file"],
                plan["input_bundle_sha256"],
            )
        )
        for name, value in plan["runtime_controls"].items():
            if name in (
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "GENERATIVEQC_LIBRARY",
                "GENERATIVEQC_JIT_SCCACHE_ROOT",
            ):
                os.environ[name] = value
        observation = device(plan, proof, out / "device-before.json")
        root = Path(source["root"]).resolve(strict=True)
        sys.path[:0] = [str(root / "python"), str(root)]
        native = importlib.import_module("generativeqc")
        if not Path(native.__file__).resolve().is_relative_to(root / "python"):
            raise RuntimeError("imported native facade escaped frozen source")
        from admission import compare, finite, reconstruct
        from model_contract import archive_expected
        from model_contract import components as model_components

        owners = {}
        records = []
        bases = None
        aux = None
        if process_id.startswith("conventional-"):
            case = calls[0]["fixture"]
            reference = __import__("json").loads(
                bundle["conventional"][f"tests/reference_data/cbs/{case}.json"][
                    "content_utf8"
                ]
            )
            reference_dir = root / "tests/reference_data/cbs"
            bases = tuple(
                native.CBSBasis(native.load_basis(reference_dir / reference[k]), n)
                for k, n in (("low_basis", 2), ("high_basis", 3))
            )
            if [b.basis.identity for b in bases] != reference["basis_identities"]:
                raise ValueError("conventional basis identity mismatch")
            for item in bundle["conventional"].values():
                data = item["content_utf8"].encode()
                if __import__("hashlib").sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("conventional embedded input mismatch")
        else:
            # This function reads local basis decimals only. No reference solver runs.
            from tools.generativeqc_validation.correlation_df_oracle import (
                basis_snapshot,
            )

            bases = tuple(
                native.CBSBasis(basis_snapshot(name, [1, 8]), n)
                for name, n in zip(
                    bundle["archive_water_basis_names"], (2, 3), strict=True
                )
            )
            aux = tuple(
                basis_snapshot(name, [1, 8])
                for name in bundle["archive_water_auxiliary_names"]
            )
            if [bases[i].basis.identity for i in (0, 0, 0, 1, 1)] != bundle[
                "archive_water_expected_basis_identities"
            ]:
                raise ValueError("archive orbital basis identity mismatch")
            if [b.identity for b in aux] != plan["archive_auxiliary_basis_identities"]:
                raise ValueError("archive auxiliary basis identity mismatch")
        for call in calls:
            outcome["stage"] = "SCI_NATIVE"
            write_new(
                out / f"attempt-{call['index']:03d}.json",
                {
                    "status": "ATTEMPT_STARTED",
                    "call": call,
                    "unix_seconds": time.time(),
                },
            )
            outcome["calls_attempted"] += 1
            if aux is None:
                atoms = reference["atoms"]
                if call["step_bohr"] is not None:
                    atoms = [
                        [
                            n,
                            [
                                x + call["step_bohr"] * call["sign"] * d
                                for x, d in zip(pos, direction, strict=True)
                            ],
                        ]
                        for (n, pos), direction in zip(
                            atoms, reference["direction"], strict=True
                        )
                    ]
                key = ("conventional",)
                kwargs = {}
            else:
                case = next(
                    c
                    for c in bundle["water_cases"]
                    if c["reference"] == call["hf_reference"]
                    and c["geometry_id"] == call["geometry_id"]
                )
                atoms = case["atoms_bohr"]
                key = (call["policy"], call["numeric_budget_bytes"])
                kwargs = {"auxiliary_bases": aux, "hf_reference": call["hf_reference"]}
            started = time.perf_counter()
            if key not in owners:
                owners[key] = native.CCSDTCBSCalculator(
                    bases,
                    device="cuda",
                    policy=call["policy"],
                    calculator_options={
                        "correlation_memory_budget_bytes": call["numeric_budget_bytes"]
                    },
                    **kwargs,
                )
            result = owners[key].singlepoint(
                atoms,
                properties=("energy", "forces")
                if call["kind"] == "EF"
                else ("energy",),
            )
            elapsed = time.perf_counter() - started
            outcome["calls_returned"] += 1
            outcome["stage"] = "TRANSPORT_RETURN"
            receipt = out / f"call-{call['index']:03d}"
            receipt.mkdir(exist_ok=False)
            core, refs = capture(result, call, atoms, elapsed)
            core_hashes = retain_return(receipt / "observables", core)
            write_new(receipt / "observables-binding.json", core_hashes)
            if core["capture_errors"]:
                raise RuntimeError(
                    "one or more returned observables could not be read; available originals retained"
                )
            from replay import recover

            core = recover(receipt / "observables")
            sample = serialized_metadata(result, refs, core)
            for field in (
                "energy",
                "forces",
                "call",
                "atoms_bohr",
                "complete_seconds",
                "complete_seconds_scope",
                "observable_availability",
            ):
                sample[field] = core[field]
            sample.update(
                numeric_representation="enriched metadata with restored primary values; original dtype/IEEE provenance is in observables",
                executed_precision_inventory="INCOMPLETE",
                semantic_work_inventory="INCOMPLETE",
            )
            enriched = receipt / "enriched"
            hashes = retain_return(enriched, sample)
            # Read the preserved JSON for arithmetic; acceptance never precedes raw publication.
            raw = read(enriched / "return.json")
            outcome["stage"] = "INTEGRITY"
            if raw["executed_backend"] != "cuda":
                raise ValueError("required endpoint did not execute on CUDA")
            if len(raw["components"]) != 5 or any(
                c["executed_backend"] != "cuda" for c in raw["components"]
            ):
                raise ValueError("constituents did not execute on CUDA")
            if raw["convergence"]["requested_policy"] != call["policy"]:
                raise ValueError("requested policy differs from fixed call inventory")
            ordered = model_components(call, raw, bundle)
            metadata = raw["basis_metadata"]
            if (
                metadata["density_fitting"] != (aux is not None)
                or metadata["hf_reference"] != call.get("hf_reference", "exact")
                or metadata["frozen_core"] != 0
                or metadata["representation"] != "spherical"
                or metadata["hf_exponent"] != 1.63
                or metadata["correlation_exponent"] != 3.0
            ):
                raise ValueError(
                    "executed CBS model differs from admitted scientific input"
                )
            outcome["stage"] = "SCI_NUMERIC"
            if not raw["converged"] or any(
                not c["converged"] for c in raw["components"]
            ):
                raise ValueError("actual CUDA endpoint/constituents did not converge")
            if call["kind"] == "EF" and aux is None:
                gates = {"energy_abs": 1e-9, "force_max_abs": 2e-6, "force_rms": 2e-6}
                compare(
                    raw["energy"],
                    raw["forces"],
                    reference["reference"]["energy"],
                    reference["reference"]["forces"],
                    gates,
                    strict=False,
                )
                from model_contract import ROLES

                for role, c in zip(ROLES, ordered, strict=True):
                    oracle = reference["reference"]["components"][role]
                    compare(
                        c["energy"],
                        c["forces"],
                        oracle["energy"],
                        oracle["forces"],
                        gates,
                        strict=False,
                    )
                    if c["method"] == "ccsd(t)" and (
                        abs(
                            finite(c["correlation"]["ccsd_t_triples_energy"])
                            - oracle["triples_energy"]
                        )
                        > 1e-9
                    ):
                        raise ValueError("independent triples gate failed")
                for field in (
                    "hf_energy",
                    "mp2_correlation_energy",
                    "delta_ccsdt_energy",
                ):
                    if abs(finite(raw[field]) - reference["reference"][field]) > 1e-9:
                        raise ValueError("independent constituent algebra gate failed")
                if any(
                    abs(sum(row[axis] for row in raw["forces"])) > 2e-7
                    for axis in range(3)
                ):
                    raise ValueError("conventional translation gate failed")
            elif call["kind"] == "EF":
                forces = reconstruct(case["oracle"])
                expected_e, expected_f = archive_expected(case["oracle"], forces)
                compare(
                    raw["energy"],
                    raw["forces"],
                    expected_e,
                    expected_f,
                    call["composite_gates"],
                )
                for i, c in enumerate(ordered):
                    compare(
                        c["energy"],
                        c["forces"],
                        case["oracle"]["component_energies"][i],
                        forces[i],
                        call["constituent_gates"],
                    )
            else:
                finite(raw["energy"])
            records.append(raw)
            write_new(
                enriched / "gate.json",
                {
                    "numeric_gate": "PASS",
                    "raw_hashes": hashes,
                    "precision_complete": False,
                    "work_complete": False,
                },
            )
        if aux is None:
            base = records[0]
            if abs(records[1]["energy"] - base["energy"]) > 1e-9:
                raise ValueError("energy-only parity gate failed")
            projection = -sum(
                f * d
                for row, direction in zip(
                    base["forces"], reference["direction"], strict=True
                )
                for f, d in zip(row, direction, strict=True)
            )
            for minus, plus in ((records[2], records[3]), (records[4], records[5])):
                derivative = (plus["energy"] - minus["energy"]) / (
                    2 * plus["call"]["step_bohr"]
                )
                if abs(derivative - projection) >= 2e-6:
                    raise ValueError("native reconverged directional gate failed")
            for row in reference["finite_difference"]:
                if abs(projection - row["derivative"]) >= 2e-6:
                    raise ValueError("retained independent directional gate failed")
        outcome["stage"] = "POST_DEVICE"
        after = device(plan, proof, out / "device-after.json")
        if after["nvidia_smi"] != observation["nvidia_smi"]:
            raise ValueError("device identity/driver/power observation changed")
        outcome.update(
            status="NUMERIC_PASS",
            science_acceptance="NUMERIC_ONLY; precision/work INCOMPLETE",
        )
    except BaseException as error:
        outcome.update(
            status="FAILED",
            failure_kind=failure_kind(outcome["stage"], error),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        write_new(out / "worker-outcome.json", outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--process", required=True)
    args = parser.parse_args()
    run(args.plan.resolve(), args.release.resolve(), args.process)
