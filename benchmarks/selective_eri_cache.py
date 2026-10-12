"""Profile master canonical buckets and select an experimental ERI cache set.

The selector is stdlib-only and never changes production routing. Its budget is
an *additional* value allowance after complete endpoint admission, not free VRAM.
Profiling requires a finite Slurm GPU allocation and the accompanying native
probe. These unscreened AO-pair diagnostics do not qualify screened shell caches
or establish complete SCF/force endpoint profitability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import typing
from pathlib import Path

from benchmarks._retention import raw_output_path

SCHEMA = "direct-eri-buckets-v1"
SCOPE = "diagnostic-unscreened-canonical"


def _integer(value: typing.Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value: typing.Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except OverflowError as failure:
        raise ValueError(f"{name} exceeds the finite timing range") from failure
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _median(row: dict, name: str) -> float:
    samples = row.get(name)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{name} requires nonempty measured samples")
    return statistics.median(_seconds(sample, name) for sample in samples)


def select_buckets(
    receipt: dict,
    cache_budget_bytes: int,
    expected_actions: int,
    *,
    fixed_setup_seconds: float = 0.0,
    maximum_states: int = 100_000,
) -> dict:
    """Solve the measured additive 0/1 model with a bounded Pareto frontier.

    Whole angular buckets are indivisible. Density-independent source work and
    a separately measured, identical contraction domain are required. Actual
    future Fock actions are unknown: ``expected_actions`` is a scenario, never
    the solver's iteration limit or a promise of measured endpoint savings.
    If exact search exceeds its bound, return no cache rather than labeling a
    benefit-per-byte greedy approximation as the optimum. Fixed setup is charged
    once, and per-bucket allocation/release costs are charged once each.
    """
    _integer(cache_budget_bytes, "cache_budget_bytes")
    _integer(expected_actions, "expected_actions")
    _integer(maximum_states, "maximum_states", 1)
    fixed_setup_seconds = _seconds(fixed_setup_seconds, "fixed_setup_seconds")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("scope") != SCOPE
        or receipt.get("status") != "complete"
    ):
        raise ValueError("selection requires a complete canonical bucket receipt")
    identity = receipt.get("identity", {})
    for key in (
        "native_sha256",
        "probe_sha256",
        "input_sha256",
        "device_uuid",
        "geometry_hash",
        "basis_hash",
    ):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"missing source/device identity: {key}")
    if (
        identity.get("operator") != "full-range-restricted-JK"
        or identity.get("precision") != "fp64"
        or identity.get("screening_tolerance") != 0.0
    ):
        raise ValueError("unsupported operator, precision or screening identity")

    candidates = []
    seen = set()
    for row in receipt.get("buckets", []):
        first = _integer(row.get("first"), "first")
        second = _integer(row.get("second"), "second")
        key = (first, second)
        if first > 6 or second > first or key in seen:
            raise ValueError("invalid or duplicate angular bucket")
        seen.add(key)
        status = row.get("status")
        if status in ("empty", "capacity-refused", "allocation-refused"):
            continue
        if status != "measured" or row.get("parity") != "passed":
            raise ValueError("bucket is not a successful parity-checked measurement")
        _seconds(row.get("max_absolute_error"), "max_absolute_error")
        if _seconds(row.get("max_scaled_error"), "max_scaled_error") > 2e-11:
            raise ValueError("bucket maximum parity error exceeds the diagnostic gate")
        values = _integer(row.get("source_values"), "source_values", 1)
        cache_bytes = _integer(row.get("cache_bytes"), "cache_bytes", 1)
        if cache_bytes != 8 * values:
            raise ValueError("FP64 value inventory disagrees with cache bytes")
        expected_work = {
            "direct_candidates": values,
            "direct_evaluations": values,
            "build_candidates": values,
            "build_evaluations": values,
            "replay_candidates": values,
            "replay_evaluations": 0,
        }
        work = row.get("work")
        if (
            work != expected_work
            or not isinstance(work, dict)
            or any(type(count) is not int for count in work.values())
        ):
            raise ValueError(
                "bucket source work is not matched or replay recomputes ERIs"
            )
        lengths = {
            len(row.get(name, []))
            for name in ("build_seconds", "direct_seconds", "replay_seconds")
        }
        if len(lengths) != 1:
            raise ValueError("timing sample counts differ")
        build = (
            _median(row, "build_seconds")
            + _seconds(row.get("allocation_seconds"), "allocation_seconds")
            + _seconds(row.get("release_seconds"), "release_seconds")
        )
        direct = _median(row, "direct_seconds")
        replay = _median(row, "replay_seconds")
        try:
            gain = expected_actions * (direct - replay) - build
        except OverflowError as failure:
            raise ValueError("gain overflow") from failure
        if not math.isfinite(gain):
            raise ValueError("gain overflow")
        if gain > 0 and cache_bytes <= cache_budget_bytes:
            candidates.append((key, cache_bytes, gain))
    if seen != {(first, second) for first in range(7) for second in range(first + 1)}:
        raise ValueError(
            "receipt must inventory all 28 buckets, including empty/refused ones"
        )
    candidates.sort()
    try:
        math.fsum(gain for _, _, gain in candidates)
    except OverflowError as failure:
        raise ValueError("aggregate gain overflow") from failure

    result = {
        "schema": "selective-eri-plan-v1",
        "scope": SCOPE,
        "identity": identity,
        "cache_budget_bytes": cache_budget_bytes,
        "expected_actions": expected_actions,
        "fixed_setup_seconds": fixed_setup_seconds,
        "selected": [],
        "selected_bucket_mask": 0,
        "selected_bytes": 0,
        "predicted_gain_seconds": 0.0,
        "optimal_for_additive_model": False,
        "production_admission": "not-qualified",
        "endpoint_performance": "not-measured",
    }
    frontier = [(0, 0.0, ())]
    for key, cache_bytes, gain in candidates:
        expanded = frontier + [
            (used + cache_bytes, benefit + gain, selected + (key,))
            for used, benefit, selected in frontier
            if used + cache_bytes <= cache_budget_bytes
        ]
        if len(expanded) > 2 * maximum_states:
            return {**result, "status": "search-bound-refused"}
        expanded.sort(key=lambda state: (state[0], -state[1], state[2]))
        frontier = []
        best_gain = -1.0
        for state in expanded:
            if state[1] > best_gain:
                frontier.append(state)
                best_gain = state[1]
        if len(frontier) > maximum_states:
            return {**result, "status": "search-bound-refused"}
    used, gain, chosen = frontier[-1]
    result["optimal_for_additive_model"] = True
    if gain <= fixed_setup_seconds:
        return {**result, "status": "no-profitable-cache"}
    return {
        **result,
        "status": "selected",
        "selected": [{"first": first, "second": second} for first, second in chosen],
        "selected_bucket_mask": sum(
            1 << (first * (first + 1) // 2 + second) for first, second in chosen
        ),
        "selected_bytes": used,
        "predicted_gain_seconds": gain - fixed_setup_seconds,
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_buckets(args: argparse.Namespace) -> dict:
    """Keep progress/failure receipts without loading any oracle integral data."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("profiling requires a finite srun GPU allocation")
    _integer(args.repeats, "repeats", 1)
    if args.repeats > 512:
        raise ValueError("repeats must be <= 512 for bounded profiling")
    _integer(args.cache_budget_bytes, "cache_budget_bytes")
    _integer(args.provider_budget_bytes, "provider_budget_bytes", 1)
    if not math.isfinite(args.displacement):
        raise ValueError("displacement must be finite")
    import ctypes as ct

    maximum_size = (1 << (8 * ct.sizeof(ct.c_size_t))) - 1
    if (
        args.cache_budget_bytes > maximum_size
        or args.provider_budget_bytes > maximum_size
    ):
        raise ValueError("profiling budget exceeds native size_t capacity")

    from benchmarks.rhf_resident_jk import read_input
    from tools.generativeqc_posthf.fixtures import source_arguments
    from tools.generativeqc_posthf.sources import NativeSource

    probe_path = args.probe.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    library = ct.CDLL(str(probe_path))
    size_pointer = ct.POINTER(ct.c_uint64)
    double_pointer = ct.POINTER(ct.c_double)
    create = library.direct_eri_bucket_create
    create.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_void_p),
        size_pointer,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
    ]
    create.restype = ct.c_int
    run = library.direct_eri_bucket_profile
    run.argtypes = [
        ct.c_void_p,
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
        ct.c_uint,
        size_pointer,
        double_pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    run.restype = ct.c_int
    destroy = library.direct_eri_bucket_destroy
    destroy.argtypes = [ct.c_void_p]
    destroy.restype = None
    linked_native = library.direct_eri_bucket_native_library
    linked_native.argtypes = []
    linked_native.restype = ct.c_char_p
    report = {"schema": SCHEMA, "scope": SCOPE, "status": "running", "buckets": []}

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    metadata = read_input(input_path, args.displacement)
    with NativeSource(**source_arguments(metadata)) as source:
        native_path = Path(source._library._name).resolve(strict=True)
        provider_path = linked_native()
        if not provider_path:
            raise ValueError("probe could not identify its native provider library")
        if Path(os.fsdecode(provider_path)).resolve(strict=True) != native_path:
            raise ValueError(
                "probe and metadata source load different native libraries"
            )
        report["identity"] = {
            "native_sha256": _digest(native_path),
            "probe_sha256": _digest(probe_path),
            "input_sha256": _digest(input_path),
            "geometry_hash": source.geometry_hash,
            "basis_hash": source.basis_hash,
            "operator": "full-range-restricted-JK",
            "precision": "fp64",
            "screening_tolerance": 0.0,
            "density": "signed-symmetric-canonical-v1",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        }
        context = ct.c_void_p()
        inventory = (ct.c_uint64 * 6)()
        device = ct.create_string_buffer(512)
        error = ct.create_string_buffer(2048)
        save()
        try:
            if create(
                source._handle,
                args.provider_budget_bytes,
                ct.byref(context),
                inventory,
                device,
                len(device),
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
            uuid, name = device.value.decode().split("\n", 1)
            report["identity"].update(
                device_uuid=uuid,
                device_name=name,
                driver_version=inventory[4],
                runtime_version=inventory[5],
            )
            report.update(
                public_nbf=inventory[0],
                canonical_nbf=inventory[1],
                provider_device_bytes=inventory[2],
                probe_device_bytes=inventory[3],
                repeats=args.repeats,
                cache_budget_bytes=args.cache_budget_bytes,
            )
            for first in range(7):
                for second in range(first + 1):
                    counts = (ct.c_uint64 * 11)()
                    samples = (ct.c_double * (3 * args.repeats + 4))()
                    if run(
                        context,
                        first,
                        second,
                        args.cache_budget_bytes,
                        args.repeats,
                        counts,
                        samples,
                        error,
                        len(error),
                    ):
                        report["failed_bucket"] = {
                            "first": first,
                            "second": second,
                            "raw_counts": list(counts),
                            "raw_samples": list(samples),
                            "status": "failed",
                        }
                        raise RuntimeError(
                            f"bucket {first},{second}: {error.value.decode()}"
                        )
                    row = {
                        "first": first,
                        "second": second,
                        "first_pairs": counts[0],
                        "second_pairs": counts[1],
                        "source_values": counts[2],
                        "cache_bytes": counts[3],
                        "status": (
                            "empty",
                            "capacity-refused",
                            "measured",
                            "allocation-refused",
                        )[counts[4]],
                    }
                    if counts[4] == 2:
                        row.update(
                            build_seconds=list(samples[: args.repeats]),
                            direct_seconds=list(
                                samples[args.repeats : 2 * args.repeats]
                            ),
                            replay_seconds=list(
                                samples[2 * args.repeats : 3 * args.repeats]
                            ),
                            allocation_seconds=samples[3 * args.repeats],
                            release_seconds=samples[3 * args.repeats + 1],
                            max_absolute_error=samples[3 * args.repeats + 2],
                            max_scaled_error=samples[3 * args.repeats + 3],
                            parity="passed",
                            work=dict(
                                zip(
                                    (
                                        "direct_candidates",
                                        "direct_evaluations",
                                        "build_candidates",
                                        "build_evaluations",
                                        "replay_candidates",
                                        "replay_evaluations",
                                    ),
                                    counts[5:],
                                    strict=True,
                                )
                            ),
                        )
                    report["buckets"].append(row)
                    save()
            for path, key in (
                (native_path, "native_sha256"),
                (probe_path, "probe_sha256"),
                (input_path, "input_sha256"),
            ):
                if _digest(path) != report["identity"][key]:
                    raise RuntimeError(f"profiling input changed: {key}")
            report["status"] = "complete"
        except Exception as failure:
            report.update(status="failed", failure=str(failure))
            raise
        finally:
            destroy(context)
            save()
    return report


def main() -> None:
    """Write raw diagnostics or a non-production selection scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--input", type=Path, required=True)
    profile.add_argument("--probe", type=Path, required=True)
    profile.add_argument("--output", type=raw_output_path, required=True)
    profile.add_argument("--repeats", type=int, default=5)
    profile.add_argument("--cache-budget-bytes", type=int, default=8 << 30)
    profile.add_argument("--provider-budget-bytes", type=int, default=1 << 30)
    profile.add_argument("--displacement", type=float, default=0.0)
    select = commands.add_parser("select")
    select.add_argument("--input", type=Path, required=True)
    select.add_argument("--output", type=raw_output_path, required=True)
    select.add_argument("--cache-budget-bytes", type=int, required=True)
    select.add_argument("--expected-actions", type=int, required=True)
    select.add_argument("--fixed-setup-seconds", type=float, default=0.0)
    select.add_argument("--maximum-states", type=int, default=100_000)
    args = parser.parse_args()
    if args.command == "profile":
        profile_buckets(args)
    else:
        receipt_bytes = args.input.read_bytes()
        result = select_buckets(
            json.loads(receipt_bytes),
            args.cache_budget_bytes,
            args.expected_actions,
            fixed_setup_seconds=args.fixed_setup_seconds,
            maximum_states=args.maximum_states,
        )
        result["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
