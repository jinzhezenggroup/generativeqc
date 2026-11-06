"""Executable lineage controls and optional scheduled scientific DF capture."""

from __future__ import annotations

import copy
import ctypes
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_cupti_source_capture import activity_record, fixture, recount, source_record

from tools.cupti_residency_capture import CuptiCapture
from tools.cupti_source_capture import SourceInventory, summarize_sources


def metric_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Model independent API joins plus an explicitly named source transform."""
    raw, activity = fixture()
    raw["catalog"] = {
        "owner": {"64": "df_metric_setup"},
        "role": {"1": "prepare"},
        "site": {
            "65": "df.metric.eigenvalues",
            "67": "df.metric.host_ready",
            "68": "df.metric.inverse_sqrt_transform",
            "69": "df.metric.scales_upload",
        },
        "payload": {
            "0": "none",
            "129": "df.metric.eigenvalues",
            "131": "df.metric.inverse_sqrt_scales",
        },
    }
    operations = [
        {
            "operation_id": 10,
            "site": 65,
            "payload": 129,
            "payload_instance": 20,
            "direction": 2,
        },
        {
            "operation_id": 11,
            "site": 67,
            "payload": 0,
            "payload_instance": 0,
            "operation_kind": 2,
            "direction": 0,
            "bytes": 0,
        },
        {
            "operation_id": 12,
            "site": 68,
            "payload": 131,
            "payload_instance": 30,
            "operation_kind": 4,
            "direction": 0,
            "dependency": 20,
        },
        {
            "operation_id": 13,
            "site": 69,
            "payload": 131,
            "payload_instance": 40,
            "direction": 1,
            "dependency": 30,
        },
    ]
    raw["records"] = [source_record(3, owner=64)]
    activity["records"] = []
    for operation in operations:
        for kind in (1, 2):
            raw["records"].append(source_record(kind, owner=64, role=1, **operation))
        identity = operation["operation_id"]
        if identity == 12:
            continue
        correlation = identity + 100
        activity["records"].extend(
            [
                activity_record(
                    5,
                    correlation_id=correlation,
                    api_name="cudaStreamSynchronize"
                    if identity == 11
                    else "cudaMemcpyAsync",
                    start=identity * 100,
                    end=identity * 100 + 50,
                ),
                activity_record(
                    39, subtype=3, correlation_id=correlation, external_id=1
                ),
                activity_record(
                    39, subtype=5, correlation_id=correlation, external_id=identity
                ),
            ]
        )
        if identity == 11:
            activity["records"].append(
                activity_record(
                    38, subtype=3, correlation_id=correlation, start=1100, end=1120
                )
            )
        else:
            activity["records"].append(
                activity_record(
                    1,
                    subtype=2 if identity == 10 else 1,
                    correlation_id=correlation,
                    runtime_correlation_id=correlation,
                    bytes=128,
                    copy_count=1,
                    start=identity * 100 + 10,
                    end=identity * 100 + 30,
                )
            )
    raw["records"].append(source_record(4, owner=64))
    recount(raw, activity)
    return raw, activity


REGIONS = {1: {"name": "scientific-df-prepare", "role": "compatibility"}}


def test_checked_host_transform_links_exact_source_generations() -> None:
    raw, activity = metric_fixture()
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"], result["issues"]
    assert result["qualified_host_round_trips"] == [
        {
            "owner": "df_metric_setup",
            "role": "prepare",
            "region_id": 1,
            "d2h_operation_id": 10,
            "host_transform_operation_id": 12,
            "h2d_operation_id": 13,
        }
    ]
    assert result["by_owner"]["df_metric_setup"]["h2d_bytes"] == 128
    assert result["by_owner"]["df_metric_setup"]["d2h_bytes"] == 128
    assert result["by_owner"]["df_metric_setup"]["source_stream_fences"] == 1
    assert result["status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None


@pytest.mark.parametrize(
    "mutation",
    ["missing", "cycle", "alias", "wrong-site", "early-fence", "failed", "shape"],
)
def test_equal_bytes_do_not_repair_broken_lineage(mutation: str) -> None:
    raw, activity = metric_fixture()
    raw = copy.deepcopy(raw)
    for row in raw["records"]:
        if row["operation_id"] == 12:
            if mutation == "missing":
                row["dependency"] = 99
            if mutation == "cycle":
                row["dependency"] = 40
            if mutation == "alias":
                row["payload_instance"] = 20
            if mutation == "wrong-site":
                row["site"] = 69
            if mutation == "failed" and row["kind"] == 2:
                row["status"] = 1
            if mutation == "shape":
                row["bytes"] += 8
    if mutation == "early-fence":
        raw["records"][1:5] = raw["records"][3:5] + raw["records"][1:3]
    result = summarize_sources(raw, activity, REGIONS)
    assert not result["source_annotations_intact"]
    assert not result["qualified_host_round_trips"]
    assert result["zero_round_trip_assertion"] is None


def capture_metric_case(name: str, output: Path) -> None:
    """Use the actual public CUDA DF-MP2 owner, not a synthetic GPU consumer."""
    from generativeqc import Calculator
    from generativeqc.profiles import file_hash

    from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
    from tools.residency_allocation import allocation_identity

    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve(strict=True)
    collector = Path(os.environ["GENERATIVEQC_CUPTI_COLLECTOR"]).resolve(strict=True)
    _, allocation = allocation_identity(library, None, None)
    meta, _ = load_fixture(name)
    arguments = source_arguments(meta)
    config = {
        "method": "mp2",
        "basis": arguments["basis"],
        "auxiliary_basis": arguments["auxiliary_basis"],
        "basis_representation": arguments["representation"],
        "precision": "fp64",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-11,
    }

    def execute(device: str) -> Any:
        return Calculator(device=device, density_fitting=device, **config).singlepoint(
            arguments["atoms"],
            charge=arguments["charge"],
            properties=("energy",),
        )

    ordinary = execute("cuda")
    independent = execute("cpu")
    native = ctypes.CDLL(str(library))
    native.cudaDeviceSynchronize.restype = ctypes.c_int
    native.cudaDeviceSynchronize.argtypes = []
    assert native.cudaDeviceSynchronize() == 0
    profiler = CuptiCapture(collector)
    inventory = SourceInventory(collector, native)
    regions = {**REGIONS, 8: {"name": "observer-fence", "role": "observer"}}
    output.mkdir(parents=True, exist_ok=False)
    capture_error = None
    cleanup_errors = []
    result = None
    graphs, source, activity = {}, {}, {}
    try:
        with profiler.region(1):
            result = execute("cuda")
    except (RuntimeError, ValueError, OSError, AssertionError) as error:
        capture_error = error
    finally:
        # A sticky CUDA error must not erase the raw failure journal or mask the
        # scientific endpoint exception with a secondary observer-fence failure.
        try:
            with profiler.region(8):
                fence_status = native.cudaDeviceSynchronize()
                if fence_status:
                    cleanup_errors.append(f"observer fence CUDA status {fence_status}")
        except (RuntimeError, ValueError, OSError, AssertionError) as error:
            cleanup_errors.append(str(error))
        try:
            graphs, source = inventory.stop()
        except (RuntimeError, ValueError, OSError, AssertionError) as error:
            cleanup_errors.append(str(error))
        try:
            activity = profiler.stop()
        except (RuntimeError, ValueError, OSError, AssertionError) as error:
            cleanup_errors.append(str(error))
    diagnostics = (
        summarize_sources(source, activity, regions)
        if source and activity
        else {"source_annotations_intact": False, "issues": ["incomplete capture"]}
    )
    expected = meta["records"]["df"]
    reference_energy = expected["hf_energy"] + expected["correlation_energy"]
    evidence = {
        "fixture": name,
        "fixture_array_hash": meta["array_hash"],
        "allocation": allocation,
        "library_sha256": file_hash(library),
        "collector_sha256": file_hash(collector),
        "solver_controls": config,
        "energy": result.energy if result is not None else None,
        "ordinary_energy": ordinary.energy,
        "cpu_energy": independent.energy,
        "fixture_energy": reference_energy,
        "energy_gate": 1e-9,
        "reference_kind": "committed-independent-same-Hamiltonian-fixture-and-native-CPU",
        "capture_error": str(capture_error) if capture_error is not None else None,
        "cleanup_errors": cleanup_errors,
    }
    for filename, value in (
        ("activity.json", activity),
        ("source-boundaries.json", source),
        ("graph-inventory.json", graphs),
        ("regions.json", regions),
        ("source-diagnostics.json", diagnostics),
        ("numerical.json", evidence),
    ):
        (output / filename).write_text(json.dumps(value, indent=2, default=str) + "\n")
    if capture_error is not None:
        raise capture_error
    assert not cleanup_errors, cleanup_errors
    assert diagnostics["source_annotations_intact"], diagnostics["issues"]
    assert diagnostics["qualified_host_round_trips"]
    assert result.converged and ordinary.converged and independent.converged
    for target in (ordinary.energy, independent.energy, reference_energy):
        assert abs(result.energy - target) < evidence["energy_gate"]


def test_failed_scientific_capture_retains_journals_and_original_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CPU doubles test failure ownership, not scientific CUDA execution."""
    import generativeqc

    from tools import residency_allocation

    binary = tmp_path / "not-a-library"
    binary.write_bytes(b"CPU-only failure ownership double")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(binary))
    monkeypatch.setenv("GENERATIVEQC_CUPTI_COLLECTOR", str(binary))
    monkeypatch.setattr(
        residency_allocation, "allocation_identity", lambda *_: ({}, {})
    )
    failure = RuntimeError("original scientific endpoint failure")
    invocations = []

    class Calculator:
        def __init__(self, **_: Any) -> None:
            pass

        def singlepoint(self, *_: Any, **__: Any) -> Any:
            invocations.append(None)
            if len(invocations) > 2:
                raise failure
            return SimpleNamespace(energy=-1.0)

    statuses = iter((0, 700))

    def synchronize() -> int:
        return next(statuses)

    raw, activity = metric_fixture()
    monkeypatch.setattr(generativeqc, "Calculator", Calculator)
    monkeypatch.setattr(
        ctypes, "CDLL", lambda _: SimpleNamespace(cudaDeviceSynchronize=synchronize)
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "CuptiCapture",
        lambda _: SimpleNamespace(
            region=lambda _: nullcontext(), stop=lambda: activity
        ),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "SourceInventory",
        lambda *_: SimpleNamespace(stop=lambda: ({}, raw)),
    )
    output = tmp_path / "failed-capture"
    with pytest.raises(
        RuntimeError, match="original scientific endpoint failure"
    ) as caught:
        capture_metric_case("h2", output)
    assert caught.value is failure
    assert json.loads((output / "activity.json").read_text()) == activity
    assert json.loads((output / "source-boundaries.json").read_text()) == raw
    numerical = json.loads((output / "numerical.json").read_text())
    assert numerical["energy"] is None
    assert numerical["capture_error"] == str(failure)
    assert numerical["cleanup_errors"] == ["observer fence CUDA status 700"]


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_CUPTI_DF_METRIC_TEST") != "1",
    reason="requires a finite scheduled real-GPU allocation",
)
@pytest.mark.parametrize("name", ["h2", "lih", "water"])
def test_actual_df_metric_round_trip(name: str, tmp_path: Path) -> None:
    directory = Path(
        os.environ.get("GENERATIVEQC_CUPTI_DF_METRIC_EVIDENCE", str(tmp_path))
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--capture",
            name,
            str(directory / name),
        ],
        check=True,
        timeout=180,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--capture":
        raise SystemExit("expected --capture FIXTURE OUTPUT")
    capture_metric_case(sys.argv[2], Path(sys.argv[3]))
