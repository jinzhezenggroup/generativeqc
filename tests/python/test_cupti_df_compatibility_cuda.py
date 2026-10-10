"""Scheduled, optional CUPTI qualification of the real DF host-tile adapter.

Each case uses a fresh process because CUPTI capture is process-global. Fixture
values are independent numerical references, not residency or lineage oracles.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc.profiles import file_hash

from tools.cupti_residency_capture import CuptiCapture
from tools.cupti_source_capture import SourceInventory, summarize_sources
from tools.generativeqc_posthf.df import DFProvider, MetricFactor
from tools.generativeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.generativeqc_posthf.mp2 import restricted_mp2
from tools.generativeqc_posthf.sources import CudaDFSource


def capture_case(name: str, output: Path) -> None:
    """Capture real calls, preserve journals, then compare source and fixture work."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU capture requires Slurm"
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve(strict=True)
    collector = Path(os.environ["GENERATIVEQC_CUPTI_COLLECTOR"]).resolve(strict=True)
    library_hash = file_hash(library)
    collector_hash = file_hash(collector)
    native = ctypes.CDLL(str(library))
    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    runtime.cudaDeviceSynchronize.argtypes = []
    assert runtime.cudaDeviceSynchronize() == 0
    profiler = CuptiCapture(collector)
    inventory = SourceInventory(collector, native)
    regions = {
        1: {"name": "source-prepare", "role": "prepare"},
        2: {"name": "host-df-mp2", "role": "replay"},
        3: {"name": "source-close", "role": "publication"},
        8: {"name": "observer-fence", "role": "observer"},
    }
    output.mkdir(parents=True, exist_ok=False)
    meta, arrays = load_fixture(name)
    source = None
    try:
        with profiler.region(1):
            source = CudaDFSource(**source_arguments(meta), tile_capacity=64)
            assert Path(source._library._name).resolve() == library
            factor = MetricFactor.from_source(source)
            snapshot = fixture_snapshot(meta, arrays, label="df", metric=factor)
        with profiler.region(2):
            with DFProvider(snapshot, source, factor, auxiliary_tile=3) as provider:
                result = restricted_mp2(snapshot, provider)
                statistics = dict(provider.statistics)
            metrics = source.source_metrics()
    finally:
        with profiler.region(3):
            if source is not None:
                source.close()
        with profiler.region(8):
            assert runtime.cudaDeviceSynchronize() == 0
        raw_graphs, raw_source = inventory.stop()
        raw_activity = profiler.stop()
        for filename, data in (
            ("regions.json", regions),
            ("source-boundaries.json", raw_source),
            ("graph-inventory.json", raw_graphs),
            ("activity.json", raw_activity),
        ):
            (output / filename).write_text(json.dumps(data, indent=2) + "\n")
        diagnostics = summarize_sources(raw_source, raw_activity, regions)
        (output / "source-diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
    energy_error = abs(
        result.correlation_energy - meta["records"]["df"]["correlation_energy"]
    )
    amplitude_error = float(np.max(np.abs(result.amplitudes - arrays["df_t2"])))
    summary = {
        "fixture": name,
        "fixture_array_hash": meta["array_hash"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "library_sha256": library_hash,
        "collector_sha256": collector_hash,
        "reference_kind": "committed-independent-same-Hamiltonian-fixture",
        "energy_error": energy_error,
        "amplitude_error": amplitude_error,
        "statistics": statistics,
        "metrics": metrics,
        "residency_status": diagnostics["status"],
        "zero_round_trip_assertion": None,
    }
    (output / "numerical-and-work.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.savez_compressed(
        output / "numerical.npz",
        amplitudes=result.amplitudes,
        correlation_energy=np.array(result.correlation_energy),
    )
    assert file_hash(library) == library_hash
    assert file_hash(collector) == collector_hash
    assert diagnostics["source_annotations_intact"], diagnostics["issues"]
    counts = diagnostics["by_role"]["compatibility"]
    assert counts["source_transfer_calls"] == metrics["host_staged_tiles"] > 0
    assert counts["source_event_fences"] == 2 * metrics["host_staged_tiles"]
    assert counts["source_stream_fences"] == 0
    assert counts["d2h_bytes"] == metrics["d2h_bytes"]
    assert diagnostics["by_role"]["lifetime"]["source_stream_fences"] == 1
    assert metrics["subsequent_h2d_bytes"] is None
    assert statistics["subsequent_h2d_bytes"] is None
    assert all(row["dependency_identity"] is None for row in diagnostics["operations"])
    assert diagnostics["status"] == "INCOMPLETE"
    assert energy_error < 1e-9
    np.testing.assert_allclose(
        result.amplitudes, arrays["df_t2"], atol=1e-11, rtol=1e-10
    )


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_real_generated_df_event_boundaries(name: str, tmp_path: Path) -> None:
    if not os.environ.get("GENERATIVEQC_CUPTI_COLLECTOR"):
        pytest.skip("requires optional CUPTI collector and scheduled qualification")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU capture requires Slurm"
    destination = (
        Path(os.environ.get("GENERATIVEQC_CUPTI_DF_EVIDENCE", tmp_path)) / name
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name, str(destination)],
        check=True,
        timeout=120,
    )


if __name__ == "__main__":
    capture_case(sys.argv[1], Path(sys.argv[2]))
