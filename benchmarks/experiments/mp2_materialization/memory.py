"""Opt-in complete-history native-heap observation, separate from timing runs.

Track imports, calculator construction, preparation and all six E+F calls in a
fresh process. This is a process-heap high-water mark, not a per-buffer estimate,
RSS measurement or GPU ledger. Retain Memray's aggregated high-water allocation
records; a full chronological event stream is not needed for this measurement.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks._retention import raw_output_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=raw_output_path)
parser.add_argument("--native-traces", action="store_true")
arguments = parser.parse_args()
output = arguments.output.resolve()
binary = raw_output_path(output.with_suffix(".bin"))
endpoint_output = raw_output_path(output.with_suffix(".endpoints.json"))

import memray

sys.argv = [str(Path(__file__).with_name("endpoint.py")), str(endpoint_output)]
with memray.Tracker(
    binary,
    native_traces=arguments.native_traces,
    file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
):
    runpy.run_path(sys.argv[0], run_name="__main__")
reader = memray.FileReader(binary)
allocations = list(reader.get_high_watermark_allocation_records(merge_threads=True))
output.write_text(
    json.dumps(
        {
            "schema": "generativeqc.complete-CPU-history-heap.v1",
            "memray_version": memray.__version__,
            "native_stack_traces": arguments.native_traces,
            "file_format": "AGGREGATED_ALLOCATIONS",
            "scope": "fresh-process imports, constructors, preparation and three cold/warm E+F histories",
            "native_heap_high_water_bytes": sum(record.size for record in allocations),
            "native_heap_allocations_at_high_water": sum(
                record.n_allocations for record in allocations
            ),
            "raw_stream": binary.name,
            "timing_qualification": False,
        },
        indent=2,
    )
    + "\n"
)
