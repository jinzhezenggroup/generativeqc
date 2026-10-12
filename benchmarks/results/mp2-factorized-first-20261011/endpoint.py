"""Frozen-source conventional CPU E+F experiment; original gates stay in raw records."""
import argparse
import ctypes
from dataclasses import asdict
import statistics
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--case", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--scientific", action="store_true")
p.add_argument("--trace", action="store_true")
p.add_argument("--payload", action="store_true")
p.add_argument("--repeats", type=int, default=1)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(a.source / "python"), str(a.source)]
from tools import validate_mp2_public_force as v
from generativeqc.autotune import source_identity

tracker = ctypes.CDLL(None)
if a.payload:
    for name in ("qc_live", "qc_peak", "qc_calls", "qc_total"):
        getattr(tracker, name).restype = ctypes.c_size_t

def measure(operation):
    if a.payload:
        before = tracker.qc_live()
        tracker.qc_reset()
    start = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - start
    receipt = {"seconds": elapsed, "process_max_rss_bytes": 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if a.payload:
        receipt.update({"initial_cxx_payload_bytes": before,
            "peak_cxx_payload_bytes": tracker.qc_peak(), "retained_cxx_payload_bytes": tracker.qc_live(),
            "cxx_new_calls": tracker.qc_calls(), "cxx_requested_bytes": tracker.qc_total()})
    return result, receipt

case = v.validation_cases()[a.case]
if a.trace:
    os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(a.output / "trace.jsonl")
else:
    os.environ.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
holder = []
def cold():
    holder.append(v._calculator(case, "cpu", 512 << 20))
    return holder[0].singlepoint(case.atoms, charge=case.charge, properties=("energy", "forces"))
base, cold_record = measure(cold)
calculator = holder[0]
warm, warm_record = measure(lambda: calculator.singlepoint(case.atoms, charge=case.charge,
    properties=("energy", "forces")))
repeated_warm = []
for _ in range(a.repeats):
    repeat, timing = measure(lambda: calculator.singlepoint(case.atoms, charge=case.charge,
        properties=("energy", "forces")))
    energy_error = abs(repeat.energy - warm.energy)
    force_error = float(abs(repeat.forces - warm.forces).max())
    assert max(energy_error, force_error) <= v.BATCH_ATOL
    timing.update({"energy_abs_error": energy_error, "force_max_abs_error": force_error})
    repeated_warm.append(timing)
changed_atoms = list(case.atoms)
element, position = changed_atoms[-1]
changed_atoms[-1] = (element, (position[0] + 0.007, position[1] - 0.003, position[2] + 0.004))
changed, changed_record = measure(lambda: calculator.singlepoint(changed_atoms, charge=case.charge,
    properties=("energy", "forces")))
raw = {"case": a.case, "source": str(a.source), "environment": v._environment_record(calculator),
    "phases": {"cold_initialization_and_EF": cold_record, "warm_EF": warm_record,
               "displaced_EF": changed_record},
    "results": {key: {"energy": result.energy, "forces": result.forces.tolist()}
               for key,result in (("cold",base),("warm",warm),("displaced",changed))}}
raw["repeated_warm_EF"] = repeated_warm
raw["warm_median_seconds"] = statistics.median(row["seconds"] for row in repeated_warm)
raw["timing_interposition"] = a.payload
raw["diagnostics"] = asdict(base.correlation)
raw["recomputed_source_identity"] = source_identity(a.source)
assert raw["recomputed_source_identity"] == raw["environment"]["library"]["source_identity"]
if a.scientific:
    record, timing = measure(lambda: v.run_case(a.case, case, backend="cpu",
        fd_steps=(0.004,0.002,0.001), budget=512 << 20))
    raw["original_qualification_record"] = record
    raw["scientific_oracles_and_all_target_calculations"] = timing
    raw["scientific_gates"] = {k: val for k,val in record["gates"].items() if k != "resource_plan"}
    raw["original_resource_gate"] = record["gates"]["resource_plan"]
    raw["note"] = "Original FAIL preserved: endpoint API explicitly reports unmeasured sentinel. External C++ requested payload and RSS are separate measurements, not production numeric telemetry."
raw["measurement_scope"] = "Payload runs interpose ordinary global C++ new/new[] across the whole process, including input/output/retained C++ state and initialization; excludes C malloc, aligned-new and allocator overhead. Timing runs omit that interposition. RSS includes all process memory and is a cumulative process high-water value. Cold timing includes Calculator construction and complete E+F, with Python module imports outside the timed scope. Scientific phase includes independent PySCF, FD, covariance, changed geometry and batch target work."
raw["experiment_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.output / "record.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
