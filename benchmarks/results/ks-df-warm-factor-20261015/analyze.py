"""Review complete #2474 populations without dropping samples or overlapping roots."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.df_component_ledger import read_trace
from generativeqc_compiler.common.evidence import block_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts, samples, histories, phases, ledgers, audits, comparisons = [], [], [], [], [], [], []
    energies, reference_energies, forces, reference_forces = [], [], [], []
    source_hashes, libraries = set(), set()

    def load(name, endpoints):
        path = args.raw / name
        data = json.loads(path.read_text())
        assert data["status"] == "measured" and len(data["records"]) == endpoints, name
        receipts.append({"path": name, "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return data

    def native(data):
        source_hashes.add(data["source"]["tracked_diff_sha256"])
        libraries.add(data["native_build"]["library_sha256"])
        assert all(row["gate"] for row in data["records"])

    def trace_operations(trace):
        path = args.raw / trace.parent.name / trace.name
        receipts.append({"path": str(path.relative_to(args.raw)), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return read_trace(path)

    for functional in ["m06-2x", "pbe0", "b3lyp"]:
        for atoms in [3, 48, 96]:
            reference = load(f"reference-{functional}-{atoms}.json", 12)
            for variant in ["baseline", "candidate"]:
                data = load(f"current-{variant}-{functional}-{atoms}.json", 12)
                native(data)
                assert not data["diagnostic"] and not data["first_k_matrix_audit"]
                assert data["protocol"] == reference["protocol"]
                identity = {"functional": functional, "atoms": atoms, "variant": variant}
                for row in data["records"]:
                    expected = next(item for item in reference["records"]
                                    if item["geometry"] == row["geometry"])
                    energies.append(row["energy"])
                    reference_energies.append(expected["energy"])
                    forces.extend(np.asarray(row["forces"]).ravel())
                    reference_forces.extend(np.asarray(expected["forces"]).ravel())
                    sample = {**identity, **{key: row[key] for key in [
                        "phase", "geometry", "replay", "seconds", "prepare_seconds",
                        "complete_seconds", "fock_builds", "energy_error", "force_error"]}}
                    samples.append(sample)
                    histories.append({**identity, "phase": row["phase"], "replay": row["replay"],
                                      "history": row["ks_diagnostic"]["history"]})
                for phase in ["cold", "warm", "moved", "moved-warm"]:
                    rows = [row for row in data["records"] if row["phase"] == phase]
                    phases.append({**identity, "phase": phase, "samples": len(rows),
                                   "median_complete_seconds": median(row["complete_seconds"] for row in rows),
                                   "fock_builds": [row["fock_builds"] for row in rows]})
                if atoms == 3:
                    continue
                traced = load(f"trace-{variant}-{functional}-{atoms}.json", 12)
                native(traced)
                assert traced["protocol"] == data["protocol"]
                for endpoint, clean in zip(traced["records"], data["records"], strict=True):
                    assert endpoint["phase"] == clean["phase"]
                    assert endpoint["fock_builds"] == clean["fock_builds"]
                    trace = Path(endpoint["trace"])
                    operations = trace_operations(trace)
                    first = next(row for row in operations if row["operation"] == "ks_begin")
                    builds = [row for row in operations if row["operation"] == "ks_iteration"]
                    providers = [row for row in operations if row["operation"] in {"ri_k", "ri_k_occupied"}]
                    assert len(builds) == len(providers) == endpoint["fock_builds"]
                    assert not any("warm_seed_audit_dense_k_builds" in row["counters"] for row in builds)
                    scopes = defaultdict(float)
                    for operation in operations:
                        name = operation["operation"]
                        if name in {"ks_begin", "ks_finish", "force_response",
                                    "one_electron_stationary_pair", "one_electron_stationary_pair_prepared"}:
                            scopes[name] += operation["host_completion_ms"]
                        elif name == "ks_iteration":
                            children = {region["name"]: region["host_ms"] for region in operation["regions"][1:]}
                            scopes["ks_jk"] += children["ks_jk"]
                            scopes["ks_xc"] += children["ks_xc"]
                            scopes["ks_other_and_final_diagnostic_wait"] += operation["host_completion_ms"] - sum(children.values())
                    complete = endpoint["complete_seconds"] * 1000
                    scopes["untraced_endpoint_remainder"] = complete - sum(scopes.values())
                    assert scopes["untraced_endpoint_remainder"] > -0.1
                    provider = providers[0]
                    provider_children = defaultdict(lambda: {"regions": 0, "gpu_ms": 0.0})
                    for region in provider["regions"][1:]:
                        if region["parent"] == 0:
                            child = provider_children[region["name"]]
                            child["regions"] += 1
                            child["gpu_ms"] += region["gpu_ms"]
                    ledgers.append({**identity, "phase": endpoint["phase"], "replay": endpoint["replay"],
                                    "begin": first["counters"],
                                    "builds": [row["counters"] for row in builds],
                                    "additive_host_ms": dict(scopes),
                                    "first_k": {"operation": provider["operation"],
                                                "gpu_ms": provider["regions"][0]["gpu_ms"],
                                                "gpu_children": dict(provider_children),
                                                "gpu_other_ms": provider["regions"][0]["gpu_ms"] - sum(child["gpu_ms"] for child in provider_children.values()),
                                                "counters": provider["counters"]},
                                    "profiler_fences": len(operations),
                                    "profiler_final_wait_ms": sum(row["final_synchronization_ms"] for row in operations),
                                    "explicit_ks_synchronizations": sum(row["counters"].get("explicit_synchronizations", 0)
                                                                         for row in operations if row["operation"].startswith("ks_"))})
        audited = load(f"audit-candidate-{functional}-96.json", 12)
        native(audited)
        for endpoint in audited["records"]:
            trace = Path(endpoint["trace"])
            for operation in trace_operations(trace):
                counters = operation["counters"]
                if "warm_seed_audit_dense_k_builds" in counters:
                    assert operation["operation"] == "ks_iteration"
                    assert counters["warm_seed_audit_dense_k_builds"] == 1
                    assert counters["warm_seed_audit_extra_occupied_k_builds"] == 1
                    audits.append({"functional": functional, "phase": endpoint["phase"],
                                   "replay": endpoint["replay"],
                                   "endpoint_energy_error": endpoint["energy_error"],
                                   "endpoint_force_error": endpoint["force_error"], **counters})
        compared = load(f"ab-{functional}-96.json", 22)
        native(compared)
        histories.extend({"functional": functional, "atoms": 96, "variant": row["selection"],
                          "phase": "interleaved", "geometry": row["geometry"], "sequence": row["sequence"],
                          "history": row["diagnostic"]["history"]} for row in compared["records"])
        comparisons.append({"functional": functional, "comparison": compared["comparison"],
                            "timings": compared["timings"],
                            "records": [{key: row[key] for key in ["geometry", "selection", "sequence", "seconds",
                                                                   "fock_builds", "energy_error", "force_error"]}
                                        for row in compared["records"]]})
    assert len(source_hashes) == len(libraries) == 1
    assert len(audits) == 30 and Counter(row["functional"] for row in audits) == {
        "m06-2x": 10, "pbe0": 10, "b3lyp": 10}
    errors = {"energy": block_error(energies, reference_energies, atol=1e-8, rtol=0),
              "forces": block_error(forces, reference_forces, atol=1e-7, rtol=0),
              "k_bound": max(row["warm_seed_audit_k_max_abs_e16_ceil"] for row in audits) * 1e-16,
              "fock_bound": max(row["warm_seed_audit_fock_max_abs_e16_ceil"] for row in audits) * 1e-16}
    assert errors["energy"]["passed"] and errors["forces"]["passed"]
    assert errors["k_bound"] < 2e-11 and errors["fock_bound"] < 2e-11
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in {"summary": phases, "samples": samples, "histories": histories,
                          "work-ledger": ledgers, "matrix-audits": audits,
                          "interleaved": comparisons, "errors": errors,
                          "raw-receipts": receipts}.items():
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"errors": errors, "libraries": list(libraries),
                      "tracked_diff_sha256": list(source_hashes), "endpoints": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
