#!/usr/bin/env python3
"""Dump the per-case diagnostics that collect.py's summary table drops.

collect.py answers "did it match the baseline and how fast". This tool answers
"what did the run actually do", which is the question a bug hunt asks. Every
field here is already present in the result JSON; nothing is recomputed from
outside the run.

Outputs, per result directory:

  <out>/cold-samples.csv   one row per case: convergence, Fock builds, the
                           precision fallback counts, and every settings field
  <out>/scf-history.csv    one row per SCF iteration per case: the full
                           ks_diagnostic.history[] trace (energy/energy_change,
                           density_change_max, physical_residual_max and the
                           nuclear/one_electron/hartree/xc components)
  <out>/work-counters.csv  one row per case: incremental_direct_jk and the
                           precision operator/work counters, which is where an
                           inconsistent counter (valid=false with all-zero
                           values, for instance) becomes visible
  <out>/failures.csv       one row per non-ok case with the exact error string
  <out>/diagnostics.json   the lot, unflattened, for anything not tabulated

Usage:
  python3 tools/scc/diagnostics.py <result-dir> [more-result-dirs ...]
  python3 tools/scc/diagnostics.py <result-dir> --out <dir>

A missing field is written as an empty cell, never as a fabricated zero, so an
absent diagnostic stays distinguishable from a real zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# The whole point of the exercise: report what the run recorded. These lists are
# the fields that carry bug signal, chosen from the observed v2 payload.
SETTINGS_FIELDS = (
    "energy_tolerance",
    "density_tolerance",
    "max_iterations",
    "diis_history",
    "screening_tolerance",
    "xc_grid",
    "xc_grid_points",
    "nonlocal_grid",
    "nonlocal_grid_inherits_xc",
    "nonlocal_density_policy",
    "force_grid_response",
    "initial_guess",
)

PRECISION_FIELDS = (
    "requested_mode",
    "complete",
    "effective_bits",
    "fallback_count",
    "conversion_count",
    "execution_retries",
    "refinement_iterations",
    "strict_refinement_applied",
    "operator_inventory_complete",
    "operator_work_counters_valid",
    "mixed_precision_fock_threshold",
    "mixed_precision_reserved_error",
    "mixed_admission_census",
    "returned_state_identity",
    "skipped_final_fock_builds",
    "post_scf_fock_builds",
    "final_residual_audits",
)

JK_FIELDS = (
    "policy_version",
    "requested",
    "active",
    "quartet_work_counters_valid",
    "anchor_full_builds",
    "delta_builds",
    "periodic_rebuilds",
    "bypass_full_builds",
    "post_scf_full_builds",
    "anchor_updates",
    "max_abs_delta_density",
    "full_candidate_shell_quartets",
    "full_rejected_shell_quartets",
    "full_admitted_shell_quartets",
    "full_admitted_quartet_tiles",
    "delta_candidate_shell_quartets",
    "delta_rejected_shell_quartets",
    "delta_admitted_shell_quartets",
    "delta_admitted_quartet_tiles",
)

SAMPLE_FIELDS = (
    "fock_builds",
    "iterations",
    "energy_change",
    "density_rms",
    "physical_residual_rms",
    "warm_start_used",
    "warm_start_fallback",
    "correction_energy",
)

HISTORY_FIELDS = (
    "iteration",
    "energy_change",
    "density_change_max",
    "physical_residual_max",
    "occupation_stabilized",
)


def cell(value):
    """Render a JSON value for CSV: integers stay integers, None is empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def is_ok(doc):
    """Accept any success status.

    The shipped result schema uses "complete" for a finished calculation and
    "failed" for an aborted one, but earlier cohorts and hand-written stubs use
    other spellings. Treat the known-bad ones as failures rather than pinning a
    single success literal, so a schema rename does not silently reclassify
    every row.
    """
    status = doc.get("status")
    return status not in (None, "", "failed", "error", "running", "missing")


def cold_of(doc):
    for sample in doc.get("samples") or []:
        if sample.get("phase") == "cold":
            return sample
    return None


def load_results(result_dir: Path):
    """Return [(name, doc)] for every *.json that is a result document."""
    docs = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
            continue
        if isinstance(doc, dict) and "schema" in doc:
            docs.append((path.stem, doc))
    return docs


def cold_row(name, doc):
    sample = cold_of(doc) or {}
    settings = sample.get("settings") or {}
    precision = sample.get("precision") or {}
    accelerator = doc.get("accelerator") or {}
    row = {
        "case": name,
        "status": doc.get("status"),
        "error": doc.get("error") or "",
        "method": doc.get("method"),
        "mode": doc.get("mode"),
        "atoms": doc.get("atoms"),
        "arm": doc.get("arm"),
        "device": accelerator.get("name"),
        "compute_capability": cell(
            ".".join(map(str, accelerator.get("compute_capability") or ()))
        ),
        "profile_accepted": cell(accelerator.get("profile_accepted")),
        "protocol": (doc.get("protocol") or {}).get("name"),
        "slurm_job_id": doc.get("slurm_job_id"),
    }
    for field in SAMPLE_FIELDS:
        row[field] = cell(sample.get(field))
    for field in SETTINGS_FIELDS:
        row[f"settings.{field}"] = cell(settings.get(field))
    for field in PRECISION_FIELDS:
        row[f"precision.{field}"] = cell(precision.get(field))
    # HF has no KS diagnostic; record that explicitly rather than leaving a
    # blank that looks like a parse failure.
    row["ks_diagnostic_present"] = cell(sample.get("ks_diagnostic") is not None)
    return row


def history_rows(name, doc):
    rows = []
    for sample in doc.get("samples") or []:
        diagnostic = sample.get("ks_diagnostic")
        if not diagnostic:
            continue
        for entry in diagnostic.get("history") or []:
            row = {
                "case": name,
                "phase": sample.get("phase"),
                "repeat": sample.get("repeat"),
                "scf_domain": diagnostic.get("scf_domain"),
                "grid_points": diagnostic.get("grid_points"),
            }
            for field in HISTORY_FIELDS:
                row[field] = cell(entry.get(field))
            components = entry.get("components") or {}
            for key in sorted(components):
                row[f"components.{key}"] = cell(components.get(key))
            rows.append(row)
    return rows


def counter_row(name, doc):
    sample = cold_of(doc) or {}
    precision = sample.get("precision") or {}
    row = {"case": name, "status": doc.get("status")}
    for field in JK_FIELDS:
        row[f"jk.{field}"] = cell((sample.get("incremental_direct_jk") or {}).get(field))
    row["precision.strict_stage_fock_builds"] = cell(
        precision.get("strict_stage_fock_builds")
    )
    row["precision.mixed_stage_fock_builds"] = cell(
        precision.get("mixed_stage_fock_builds")
    )
    row["precision.scf_fock_timeline_len"] = cell(
        len(precision.get("scf_fock_timeline") or ())
    )
    row["precision.operators_len"] = cell(len(precision.get("operators") or ()))
    row["jk_consistent"] = cell(jk_consistent(sample.get("incremental_direct_jk")))
    return row


def jk_consistent(jk):
    """Flag the counter state that is internally contradictory.

    Returns None when there is nothing to judge, else a short verdict. A
    declared-invalid quartet census that nonetheless reports real delta work is
    fine; a declared-invalid census that reports zeros everywhere while the
    solve clearly did delta work is the inconsistency worth surfacing.
    """
    if not jk:
        return None
    valid = jk.get("quartet_work_counters_valid")
    if valid:
        return "valid"
    delta_builds = jk.get("delta_builds") or 0
    census = [
        jk.get("delta_candidate_shell_quartets"),
        jk.get("delta_admitted_shell_quartets"),
        jk.get("full_candidate_shell_quartets"),
    ]
    if delta_builds > 0 and all(v == 0 for v in census):
        return "INVALID-but-all-census-zero"
    return "invalid"


def write_csv(path: Path, rows, fields=None):
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv[1:])

    for result_dir in args.result_dirs:
        if not result_dir.is_dir():
            print(f"ERROR: {result_dir} is not a directory", file=sys.stderr)
            return 2
        docs = load_results(result_dir)
        if not docs:
            print(f"ERROR: no result JSON in {result_dir}", file=sys.stderr)
            return 2
        out = args.out or result_dir

        cold = [cold_row(name, doc) for name, doc in docs]
        history = []
        counters = []
        failures = []
        for name, doc in docs:
            history.extend(history_rows(name, doc))
            counters.append(counter_row(name, doc))
            if not is_ok(doc):
                failures.append(
                    {
                        "case": name,
                        "status": cell(doc.get("status")),
                        "method": cell(doc.get("method")),
                        "mode": cell(doc.get("mode")),
                        "error": doc.get("error") or "",
                    }
                )

        print(f"\n=== {result_dir} ({len(docs)} result files) ===")
        write_csv(out / "cold-samples.csv", cold)
        write_csv(out / "scf-history.csv", history)
        write_csv(out / "work-counters.csv", counters)
        write_csv(
            out / "failures.csv",
            failures,
            fields=["case", "status", "method", "mode", "error"],
        )

        bundle = {
            "result_dir": str(result_dir),
            "results": {name: doc for name, doc in docs},
        }
        (out / "diagnostics.json").write_text(json.dumps(bundle, indent=1))
        print(f"wrote {out / 'diagnostics.json'}")

        # Flag the two conditions this tool exists to detect, without failing
        # the run: they are observations, not errors.
        no_ks = [r["case"] for r in cold if r["ks_diagnostic_present"] == "false"]
        if no_ks:
            print(f"\nNOTE: no ks_diagnostic (RHF path) for: {no_ks}")
        bogus = [
            r["case"]
            for r in counters
            if r.get("jk_consistent") == "INVALID-but-all-census-zero"
        ]
        if bogus:
            print(f"NOTE: quartet census invalid but all-zero for: {bogus}")
        if failures:
            print(f"NOTE: {len(failures)} non-ok case(s): {[f['case'] for f in failures]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
