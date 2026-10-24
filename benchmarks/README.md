# Benchmarks

Use maintained runners for new measurements and `results/` for reviewed evidence.
Write each run to a new directory under `.artifacts/benchmarks/`; output into
`results/` is rejected. Publication is a separate review step governed by the
[evidence retention policy](../docs/maintainer/evidence_retention.md).

| Task | Maintained entry points |
| --- | --- |
| Public DFT/HF comparisons | `readme_defaults.py`, `compare_gpu4pyscf.py`, `compare_df_direct_endpoint.py` |
| CPU endpoint and linear algebra sweeps | `cpu_hf_high_l.py`, `cpu_linalg_sweep.py`, `native_cpu_ks_force_endpoint.py` |
| Correlated/CBS endpoints | `ccsdt_cbs_endpoint.py`, `df_cbs_qm7_endpoint.py`, `ccsdt_prepared_endpoint.py` |
| Prepared execution and cache behavior | `prepared_lifecycle.py`, `runtime_cache_state.py` |
| Qualification and publication | `readme_defaults_qualification.py`, `../tools/evidence.py` |

Read the selected runner's `--help` and its protocol before measuring. Use finite
Slurm allocations for all real GPU work and preserve scheduler device visibility.
An old source-matched timing is not evidence for a new checkout.

## Historical experiments

Closed experiment drivers, copies of measured scripts, source patches and raw
diagnostics have a different lifetime from maintained runners. The
[granularity migration](results/retention-granularity-20261024/README.md) records
retired paths with a published master commit, original byte lengths and SHA-256.
It also lists the retired issue-specific entry points; restore their source-era
snapshot to use them. There is no implicit Git fetch or external artifact service.

Consolidated campaigns retain all their JSON measurements as JSON **values** in
`measurements.json.xz`, with one authenticated storage manifest. Every number and
string token and array order is preserved, including failed/negative runs. Small
decisions and readable reports remain alongside them. Inspect without history:

```bash
python tools/inspect_retained_benchmarks.py benchmarks/results/issue382-packed-df
python tools/inspect_retained_benchmarks.py benchmarks/results/issue382-packed-df \
  final/384-clean.json --output .artifacts/benchmarks/384-clean.json
```

The logical record path is the original path inside that campaign, including its
original compression suffix. Extraction writes JSON, preserving its tokens; use
the history recovery tool for original formatting/compression or old scripts.
Collections are a migration format for reviewed historical records, not permission
to commit new raw run bundles. Current numerical fixtures remain directly
available to their ordinary offline tests.
