# #2054 hybrid provider crossover pilot

This directory owns a benchmark protocol, not a crossover conclusion. The
source-matched runner is [`tools/benchmark_hybrid_provider_crossover.py`](../../../tools/benchmark_hybrid_provider_crossover.py).
The current public CUDA KS contract supports exact Direct J/K and full DF-JK
with the existing occupied exchange reuse path. It does not expose DF-J with
exact K. The runner records that arm as `UNSUPPORTED` and never substitutes a
different approximation.

## Frozen question

- PBE0 RKS, def2-SVP spherical, FP64, `GridSpec()` (48 radial × 16 polar × 32
  azimuth points per atom), screening `1e-12`, energy `1e-12`, density `1e-10`,
  180 maximum SCF iterations, E+force including host return.
- Cases: three-atom smoke; 48-atom nested water; non-water formaldehyde holdout.
  The 96-atom case is only attempted after resource and time review. Nested
  water geometry comes from `benchmarks/readme_hf_scaling.py`; formaldehyde
  comes from `benchmarks/dft_force_matrix.py`.
- DF uses explicit def2-SVP auxiliary basis and a 1 GiB DF device workspace
  hint. These are recorded as the DF approximation identity, separately from
  the common orbital problem. Direct has no auxiliary basis in its provider.
- Cold time includes `prepare_batch` and first complete `batch.execute`;
  warm, changed-geometry, and moved-warm each include complete E+force and
  host-return publication. The ABBA process order is Direct, DF-JK, DF-JK,
  Direct, with one independently recorded unsupported-arm disposition. Every
  run is a separate process; retries receive new output directories.
- Native convergence requires finite E and every force component, the CUDA
  backend, the live provider proof, and physical residual RMS at most `1e-8`.
  Independent PySCF PBE0 uses the exported frozen quadrature and grid response
  with the same orbital/auxiliary basis and requires convergence, energy error
  at most `1e-8` Hartree, force max absolute error at most `3e-7` Hartree/Bohr.
  `summarize` rejects absent or mismatched records. No tolerance is relaxed
  after seeing a result.

## Reproduction

Run only on a source-matched installed build in a finite Slurm GPU Job. Set
`PYTHONPATH=python:.` and `GENERATIVEQC_LIBRARY` to the loaded library path.
The output below is ignored working evidence; review and compact small records
before adding any to Git.

```bash
python tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-3 water-48 formaldehyde \
  --output .artifacts/benchmarks/hybrid-provider-2054/native-1
```

Run each supported native record's independent oracle in a separate process,
preferably a CPU HPC allocation with the same source and PySCF version:

```bash
python tools/benchmark_hybrid_provider_crossover.py run-reference \
  --native .artifacts/benchmarks/hybrid-provider-2054/native-1/water-3-direct-0.json \
  --output .artifacts/benchmarks/hybrid-provider-2054/oracle-water-3-direct-0.json
```

In a separate GPU diagnostic process per supported arm, run `run-profile`
with `--case`, `--arm`, `--trace` and `--output`. It records actual trace
counters and requires a completed occupied projection reuse receipt for the
DF-JK arm. Never add profile durations to clean ABBA timings.

For each ABBA pair, pass its Direct, DF-JK and unsupported records, the
matching two PySCF records via `--oracle`, and both diagnostic records via
`--profile` to `summarize`. Review both pair summaries and every raw losing
record. A passing pair is a **pilot** only;
full #2054 still needs a supported third arm or explicit issue disposition,
96-atom and holdout coverage, measured memory/work decomposition, and
reviewed complete-endpoint crossover evidence before any policy discussion.

Grid NPZ exports and full raw attempts remain in task-owned shared storage.
Provider metric records describe allocations and planned peaks; they are not
measured device peak memory. Missing work counters stay absent. Diagnostic
traces and profiles must be collected in a separate pass and never mixed with
clean endpoint times.
