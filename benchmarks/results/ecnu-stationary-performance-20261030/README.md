# ECNU A100 stationary force qualification

Complete host-returned energy + analytic-force endpoints on one LSF-allocated
A100-PCIE-40GB, eight math threads, FP64, CUDA 12.9, Release/portable SM80.
Each population has one cold call, three frozen-density warm calls, one moved
call and three moved-warm calls. GPU4PySCF 1.7.6 / PySCF 2.13.1 / CuPy 14.1.1
uses its separate environment without cuTENSOR. Its original reference receipts
are reused with verified protocol and file hashes.

These are **matched-grid implementation diagnostics**, not a grid-converged
production ranking. Use spherical def2-SVP, or the prescribed def2-mTZVPP plus
D4/gCP for r²SCAN-3c. All force response and corrections remain in the timed
endpoint. Gates are 1e-8 Eh energy and 1e-7 Eh/Bohr maximum force error.
WB97M-V/96 has no complete reference cohort and is excluded, not ranked.

Warm medians in seconds (all calls include complete E+F):

| Case | Before | Fixed | GPU4PySCF |
| --- | ---: | ---: | ---: |
| r2scan-3c-48 | 8.260 | 5.705 | 6.392 |
| r2scan-3c-96 | 29.116 | 16.226 | 15.269 |
| m06-2x-96 | 11.672 | 9.835 | 7.419 |
| pbe-96 | 6.435 | 5.254 | 5.345 |

All 13 cohorts / 104 native calls pass. The 1024-point controls give 14.169 s
for r²SCAN-3c/96 and 7.799 s for M06-2X/96; the latter still trails the
7.419 s reference. These controls are separate from the fixed-default table.

Every cold/warm/moved sample, error, residual, iteration/Fock count, executed
control, scheduler receipt and source/binary identity is in
`observations.json.xz`. `anchor-forces.npy.xz` holds independent cold/moved
oracle vectors for every case and native vectors for the four baseline
ablations. All replay vectors were checked before compaction; their complete
raw files remain hash-bound in the ignored run directory. Reference potential
invocations and native Fock builds are distinct work counts.

```bash
python benchmarks/results/ecnu-stationary-performance-20261030/summarize.py
# Restore the exact frozen input; output belongs in ignored run storage.
xz -dc benchmarks/results/ecnu-stationary-performance-20261030/def2-svp.json.xz > .artifacts/def2-svp.json
```

Reproduce through a genuine finite GPU allocation, preserving its device
visibility. Run `python -m benchmarks.readme_dft --method METHOD native --atoms
N --basis-file BASIS --repeats 3 --allow-portable-build --reference REF --output
OUT` in the native environment; generate REF with the driver's `reference`
subcommand in the isolated GPU4PySCF environment. r²SCAN-3c uses
`python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json`.

Raw receipts and reproduction drivers remain in
`.artifacts/ecnu-stationary-performance-20261030/`; ECNU retained the corresponding
`results/stationary-performance/` directory. Additional controls use the
existing `KsOptions(tile_points=1024)` or the native minimum-workspace limit;
they do not change defaults. See the [decision note](../../../.agents/notes/implemented/performance/2026-10-30-stationary-ao-capacity-cliff.md)
for admission, fallback, work attribution, test results and retained failures.
