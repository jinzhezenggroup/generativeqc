# Decision: use shell derivatives for eligible source-only Coulomb response

Status: implemented
Date: 2026-10-21

## Problem

The source-only pure-J response needs no retained raw or fitted tensor, but its
default derivative selection still required the resident-response profile.
The [source batching investigation](2026-10-21-batched-raw-coulomb-response.md)
isolated generic derivative contraction as a separate large cost: on DF96,
explicit shell execution took about 32.9 seconds versus 471.4 seconds for the
generic contraction. These are component diagnostics, not endpoint speedups.

## Decision and invariants

Admit full-rank source-only pure-J BLAS response to the existing generated shell
consumer, using the existing architecture-independent work threshold. Exclude
borrowed, raw-resident, fitted-resident, occupied/exchange, truncated-metric and
diagnostic-algebra owners. Preserve existing upload/scatter/serial exclusions.

This admission is separate from the resident derivative profile. It does not
enable folded pairs, screening, compact schedules, primitive packets or mixed
arithmetic. Metric order, source regeneration and application precision remain
unchanged. Preflight the ordinary shell metadata, coordinates, output and minimum
response workspace against both existing resource allowances; retain generic
execution if it does not fit. The explicit
`GENERATIVEQC_DF_WEIGHTED_EXECUTION=generic` rollback remains available.

## Evidence

The final candidate is base `f5ca5398f25ed7b31285367b5b8fd829d3b024b5` plus the
retained source patch. Its frozen native library SHA-256 is
`d3f883dc0ae0de87ea7759cbb2dfbfc977d77f5a55f8c88ea8ff60d32b5c4e85`.
All 5,852 measured source/asset inputs match the implementation before commit.
Slurm GPU tests pass 41 cases, including independent PySCF PBE/r2SCAN-3c
energy/force endpoints, the explicit generic rollback, auxiliary g, constrained
storage, signed multidensity adjoints and displaced-geometry lifetime checks.

The earlier candidate's r2SCAN-3c DF48 Standard and Fast complete cold/one-warm
endpoints pass unchanged independent gates. With forced dense values and a
4 GiB DF allowance, their cold/warm times are 188.119/29.931 and 187.013/29.770
seconds, respectively, at 16/1 cycles. That candidate predates the final
budget/algebra guards and is identified separately in the retained local
`source-coulomb-shell-v1` receipts. It is not a final-source README population.
Final-candidate receipts, launchers and checks are retained under ignored
`.artifacts/source-coulomb-shell-v2` and the corresponding n1 directory.

## Remaining work

Shell selection removes repeated derivative work; it does not remove repeated
raw three-center generation during SCF. Fast DF still uses the strict native
fallback where relaxed work has no qualified production contract. A separate
experimental stationarity scan must not be included in the README as native
Fast. Rebenchmark every README population on one frozen production candidate;
retain failed, slower and incomplete cases and do not infer universal leadership
from the isolated derivative improvement.
