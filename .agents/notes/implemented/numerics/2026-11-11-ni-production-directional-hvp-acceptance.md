# Accepted production Ni UKS directional HVP, not a full-Hessian qualification

Date: 2026-11-11

## Scope and source identity

The canonical asymmetric neutral NiH3 doublet now passes its frozen independent
action gates using repository production callables, without source rewriting,
callable replacement, displaced production SCFs or reference-derived production
integrals. The first/directional/shared-spin and grid mechanisms are implemented
in `591c2ff40235f48f3201366005e42a6e602ad26b`; see
`../performance/2026-11-11-first-source-and-grid-hvp-work.md`.

The actual run started against parent
`de25a2604248ff0e82f8ac7a7c8e2fc499bece64` plus dirty numerical files later
committed byte-for-byte in `591c2ff40`. Preserve that original parent-plus-dirty
receipt rather than relabeling it as a clean later-commit run. Every recorded
production source/script/binary hash is unchanged through acceptance. The
rebuilt native library SHA256 is
`a451331d1b2fb2d613d0bf4c80dd0dc4a36cd08bd57183c0a0e1532fbcb668d4`.
This library includes the imported master native policy, unlike the previous
experimental action's older binary.

The method remains B3LYP/VWN-RPA, real-spherical external def2-SVP, native default
Treutler grid, CPU direct FP64 and no density fitting. Ni/H positions in Bohr are
`(.01,-.02,.03)`, `(1.8,.1,.03)`, `(-.5,2.4,.3)`, `(-1.7,-.8,1.1)`.
There are 46 public AOs, 51 Cartesian AOs, 20 shells and 50,064 grid points;
alpha/beta occupations are `(16,15)`. The native SCF takes 123 iterations and
returns energy `-1509.5720618893429` Hartree.

## Independent gates and observed values

The pre-frozen protocol is `/tmp/gqc-continue-ni-second-order-gates.json`, SHA256
`8f15ecb93a8095c07861ad85e24498e670323306a8f5578ae3201d88a96cda95`.
It is not edited or relaxed for this run.

| Quantity | Observed | Original gate |
| --- | ---: | ---: |
| Central density max error | 7.872758001070679e-12 | 2e-7 |
| Central energy error (Hartree) | 2.7284841053187847e-12 | 2e-8 |
| Independent orbital-gradient norm | 4.003441143467981e-11 | 2e-9 |
| Alpha-induced Fock max error | 3.3134606169937797e-13 | 2e-9 |
| Beta-induced Fock max error | 2.987263214571101e-13 | 2e-9 |
| True coupled response residual | 2.83622151735528e-12 | 2e-10 |
| Raw translation error | 2.0192181260370035e-15 | 2e-8 |

One coupled nuclear response solve takes 23 solver iterations. All eight source
components are present: one-electron, Coulomb, XC AO/grid/weight, exact exchange,
overlap/Pulay and nuclear. Their sum agrees with the returned unprojected action
at 2e-12 absolute tolerance. Displaced production SCF count is zero.

All 72 selected independent state receipts and array hashes are reverified.
The retained independent full matrices, not a native full matrix, supply three
separate actions at the fixed direction:

| Independent gradient step (Bohr) | Raw matrix-action max error |
| --- | ---: |
| 0.001 | 1.256752251155291e-7 |
| 0.0005 | 3.237057666360954e-8 |
| 0.0003 | 9.918640075934348e-9 |

All pass the original 1e-5 gate. The older experimental action and its distinct
source identity remain preserved in
`../../proposed/2026-11-11-ni-complete-experimental-hvp-acceptance.md`.

## Complete process timing, resources and semantic work

The successful validation process exits zero in 2148.69 seconds (35:48.69),
including native SCF, independent central-state/spin checks, HVP and acceptance.
The HVP assembly timer is 1934.8474994786084 seconds; peak process RSS is
2,992,292 KiB. The explicit 2 GiB grid lease is only a sub-owner budget, not a
claim that every process allocation fits within 2 GiB.

The first-source diagnostics report 1,959 raw-first calls, 26,199 scalar-direction
calls, 494,532 primitive records, 9,426,123 primitive-component evaluations and
485,661 factorized record-local geometry evaluations. These are successful
source work, not the number of downstream alpha/beta Fock scatters. The HVP
records 48 grid actions and 48 point-kernel reuses, with no point-kernel build or
fallback during the HVP itself. The earlier independent induced-Fock checks
already warmed that owner-local kernel. Its panel is 40,051,200 bytes with an
80,102,400-byte reservation across the feature/scalar budgets.

This is a warmed-artifact native-plus-reference validation process, not a cold
installed public endpoint benchmark or a paired performance comparison with the
previous experiment. Do not infer a molecular speedup from the timing difference.

## Receipts, failure preservation and local recovery

Successful raw receipts/arrays are under
`/tmp/gqc-ni-production-directional-grid-no-watchdog/`; the driver is
`/tmp/gqc-ni-production-directional-grid.py`. Its log, time and cache-stat files
use the same successful-directory prefix.

The earlier signal-11 run at the added traceback watchdog's 900-second deadline
remains failed under `/tmp/gqc-ni-production-directional-grid-final/`, including
its exact original harness archive. A compiler-only GDB reproducer isolates the
watchdog-thread `dump_frame` crash without native integral execution; the
no-watchdog compiler control exits normally. The successful run removes only
that added diagnostic; every scientific gate and the external 3,600-second
process timeout remains. See the linked implementation note for exact logs.

Ignored local recovery files are
`build/qualification/ni-production-hvp-20261111/local-recovery.tar.gz` and
`manifest.json`. The archive is 60,139,611 bytes, contains 211 verified members,
and has SHA256
`cecb88944dec7fddb5baa5cff78f7bdd26eb6f899be881fc3cd5bd3637a9c103`.
It preserves the production code/binary, exact successful and failed drivers,
reference inputs/arrays, raw test and acceptance receipts and the prepared
full-matrix driver. Its input-manifest SHA256 is
`5d62814679a7a90223134dd3878cd8fc401507d9fa433b76391a397cea36bd27`.
Verification is `/tmp/gqc-production-ni-hvp-recovery-verification.json`.
Nothing is uploaded as a Release, external binary asset or new backup host.

## What is still not qualified

This accepts one complete production Ni directional action. It does not supply
a native raw full matrix, independent native H@v/bilinear/full-translation gates,
an installed public Ni curvature endpoint, D3/ECP totals or a TS/frequency result.
No capability or issue-closing authority follows from this action alone.

The next separate protocol, `/tmp/gqc-ni-production-full-matrix.py`, calls the
real serial `composite_rks_hessian` endpoint for all twelve coordinates and then
an independent second native direction for H@v/bilinear checks. It starts from
clean `591c2ff40`, rechecks the central state/spin response, retains all frozen
matrix gates and a finite eight-hour process limit, and writes under
`/tmp/gqc-ni-production-full-matrix/`. A running protocol is not an accepted
full matrix. Mixed Ni/I/Cs and unrestricted ECP acceptance, their exactly-once
D3 totals, installed qualification and fixed-method TS/final frequencies remain
required for issues #2844–#2848.

The subsequent clean-start serial full-matrix run now finishes and passes its
original complete gates. See the separate
[full-matrix acceptance](2026-11-11-ni-production-full-matrix-acceptance.md)
for that later evidence boundary; the directional receipt above is unchanged.
