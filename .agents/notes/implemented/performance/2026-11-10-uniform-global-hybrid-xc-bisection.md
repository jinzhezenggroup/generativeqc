# Decision: use one spatial policy for pure and global-hybrid semilocal XC

Status: implemented
Date: 2026-11-10

## Problem

The [earlier extension](2026-11-10-default-xc-bisection-functional-extension.md)
admitted pure GGA and global meta-GGA hybrids but excluded global GGA hybrids
with fewer than four work jets. That distinction encoded an endpoint speedup
threshold, not a scientific, resource or ownership requirement. The explicit
PBE0/B3LYP pilot already passed numerical checks and saved about 0.41/0.47 s;
exact exchange merely diluted the relative gain of the same XC optimization.

## Decision

Remove the exchange/work-jet gate and its policy arguments. Pure and
global-hybrid GGA/meta-GGA now use the same automatic bisection policy. There
is no functional-name allowlist, exchange-fraction threshold, or minimum
percentage speedup for admitting an otherwise qualified functional. Endpoint
non-regression and independent numerical acceptance still matter.

Keep the existing strict-FP64 RKS, four-AO-jet, 512-AO, 524288-point, CC8+ and
low-FP64-throughput prerequisites. Range separation and nonlocal correlation
are not promoted: their additional consumer/ownership contracts have not been
qualified by the global-hybrid endpoints. Resource-ledger/reference-budget,
mixed-precision, UKS, smaller-owner and unqualified-device fallbacks remain.
The resident-grid/local-AO and optional host/device storage admission is
unchanged. Explicit selectors 1/2/0 retain center/bisection/canonical behavior.

## Invariants

Only execution point order changes. Quadrature points/weights, AO cutoffs,
precision, guesses, stopping gates, exact exchange and DF approximation do not.
Canonical moving-grid force ownership, graph replay, transactional publication
and bounded optional-storage refusal remain authoritative. Bisection can
increase retained arena capacity; this is not a peak-memory reduction claim.

## Evidence

Base: master `d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`.
Follow-up source identity:
`b19e22e321aa9e541a98a95ff12da911fb47c29f0dd4121393a89124fea2ae88`.
Production library SHA256:
`581b7114c7610d6f7a3a6c45bc50c68128dea7bb3dd4cba3fa9b2f1fdbf189ef`.

The isolated follow-up build replays the frozen CMake recipes with verified
sccache 0.16.0 and relinks the changed-header C++ closure: `cuda_ks.cpp`,
`dft_method.cpp`, `cuda_rhf.cpp` and the build-identity translation unit.
Individual SHA256 receipts authenticate borrowed unchanged CUDA/generated
objects. This is **not a clean full rebuild**. Native source/include inputs
still differ from the frozen master inputs only in `cuda_ks.cpp` and
`ks_tile_policy.hpp`; the source inventory independently matches this checkout.
Exact launcher/compiler/link commands and before/after cache statistics remain
in the local receipts. Test-hook objects do not replace the production library.

The follow-up qualifies genuinely unset selection against explicit center
grouping in the same source-bound binary, including repeated complete PBE0 and
B3LYP energy/analytic-force endpoints and independent GPU4PySCF stock-grid
references. Earlier source populations remain separately identified rather than
being relabeled as final-policy measurements.

Final-policy cold medians in seconds (center / unset):

| DF endpoint | Independent process pairs | Center | Unset | Saving |
| --- | ---: | ---: | ---: | ---: |
| PBE0, 96 atoms | 3 | 20.491549 | 20.069346 | 0.422203 |
| B3LYP, 96 atoms | 3 | 20.645178 | 20.181458 | 0.463720 |
| PBE, 96 atoms | 1 | 10.327456 | 9.839587 | 0.487869 |
| r2SCAN-3c, 96 atoms | 1 | 23.634940 | 21.869235 | 1.765704 |
| M06-2X, 96 atoms | 1 | 24.191744 | 23.258925 | 0.932819 |

Only the two newly admitted hybrids have repeat-qualified follow-up gains;
the one-pair rows are preservation controls, not repeat-qualified timing claims.
The earlier policy's repeated r2SCAN-3c/PBE/M06-2X/direct-PBE samples remain
separately identified in the compact summary and the linked earlier note.
Cold iteration/Fock-build counts are 9/9 for PBE0/B3LYP and 10/10 for the
three controls under both arms; all warm calls use 1/1.

The endpoint protocol is unchanged: README water clusters and frozen auxiliary
inputs, spherical def2-SVP (r2SCAN-3c uses def2-mTZVPP), stock grids and scientific
controls, fresh processes with one complete cold E/F call followed by three
frozen-own-density warm calls. Cold includes construction through host export,
including SCF, analytic forces and applicable D4/gCP; process/import/context
startup and cache-miss compilation are excluded. A native import guard rejects
PySCF/GPU4PySCF. Independent references run in separate processes. The n4
RTX5090/sm120/CUDA 12.9.1 job preserves Slurm visibility and one numerical host
thread, alternating/reversing center and unset arms. All samples, including
non-headline controls, enter the scientific gates.

Paired E/Fmax/Frms gates are `1e-8` Eh / `1e-7` / `3e-8` Eh/Bohr.
Independent stock-grid gates are energy/atom `1e-5` Eh, force max `1e-4` and
RMS `3e-5` Eh/Bohr. Both cold and warm median non-regression gates allow at
most 5%; a minimum 2% gain is deliberately **not** a per-functional admission
rule. Stock-grid agreement is not a new dense-grid convergence certificate or
a GPU4PySCF timing-ranking claim.

The clean follow-up (n4 Slurm1559, 35-minute finite allocation) completes
18 native processes / 72 E/F calls plus five independent processes / ten E/F
calls. Every export passes the paired and independent gates. Paired maximum
energy/force-max/force-RMS errors are `9.094947017729282e-13` Eh /
`1.5509815654013437e-12` / `3.9335156078617157e-13` Eh/Bohr. The maximum
independent force error is `1.3492184236510094e-6` Eh/Bohr. Hybrid warm median
ratios are 0.991327 (PBE0) and 0.990747 (B3LYP); none of the control cold/warm
medians regresses. No failed or interrupted clean follow-up populations are
excluded.

The native/resource suite passes 448 tests plus four subtests; the removed
exchange/work-jet arguments collapse five redundant policy rows, not numerical
or resource coverage. The compiled spatial contract passes independent CPU E/V,
RKS/UKS, exact point permutations and tails, changed-density graph replay,
canonical force ownership and exact one-byte resource refusal. Compute
Sanitizer memcheck reports zero errors. All GPU execution uses finite-time
`srun` allocations, not the maintenance node n3.

The final-source PBE0 geometry lifecycle adds ten native cold/warm/moved/
moved-warm/fresh-moved E/F calls. Automatic-versus-center parity passes at
every corresponding phase (maximum force difference `6.377121053446899e-13`
Eh/Bohr). Independent moved-grid gates pass (maximum force error
`1.7817805941788833e-6` Eh/Bohr). Both schedules retain the same moved-owner
versus fresh-owner discrepancy, about `1.832938e-6` Eh/Bohr: this is **not**
a `1e-7` fresh-owner parity pass or a bisection regression/fix.

Separate PBE0 diagnostics confirm the shared XC work reduction per Fock:
point-AO visits `206131264 -> 178102976`, point-AO-square work
`41225797696 -> 30228019776` (-26.68%), admitted point batches `8 -> 16` and
retained arena bytes `87338648 -> 133113992`. The arena is not whole-endpoint
peak memory; optional bounded capacity/OOM fallback remains necessary.
Intrusive diagnostic samples are not included in clean timing medians.

n1 Slurm17103 passes the native suite, spatial contract, memcheck and native
lifecycle, then exits at the reference wrapper's unsupported `--moved` flag.
The failed CLI log is retained; it is not a failed scientific gate or a dropped
timing sample. A geometry-only wrapper around the maintained independent
endpoint fixes that harness error. Slurm17104 reruns the independent moved
reference and the separate diagnostics successfully, with the same authenticated
production library. Earlier r2SCAN-3c/PBE lifecycle caveats in the linked notes
are not reclassified as passing tight fresh-owner parity.

## Retention and reproduction

`benchmarks/results/xc-spatial-default-20261110/summary.json` groups the earlier
and follow-up raw timing values under distinct source/library identities. It
preserves every earlier raw timing value and the explicit pilot identities,
adds every final-policy cold/three-warm timing and iteration/Fock count, and
hash-binds the complete local record inventories. Ancillary work/lifecycle/test
details remain in their existing decision notes and the evidence above rather
than duplicating those fields in a nearly full bounded evidence slot. No cap
is raised, unrelated evidence removed, or external release/backup published.

Full E/F arrays, scripts, exact build commands, cache statistics, Slurm receipts,
tests, memcheck and failed CLI receipts remain under ignored
`.artifacts/xc-spatial-uniform-default-20261110/` and the matching n1/n4
`/data/jzzeng/xc-spatial-uniform-default-20261110/` roots. The earlier roots are
not overwritten. The maintained reproduction endpoint is
`benchmarks.readme_defaults`, with `standard --method pbe0` or `b3lyp`,
`--mode df --atoms 96 --warm-repeats 3`, the frozen `--inputs` directory and
separate output files. Use finite-time `srun`, separate fresh selector-1 and
selector-unset processes, alternating order, and separate `gpu4pyscf` oracle
processes. The local follow-up scripts provide the full scheduled campaign and
all-sample gate recomputation.

## Consequences and revisit conditions

One XC scheduling policy replaces a profitability-based distinction between
global hybrids. Revisit if representative complete endpoints regress or a real
numerical/resource/consumer precondition requires a narrower domain, not merely
because exact exchange makes the percentage gain smaller. Broadening the
remaining precision/spin/device/nonlocal domains still needs independent
qualification.
