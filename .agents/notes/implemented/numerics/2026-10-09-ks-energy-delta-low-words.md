# Decision candidate: retain CUDA KS energy low words with smaller K pair caches

Status: implemented; frozen candidate qualified; compatibility repair host-validated
Date: 2026-10-09

## Problem

The qualified unified K selector at `dee3d522b` spends substantial work in
`ddds`. The three preserved pair-cache prototypes improve prepared K but fail
at least one predeclared complete Cold E+F gate. The final two-class prototype
changes 48-atom Fock counts from `[19,19,19]` to `[19,20,20]`.

Both extra-build rows have iteration-19 energy change
`1.3642420526593924e-12 Eh`, above the unchanged `1e-12 Eh` gate. Their density
and physical-residual histories nearly match the 19-build baseline. This
directly identifies an energy veto, not slower density/DIIS convergence. It
does not prove that a particular atomic update is its unique source.

Existing CUDA diagnostics already compensate the three electronic AO traces.
They nevertheless collapse each trace's low word into a large binary64
component, add large totals including fixed nuclear repulsion, then subtract
rounded totals. Independent actual-kernel and Decimal tests expose the lost
low words. Such rounding can cause both false passes and false vetoes.

## Candidate

Retain the diagnostic trace sums and corrections through the electronic-energy
sum and its difference. Nuclear repulsion cancels exactly within an immutable
geometry owner. A shared host/device two-sum helper preserves subtraction's
roundoff and both corrections. Cold first evaluations retain the infinite
baseline; compatible host-controlled warm baselines retain both words.

Reported components and total energies, FP64 products/matrix storage, operator,
DIIS, density/residual/max-residual gates, iteration budgets, precision policy,
screening and production oracle boundaries stay unchanged. This is improved
energy-difference evaluation, not a tolerance floor, residual-only stopping or
equal-iteration assumption. Provider/atomic and product-rounding noise can
remain; fewer Focks and bitwise reproducibility are not guaranteed.

Two extra doubles per scalar record and one per device control are bounded
private payload. Existing arena/transfer accounting uses `sizeof`; padding may
change its charge. No new launch, matrix sweep or allocation object is added.
The diagnostic test's canary moves after the enlarged 136-byte private record.

Retain the sm120 `ddds` canonical/materialized/rolled value schedule and the
smaller `dppp` table. In the dense emitted-subset probe, `ddds` expensive
order-four calls drop 64 to 16, with four order-two calls and 64 products.
This is a source work count, not executed FLOPs or a guarantee for pruned work.
`dppp` retains eight/four calls and 32 products, caching four instead of eight
entries. Force, non-Fock schedules, queue ownership and portable profiles stay
unchanged. Reuse the existing coefficient algebra, not another recurrence.

### Subsequent source review: mixed arithmetic compatibility

Exact regeneration of `9db85cf78451134d037513f73b583751e1c84ce7` found that
canonical orientation moved the mixed `ddds` sign declaration into an
indentation-sensitive `double` to `float` replacement. Its coefficient/product
chain therefore narrowed unintentionally, even though the strict-FP64 campaign
passed. The curated mixed-method summaries do not establish per-class mixed
`ddds` oracle coverage. The source repair explicitly retains the previous wide
products using the default-false `mixed_pair_products_fp64` schedule option,
enabled only in the tuned `sm_120` `ddds` Fock schedule. A global class exception
was rejected because it would also widen existing custom canonical schedules.
The flag round-trips through manifest/tuning serialization and derived schedules;
ordinary host evaluator tests compare exact binary32-input products with an
independent Decimal oracle. Float accumulator storage remains unchanged.

The earlier unrolling description was also incorrect: the frozen emitter used
the unchanged force schedule's flag and emitted `#pragma unroll 1` in the Fock
helper. Metadata now reflects that rolled source without changing its pragmas.
Frozen measurements and binary identities below are retained unmodified. No
new real-GPU or performance qualification is claimed for the repaired source.
Its source identity is
`6d517c95ad467049382bee31d10630150fd706a6e594988423f7d48564c09b8e`.

## Acceptance and provenance

Frozen candidate identity/library SHA256:
`d087c507d3e02b69b224a01d3dbb6bac478f152670452fc7a3710e38d152f9aa` /
`5c9ac43c8c3edf32e1cb85fc8ee872a84a3794277b70019f26aec903265fcba0`.
Baseline library:
`1426e40973b32359dca90d0cfa106eb62d1052eddb07af50ce12c4b86f079376`.
CUDA 12.9.1/GCC12, verified ccache 4.5.1 with the existing cache, RTX5090/node4,
finite exclusive Slurm allocations and preserved scheduler visibility.

Host arithmetic/energy probes: 33 passed, 45 GPU resource skips; real GPU
energy/trace suite: 71 passed, memcheck zero errors. Native focused existing
cases cover LDA/PBE/r2SCAN RKS/UKS, PBE0, RSH, public WB97M-V, mixed/FP64,
warm/seed/chunk/replay and failed-state behavior against independent CPU
references. Integral profile/codegen/recurrence/task tests: 246 passed;
compiler structure: 504 modules, zero errors. All 22 non-Fock definitions and
21 streaming queue bodies, plus tested portable profiles, retain their source.

No full stock native-suite pass is claimed. Its ledger/local-map route-5
assertion also fails on the unchanged frozen baseline. A diagnostic-only copy
excluding that assertion then reproduces its legacy WB97M-V final-model
rejection on that same baseline. Production tests and model/admission policy
are not changed to fix unrelated failures. The focused native harness reuses
unchanged cases; original failures and diagnostic copies remain in raw evidence.

The predeclared promotion still requires >=1% complete mean AND median gain
at both 384/768 AOs: five Cold observations/policy at 96 atoms, three at 48,
alternating policies. Preserve all actual Focks, including slower valid rows.
Four warm/moved owners retain 32 holdout rows, resupplying moved coordinates
every moved-warm call; no phase may regress more than 3%. Prepared-K includes
upload/transforms/K/projection/export/sync and excludes preparation.

Independent acceptance remains K error <1e-8, E error <=1e-8 Eh and max force
error <=1e-7 Eh/Bohr, with strict E/D thresholds 1e-12/1e-10, screening 1e-12,
spherical def2-SVP and the same 48x16x32 PBE0 grid. Priming is scientific
sanity/cache preparation, never a Cold promotion sample.

## Complete measured result

Frozen clean jobs 688/695 retain 144 prepared-K wall observations, 16 Cold E+F
rows and 32 holdout rows. Every numerical gate passes. Prepared-K full-density
mean/median gains are 8.14%/7.97% at 384 AOs and 7.80%/7.67% at 768 AOs.
Dense admission totals stay 23,480,495/81,907,624 generated quartets, not
primitive/root or native-fallback counts. The isolated single-launch `ddds`
diagnostic is 143.638638 to 60.513141 ms (57.87% less time); static resource
reports are unchanged, not evidence of executed FLOPs, spills or occupancy.

| Atoms | Baseline mean / median | Candidate mean / median | Mean / median reduction |
| --- | ---: | ---: | ---: |
| 96 | 102.388464 / 105.571264 s | 99.333829 / 96.292204 s | 2.98% / 8.79% |
| 48 | 42.130438 / 42.079572 s | 41.242694 / 41.233890 s | 2.11% / 2.01% |

Actual Focks: 96 baseline `[19,17,19,17,19]`, candidate `[17,19,19,17,17]`;
48 both `[19,19,19]`. Every valid row is retained; endpoint gains include
changed SCF work. Statistical significance and universal profitability are
not claimed. The final 48-atom priming sanity still needs 20 Focks with a
compensated iteration-19 delta of `1.2369186011243525e-12`; this is retained
outside the promotion population and demonstrates remaining input noise.

Warm/moved-warm mean/median gains stay below 1%; moved-first gains 2.19% with
13 Focks for both policies. No holdout regresses. Maximum E/F errors are
`9.095e-12 Eh`/`4.209e-11 Eh/Bohr`; independent K max error is `6.058e-15`.
320 public/prepared K cases, 864 production task matrices, six task/union
sanitizer logs, native through-f response and Cartesian order-two finite
differences all pass. Focused method/state qualification and energy memcheck
also pass; the stock-suite failures above are still not relabeled as passes.

Curated complete timings/work/accuracy and negative populations are under
`benchmarks/results/ks-energy-expansion-20261009/`; full raw forces, histories,
prototypes and reproducible source patches remain in ignored local archives
identified there by checksum. No Release or external backup is published.

## Retained failures and revisit conditions

Keep every earlier rejected population and initial unformatted diagnostics.
The initial private canary expectation is repaired, not its numerical gates.
A first-sanity driver using QA's one-thread environment correctly fails the
independent eight-thread reference protocol before any complete observation;
the corrected driver restores its matching policy environment. Never bypass
source identity or reference-protocol checks.

Revisit remaining input/provider/product roundoff if complete endpoint or
holdout work still regresses. Do not rerun a fixed population until a favorable
subset appears, remove slow valid trials or weaken physical stopping criteria.
