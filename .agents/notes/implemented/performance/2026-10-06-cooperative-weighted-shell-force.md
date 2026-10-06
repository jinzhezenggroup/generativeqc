# Cooperative weighted contraction for the four heavy s/p/d force classes

Status: implemented default-off candidate; frozen-source qualification passed
Date: 2026-10-06

## Problem and measured selection

The retained 96-atom PBE0 Combined pilot profile (`14afa7a53`) attributed
2.052390 s to angular order six and 1.683450 s to order seven, versus 0.542751 s
to order eight. The order-six/seven workers reported 255 registers, 87,184 bytes
of local memory per thread and one active block per SM. Order six contains three
canonical s/p/d classes, order seven one. These four are the first coverage;
the measurements are motivation, not evidence for this candidate's speed.

## Decision

Reuse `build_coulomb_derivative_algebra(8)` and schedule its existing interned
DAG by dependency level. The compiler emits packed instructions, leaf mappings
and n=0 spatial-state mappings. CTA lanes execute independent nodes per level,
with publication barriers between levels. Lane-varying instruction/output
tables use read-only global loads; placing those tables in constant memory
would serialize distinct lane addresses. Uniform levels/constants retain
constant storage. The existing Boys evaluation, cached
product-center seeds, Dual3 Hermite response and Cartesian state consumer retain
their mathematical ownership. No new derivative recurrence is introduced.

One CTA owns a shell quartet and shares primitive-pair geometry and Coulomb
states. Hermite axes are prepared by three lanes. Each lane owns at most three
AO weights per source, contracts each primitive response immediately into
N-1 atom gradients, and retains neither a private Coulomb auxiliary nor raw
component-gradient arrays. Warp/CTA reduction publishes at most 3*N_unique
force atomics per source; translation recovers the final atom's contribution.
This changes summation order and needs independent numerical qualification.

A dedicated bounded-queue specialization is selected only for angular orders
seven, full-range operators, a proved maximum shell angular momentum <=2,
a resident pair cache and the retained derivative mode. It never instantiates
the generic per-AO force consumer, so that consumer's private frame cannot set
the candidate's resource footprint. Unsupported bases/schedules retain the
existing bounded consumers. Selection is a separate default-off preparation
switch; materialized dddd selection remains independent.

The initial constant-table prototype (`3bede7366`) failed native device linking:
`File uses too much global constant data (0x14870 bytes, 0x10000 max)`.
Header-emitted support reached multiple translation units; lane-indexed DAG
instruction/output data must not consume the module's 64-KiB constant bank.
The revised lowering uses read-only global arrays for those tables, retaining
constant storage only for the small uniform level and literal tables. This is
also the appropriate load layout for different instruction addresses per lane.
The failure was a CPU build result; no GPU benchmark of that prototype ran.

## Validation and pending measurements

Host lowering checks compare all 165 spatial roots with independent evaluation
of the original DAG, including arbitrary Boys leaves and zero rho. Dependency
checks require every arithmetic input to come from an earlier published level.
Production-host dispatch tests cover both layouts, cache/schedule guards, f and
unproved angular bounds, resident ownership, workspace and submission failures.
The native gate compares the four classes against retained raw AD derivatives
and an independent symmetry-orbit force contraction, both spins, signed J/K,
repeated atoms, triangular domains, screened/empty claims and output canaries.
The Libcint gate covers both layouts and prepared replay at a fixed density.

No default promotion is claimed. The initial frozen candidate has been
built and run; production selection is narrowed as recorded below. Compare
complete 96-atom PBE0 warm/moved-warm endpoints with
tracing disabled; collect J/K/force regions and admitted shell/AO/primitive work
in separate diagnostic samples. Record registers, local bytes, shared workspace
and occupancy explicitly, without equating theoretical and achieved occupancy.

## Initial all-four-class device evidence and selection correction

Frozen source `369d1eb2577ec32ced0fc8b5ef22c2645e85acb7` built successfully
with ccache. Slurm job 6273 passed the native test (48.99 s including compilation)
and eight fixed-density Libcint cases (49.40 s). Native preparation/contraction/
publication counters passed for all four classes, both output layouts and spins.
All six 96-atom endpoint records in job 6275 passed the retained independent
reference gates. The experiment retained the materialized dddd opt-in and the
same angular/resident schedule as the earlier materialized-enabled baseline.

The all-four-class experiment regressed. Its plain warm/moved-warm endpoints
were 26.565862 / 26.523096 s, versus 25.780899 / 25.879465 s for the earlier
materialized-enabled source. Each plain phase is a single sample. Diagnostic
warm two-electron force was 7.885071 s versus 7.067407 s. The measured order-six
worker rose from 1.995230 s to 3.480160 s; order seven fell from
1.650240 s to 0.950137 s. Worker timing covers all classes in that
angular pass and must not be added once per class in the ledger.

Both cooperative workers use 221 registers/thread, 144 local bytes/thread,
3,088 static shared bytes and 42,984 dynamic shared bytes per block, and one
theoretical active block/SM. The private-frame reduction from approximately
87 KiB does not imply an endpoint benefit or improved achieved occupancy.
The order-six shared execution cost exceeds its saved per-component work at
this workload. Keep production order six on the retained consumer; select the
cooperative worker only for order seven. Preserve all-four-class lowering and
independent native qualification so order six can be revisited with a better
schedule. This is a measured coverage restriction, not a numerical fallback.


Measured warm admissions for the four lowering classes (job 6275):

| Shell class | Admitted shell quartets | AO quartets | Primitive-AO quartets |
| --- | ---: | ---: | ---: |
| 14 | 116,944 | 37,823,760 | 74,979,216 |
| 17 | 84,800 | 19,535,760 | 40,467,600 |
| 18 | 59,832 | 9,306,432 | 16,324,416 |
| 19 | 29,328 | 14,027,904 | 19,115,136 |

Warm totals equal the materialized-enabled baseline exactly: 92,233,228 shell,
1,263,186,780 AO and 5,944,643,268 primitive-AO quartets. Moved-warm totals
also match exactly: 92,233,517 / 1,263,188,812 / 5,944,656,598. Maximum pilot
errors are 1.06e-10 Eh and 2.16e-11 Eh/Bohr. This establishes numerical and
admission parity, without substituting admission counts for later execution.

Later master integration retains the generic-only 128-thread force selector
and actual-block-width queue admission/drain. The angular cooperative consumer
still uses its fixed 256-lane component ownership. Device results above apply
to the frozen pre-integration all-four-class candidate, not to subsequent
source hashes. The narrowed selection needs its own clean endpoint result;
no final endpoint speedup is inferred by subtracting the pilot's worker times.

Receipts are retained under `.artifacts/cooperative-force/evidence/`
(`gates-6273`, `pbe0-endpoint-6275`) in the author's checkout and under
`.artifacts/force-jk/` in the frozen n1 checkout. Production ledgers count
admitted shell/AO/primitive-AO work. They do not directly count weighted
recurrence calls or force atomics; native actual counters and analytical
scatter bounds must remain labeled separately.

A small native Nsight probe was attempted through Slurm to inspect achieved
occupancy and local-memory transactions. Job 6277 matched no kernels because
Nsight spells template booleans as `(bool)0/1`; corrected job 6278 reached the
consumer but returned `ERR_NVGPUCTRPERM` (performance-counter access denied).
No achieved occupancy or measured spill-transaction value is available. The
native executable still passed its numerical/work checks. These small-grid
probes would not establish representative production occupancy even if counters
were accessible; the endpoint launch-resource ledger remains theoretical.

## Integrated order-seven-only qualification

Frozen source `e1e26fa0e98e668a02beedc1e1cd4887df840c25` incorporates the
latest #2033 base and the review's constant-payload guard plus live Separate
RHF/UHF/third-component-packet cases. Full C++/CUDA build passed with explicit
ccache launchers; the subsequent review-only update reused the build/cache
and completed in seven steps. Slurm job 6280 passed the native gate (49.89 s,
including compilation) and all eight Libcint cases (48.75 s). The integrated
host suite passed 15 tests, compiler structure checked 481 modules with zero
dependency errors, and Ruff/format/diff checks passed.

The final minimal endpoint run (same Slurm job 6280) used materialized dddd plus
order-seven-only cooperative selection. It reused the earlier materialized-on
baseline rather than running another control cohort. Each geometry again has
one diagnostic replay followed by one plain replay; all replays below used
one SCF iteration and one Fock build.

| Measurement | Earlier materialized-on baseline | Integrated order-seven-only |
| --- | ---: | ---: |
| Plain warm complete endpoint (s) | 25.780899 | 24.862497 |
| Plain moved-warm complete endpoint (s) | 25.879465 | 24.829471 |
| Diagnostic warm J (s) | 2.456153 | 2.455658 |
| Diagnostic warm K (s) | 1.944387 | 1.943988 |
| Diagnostic moved-warm J (s) | 2.396846 | 2.428518 |
| Diagnostic moved-warm K (s) | 1.948154 | 1.956800 |
| Diagnostic warm two-electron force (s) | 7.067407 | 6.350618 |
| Diagnostic moved-warm two-electron force (s) | 7.088373 | 6.360110 |
| Profiled warm order-six worker (ms) | 1995.230 | 1988.470 |
| Profiled warm order-seven worker (ms) | 1650.240 | 942.223 |
| Profiled moved-warm order-seven worker (ms) | 1668.080 | 942.098 |
| Profiled warm dddd worker (ms) | 387.067 | 386.830 |

Order-seven worker time falls about 43–44%; the diagnostic force region falls
about 10%. Plain endpoint samples decrease 3.6% / 4.1%, but they are single
samples against an earlier frozen-source baseline, not a robust median or
statistical default-promotion campaign. J/K work is unchanged and their timings
remain close. Cold/moved used 26/12 SCF iterations versus the baseline's 27/16;
do not use those wall times to claim a kernel speedup.

All six final records passed the independent-reference gates; maximum errors
are 1.03e-10 Eh and 2.99e-11 Eh/Bohr. Warm and moved-warm shell/AO/primitive-AO
admission totals match the earlier baseline exactly, including the per-class
counts above. The order-seven worker retains 221 registers, 144 local bytes,
3,088 static + 42,984 dynamic shared bytes and one theoretical active block/SM.
Order six retains its original generic frame and approximately 2-s worker.
Both materialized and cooperative switches remain default-off; this evidence
qualifies the explicit angular/resident opt-in, not the ordinary default route.
Final receipts are in `gates-6280`, `pbe0-endpoint-6280` and `order7-build` under
the local evidence directory; the frozen n1 checkout is
`/data/jzzeng/qc-pbe0-cooperative-order7-20261006`.

## Revisit when

Broaden coverage only after measured endpoint benefit and independent numerical
gates. A class-indexed scheduler could remove the angular partition's repeated
enumeration, but is separate work. Final Fock/force fusion also requires proving
that the last Fock build and force consumer use exactly the same density frame.
