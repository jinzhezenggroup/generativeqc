# Decision: recover occupied XC work avoidance through audited SCF proposals

Status: implemented, opt-in; qualification is source-bound separately
Date: 2026-11-04

## Problem

PR #2971 correctly observed that a small feature discrepancy alone is not an
XC energy/potential/force certificate. Its exact-mode repair necessarily does
both orbital projection and dense original-D work. Merely removing the exact
comparison would revive the scientific problem, not recover a safe fast path.

## Decision

Change the consumer contract, not the scientific acceptance tolerance. Treat
occupied contractions as a bounded nonlinear SCF initializer. The explicit
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` route may borrow the current canonical frame
and use an admitted finite feature proposal without
reading the original matrix at that point. Unsupported points recompute the
same live D. Default/unset/invalid controls remain matrix-only.

Cheap finite projection/feature, coefficient-range and capacity checks guard
initializer work. Proposals do not compute absolute-dot bounds: those bounds
do not certify a published physical value under this consumer contract, and
their directed-rounding cost needlessly erodes the shortcut. Exact-mode
callers retain the original coefficient/jet/absolute-dot proof envelope,
stricter prefilter and complete bitwise verification.

Reuse the existing bounded initializer handoff and strict target machinery.
The proposal stage ends on residual RMS at most 1e-6, apparent convergence, or the
finite warmup cap (at most eleven iterations, leaving a fresh target build).
These switches schedule work; they never authorize a result. Before any
publication, reset the energy epoch and run the
original stored-D target under the original physical and canonical gates.
DIIS history remains a density predictor for at most six unsuccessful strict
sweeps and is then cleared; no stored Fock ever certifies a physical result.
Near-stationary inputs with prior residual RMS at most 1e-6 do not admit a
candidate at all. This avoids needless initializer/epoch work on warm replays.
Numerical proposal failure retries that same D strictly and never retries
another proposal. The overall SCF iteration limit remains finite.

`CudaXcOccupiedDensity.provisional` is explicit and defaults false. The native
plan refuses it in `StrictAudit`, response and unsupported precision/layout
contexts. There is no new C/Python public method, CPU oracle in production,
global scratch allocation, scientific kernel launch or synchronization.

## Invariants

- No provisional E, Fock residual, warm seed, final density or force certifies
  convergence. Final-state and derivative consumers use original matrix work.
- Frame ownership/generation, stream ordering and one-use tmp1 lease survive.
  Failed/rejected proposals and new solves revoke that lease.
- Exact-mode orbital and absolute-bound panels require `2 * rank <= active`.
  Proposals use only the orbital panel and require `rank < active`, reducing
  contraction work without reserving an unused second panel. Compact AO mapping
  and original-D fallbacks remain.
- FP64 reassociation is not reported as FP32 arithmetic. Traces expose the
  provisional frame generations; final traces must return to matrix work.
- Performance evidence must include handoff, dense refusals and strict
  refinement in complete cold/warm/moved E+F, not just projection throughput.

## Rejected alternatives

Do not invent a universal XC Lipschitz constant from the feature bound, relax
scientific gates, sample a few original points as a proof, restore the old
unverified default, or call full per-point dense verification an acceleration.
Unconditionally clearing DIIS at handoff was independently tested at
`681f4c77962e1d30ae7d056dae54e99e218d05a7`. Its 48 complete native endpoints
passed the scientific gates but added two RKS cold/moved iterations and three
UKS warm iterations, losing the projected work savings. Keep this rejected
population separate rather than calling projection-only savings a speedup.
Keeping the absolute-dot feature budget in a proposal-only route also proved
unhelpful: it incurs directed-rounding work but does not replace strict target
acceptance. A finite proposal is never treated as an energy/force certificate.
Adding interval derivatives for every supported Libxc point program is a much
larger project; initializer auditing avoids requiring that contract at all.
Retaining the two-panel rank gate after removing absolute bounds would reject
valid one-panel proposals on localized AO tiles. Capacity and work admission
must follow the storage actually consumed, not the exact-mode scratch layout.

## Evidence and promotion boundary

The complete 96-atom r2SCAN-3c regression at
`6918c4584b85a5571dc69cbab35fde77fade09d7` isolated another cost: passing null
absolute storage to the certified tiled specialization still retained costly
runtime-predicated directed-rounding instructions. Its occupied producer took
5.158219 GPU seconds for 231 launches, plus 132 matrix launches, versus
5.212594 seconds for all 363 matrix launches. Compile proposals through the
existing non-certifying specialization instead. Compact matrix fallback shares
that specialization's panels; exact callers retain their certified producer.

At `07ec901f833c584518a6cc8481efef5649ba90ab`, three fresh-process alternating
pairs on the original 96-atom Direct Standard workload give complete cold E+F
medians 30.500666 → 28.909603 seconds (5.22% less time, above the retained 2%
variability floor), with 10 iterations/Fock builds on both routes. Frozen warm
11.139000 → 11.232632 seconds is below the floor and is not a speedup. The
independent default-grid small cases and audited dense-grid scaling oracles
retain their original gates. A separate same-image profile gives combined
density-product device time 5.216700 → 3.674313 seconds across cold+warm,
including matrix refusals and the strict tail, not just an isolated proposal.
Source-bound vectors, all rejected populations, work, sanitizer/resource data
and offline readers are retained in the
[qualification capsule](../../../../benchmarks/results/occupied-xc-audited-proposals-20261104/README.md).
The superseded October 23 full report is byte-verifiably recoverable from
published `ffa6936968d5c5a33194bd9bc09435a9f2ee0d05`; current synthetic profile
tests do not require its payload. This keeps new full-precision qualification
inside the unchanged checkout budget without an external archive or lost
historical observations.

Executable host/device poison-D probes distinguish actual work avoidance from
exact verification; ordinary exact calls must still read and reject poison D.
Device tests retain independent stored-D comparisons, Gram underflow,
cancellation, RKS/UKS/empty spins, mapped/tail/scalar/warp domains and compact
scratch/capacity fallback. Native tests require a strict matrix tail before
publication and independent original/displaced default-grid E+F gates of
2e-9 Eh and 1e-7 Eh/Bohr, without production CPU derivatives.

No default promotion or all-functional speedup is implied. Retain source,
binary, scheduler, complete endpoint/work and sanitizer evidence with any
measured claim; historical #2971 receipts are not new-source evidence.

Supersedes the production exclusion in the
[matrix-fallback decision](2026-11-04-occupied-xc-production-matrix-fallback.md),
not the lower-level
[exact-identity contract](../numerics/2026-11-04-occupied-xc-exact-identity.md).
