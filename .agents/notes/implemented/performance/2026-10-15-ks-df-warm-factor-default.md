# Decision: guarded hybrid KS warm-factor replay by default

Status: implemented
Date: 2026-10-15
Related: #2474
Supersedes: [opt-in candidate policy](2026-10-15-ks-df-warm-factor-candidate.md)

## Problem

The opt-in policy was appropriate before independent scientific and complete
endpoint qualification. That qualification is now retained. Continuing to require
an environment switch withholds the measured benefit even from seeds that satisfy
the existing exact-factor and provider-lifetime guards. Lack of a universal
speedup or a global allocator-peak measurement is not a correctness reason to
disable this allocation-free, guarded path everywhere.

## Decision

An unset `GENERATIVEQC_CUDA_KS_WARM_FACTOR` permits exact-factor publication and
replay, as does `1`. Keep `0` as explicit rollback; other values, including an
empty string, disable the path. This promotes the selection policy only, not the
eligibility domain or numerical acceptance gates. Diagnostic matrix auditing
remains off by default.

## Invariants

- Preserve exact density-generating AO coefficients, fixed occupations and the
  original density kernel; do not substitute final canonical orbitals.
- Require the same owner/source/device/stream/shape, valid epoch/generation/rank,
  strict FP64 RKS and existing complete final fitted-projection admission.
- Rebuild U through the shared provider; never borrow stale projection scratch.
- Retain dense first K for UKS, mixed precision, absent capacity, truncated
  metrics, imported or mismatched seeds, and invalidated identities.
- Add no retained numeric allocation and do not change convergence, energy,
  force, physical-residual or final-state export gates.
- A disabled factor-backed cache still reconstructs its original density.
  Re-enabling a density-backed cache must not invent a factor certificate.

## Evidence

The immutable [qualification publication](../../../../benchmarks/results/ks-df-warm-factor-20261015/README.md)
contains 462 native E/force endpoints against independent DF references, strict
first-K/Fock audits and complete-endpoint comparisons for M06-2X, PBE0 and B3LYP.
Its measured source and explicit baseline/candidate binary are retained there;
its historical decision does not itself claim this later policy change.

The 96-atom clean frozen-warm medians improve from 36.562052 to 21.402944 seconds
for M06-2X, 22.820600 to 12.501131 for PBE0 and 20.497722 to 13.205794 for B3LYP.
The 48-atom ordered/tiled provider remains ineligible and follows its incumbent
path. These are scoped measurements, not a guarantee for arbitrary inputs.

Regression coverage distinguishes genuinely unset, explicit on/off, invalid and
empty selections, and selection changes across publication/replay. The focused
native entry point exercises explicit `0`, explicit `1` and unset RKS/UKS cases.

Policy validation passed 146 host regressions, including publication checks. On
n1, Slurm job 8603 used one scheduler-assigned RTX 5090 with a finite ten-minute
limit and passed all six focused native cases. Explicit-on and unset-default
matrix checks both reported K error `2.22045e-16`, zero full-Fock difference and
independent CPU-DF error `6.66134e-16`.

That device check incrementally rebuilt the frozen issue-source port with only
the selector/comment delta in production C++, using verified sccache 0.16.0; the
new source SHA-256 is
`8a30b7a8aaa9c451214528a39b16c4645f68f80c982042c29ecac06eba45f229`.
This is a policy regression, not a new endpoint-timing campaign or qualification
of unrelated changes in the older local checkout.

PR integration onto `cb8781fb1bdcfb9166ef07cfed463c0482dea752` preserves the
upstream radial-source energy anchors, compensated stopping history, nonlocal
initialization and mixed-point precision tracking. The targeted host, phase and
default-inventory suites pass 414 tests with 38 opt-in probes skipped. These
integration checks do not relabel the frozen-source GPU timings as measurements
of this later master revision.

## Rejected alternatives

- Permanent opt-in after qualification hides the benefit without strengthening
  any runtime correctness condition.
- Unconditional occupied K, a size whitelist or new full-U reservation would
  remove the bounded fallback and exceed the qualified resource contract.
- Rewriting the historical publication would misrepresent which source and
  policy its checksums, timings and acceptance decision actually describe.

## Revisit when

New eligibility domains require independent numerical and endpoint qualification.
A measured regression can use explicit `0` while a narrower admission policy is
investigated. Preserve the historical evidence rather than generalizing its
performance claim to unmeasured sources, devices or layouts.
