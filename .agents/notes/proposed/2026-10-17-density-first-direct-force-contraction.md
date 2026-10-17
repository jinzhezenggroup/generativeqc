# Trial: contract pure-J densities into shared Hermite responses

Status: proposed; implemented in an isolated default-off qualification export
Date: 2026-10-17

## Problem and boundary

Shell-shared order-six preparation removes repeated primitive and Coulomb/Boys
work, but the cooperative consumer still repeats a Cartesian response
convolution for each admitted AO quartet. The corresponding order-seven
consumer has the same remaining arithmetic. Another page/CTA-size experiment
does not remove that work.

This trial extends the source-frozen master `a6ac79757e38be772f3a22c5e0b5b4bea314fad3`
plus the six scientific compiler files from force PR #2558
(`719214668aeb69146174a7039490fd95d5607533`) and the separate order-six relative
drain. It neither enables occupied XC nor changes FP64, screening, convergence,
basis, grid, method parameters, or SCF trajectories deliberately. Both force
PR dependencies and their original source-specific measurements remain distinct
from current master.

## Decision

For a complete AO-Schwarz mask, pure-J weights factor into symmetrized total
density pair weights. Contract the bra and ket densities and fixed AO
normalizations into primal and three relative-response Hermite coefficients
before joining them through the Coulomb recurrence. For a repeated shell pair,
half the ordered product reproduces the original triangular canonical orbit.
The independent host orbit test includes nonsymmetric RKS and UKS densities.

The compiler owns the projection, contraction and response/scatter algebra.
The Gaussian, primitive-pair, relative-Hermite and Coulomb/Boys traversal is
extracted into one shared lowering used by both consumers; an opt-in one-off
comparison confirms byte-identical emission of the retained component consumer.
That historical byte comparison is not a permanent CI source fingerprint.

Runtime code only selects a bounded specialization and its workspace. Explicit
`GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=relative-density` captures this experiment
with the prepared source. Automatic selection is unchanged. Order six retains
its independent class-domain drain; order seven retains its existing page size
and queue. Other angular classes and low-order consumers do not acquire the new
workspace. No additional resident allocation or public ABI is introduced.

## Invariants and fallbacks

- Only exact full-range pure J is factorized. Nonzero exchange and bilinear
  seeds retain the original component contraction, including separate sources.
- One failed AO-Schwarz admission rejects factorization of the entire quartet.
  The original screened component consumer then runs without relaxing a gate.
- Unsupported angular domains decline before reading a bounded projection
  array. Only order-six/seven s/p/d quartets are admitted.
- Non-finite input/projection/gradient/reduction intermediates decline before
  any force publication. The retained consumer receives the original inputs;
  no partial scatter is replayed or silently dropped.
- Source signs, independent/repeated atom ownership, canonical symmetry,
  translation recovery, Cartesian/spherical projection and strict FP64 remain
  unchanged. Separate source mode leaves the zero K channel untouched.
- Resource bounds remain below 44 KiB for the generated recurrence frame.
  Noneligible specializations retain the old smaller frame.
- Diagnostic counters distinguish density projection terms and Hermite-state
  products from AO contractions. Declined attempts remain visible as executed
  preparation work; no iteration/cardinality is advertised as a hardware FLOP.

## Preliminary evidence

Slurm node2 job 3337 independently checks forces against retained raw AD
derivatives and a separate host symmetry orbit. It covers both spins, combined
and separate channels, positive/negative exchange, pure J, partial/empty AO
screening masks, repeated/coincident atoms, repeated shell pairs and a finite
original-weight case whose reordered ket-density sum overflows. Memcheck,
synccheck and racecheck each report zero errors/hazards. This is not full native
dispatch or target-GPU E+F qualification.

The formal probe has 31,720 bytes of shared workspace. Its combined RKS
specialization uses 118 registers, a 416-byte stack and zero spills; separate
and UKS resource receipts are retained too. These are resource bounds, not
measured occupancy, traffic or stall counters. The preliminary pure-J event
experiment on an assigned RTX PRO 6000 is not the target RTX 5090 endpoint.
The earlier accidental order-eight probe exceeded the degree-eight Coulomb
response contract and was discarded, not presented as an optimization result.

The 454 focused host tests and 567-module compiler ownership check pass. The
full linked native image, independent Libcint source exports, source-matched
complete E+F populations, uninstrumented timings, convergence counts, diagnostic
work/profile accounting and target RTX 5090 qualification are still required.

## Revisit and reproduction

Do not promote a default or claim GPU4PySCF parity from these probes. Retain a
candidate only after the complete endpoint and independent 1e-8 Eh / 1e-7
Eh/Bohr gates pass with original population/time limits and all twelve calls.
If complete endpoints regress, retain the negative result and do not publish
an isolated-kernel improvement as an accepted production optimization.

Ignored raw commands, exports, source/cache/resource receipts and logs live in
`.artifacts/direct-parity-20261017/`; this separate qualification source is
`/data/jzzeng/gqc-direct-parity-20261017/relative-density/source`.
