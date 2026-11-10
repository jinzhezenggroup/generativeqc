# Candidate: lower only the first directional source consumed by a nuclear RHS

Status: proposed
Date: 2026-11-11

## Observed work amplification

The source investigation in
`2026-11-11-ni-uks-hvp-first-integral-work.md` identifies raw first-integral
construction before the observed coupled response solve. Record-local geometry
reuse is implemented separately in
`../implemented/performance/2026-11-11-factorized-host-record-geometry-reuse.md`.
It does not remove two additional kinds of repeated work:

1. The full factorized primitive computes second directional jet fields even
   for a raw-first consumer.
2. The nuclear RHS immediately contracts twelve independent slot-gradient
   values with one fixed direction, then repeats the same raw integral stream
   for the other spin with different density weights.

The second observation permits a stronger first-order lowering than simply
removing the unused second derivative: emit only value plus directional value,
then scatter that scalar source to both spin Fock consumers. Preserve same-spin
exchange factors and the common Coulomb/core/overlap source explicitly.

## Isolated experiments, not repository behavior

Scratch scripts leave the four numerical geometry-reuse source files unchanged
while the original finite Ni diagnostic runs. They reuse the existing Graph
roots and generated product rules; they do not implement another ERI recurrence.
In-memory source substitutions are exploratory tooling, **not a proposed
production mechanism or permanent source-spelling test**.

- `/tmp/gqc-first-only-factorized-probe.py` removes unused directional/JVP
  outputs from raw-first lowering. Its independent primitive, bounded-resource
  and failure-publication suite passes six tests in 151.64 seconds. The selected
  before/after raw-first and weighted-HVP arrays are bitwise equal. Local timing
  samples are noisy; no complete-endpoint speedup follows from that probe.
- `/tmp/gqc-shared-spin-factorized-probe.py` streams a raw first source once to
  a bounded one/two-spin Fock panel, with one shared overlap matrix. Six complete
  coupled UKS tests pass in 394.73 seconds, including independently converged
  multistep-gradient full raw Hessians in both representations. Its retained
  semantic work is 1,368 first-component calls, 110,808 primitive records and
  110,808 primitive-component evaluations. These are that test run's counts,
  not a Ni measurement or a baseline-relative speedup.
- `/tmp/gqc-directional-spin-factorized-probe.py` lowers only scalar first
  directions and shares them between spins. Independent-slot directional parity
  for ssss, d/p/s/p and ffff, including signed two-exponent contractions and up
  to 64 components, has maximum errors `8.67361738e-18`, `2.16840434e-18` and
  `3.68628739e-18` against the separately qualified raw gradients. Complete
  moving-grid UKS regression also passes all six tests in 448.03 seconds,
  including full raw independent finite-difference matrices. The frozen Ni
  endpoint remains a separate gate. This run reports 867 raw-first calls and
  507 scalar-direction calls, with 110,904 primitive records and 114,616
  primitive-component evaluations across all tested consumers; do not mistake
  those complete-suite counts for a pure one-RHS measurement.

The Ni trial is `/tmp/gqc-ni-hvp-directional-spin-probe.py`. It retains the
canonical asymmetric fixture, default grid, 2 GiB explicit grid lease, independent
central-state checks, all frozen second-order gates and no displaced production
SCF. Its sidecar hashes the production dirty sources, exploratory scripts and
emitted directional C++; pipeline timing includes exploratory generation.
`experiment_only=true` and `public_curvature_qualified=false` remain explicit.
An independent topology inventory predicts 19,881 nontrivially translating
canonical shell quartets, 26,199 component-page calls, 485,661 record bindings
and 9,383,823 component evaluations for one full-range first-direction pass.
These are analytical schedule counts, not a completed endpoint receipt.

The independent full-matrix validation protocol is separately frozen in
`/tmp/gqc-ni-independent-hessian-columns/manifest.json`: 72 cold native/reference
state pairs cover all twelve columns at the original three step sizes and
unchanged state gates. Its worker cap is twelve, each point has a finite
900-second limit, and failed points remain explicit. Successful independent
columns do not supply a native Hessian or justify capability promotion.

## Production requirements before adoption

- Express the active jet outputs explicitly in the compiler, not by rewriting
  source strings, intercepting private emitters or inspecting target spelling.
- Share one transient geometry binding of the correct record/family identity;
  retain canonical raw-first and weighted-HVP behavior and CUDA emission.
- Select scalar first-direction execution explicitly, not by inferring it from
  zero direction values or reinterpreting a different operation's output ABI.
- Keep native input/capacity checks and late-error atomic publication. Include
  the new operation in required-byte admission and native failure injection.
- Validate one/two spin panel shapes and density weights before compilation or
  native execution. Preserve restricted callers and non-full-range fallbacks.
- Publish actual first-source work counts in complete HVP/Hessian diagnostics.
- Require independent raw full-matrix, H@v, translation and state/response gates
  before opening Ni capability admission. A qualified HVP alone is insufficient.

The exploratory RHS implementation does not establish public UKS, ECP or TS
completion for #2844--#2848. It also does not justify pruning integral records,
weakening thresholds or moving a failed molecular fixture.
