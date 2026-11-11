# Candidate: lower only the first directional source consumed by a nuclear RHS

Status: proposed
Date: 2026-11-11

Production integration is recorded separately in
`../implemented/performance/2026-11-11-first-source-and-grid-hvp-work.md`.
The experimental receipts below retain their original source provenance and
do not qualify the later production implementation.

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

## Completed independent matrix and negative endpoint follow-up

The original cold matrix protocol accepted 67/72 states. Native SCF failed
at indices 1, 39 and 61; independent SCF failed at 22 and 70. Keep those
receipts and the separate failed Newton retry for 22 unchanged. A new cold
protocol, `/tmp/gqc-ni-hessian-cold-retries/manifest.json`, uses native DIIS
history 16 and a 1,200-cycle bound, independent minao initialization/DIIS
space 16 and a 1,200-cycle bound. It does not seed either engine from the
other, alter the method/grid/state gates or relabel a false convergence flag.
All five retries pass their original state gates.

The separate collector `/tmp/gqc-ni-collect-independent-matrices.py` selects
67 original and five retry receipts, retaining hashes and source revisions
per point. All 72 pass the pre-frozen occupied cross-geometry overlap gate
of 0.99; the minimum singular value is 0.999871675434569. The three complete
independent raw matrices have maximum step spread 1.7779318757e-6 Eh/Bohr².
Their raw symmetry errors are 6.6138872e-7, 1.6878232e-7 and 8.8070292e-8;
these are finite-difference oracle errors, not native raw-symmetry admission.
Matrix-direction products match the independently displaced directional
gradients within 1.2422625e-7 Eh/Bohr. Complete receipts and matrices are in
`/tmp/gqc-ni-independent-complete-matrices/`. They remain independent
validation assets, not a returned native HVP or molecular Hessian.

The scalar-direction/shared-spin trial without gap conditioning reaches its
3,600-second limit without an HVP, with peak RSS 2,070,304 KiB. Its last
scientific receipt is `native-hvp-start`; no response residual or iteration
count follows from the last first-source progress marker. Source provenance
is parent `b734dc887d` plus the recorded dirty numerical sources, not a
clean-parent execution or a later restamped commit.

The spin-gap trial also reaches its 3,600-second limit without returning an
HVP, at reported peak RSS 2,081,064 KiB. Its explicit response-complete
marker records 46 actions and 635.076322 seconds, but supplies no returned
HVP diagnostic or numerical residual. The frozen compiler-only point-panel
trial separately records 46 response actions in 442.130459 seconds, then
also times out at 3,600 seconds without an HVP, at peak RSS 2,258,664 KiB.
Its final timeout receipt is retained beside its frozen source provenance.
These are response-phase observations from
different overlapping experiments, not paired complete-endpoint speedups.

Spin-gap and compiler-only point-Jacobian reuse have separately entered the
production adapter; see
`../implemented/performance/2026-11-11-fixed-state-semilocal-response-reuse.md`.
The directional/shared-spin integral lowering in this proposal remains an
experiment. Production integration and complete Ni raw-Hessian acceptance
are still required.

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

## Subsequent complete experimental action

After combining the committed one-electron second-integral reuse with these
scratch lowerings, the canonical NiH3 action returns and passes all three
pre-frozen independent matrix-action comparisons. The exact source provenance,
timing/work counts and explicit remaining production/full-matrix boundaries
are recorded in `2026-11-11-ni-complete-experimental-hvp-acceptance.md`.
This does not convert the source-rewriting experiment into production code.
