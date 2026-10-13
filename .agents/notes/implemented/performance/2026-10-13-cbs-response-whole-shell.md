# Decision: reuse whole-shell recurrence in compensated RHF response

Status: implemented
Date: 2026-10-13

## Problem

Complete correlation-only DF CCSD(T)/CBS forces still consume an exact RHF
reference and exact four-center orbital/nuclear response. An optional complete
canonical Cartesian ERI lease is capped at 8 GiB. The pinned QM7 C5 cc-pVTZ
frame has 318 public / 355 Cartesian AOs and needs 15,972,157,160 bytes for that
lease. Its refusal selects bounded recomputation, not an out-of-memory error.

Before this change, the compensated fixed-mask consumer bypassed both generated
ordinary J/K and the existing whole-shell materialized recurrence. It prepared
the same Hermite/Coulomb source separately for each AO component. A separate
node4 diagnostic of the frozen baseline measured roughly 73 seconds per exact
Arnoldi action, despite only 18 response iterations and zero restarts. Complete
C5 E+F took 2660.193940 seconds on node1; the independent node4 diagnostic took
2654.466469 seconds. The TZ group dominated, not DZ CCSD(T).

## Decision

Let compensated, fixed-mask J/K reuse the compiler-owned whole-shell recurrence
for all s/p/d/f combinations. Keep immutable shell-row candidate selection and
the original per-AO geometry-only Schwarz predicate. The compiler scatters
positive J and positive K through distinct compensated sinks. Native ownership
clears/folds these planes exactly once together with any retained-value writers,
before Cartesian-to-public projection. No new integral algebra is introduced.

Resident buckets keep priority, including explicit partial selections. Without
prepared shell storage the original component-wise CUDA evaluator remains.
Fixed-mask requests tighter than the prepared rows remain invalid; they must
not silently inherit an incomplete candidate domain. Ordinary uncorrected J/K
keeps its prior order-three-through-nine routing.

The response option `materialized_jk=false` is a private ablation. The final
scalar residual explicitly disables both source residency and materialization,
so its independent lowering and zero-screening acceptance remain meaningful.
An uncached action alone no longer implies an independent component evaluator.

## Invariants and rejected shortcuts

- Do not use density-dependent Fock screening on signed Krylov vectors: a
  changing mask is not the fixed linear operator required by the solve.
- Keep compensated FP64 accumulation and the independent residual gate at
  `1e-10`; do not trade their removal for a faster benchmark.
- For spherical test frames, form the independent Schwarz mask in normalized
  Cartesian AOs **before** libcint's spherical projection. Masking a public
  spherical ERI tensor defines a different operator for d/f shells.
- Do not increase the full ERI ceiling to hide the C5/C7 cliff. C7 needs about
  51.7 GiB for this optional lease, not for mandatory endpoint storage.
- Partial residency is already available explicitly. Its automatic cost policy
  needs new endpoint evidence after reducing recomputation cost; filling an
  8-GiB allowance is not itself a performance improvement.
- This is not full DF-HF/RI-JK response. Correlation fitting and CBS equations are
  unchanged. A fully fitted reference requires its own consistent auxiliary and
  metric derivatives and independent force qualification.

## Numerical evidence

Source: `b400ac1fb9ba232d29e0069ea2c759ac5b3bc926` plus the task patch.
CUDA 12.9.1, Release sm120, eight AOT shards, fast compile off, direct device
link, verified sccache 0.16.0. The library-only CMake target builds successfully.
An initial all-target build encountered an unrelated existing COSX native-test
compile error; no COSX sources were changed.

- 191 focused host tests pass, including compiled dyadic cancellation tests for
  separate raw J/K correction planes and executable complete s/p/d/f slot bounds.
- Node4 Slurm 828: 152 response tests and 8 complete DF-MP2/DF-CBS tests pass.
  Tests cover signed linearity/self-adjointness, spherical/Cartesian oracles,
  nonzero f-shell response with reconverged energy differences, fixed masks,
  stricter-mask refusal, resident/partial/bounded paths, and allocation refusal.
- Node4 Slurm 829: through-f exact Cartesian and screened spherical actions
  pass both Compute Sanitizer memcheck and synccheck with zero reported errors.
- Signed J/K gates: `2e-11` absolute / `2e-12` relative. Independent response
  force directions use steps `1e-4` and `3e-5` Bohr and `3e-7` absolute/relative
  tolerances; the complete DF endpoint retains its existing independent gates.

Frozen candidate library SHA-256:
`2d36d6df0b66090ad5f7d93ee2ae5832aabea5bfd61944928a8bdee9714febc2`.
Production patch SHA-256:
`46e2f46ad9997502f6876470764815bf06cd78fbc6aadbb5d20b10e6dfcbee57`.
Ignored raw evidence lives under `.artifacts/cbs-response/` in the task worktree
and `/data/jzzeng/readme-cbs-20261013/` on nodes1/4. Primary endpoint timing runs
are separate from diagnostic/profiling replays.

## Complete endpoint evidence

Frozen exact-reference/correlation-RI C1–C7 E+F observations on node1 RTX 5090:

| QM7 carbon count | Baseline seconds | Whole-shell seconds |
| ---: | ---: | ---: |
| 1 | 14.541877 | 15.535868 |
| 2 | 38.491684 | 40.969052 |
| 3 | 117.739450 | 117.408148 |
| 4 | 309.237849 | 308.005981 |
| 5 | 2660.193940 | 1084.774018 |
| 6 | 5579.069581 | 2456.269825 |
| 7 | 10111.064128 | 4540.704994 |

Baseline source is `05844f847b447e426523d4d9a559991ae8ac640b`, library SHA-256
`f814ddfb406b2f97fe1f7fd178a71d9f19d4e47219f13008ac64688319f871a4`.
Both series use one API-cold call, spherical cc-pV[D,T]Z/RI, DZ correction,
all-electron FP64, 64-GiB complete numerical budget, identical fixed QM7
geometries and timing exclusions. They are not repeated statistical trials.
Every component/composite energy is unchanged; the maximum component force
difference over C1–C7 is `7.435e-13 Eh/Bohr`. This parity is not an independent
oracle for the large molecules. The small C1/C2 single samples are slower;
this is a qualified bounded-source optimization, not a universal speedup.

The change reduces repeated source preparation but does not eliminate exact
nuclear traversals or the deliberately scalar final audit. Diagnostic C5
actions improve from about 73 s to 16.7 s; independent audit and nuclear work
remain about 73 s and 93 s respectively. Separate profiling found whole-shell
orders 0–2 slower than the component schedule (about 2.43 s/action potential
savings from hybrid dispatch). No hybrid policy was implemented: validate a
complete endpoint before adding such a cutoff.

Full RI-JK is an explicitly different method, documented in
[the full-DF decision](../numerics/2026-10-13-full-df-cbs-reference.md).

## Revisit when

Changes to the source/cache/derivative ownership alter this balance. Measure
complete same-protocol endpoint work, including nuclear derivatives and the
independent audit, rather than assuming the former bottleneck still dominates.
