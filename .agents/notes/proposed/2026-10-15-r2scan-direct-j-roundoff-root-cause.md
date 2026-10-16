# Investigation: Direct-J scatter noise precedes the r2SCAN SCF divergence

Status: proposed (component diagnosis established; production mitigation unqualified)
Date: 2026-10-15

## Problem

The retained #2464 r2SCAN-3c direct/96 moved endpoint changes from 14 to 23
Fock builds despite faster force work. Earlier investigation established that
the unchanged 1e-12 Eh energy gate can keep an otherwise stationary SCF solve
active. That explains the additional work, but does not identify the originating
numerical operation. Repeated fixed-incumbent reconvergences also differ in
their work counts, so a candidate-only explanation is insufficient.

The earlier ordinary-FMA point probe and serial compensated-XC experiment are
not the answer. Their limitations are preserved in
`../rejected/2026-10-15-r2scan-cse-rounding-and-xc-totals.md`.

## Experiment boundaries

This investigation uses frozen incumbent revision
`5f80ba629b5f20412215cdc690f8fa458ac3fdf8`, the original native checkpoint,
the existing independent-oracle protocol, and the original max_iterations=100,
energy_tolerance=1e-12, density_tolerance=1e-10, screening_tolerance=1e-12,
basis, geometry, and grid. No convergence gate is weakened.

Private host instrumentation records the first three physical KS iterations,
then deliberately exits with code 86. Another mode replays the actual primary-J
operation three additional times with the identical live density pointer and
owning KS dispatch, then exits 86. These are intrusive diagnostics, not successful
SCF/force endpoints or timing evidence. The old public `FockPlan.evaluate`
standalone test is not used: it invokes a different J implementation.

Slurm job 8604 runs the baseline modes; job 8609 runs the numerical control.
Both use node1 / main / gpu:5090:1 with finite 20-minute allocations, retain
Slurm's CUDA visibility, and execute each mode in a fresh process/owner.

To avoid recompiling unchanged numerical CUDA kernels just to add host snapshots,
the diagnostic DSOs reuse the frozen incumbent's read-only objects and static
libraries. Only two host C++ objects are replaced in the baseline diagnostic;
the control additionally replaces `direct_coulomb.cpp`. Build receipts retain
the exact commands, replacement source/object hashes, compiler dependency hashes,
and reused link-input hashes. No predecessor source, build, library, or endpoint
is modified. A separate full build in this campaign is not used by these runs
and is stopped once the source-bound diagnostics complete; its partial outputs
are not associated with the diagnostic library hashes. Completed compiler-cache
entries and intermediate objects are retained.

## First differing operation

The actual resolved dispatch is generated J, not MD, a resident value source,
canonical J, mixed precision, or density fitting:

```text
md_j=0 resident=0 generated_j=1 generated_k=0
canonical_j=0 canonical_k=0 fixed=0 mixed_j=0
n=1088 spins=1 source=0 fitted=0 mixed=0 chunk=0
```

Two independent native traces have identical first-iteration density hashes
(`9be9d009d7552dca8e499906467eaa1df6beecb1914dd10308b4692501acefa2`).
Hcore, overlap, orthogonalizer, the complete Vxc matrix, and all three XC totals
are also bitwise identical. J is the first differing physical component:
230,058 of 1,183,744 entries differ, with maximum absolute difference
3.268496584496461e-13. Fock, residual, DIIS effective matrix, transformed
eigensolver input, eigenvectors, and the next density then differ downstream.
The first proposed density has maximum difference 1.503897e-11.

The first transformed eigenproblem input differs by 5.083312e-11, versus
3.268497e-13 in AO J. The overlap minimum eigenvalue is
4.7857941194755205e-4 and its condition number is about 1.1134e4. This is
observed amplification through the existing orthogonalization/SCF pipeline,
not evidence that the eigensolver is the originating nondeterministic operation.

The compensated electronic-energy differences between these two traces are:

| Physical iteration | Trace B minus trace A, Eh |
| --- | ---: |
| 1 | +3.782335485720681e-13 |
| 2 | -2.498880682782982e-12 |
| 3 | -1.1079746569122712e-12 |

At iteration 1 the separately rounded public one-electron, Hartree, and XC
scalars are all equal; their equality conceals the difference in the compensated
electronic low word. Comparing only those rounded scalars misses the first
energy divergence. By iteration 2 the between-run energy difference already
exceeds the unchanged SCF energy tolerance.

## Same-input replay and causal control

Four evaluations of the same live KS density produce different J matrices in
the unmodified numerical incumbent. Relative to the first evaluation, the three
repeats change 230,440 / 230,251 / 230,694 entries, with maximum absolute
differences 2.842171e-13 / 3.836931e-13 / 2.415845e-13. Evaluating
`0.5 * fsum(D * (J_repeat - J_first))` gives Hartree differences
+4.005518e-13 / +6.411651e-14 / +3.339311e-13 Eh. Thus neither a changed
density nor an XC/eigensolver difference is needed to produce the J noise.

The source owner is the generated streaming Coulomb implementation in
`src/scf/cuda/direct_coulomb.cpp`. Shell workers share a Cartesian output and
scatter density-weighted ERIs through FP64 floating atomics emitted by
`python/generativeqc_compiler/integral/lowering/fock_accumulation.py`.
The `CompensatedOutput` pointer-only contract takes the ordinary atomic branch
in `src/runtime/compensated_atomic.cuh`; its correction pointer is null in this
production caller. Concurrent worker arrival order therefore changes rounding
before AO projection and before the already compensated energy trace.

The private control changes only these generated scatters to use the existing
rounding-residual plane and folds that plane on the same stream before AO
projection. It charges the extra bounded Cartesian plane to the same optional
provider budget; it does not change points, screening, density, integral
recurrences, thresholds, or the resolved generated-J route. The retained native
DDDD compatibility writer still uses ordinary FP64 atomics: this is deliberately
not claimed to be complete or bitwise-reproducible compensation.

With that intervention, same-input replay changes only 137 / 129 / 159 J
entries, with maximum absolute differences 6.661338e-16 / 1.540434e-15 /
1.540434e-15. The corresponding Hartree differences are
+7.178647e-20 / +1.784232e-19 / -2.997643e-19 Eh. In this experiment, the
maximum fixed-density Hartree variability falls by about 1.3 million times.
This targeted intervention supports the floating-scatter diagnosis; it is not
an independent accuracy oracle or an endpoint performance qualification.

The control's first proposed-density difference is 3.038889e-14, versus
1.503897e-11 in the ordinary-scatter traces. Its compensated electronic-energy
differences after iterations 1/2/3 are +9.019577e-19 / +2.056571e-13 /
+3.828277e-13 Eh. Remaining differences are explicitly retained: the intervention
does not establish bitwise-deterministic SCF or eliminate all feedback rounding.

## Consequences and remaining gates

The first differing operation is a generic Direct-J accumulation, not r2SCAN's
point evaluation or XC total reduction. The observed SCF feedback raises its
effect above the stringent energy gate, giving a concrete upstream origin for
the previously observed energy-blocked tails. This does not prove that every
additional Fock build in every historical run has one exclusive cause.

Do not solve this by weakening convergence, enabling global CSE on inadequate
evidence, or adding a functional-specific escape. A production mitigation should
address the shared J accumulation and retain explicit bounded fallback behavior.
The existing compensated primitive improves accuracy but explicitly does not
promise a deterministic reduction contract. Native DDDD, generated exchange,
other J providers, response semantics, and budget refusal need separate treatment
where relevant; this experiment does not qualify them by analogy.

Before adopting any numerical change, require independent J/molecular oracles,
the unchanged energy/force acceptance gates, repeated complete endpoint timings,
and semantic work counts. The control here exits before convergence and forces,
so no production numerical patch or performance promotion follows from it.

## Evidence

Full private evidence and raw binary snapshots are retained on n1 at
`/data/jzzeng/qc-r2scan-root-20261015`:

- `evidence/incremental-probe-build.json`, `incremental-control-build.json`:
  actual diagnostic build provenance and object reuse receipts.
- `instrumentation.patch`, `control-instrumentation.patch`: bounded temporary
  native diagnostics and the experimental numerical intervention.
- `results/{replay-j,trace-a,trace-b}` and matching `control-results`: exact
  inputs/stages, scalar hexadecimal logs, partial native journals, exit receipts.
- `evidence/comparison.json`, `control-comparison.json`, `root-cause-summary.json`:
  bitwise comparisons and compensated component deltas.
- `evidence/overlap-conditioning.json`: CPU-only spectral conditioning census.
- `evidence/audit.json`, `snapshots.sha256`: six expected diagnostic exits,
  matching checkpoint/protocol/source/library identities, and raw snapshot hashes.

The compact receipt/source archive, excluding the large raw snapshots and
unchanged incumbent objects, is `diagnostic-evidence.tar.xz`, SHA256
`c4b64df60c82db9bb8047c63cb12a60e5ed64cd1e64164779965afe8e18acbff`.
Its checked local copy is in the ignored campaign directory; the complete raw
snapshot population remains at the n1 root above.

The local ignored campaign directory is
`.artifacts/r2scan-root-cause-20261015/`. Temporary native diagnostic hooks are
not a production change and must not remain in a submitted patch.

## Production follow-up (2026-10-15)

The separate [shared Direct-J compensation decision](../implemented/numerics/2026-10-15-generated-direct-j-compensation.md)
records the production implementation, independent generated/native/bounded J
oracles, memory-safety gates and repeated complete SCF+force endpoints. It
supersedes the mitigation-pending status of this investigation, not its
historical experiment boundaries: the exit-86 traces above remain diagnostics
and are not timing or full-convergence evidence.

The [coupled J/XC follow-up](../implemented/numerics/2026-10-15-coupled-j-xc-compensation.md)
retains the J-only cold-start negative and investigates scalar-total roundoff
with J held corrected. It does not relocate the first differing operation in
the original traces or turn the earlier partial diagnostic into qualification.
