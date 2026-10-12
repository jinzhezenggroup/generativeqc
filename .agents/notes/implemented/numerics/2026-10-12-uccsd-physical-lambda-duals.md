# Decision: use physical excitation duals for native UCCSD Lambda

Status: implemented
Date: 2026-10-12

## Problem

The unrestricted equation DAG stores five full amplitude/residual arrays, but
the physical excitation coordinates store independent signed representatives.
Ordinary amplitude packing is not the transpose of expansion. Treating it as
a cotangent pullback gives a factor-of-four error in same-spin doubles and can
still appear plausible in a restricted or singles-dominated molecule.

Issue #1824 needs solved Lambda and explicit parameter adjoints. Maintaining
separate CPU/CUDA derivative equations or differentiating DIIS iterations would
create another scientific authority and impede #1825 corrected response.

## Decision

If `U` is signed expansion, `U*U=w`, with `w=4` for aa/bb doubles and one
otherwise. Add `pack_dual=U*` and `unpack_dual=U/w` to the existing layout.
Use paired signed sums in same-spin pullbacks so lifting then pulling back is
bit-exact when quarter-scaled representatives remain exactly representable.
Repeated-index cotangents have no physical
coordinate and do not contribute; full bars need not be antisymmetric.

Solve `J*lambda=-grad(E_corr)` in unweighted independent excitation coordinates
with shared generic TensorIR AD and the existing checked transpose/GMRES owner.
Full published Lambda is `U lambda`, whereas a full residual VJP seed is
`U lambda/w`. Keep that distinction explicit in APIs, docs and provenance.
Both native backends execute the parent's own physical graph and its exact
derivatives. The independent determinant-space/CAR oracle and PySCF remain tests,
not production consumers.

Borrow the live UCCSD parent and freeze admitted identities/configuration.
Admit logical numeric storage, worst-case complete contraction work and Krylov
storage before amplitude copies/native execution. Freshly qualify CC, solve with
a true-residual check, then qualify combined Lagrangian stationarity. Parameter
requests recheck identities and actual standard-CCSD stationarity, then execute
one bounded canonical independent input VJP.

## Rejected alternatives

- Ordinary amplitude packing of full bars: omits the signed-orbit sum.
- Multiplying or dividing published Lambda by four: confuses physical duals with
  minimum-norm full residual seeds and breaks independent Lambda comparison.
- RCCSD pair/sqrt weighting: its pair symmetry is not unrestricted antisymmetry.
- Differentiating iterative convergence or proposal denominators: gives an
  algorithm-dependent derivative rather than the stationary physical equation.
- Dense production amplitude Jacobians or hidden PySCF/interpreter fallback:
  violate bounded matrix-free/native ownership and independent-oracle separation.
- Calling a fixed-MO graph input bar a relaxed RDM or force: skips physical
  input bindings, Fock normal ordering, HF and shared UHF/orbital/overlap chains.
- Counting only memory or one transpose action: does not bound a complete solve.

## Invariants

- Exact five-block signed primal maps and FP64 finite input/output contracts.
- Unit unweighted physical excitation pairing; quarter same-spin residual seeds.
- Same parent physical equation identity on CPU and CUDA, no CPU oracle on CUDA.
- Fresh parent residual/energy gates, checked true adjoint residual, and freshly
  generated combined amplitude stationarity before immutable publication.
- Combined/component stationarity use one DAG and are not independent equations.
- Explicit native, state-map and Krylov bounds and total semantic work/action caps.
- The parameter owner enforces standard CCSD Lambda, not corrected-(T) Lambda.
- Borrowed parent/executor/solver must remain live and admitted; providers may close.

## Evidence

`tests/python/test_uccsd_response.py` compares the signed dual maps, native
JVP/VJP transpose actions, and all canonical Hamiltonian input bars against an
independent complete determinant-space/CAR Lagrangian finite difference. It
deliberately injects the wrong same-spin seed multiplicity and requires failure.

`tests/python/test_uccsd_lambda.py` compares every multiplier block against
independent PySCF for Li, OH, NH2, broken H4 and asymmetric H6, checks the
restricted-limit full-Frobenius RCCSD dual, and compares Fock directions with
reconverged independent fixed-orbital CC energy finite differences. It also
exercises stale/forged states, false solver success, resource/action admission,
nonfinite/native failures, immutable publication and empty excitation spaces.

`tests/python/test_uccsd_native.py` uses native UHF and streamed shell integrals
through complete native energy, Lambda and parameter response. Exact shared
shell definitions avoid comparing rounded basis tables. Independent HF frames
can differ in MO phases/degenerate rotations, so use one AO one-body direction
and transform it with each reference's own coefficients instead of comparing
their Lambda arrays directly. Choose a symmetry-preserving normalized
core-Hamiltonian direction. A generic random AO perturbation on OH stagnated
the independent fixed-frame oracle at a proposal norm of about `1.1e-5` after
400 cycles, even when starting from independently converged CC. The
core-Hamiltonian direction converges both signs to proposal norms below
`9e-12` with unchanged tolerances. Avoid hiding that failure by relaxing
convergence, comparing unconverged energies, or identifying degenerate
electronic-state orientations across independently converged HF frames.
The diagnostic is retained locally in
`.artifacts/uccsd-lambda-oh-fd-diagnosis.log`.

The real-CUDA tier checks NH2, broken H4 and asymmetric H6 using CUDA transforms,
converged CC, Lambda and Fock/aa/ab/bb input VJPs in finite Slurm allocations.
Qualification logs remain ignored under `.artifacts/`; source-matched terminal
results belong to the change's PR evidence, not a production performance claim.

## Consequences

Generic AD retains more intermediate arrays/work than a hand-fused optimized
Lambda implementation. Logical admission is deliberately conservative. The CPU
emitter supplies a common metadata scalar-loop work model on CUDA, not runtime
CPU work or hardware FLOPs. Host Krylov and coordinate work remain explicit in
endpoint timing but are not part of semantic contraction-work counts.

The owner establishes amplitude response and fixed-orbital graph input bars;
#1824 still requires physical binding/normal-ordering pullbacks, shared UHF
orbital/overlap response and complete independently qualified analytic forces.
#1825 additionally requires triples VJPs, corrected Lambda RHS/stationarity and
the explicit orbital-energy-to-Fock-diagonal chain. Its corrected multipliers
must not bypass this owner's standard-stationarity check.

## Revisit when

Resident or tiled derivative actions change native storage/data movement, or
physical UHF input binding is added. Preserve physical coordinates and independent
acceptance gates while updating complete endpoint admission and evidence. Avoid
replacing this foundation with a second Lambda algebra or silently relaxing gates.

## References

- #1824 and #1825; native parent foundation in PR #2320.
- `docs/developer/uccsd_lambda.md`.
- Parent source `f7ceb6a9a4cbef95574c6024fbbdb61d2dcaca53` plus this change.
- Integration replays the unmerged energy owner over master
  `a47fa9cc8227df8bde46bcb878fe553d8c9d1ac2` after #2318/#2319 merged. Existing
  PR heads remain untouched; final-source gates are rerun on the integrated tree.
