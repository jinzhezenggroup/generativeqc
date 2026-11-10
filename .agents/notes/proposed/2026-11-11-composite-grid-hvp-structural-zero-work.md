# Proposal: remove structurally zero and invariant composite-grid HVP work

Status: proposed
Date: 2026-11-11

## Observation

The live complete Ni action experiment
`/tmp/gqc-ni-complete-source-reuse-probe.py` has returned its one-electron and
Coulomb second-integral sources, but has not yet returned the complete HVP.
The source markers alone do not identify a unique remaining bottleneck.

Independent inspection of `_CompositeGrid.hvp` nevertheless identifies exact
redundancy that does not depend on a runtime numerical threshold:

- `direction(vector, delta_density)` already returns the right AO motion. The
  HVP retains that result under `_`, then reconstructs the identical right
  motion for every coordinate and each AO/grid/weight component. Reuse the
  returned array rather than retaining one copy and rebuilding another.
- The weight component supplies zero left AO/point motion and zero mixed AO
  motion. Both its first and mixed density-feature tangents are identically
  zero, even with a nonzero right density response. Only the explicit first
  and mixed partition-weight fields contribute; computing zero feature
  contractions does not improve their scientific completeness.
- `_mixed_program` currently differentiates a first feature program with a
  left density tangent, even though `_mixed_features` always supplies zero
  `left_density` and `d_left_density`. Linearize the canonical primal in the
  AO inputs only for the left leg, then include density and both AO namespaces
  in the right JVP. This eliminates known-zero contractions without hand-coding
  another bilinear/derivative algebra.

The general first feature program must still support a right density tangent.
A separate pure-density action may reuse the primal program with the density
replaced by its tangent because every declared feature is linear in density at
fixed AOs. Any interface change needs an explicit physical contract and caller
migration; do not infer a zero tangent from object identity or array shape.

## Isolated evidence

`/tmp/gqc-ni-grid-feature-work-audit.py` uses the retained canonical NiH3 state,
46 spherical AOs and 50,064 native-grid points. For each spin it executes the
existing feature programs with zero left/mixed AO tangents, a nonzero right AO
motion and an independently generated nonzero symmetric right density tangent.
Both first and mixed feature outputs are exactly zero in all five channels.

The first contractions take 4.737689 and 4.740405 seconds; the mixed contractions
take 14.230219 and 14.208373 seconds. These isolated observations overlap another
experiment, are not paired endpoint timings and cannot be summed into an
accepted molecular speedup. The source hashes, numerical outputs and finite
process receipt are retained under `/tmp/gqc-ni-grid-feature-work-audit*`.

An isolated compiler candidate changes only the left JVP seed set to AO inputs
and removes the corresponding absent feed entries. It passes all **13** existing
compiler grid-second-order tests in **2.63 seconds**, including multistep feature
differences. At the same Ni source size its mixed zero contractions remain
exactly zero and take 9.488261 / 9.442187 seconds; first contractions remain
unchanged. This is still an overlapping source experiment, not a paired endpoint
claim or production implementation. The candidate and exact source SHA are
retained in `/tmp/gqc-stationary-grid-second-ao-only-candidate.py` and
`/tmp/gqc-ni-grid-feature-ao-only-candidate-audit.json`; the running complete HVP
continues to use unchanged production numerical source bytes.

## Required qualification

Preserve AO/grid/weight component decomposition, mixed density response,
partition branches and native/nonlocal behavior. Verify canonical feature JVPs
against multistep primal differences, independent Cartesian/spherical molecular
Hessians, public PBE0/B3LYP endpoints and failure/budget recovery. Keep finite
input admission before skipping structural zeros. Reusing the already returned
right motion should reduce, not enlarge, the current live-buffer peak.

After source-level validation, rerun the complete Ni action with explicit timing
around every contributor, not only the second-integral functions. Existing
negative protocols and unchanged independent gates must remain retained. This
proposal is not production code or Ni/full-matrix/installed acceptance.

## Subsequent endpoint result

The unchanged live experiment subsequently returns its complete HVP and passes
the pre-frozen independent action gates. See
`2026-11-11-ni-complete-experimental-hvp-acceptance.md`. This proposal's structural
zero/invariant optimizations remain scratch-only and are not responsible for
that accepted result; do not attribute the endpoint timing to this candidate.
