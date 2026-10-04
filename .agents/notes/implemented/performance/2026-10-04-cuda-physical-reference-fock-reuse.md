# Decision: retain the converged physical Fock for CUDA RHF references

Status: implemented
Date: 2026-10-04

## Problem

Detached physical-reference export unconditionally rebuilt the final Fock, even
when the SCF iteration already owned the target-precision pair P_n/F(P_n).
Correlated consumers need a physical canonical reference, so the ordinary
energy-only density-step reuse criterion cannot by itself license publication.

## Decision

Use the existing converged-density retention policy for physical-reference
exports. Preserve P_n and its un-extrapolated F(P_n), diagonalize that physical
Fock, and independently validate the resulting detached F/P/C/epsilon frame.
The validator checks reconstructed density, the physical commutator,
F C = S C epsilon, metric orthogonality, and C^T F C canonicality at its existing
1e-8 maximum-entry gate. DIIS orbitals and density RMS alone are not this proof.

Only a numerical/canonical validation rejection of a retained candidate permits
one retry: copy the already available P_{n+1}, rebuild the exact physical Fock,
repeat final diagonalization and energy evaluation, and validate again. A second
rejection propagates. Already-rebuilt candidates, transport errors, allocation
errors, malformed shapes, and solver failures do not restart this sequence.
Nothing is published until the detached reference has passed validation.

The explicit GENERATIVEQC_FINAL_FOCK_REBUILD diagnostic preserves the legacy
rebuild route for A/B comparisons. Exact incremental Direct-J/K also retains its
ordinary final rebuild. This change neither caches Fock operators across solves
nor changes the physical-reference acceptance tolerances.

## Invariants

- Reference export admits one restricted system and zero screening tolerance.
  Exact-target refinement must precede reuse if a mixed iterative route is
  admitted; an approximate iterative operator alone never licenses publication.
- F(P_n) remains separate from the DIIS-extrapolated eigensystem, and the
  convergence transition preserves the density generation that produced it.
- Final diagonalization can replace C and epsilon, but cannot silently replace
  P without rebuilding its physical Fock. The legacy retry explicitly does both.
- Each execution initializes convergence and evaluates the current operator.
  Changed geometry reconstructs geometry-bound inputs. A warm density is a seed,
  not permission to reuse another execution's Fock.
- Reference matrices are owned host vectors. Download synchronizes outstanding
  copies before validating or releasing staging storage, including exceptional
  exits. The exported reference does not borrow the retained CUDA plan.
- Successful strict-FP64 diagnostics distinguish iteration Fock applications,
  explicit post-SCF builds, and skipped final builds. Failed partial work and
  incompletely instrumented mixed work must not be certified as complete.

## Rejected alternatives

Always rebuilding discards an already evaluated operator even when the stronger
reference contract passes. Reusing based only on a small density step leaves the
canonicality obligation unproven. Swapping the exported density to P_{n+1}
without rebuilding F breaks density/operator provenance. Repeated retries would
hide an unresolved numerical failure and introduce unbounded finalization work.

## Evidence and limits

`tests/python/test_rhf_reference_residency.py` specifies device qualification for
cold and warm references, changed geometry on a retained plan, forced rebuild
comparisons, optional ERI admission/fallback, and Cartesian/spherical frames.
The new strict cases compare density, Fock, orbital energies and energy against
an independent CPU solve and require the expected one-build work difference.

`tests/python/test_rhf_reference_retry.py` executes the production export block
and download helper with deferred host transport and fault-injected numerical
actions. Its 17 cases check bounded retry ordering, paired density/Fock inputs,
publication/work counters, second-rejection propagation, non-retry error
categories, solver failure status, and exceptional transport draining. These are
host control-flow tests, not GPU mathematics or chemistry qualification.

No complete correlated-endpoint timing or runtime speedup is established by
these host tests. Device numerical coverage and complete MP2/CC endpoint benefit
must be reported from actual runs with their source identity and work counts;
missing supplementary runs must not be described as passed.

## Revisit when

The reference validator, final-state precision/ownership contract, or admitted
batch/spin domain changes; or complete correlated-endpoint measurements show
that validation and fallback costs outweigh the eliminated operator work.

## References

- Issue #1858 and PR #1862
- `src/scf/cuda/reference_export.cuh`
- `src/scf/cuda_rhf.cpp`
- `src/scf/rhf.cpp`: validate_physical_reference
