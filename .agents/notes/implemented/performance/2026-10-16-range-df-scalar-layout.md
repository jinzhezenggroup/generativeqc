# Decision: bound and reuse a scalar DF exchange layout

Status: implemented
Date: 2026-10-16

## Problem

Restoring an explicit fitted-LR opt-in addresses the unqualified default in
PR #2526, but does not fix the measured scalar exchange bottleneck. An unprofiled
same-Hamiltonian comparison must distinguish an actual optimization from merely
selecting the faster, scientifically different Direct-LR Hamiltonian.

The diagnostic provider interposer attributes approximately 12.94 ms to each
fitted LR build, versus 2.20 ms for the cached Direct LR build on water/TZVP.
Each warm endpoint performs four such builds. The unchanged full-range DF owner
costs about 13.19 ms per build in both arms. These instrumented numbers explain
the historical endpoint regression; they are not clean speedup measurements.

The old scalar contraction reads AO-pair-major B with an auxiliary stride inside
the matrix products. A Q-major copy already exists for the external dense-LA
provider but was not prepared or consumed by the scalar schedule.

## Decision

- Reuse that Q-major representation for scalar preparation at 16 or more AOs,
  provided its retained tensor is at most 16 MiB per owner. Existing retained
  numeric accounting includes the copy. Preserve the external provider's policy.
- Traverse independent output lanes contiguously for B times density. Transpose
  one auxiliary matrix into temporary scratch for the second product, using one
  temporary output row before accumulating the auxiliary contribution.
- Keep every output element's kappa, lambda and outer auxiliary accumulation
  order, FP64 arithmetic and full auxiliary dimension. This changes data movement,
  not the exchange equation or asymptotic cubic work per auxiliary matrix.
- Keep the cached dot-product schedule if optional transpose/row allocation
  fails. If the optional retained cache cannot be admitted or allocated, keep
  the original strided contraction. Neither fallback changes Hamiltonians.
- Validate every mutable cache element against its pair-major source; traverse
  the source contiguously without removing finite/stale-cache rejection.

The temporary inventory is one AO matrix plus one AO row, in addition to the
existing exchange and transformed-density matrices. The retained scalar cap is
per owner, not a batch or whole-process memory bound. Public callers supplying
their own cache retain their existing tensor contract.

## Rejected alternatives

- Enabling fitted LR by default or using different selectors for energy and
  forces does not repair the implementation. Keep the independent
  [opt-in selection decision](../compatibility/2026-10-16-range-df-explicit-opt-in.md).
- The first cached layout improves the water/TZVP endpoint by 3.66%, but leaves
  a 1.74% fitted-versus-Direct replay slowdown in its separate comparison.
- An exact-symmetry schedule adds a check and a specialized row product but does
  not improve the measured replay regression. Its same-Hamiltonian improvement
  is 2.98%, and fitted-versus-Direct replay remains 2.35% slower. That follow-up
  was stopped before complete geometry/oracle qualification; partial receipts
  remain negative evidence, not an accepted campaign.
- Skipping columns beyond `effective_rank` is invalid: the retained whitening
  matrix is a symmetric square inverse root, not a compact eigenvector basis.
  Rank truncation does not make the corresponding stored columns zero.
- Assuming symmetric B is unnecessary. The chosen transpose supports mutable
  nonsymmetric public tensors without weakening the scientific API.

## Evidence

The source-matched CPU Release build uses scalar linear algebra, one CPU/BLAS
thread and the retained cc-pVDZ-JKFIT auxiliary basis. Water uses a 24 x 6 x 12
unpruned grid per atom (5,184 points), not the production default grid.

Seven interleaved pairs compare the pre-optimization fitted Hamiltonian with
the optimized fitted Hamiltonian: warm water/TZVP improves from 1.173277 s to
1.111625 s (5.25%), above the unchanged 2% significance floor. Both arms retain
two SCF iterations and four physical Fock builds. A separate comparison using
the same optimized binary observes 1.107172 s Direct LR versus 1.115258 s fitted
LR: the residual slowdown is 0.73%, not zero and not a default-promotion pass.

The larger 86-AO water dimer retains the same two-iteration/four-Fock-build warm
work and improves from 7.952534 s to 5.901382 s (25.79%) across five interleaved
pairs, with independently reconverged fitted energies in both arms. Water/SVP
improves only 0.99% in its seven-pair same-Hamiltonian comparison, below the
significance floor; its separate fitted-versus-Direct residual is 0.45%.
Five-pair SVP geometry replay preserves 11 iterations and 13 Fock builds, and
checks both moved and original energies independently. Larger cache-envelope
fallback tests are numerical/resource checks, not larger-system speed claims.

Native tests compare cached and uncached RHF/UHF exchange with an independent
reconstructed four-center oracle at 7, 16, 24, 43 and 86 AOs, for symmetric and
nonsymmetric tensors and nonsymmetric spin densities. The unchanged matrix gate
is 2e-12. Separate identity-metric tests straddle the 16-AO admission and 16-MiB
retention boundaries at 15/16 and 1024/1025 AOs. Existing finite stale-cache
negative controls remain enabled.

Final validation reports 121 Python passes, 17 CUDA skips and five non-DF
native-force eligibility failures reproduced unchanged against the preserved
baseline. This build has stationary CPU force AOT disabled; do not label it a
passing native non-DF force campaign. All fitted-force/oracle and relevant host
contracts pass, as do 11 related native CTests, formatting, strict documentation
and native API/ownership checks. Optional allocation refusal is a retained
fallback, not a newly fault-injected qualification.

Raw timings, source patches, actual loaded-library identities, diagnostic-only
profiles, rejected attempts and independent numerical receipts are retained in
the [measurement family](../../../../benchmarks/results/range-df-scalar-layout-20261016/README.md).

## Consequences and revisit conditions

This is a CPU layout optimization, not a new GPU qualification, lower-rank
approximation, default-grid throughput claim or force-kernel speed claim. VV10
grid-pair work and the fitted-versus-exact Hamiltonian distinction remain.
Revisit the cache cap or larger-system policy only with complete endpoint and
resource evidence, preserving the bounded uncached path and independent gates.
