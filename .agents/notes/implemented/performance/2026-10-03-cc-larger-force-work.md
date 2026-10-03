# Decision: bound complete force memory and reduce repeated ERI derivative work

Status: implemented
Date: 2026-10-03

## Problem

The native CCSD(T) force owner stopped at 12 AOs. Its complete resource model
already composed the triples, Lambda, parameter, raw-Hamiltonian, response and
derivative lifetimes, but larger execution had two avoidable work amplifiers.
The full MO Hamiltonian used shell-clamped width-two source tiles, repeating its
transform for every tiny tile. The final derivative consumer evaluated every
ordered shell quartet even though ERIs and their derivatives have eight exact
permutation symmetries.

Merely lifting the public dimension guard exposes those costs. A 14-AO CPU
probe spent about 38 of its 43 seconds in the derivative consumer. Expanding the
source tile alone cannot remove that derivative work. Restricting wider tiles
to unused space beneath the old selected peak also prevents useful CUDA tiles
when the raw provider itself dominates that peak.

## Decision

Qualify the native CCSD(T) force owner through 28 AOs. Preserve the separate
12-AO CCSD and Python complete-gradient boundaries and all physical stability,
canonicalization, independent residual and memory gates. Larger dimensions
remain experimental until independently qualified.

Compute the complete endpoint's minimum peak with a width-one raw provider.
Select the largest basis-wide source tile fitting the caller's complete budget,
using the same generated provider capacity formula in planning and execution.
Report the selected peak and retain the minimum separately. A one-byte reduction
from a roomy selected peak can shrink a tile; one byte below the minimum must
fail before triples buffers or source reads. Do not confuse these two tests.

For dense final ERI adjoints, project the MO weights onto all eight ERI index
permutations with equal weights. The same coefficient matrix transforms every
slot, so this projection commutes with the MO-to-AO pullback. Generate the
projected weight in the first transform reduction without allocating another
rank-four tensor. Then visit canonical shell quartets and multiply each weight
block by the number of distinct shell permutations: 1, 2, 4 or 8, according to
repeated pairs and shells. Full component blocks within repeated shells remain
present. The primitive consumer still differentiates each slot independently;
physical atom scatter happens afterward, including coincident centers.

Factorized two-electron weights retain their existing ordered schedule. The
one-electron and nuclear contributions, derivative kernels and response equations
remain the same. No screening, force projection or reference-oracle result enters
the production calculation.

## Resource regression found at larger dimensions

The 14-AO allocator probe exposed 97 bytes of diagnostic-string allocation on top
of the triples arena's exact numeric peak. Defer the triples program-hash and
completion-reason publication until that dead arena is released. This restores
the allocation-observed bound without increasing a budget or weakening the
one-byte-short rejection. Small systems previously hid that metadata beneath
another phase's larger peak.

## Qualification

- The native derivative contract uses deliberately nonsymmetric MO weights on
  two-AO and five-AO s/p fixtures, including repeated shells and two shells on
  one atom. It compares canonical traversal against an independent full-coordinate
  AO pullback contracted with dense integral derivatives. It checks the exact
  canonical shell count, including orbit degeneracies.
- Host planner tests cover CPU/CUDA layouts, full-width selection, tighter-budget
  shrinkage, width-one admission and minimum-minus-one rejection. Native force
  allocator tests cover H2O, NH3 and the 14-AO water dimer, including exact source
  read/value counts and retained-input capacity changes.
- Public tests use pinned PySCF 2.14.0 corrected CCSD(T) Lambda forces with the
  exact repository basis-pack primitives. Cold/warm/changed-geometry reuse and
  two-step 14-AO directional energy differences protect the relaxed force.
  `benchmarks/ccsdt_cluster_oracle.py` regenerates independent references.
- Complete endpoint experiments retain baseline and candidate binary/source
  identities, all measured repeats and work counts. Real-device execution and
  memcheck use finite Slurm allocations. Unfinished or failed 56-AO experiments
  are not evidence for the public 28-AO boundary.

The [retained complete-endpoint comparison](../../../../benchmarks/results/cc-large-force-20261003/summary.json)
contains all cold, twice-warm and changed-geometry outputs. The experimental
comparator raises the old AO guard and allows raw tile widening only under its
prior peak; it retains ordered derivatives. It is not an old supported public
14/28-AO endpoint. Its exact patch and baseline commit are retained for replay.

Warm CPU means decrease from 43.01 to 10.07 seconds at 14 AOs and from 750.35 to
193.41 seconds at 28 AOs. Warm GPU means decrease from 11.46 to 1.62 seconds and
from 182.46 to 27.19 seconds, respectively. At 28 AOs the candidate cold/changed
endpoints take 195.52/194.07 seconds on CPU and 75.06/68.95 seconds on GPU. Nodes
were shared, and the 28-AO CPU pair used different pinned cores. These are
complete endpoint measurements, not device-residency claims.

The derivative traversal visits 1,540 rather than 10,000 shell quartets at
14 AOs, and 22,155 rather than 160,000 at 28 AOs. These are exact traversal
counts, not nonzero primitive or launch counts. The 14-AO native allocator
probe observes one full-width source read and 2,151,296 transform FMAs at the
selected 11,618,144-byte peak. At the 11,343,144-byte minimum, it observes 16
source reads, the exact measured bound, and forces agreeing within 1e-11.

All recorded energy, triples and force outputs pass independent PySCF gates;
maximum errors are 1.5e-12 Eh, 1.1e-14 Eh and 6.7e-8 Eh/bohr. All 39 public
CPU/CUDA tests pass on node1, including the new 14-AO force/finite-difference
test. Complete 14-AO CUDA force memcheck reports zero errors. Native tests
also preserve the minimum-minus-one refusal before numerical work.

## Rejected alternatives and revisit conditions

The experimental 56-AO water cluster fails the existing canonical occupied/virtual
subspace degeneracy gate (node2 Slurm job 2066). The detailed single-point error
was `degenerate RCCSD(T) canonical occupied/virtual subspace`; the batch wrapper
only reported numerical failure. Keep the 1e-10 same-space gap gate and the public
28-AO boundary until degenerate-subspace response is independently qualified.

The [follow-up diagnostic](../../../../benchmarks/results/cc-large-force-20261003/degenerate-frontier.json)
uses the faster reference owner from #1728 but preserves the numerical gate.
Independent exact-basis PySCF RHF and native response find the same 14 occupied
or virtual degenerate pairs, with gaps around 1e-13 to 1e-15 Eh. Native null-space
stationarity is at most 7.42e-13, yet dividing it by these gaps would create a
spurious Fock cotangent up to 190.94. Pairwise diagonal denominator cotangents
agree within 2.6e-16, but this does not establish the full block adjoint or make
zeroing off-diagonal response a generally valid fix. A future extension needs
a gauge-invariant triples-denominator response, with independent gradient and
orbital-rotation checks. Do not weaken the gate or perturb away this fixture's
symmetry merely to declare larger force support.

Increasing a memory limit without changing source/derivative traversal does not
reduce repeated work. Applying a constant factor of eight is incorrect when
shells repeat. Symmetrizing only two MO indices is insufficient for arbitrary
generated adjoints. Folding physical atoms before differentiation loses distinct
slot contributions. Materializing full coordinate-major derivatives or another
AO rank-four weight cache is unnecessary for this bounded transformation.

The complete force remains quartic in retained ERI/weight storage, and parts of
the CUDA response and weight preparation remain host-owned. Revisit larger
qualification only with independent forces, finite differences, resource tests
and complete endpoints. Further CUDA speedups should measure the retained MO
provider and primitive submission/response work instead of assuming that fewer
derivative quartets dominate every backend.
