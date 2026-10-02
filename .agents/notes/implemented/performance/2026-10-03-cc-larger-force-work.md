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

## Rejected alternatives and revisit conditions

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
