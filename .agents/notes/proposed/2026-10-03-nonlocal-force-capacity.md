# Proposal: admit complete-grid nonlocal forces by their actual native capacity

Status: proposed; device and complete default48 endpoint qualification pass
Date: 2026-10-03

## Problem

The 48/96-atom full-grid WB97M-V comparison exposed two independent admission
cliffs. The default 256 MiB nonlocal cap rejects the SCF provider during prepare.
An explicit 1 GiB cap admits SCF but the force driver then truncates the cap to
one quarter of its 1 GiB device allowance, irrespective of its other live owners.
The resident force arena needs about 240 bytes per complete-grid point. It cannot
be shrunk by reducing the AO tile, and spare total force capacity was unusable.

Node2 job 2045 reached a separate force-JIT failure: its default GCC 12 installation
lacks cc1plus. The task uses the existing complete GCC 11 through NVCC_CCBIN;
no system compiler or driver was modified. Node5 job 1387 was stopped after the
predictable force cap was identified. Explicit 4 GiB force-budget diagnostic runs
are separate from default-capacity qualification and cannot establish it.

## Decision

The native owner exposes a GPU-free required-byte query shared with its allocator.
It charges 22 coexisting I/O arrays, the canonical pair workspace (including
ordered active-partner metadata) and three sticky error flags. Python asks this
owner instead of duplicating its inventory or guessing a fraction of another
budget. The existing composite planner reserves the exact result, then admits
the AO, two stationary accumulators, native derivative allowance and conservative
host inventory under the original 1 GiB device / 2 GiB host totals.

The KS nonlocal default becomes a finite 1 GiB cap. Allocation remains sized by
the grid. Explicit user caps are unchanged and are checked before force JIT or
allocation. Overlarge full-grid requirements still fail; there is no unbounded
allocation, CPU fallback, threshold change or implicit reduction in work.

## Evidence and limits

Host tests cover ABI extent validation, a device-free query, missing-query failure
and preservation of explicit budgets through semantic options and the native ABI.
Real-device tests compare reported bytes with actual retained allocation, including
48/96-atom full-grid extents and partner-block tails. Exact capacity must succeed;
one byte less must fail. These allocation tests do not evaluate quadratic pairs and
are not complete energy/force or performance qualification.

Node1 passes all 16 capacity/ABI tests, including seven real allocation extents
through 2359296 points. All six independent complete RKS/UKS WB97M-V and
reconverged displaced-energy tests also pass. The initial combined run expected
MemoryError for one-byte-under-budget rejection, while the established Python ABI
raises RuntimeError with native OUT_OF_MEMORY (7). That test expectation was
corrected and all capacity tests rerun; production behavior was not changed.
The qualified library SHA256 is
`f1253b0aa722ff119f6dc99ff61731f0da1ac473e38f53a8ef2233beb24a9c47`.
Node1 job 5382 completes the full 48-atom comparator with public defaults:
384 spherical def2-SVP AOs, 1179648 grid points, WB97M-V RKS, unchanged total
1 GiB device / 2 GiB host force limits. All three pairs (cold, priming, one timed
warm repeat) pass, with maximum energy error 4.525e-11 Eh and force error
1.707e-10 Eh/Bohr. Native cold/warm times are 2092.459/284.679 seconds versus
GPU4PySCF 520.076/175.466 seconds. Cold SCF iterations are 30 versus 18 and
warm iterations one versus four. This qualifies default-capacity completion;
it is not a speedup claim or a robust multi-repeat performance estimate.
The 96-atom native-allocation boundary test does not establish a default96
complete endpoint; that qualification remains separate.

The build uses verified ccache and explicit CXX/CUDA launchers with checkout-root
CCACHE_BASEDIR. Build receipts and subsequent endpoint results are retained under
ignored `.artifacts/wb97m-capacity/` and the authorized remote task directory.

## Revisit when

Borrowing immutable SCF grid/features or reusing nonoverlapping force buffers could
lower the actual requirement. Such a change must prove lifetime/stream identity,
update the same native inventory, preserve exact-budget tests and retain a bounded
independent owner when borrowing is unavailable. Raising caps does not reduce the
quadratic nonlocal work or establish a GPU4PySCF performance advantage.
