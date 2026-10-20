# Decision: use the shared phased derivative for Treutler partitions

Status: implemented
Date: 2026-10-20

## Problem

The initial Treutler grid implementation deliberately retained the bounded
serial/cooperative derivative, even when a large stationary owner had already
reserved phased scratch. Its pair-coordinate graph supported radius adjustment,
but the phased CUDA entry did not receive the elemental radii. Removing the
dispatch guard alone would therefore have silently produced equal-radius forces.

## Decision

Pass the topology owner's immutable radii through `PhasedBeckeInput` to the
existing `PreparedCenterGeometry` in both primal and reverse pair kernels.
The common generated scalar graph handles clipping, the Treutler adjustment
and its chain rule. No new partition algebra, pair layout or scratch allocation
is introduced. Atom products, normalization, ordered gather and point motion
retain their existing phase boundaries.

Admitted Treutler owners now use phased execution and report retained pair work.
Native automatic admission still starts at 48 atoms, and all existing device,
lane, cache and concurrent-byte conditions remain. Missing optional storage
retains the bounded traversal. Ragged owners are supplied explicitly or resolved
from the existing atom-prefix topology, including point tails and external VV10
seeds. Geometry rebinding changes center geometry, never elemental radii. The optional
FP32 transport still executes bounded kernels, so its pair counter ignores
reserved FP64 phase panels and counts the actual selected traversal.

The optional coefficient and normalized-adjoint reverse schedules compose with
the same adjusted-coordinate graph. Binary-scaled products remain excluded for
adjusted partitions; this change does not independently promote that different
arithmetic policy. Mixed/second derivatives retain their existing consumers.

## Invariants and qualification

- Independent 65-digit Decimal direct products and moving-point differences
  check the radius caps, pair orientation and chain rule at 1, 3 and 5 switch
  iterations. The directional derivative gate is 3e-14.
- Host bounded/phased comparisons include 2--128 atoms, signed and zero seeds,
  moved centers, exact zeros, invalid-input publication and semantic pair work.
- Real stationary CUDA owners cover 48, 96 and 128 atoms, explicit and ragged
  implicit owners, external VV10 seeds, ordinary/coefficient/normalized reverse,
  serial/cooperative normalization, zero-seed controls, geometry reuse and
  failure-before-publication. The source gate is 2e-11 absolute + 5e-12 relative.
- Complete 48-atom water/STO-3G level-0 energy/force gates exercise native and
  composite owners against independent PySCF, reconverged finite differences,
  cold/warm/moved execution, and the native constrained-memory fallback. Gates
  are 2e-8 Hartree, 3e-7 Hartree/Bohr and 2e-7 directional finite difference.
  This deliberately coarse discrete quadrature tests schedule correctness;
  it does not qualify production grid convergence or refresh README benchmarks.

The scientific implementation at `eaaf103c2`, based on merged grid commit
`27b79cc80`, passed all four PBE0/WB97M-V Direct/DF endpoint gates on n1
under finite Slurm allocations (RTX 5090, CUDA 12.9, sm_120). Maximum errors
were 6.14e-12 Hartree and 2.14e-9 Hartree/Bohr. Each grid had 37,216 real points;
WB97M-V used that size independently for XC and VV10. Composite-owner counters
confirmed 41,979,648 primal pair visits per complete force at 48 atoms.
The native PBE0 endpoint also passed the minimum-capacity fallback.

All 30 adjusted CUDA owner cases and 366 legacy owner cases passed. Two
adjusted cases (48 and 128 atoms) passed each of memcheck, racecheck, initcheck
and synccheck without errors or race warnings. Independent host derivatives,
resource/dispatch regressions and the compiler/ownership checks also passed.
A stale legacy test expectation was corrected: a valid equal-spin proof selects
the restricted point contraction in both inline and bulk evaluation; phased
storage controls only admission of the bulk producer.

The subsequent FP32 work-counter correction changes no scientific arithmetic.
Its executed-schedule truth table and transport regressions passed 77 host
checks. The final rebuild also passed all 30 adjusted owner cases and repeated the
native PBE0 Direct endpoint, including its bounded-memory fallback. Final
native library SHA-256 is
`781ece67ae51c6e2627cddabe0f9c9efca7cfa4954ae685c94d1ed980018d2b6`;
the standalone PBE0 stationary library is
`537506fac0a26c6ab0a10be0e052982cea412715fba4df4f2d6293d88d7b52e3`. Raw logs, source identities, compiler commands and cache statistics are
retained in `.artifacts/treutler-phased/`; n1 receipts are under
`/data/jzzeng/treutler-lebedev-2609/receipts/treutler-phased/`.
CMake and runtime compilation used verified sccache 0.16.0 without clearing its
shared cache; local host probes used verified ccache 4.5.1.

## Consequences

Each phased point retains one primal state per unordered pair rather than
regenerating the bounded multi-strip traversal's pair state during reverse.
The four-word pair panels and seven phase boundaries are unchanged. More pair
parallelism trades optional resident scratch for work reuse; no endpoint speedup
is inferred from isolated pair counts.

## References

- [Initial grid decision](../numerics/2026-10-20-treutler-lebedev-default.md)
- `tests/python/test_treutler_phased.py`
- `tests/python/test_treutler_phased_cuda.py`
- `tests/python/test_treutler_phased_endpoints.py`
