# Decision: separate SCF capacity from derivative leases and retain pure-J streaming

Status: implemented
Date: 2026-10-21

## Problem

The Cartesian capacity tightening in #2660 was intended for the lazy exact LR
derivative provider in mixed DF/VV10 forces. Its predicate used derivative order,
but the ordinary Direct SCF provider also owns first-order resources. Tightening
that provider's allowance evicted its optional MD-J source for r2SCAN-3c/96.
The exact fallback remained correct but each J application became much slower.

A second independent cost affected Fast r2SCAN-3c/48. Automatic residency built
1,911,873,962 FP32 integrals and rebuilt the same number in FP64 at the strict
handoff. Their admitted additional capacities were 8,148,720,620 and
15,796,216,244 bytes. The two preparation scopes took 8.83 and 18.42 seconds;
their complete cold endpoint was slower than using the existing J provider.

## Decision and boundaries

Distinguish the primary SCF provider from a derivative-only provider explicitly.
Only the two lazy LR capacity queries use the tighter actual-Cartesian envelope.
The primary SCF owner retains its previous shape-based allowance and bounded
optional-storage fallback, even when its derivative order is one. This restores
an established allocation envelope; it does not prove that every optional MD
shape fits, borrow an uncharged force reserve, or change the CUDA stack policy.

For automatic pure-J through-d KS, retain an available generated strict
provider instead of building a second quartic source. The incumbent dispatcher
can use MD-J within that provider. Fast's other admitted mixed operations and
physical FP64 closure remain active. Explicit residency, mixed hybrids,
uncovered high-angular operators and range corrections retain their previous
policies. Missing generated capability still permits bounded source admission.

No grids, convergence thresholds, auxiliary basis, precision gates or force
components change. This supersedes the overly broad capacity predicate in
[the streamed-Coulomb decision](2026-10-21-streamed-coulomb-and-driver-memory.md).

## Diagnostic evidence

All real-device calls used finite Slurm allocations on n1/node1 RTX 5090, with
CUDA 12.9.1, GCC 13.3 and verified sccache 0.16.0. Cold includes construction,
preparation, SCF, forces and export; warm uses the calculation's own frozen
cold density. These are complete endpoints, not isolated J-kernel timings.

| r2SCAN-3c Direct | Cold seconds | Warm seconds | Target iterations |
| --- | ---: | ---: | ---: |
| Previous v4 Standard/96, fresh replay | 83.360 | 19.188 | 22/1 |
| README binary Standard/96, clean replay | 247.595 | 26.268 | 22/1 |
| README binary Standard/96, traced replay | 248.933 | 26.514 | 22/1 |
| Capacity fix v1 Standard/96, traced | 82.373 | 18.906 | 22/1 |
| Capacity fix v1 Fast/96, traced | 59.582 | 18.678 | 14/1 |
| Capacity fix v1 Fast/48, traced | 45.582 | 5.315 | 10/1 |
| Same v1 Fast/48, residency disabled | 15.326 | 5.166 | 10/1 |

Warm is median-of-three for v1, one observation for the earlier replays. The
untraced/traced README comparison reproduces the regression without attributing
it to measurement fences. Previous/current Standard densities yield energy
agreement to 4.7e-11 Eh and force agreement to 3.0e-11 Eh/Bohr. Every completed
v1 population passes the unchanged independent README energy/force gates,
including DF WB97M-V/12. These observations do not establish universal leadership.

The unchanged mixed-DF GPU tests using `density_fitting="auto"` fail during
auxiliary-handle role admission on both the baseline and v1, before the changed
provider code. Preserve those failures. Separate diagnostic executions use
explicit backend selection and retain the original independent displaced-energy,
force, warm-replay and capacity assertions; they are not claimed as a passing
unmodified AUTO test.

The combined v2 clean sweep completes all 64 Direct populations: eight methods,
four sizes, Standard/Fast, and one cold plus three frozen-seed warm calls each.
All 256 energy-plus-force calls complete. Independent gates qualify 62
populations; both M06-2X/3 defaults retain their pre-existing grid-accuracy
failures. Against the unchanged, independently qualified GPU4PySCF receipts
from #2662, the observed Direct cold/warm wins are 18/31 and 26/31 for Standard,
21/31 and 27/31 for Fast. Small latency differences do not establish statistical
significance, and the remaining slower cases are retained.

Final clean r2SCAN-3c/96 cold/warm medians are 79.359/19.056 seconds (Standard)
and 52.520/18.777 seconds (Fast). Fast/48 is 15.060/5.152 seconds, versus
44.821/5.149 seconds in the published README campaign. The numerical controls
and target iteration counts remain unchanged. These diagnostics do not update
the README's complete Direct/DF campaign or its source-bound figures.

The separate DF WB97M-V/96 check also completes and qualifies all four calls:
708.743 seconds cold and 110.246 seconds median warm. Thus all 65 v2 populations
and 260 energy-plus-force calls complete, with 63 qualified populations and
the two retained M06-2X/3 failures. The lazy derivative capacity remains usable
at the largest README geometry.

Validation also includes the 1,549-test host allocation/bridge suite, 24
phase-source/precision/bridge checks, 37 interface/resource checks, both explicit
DF-backend displaced-force cases, all pre-commit hooks, and a complete Fast
r2SCAN-3c/12 cold plus three warm endpoints under Compute Sanitizer memcheck
(zero errors). The diagnostic compiler commands both invoke the verified
sccache launcher; cache statistics and source checks are retained.

## Source binding and retained artifacts

The v1 source is master `5caddea2215e8fa60d44b9ab17b5d109880e26ba` plus
the archived capacity-only patch; its core SHA-256 is
`f4c801fd8fcf4ff366a68e5198b6a6c435c48a22e207efd951b45ccce0fbeeca`.
The combined v2 core SHA-256 is
`f95ed636f503ab0f8e1db55c1beb2157c8b4896098b02b6b62a71c97eb96f16d`.
Both retain manifests covering all 5,852 source/asset inputs, patches, compiler
commands/cache statistics, launchers, complete receipts and failed attempts.

Ignored local `.artifacts/provider-allowance-v{1,2}/` and remote
`n1:/data/jzzeng/scf-performance-20261021/provider-allowance-v{1,2}/` retain these
artifacts. The separate baseline comparison is retained under
`qc-scf-performance-20261021/.artifacts/direct-regression-v1/`. Production README
measurements in #2662 remain bound to their original binary; diagnostics do not
silently replace its points or relabel older sources.

## Revisit when

Replace the primary allowance with a tighter bound only after inventorying its
optional J owner and checking complete endpoints across all affected shapes.
Reconsider pure-J residency if a generated value source removes its preparation
cost, or a consumer demonstrates enough repeated work to recover that cost.
Memory capacity and a maximum iteration count alone do not demonstrate reuse.
