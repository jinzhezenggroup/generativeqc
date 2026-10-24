# Decision: separate spatial XC execution from canonical moving-grid ownership

Status: implemented, opt-in
Date: 2026-10-19

## Problem

Continued PR #2609 source reading found that GPU4PySCF's `Grids.build` sorts
points by nearest atom after a greedy atom walk. The matched benchmark replaces
its atomic rules, but does not disable this default sorting. Native retains
atom/radial/angular order. Identical point sets therefore create very different
local AO unions and contraction work. Larger blocks alone made native slower.

On n4:1176, with identical native core density and upstream AO cutoff1e-16,
GPU4PySCF complete dense XC is5.953666s unsorted vs2.435611s sorted; occupied
XC is3.929156s vs1.909951s. Maximum block AO count is932 vs458. All matrices
agree within5.80e-14. This is a causal ablation, not an assumption about sorting.

On n4:1177, permuting only native generated points/weights changes complete
256-point fixed-D XC from3.988047s to3.016372s; point-AO-square work drops from
164,382,040,064 to122,486,172,672 and maximum active AO767 to450. Its matrix
agrees with original-order native within6.22e-15. Sorted1024 is3.518009s and
is not promoted. Setup is separate in these diagnostics; endpoint timings below
must include it.

## Decision and ownership

`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=1` prepares an optional exact execution copy
before local maps. The default is unchanged. A deterministic nearest-neighbor
center traversal plus stable counting permutation groups nearby points; no
point, quadrature weight, threshold or numerical precision changes. This is
scheduling over existing native grid values, not CPU/reference quadrature.

The original immutable `MolecularGrid` remains the canonical owner. `CudaXcPlan`
retains both views; SCF uses the spatial copy and final stationary bindings use
the authenticated canonical view. This is essential: force atomic measures and
`points_per_atom` ownership cannot be attached to sorted indices. Nonlocal
consumers remain canonical because their force/cache identities require separate
qualification. Same dimensions alone cannot authorize a different grid owner.

Allocate four doubles per point as separate optional XC storage; report its
actual bytes in the owner's resource count. Explicit host allowance covers
staging plus two point-index/four atom-index arrays. No replacement fixed cap is
introduced. Device OOM/insufficient allowances preserve canonical execution;
host metadata OOM and unrelated runtime failures propagate. Public ledger-bound
owners remain excluded until the fleet planner reserves this lifetime. Setup
publishes pointers only after upload/synchronization; failure drains before
staging release. Destruction drains before freeing the execution copy. Replay
requires no allocation or host ordering, and geometry changes rebuild the plan.

## Validation

n2:3480 and n4:1178 execute independent CPU PBE/r2SCAN E/V gates, RKS/UKS,
nontrivial spherical AO/tail tiles, changed-density graph replay, exact budget
and one-byte-short host/device refusals, immutable map rejection and distinct
canonical/execution pointers. Small cases pass all four compute-sanitizer tools;
memcheck/initcheck/synccheck report zero errors, racecheck zero errors/warnings/
hazards. This is not large-system sanitizer evidence. Existing host grid,
stationary ABI/admission, point batch resources, XC selector and DIIS integration
suite:103 passed. Compiler structure:568 modules, zero dependency errors.

Source identity: `11547d49f90d375d15a43c0365e0a037723a966d2771d8d97cce1aba3d63b145`.
Library SHA256: `1f5c2e62d6634814946daf30c403201e8a5ce795a936edf2093be6747917768d`.
Receipts: `.artifacts/pr2609/spatial/`, with separate source/runtime snapshots on
n2 and n4. These supersede no historical source identity.

## Complete endpoint evidence

n4:1178 clean E/F uses primitive J plus spatial XC. Cold includes construction,
ordering, maps, SCF, full moving-grid force, corrections and host publication.
Protocol is the same96-atom FP64/direct/noDF48x16x32 grid as the primitive note.

| Endpoint (s) | Primitive, canonical grid | Primitive, spatial grid | GPU4PySCF baseline |
|---|---:|---:|---:|
| r2SCAN-3c cold | 180.891338 | 157.723791 | 153.635193 |
| r2SCAN-3c warm | 27.006406 | 26.050681 | 31.035172 |
| r2SCAN-3c moved | 109.797212 | 102.479027 | 107.229452 |
| r2SCAN-3c moved-warm | 26.978976 | 26.020513 | 26.641672 |
| PBE0 cold | 67.667604 | 62.549000 | 76.804166 |
| PBE0 warm | 10.615096 | 10.365096 | 13.853463 |

Cold/moved n=1; warm medians n=5. All18 new endpoints pass independent gates.
Max r2 E/F errors are1.00e-11Eh/4.58e-11Eh/bohr; PBE0 errors are
5.46e-12/2.45e-11. r2 cold remains24Fock; moved changes13to14, and PBE0
cold18to17, illustrating floating-point ordering's effect on solver trajectories.
Warm has one Fock. Report those counts with timing; this is not equal-work
kernel ranking or a universal/grid-converged production comparison.

Relative to236.820267s after the capacity fix, the combined cold is33.4% faster,
but remains2.7% slower than the original upstream cold sample. The existing DIIS
reducers and a fresh upstream cold are separate follow-up measurements, not
silently included in this spatial-grid result.

## Rejected alternatives

Globally sorting MolecularGrid without owners/atomic measures breaks derivatives.
Reusing a differently ordered VV10 final cache is also unsafe. The chosen private
copy avoids changing those contracts, at a retained-memory cost (72MiB for the
2,359,296-point case) and an explicitly timed cold preparation cost.

Blanket cuBLAS promotion is rejected by n4:1174: generated256 XC3.988s,
density BLAS3.744s, potential BLAS6.797s,both5.991s, with identical science.
The optional provider disables compact batching, so these are complete provider
choices, not isolated GEMM rankings. Occupied factors, shell sorting, local AO
threshold and independent density/potential batching remain different upstream.

## Revisit when

A fleet resource plan explicitly reserves optional grid copies; nonlocal force
bindings can authenticate execution order; or further endpoint data supports
an automatic ordering policy. Do not infer universal speedup from one water
basis or enable unqualified library routes because upstream uses BLAS.

See [the complete source audit](2026-10-19-gpu4pyscf-cold-source-audit.md) and
[primitive-J decision](2026-10-19-primitive-complete-md-j.md).


## Fresh upstream and independent follow-up controls

n4:1180 repeats complete GPU4PySCF1.8.1 cold:153.314540s,60JK, E/F pass.
The final retained native157.723791s is2.9% slower than this repeat; it is not
claimed to beat upstream. Existing ordered/cooperative incremental DIIS in
n4:1179 gives156.680948/154.634718s,24Fock, with cold and one warm gate each
passed. These selectors remain explicit and their samples stay separate.
The [local FMA candidate](../../rejected/2026-10-19-md-j-explicit-fma.md) was
numerically valid but unprofitable and was reverted. The retained spatial source
identity above remains the final implementation.

## Subsequent default promotion

The opt-in selection decision above is superseded by
[the default-promotion decision](2026-10-19-md-j-spatial-defaults.md).
Historical measurements and source identities above remain unchanged.
