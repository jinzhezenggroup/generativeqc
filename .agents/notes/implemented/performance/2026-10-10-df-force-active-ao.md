# Decision: qualify native AO-jet bitmasks for fitted stationary forces

Status: implemented
Date: 2026-10-10

## Problem and causal evidence

The preceding retained-charge/geometry admission change removes the large DF-J
force refit, but its 96-atom PBE0-DF cold E+F median is still 99.337125 s versus
76.572512 s for GPU4PySCF. Those historical populations use the earlier frozen
source, not the current-master control below. See
`2026-10-10-df-coulomb-force-and-grid-admission.md` in this directory.

A new force-only Nsight Systems capture of the qualified predecessor attributes
8.978471 s to 9,280 NN GEMM kernels, versus 1.862881 s to point evaluation,
1.803694 s to cooperative AO geometry and 1.341925 s to primal Becke pairs.
These are intrusive kernel totals, not complete endpoint timings. The largest
remaining force consumer is therefore the dense AO density projection, not the
Becke pair domain. The public force selector excludes every fitted provider even
though DF does not change the moving-grid AO derivative.

## Decision and scientific boundary

Add a fitted ordinary-force profile for the existing native
`exact-jets-native-bitmask` consumer, above the same continuous dense
point-times-AO-squared crossover as the incumbent large direct-force profile.
Keep derivative orders 1/2, one/two spin blocks, all-electron/resident-grid
requirements, fixed 256-point or budget-auto tiles, and the existing 512/256 MiB
additional-consumer budget requirements. No molecule, functional, auxiliary-basis,
GPU product or atom-count whitelist is introduced.

Retain the existing force AO cutoff **1e-16**, not the SCF density or grid-weight
threshold. Every requested AO jet at every point in the tile participates in
the strict cutoff predicate. In particular, order-two forces inspect second
derivatives themselves and do not reuse the SCF's order-one inventory. "Exact"
means exact labels at that explicit sampled-jet cutoff; it does **not** mean
unscreened mathematics. All grid points, weights, Becke primal/reverse pairs,
DF integral response, AO cross terms inside the selected domain, and public
energy/force acceptance gates remain present.

The native producer evaluates the authoritative generated AO jets without
writing a full discovery panel or looking up labels on the host. The existing
grid owner maintains compact labels, matrix gathers, stream ordering and
token-checked lifetimes. The compiler owns the unchanged AO mathematics and
predicate lowering; this change only qualifies an already implemented consumer.

## Bounded fallback and resources

The optional requested map allowance is 64 MiB. The exact-bitmask resource
planner charges its actual bounded inventory to both host/device admission,
without spending the native derivative provider's independently reserved bytes.
For the 96-atom fixture this charge is only 1,038,352 bytes, not 64 MiB.
The complete map inventory must have average active fraction <= 0.8. Missing
native capabilities, insufficient optional budget, allocation failure or
occupancy rejection retain the existing dense route. Unknown/nonresident
providers, composite/ECP consumers, insufficient complete-owner budgets and
work below the qualified crossover are not newly selected.

Geometry or basis changes invalidate/rebuild the map. Same-geometry replay
reuses it only under the existing grid token and generation checks. Retained
maps must not weaken failed-task publication or borrowed-stream drains.

Additional-consumer bounds explicitly exclude retained SCF/snapshot storage and
the independent DF response owner. They are not measured whole-force allocation
peaks and do not qualify whole-KS resource estimation.

## Qualification protocol

The source base is frozen master `464df951f`; the performance change touches
only `python/generativeqc/_force_active_ao.py`. Both arms use the newly built
current-master main library SHA-256
`5757f82f923cf30fb1519b7942a3388767008069cf96d22208cd75da94661751` and its
matching stationary AOT siblings. Compiler caching is verified ccache 4.5.1,
with explicit CXX/CUDA launchers and retained pre/post statistics. Reconfigure
does not clear the cache. The subsequent master #2176 changes OpenBLAS configure
probes, not this CUDA consumer; it does not relabel the frozen measurements.

The full endpoint fixture remains 32 waters/96 atoms, spherical def2-SVP
(768 AOs), cc-pVDZ-JKFIT (3,712 auxiliaries), FP64 PBE0-RKS and the unpruned
48x16x32 grid (2,359,296 points). Fresh processes, owners and densities retain
disk caches. Preparation through the first synchronized public E+F result is
timed; imports, CUDA initialization, Calculator construction and post-result
teardown remain outside both arms. GPU4PySCF is 1.8.1. Energy/force gates are
1e-8 Eh and 3e-7 Eh/bohr over every component, with native physical residual
<= 1e-9. No normalization by Focks substitutes for the endpoint.

Independent CUDA tests exercise LDA/PBE/PBE0, RKS and unequal-spin UKS, spherical
s/p/d orbitals with unequal orbital/auxiliary bases, cold/replay/moved/replay,
and both zero-budget and occupancy-declined dense fallbacks. Test-only lowered
work/occupancy admission exercises small fixtures without expanding the
production profitability domain. CPU derivative entry points are forbidden
inside native execution. The original tetramer-cation LDA reference fails to
converge in 200 cycles; the retained failure is not a native regression. The
open-shell oracle uses the independently convergent single-water cation instead,
without loosening any convergence or force gate. A second initial test exercises
the intended small-system occupancy fallback; the final routing test separately
admits the inventory and tests the fallback explicitly.

## Evidence retention and revisit conditions

### Matched complete cold population

Nine fresh processes run on one Slurm-assigned GPU in the order
`dense0/native0/reference0/reference1/native1/dense1/dense2/native2/reference2`.
All nine endpoints pass the complete numerical gates. Do not discard the last
reference's extra SCF iterations or replace it with an older faster population.

| Arm | Complete E+F samples (s) | Median (s) | Force median (s) | Iterations / Focks |
| --- | --- | --- | --- | --- |
| Current-master dense force | 96.754534, 96.498385, 96.108128 | 96.498385 | 26.004491 | 24 / 24 each |
| Fitted force bitmask | 86.909555, 86.942887, 86.973079 | 86.942887 | 16.440103 | 24 / 24 each |
| GPU4PySCF | 75.654161, 76.728038, 89.320742 | 76.728038 | 10.431943 | 37/38, 37/38, 46/47 |

The change reduces the matched complete endpoint by **9.9022%** and its force
phase by **36.7798%**. The native endpoint is still **13.3131% slower** than the
reference median; this does not establish parity. GPU4PySCF's wide endpoint
range is accompanied by actual iteration/Fock variability, not hidden through
per-Fock normalization. Native preparation medians are 18.244910/18.268778 s;
this force-only change does not claim a preparation or SCF speedup.

Maximum independent energy/force errors across both native arms are
2.273737e-11 Eh / 1.452798e-10 Eh/bohr. The maximum candidate-versus-dense force
change over all 288 coordinates is 2.289818e-13 Eh/bohr. Each native residual is
5.972529e-13. The earlier baseline's 25 iterations are not relabeled as these
current-master 24-iteration measurements.

### Work, budgets and validation

The force inventory records 1,516,176 selected tile AO entries (min 0, max 553),
768 empty tiles, 388,141,056 point-AO visits and 81,057,099,776 point-times-AO-
squared work versus 1,391,569,403,904 dense work. Density projection retains four
jets, 8,448 passes, 33,792 matrices and 324,228,399,104 FMA pairs. Discovery visits
all 18,119,393,280 required AO jet values once, writes no discovery AO panel,
uses no host AO-label lookups, and copies only 73,740 discovery bytes to the host.
The 80% occupancy gate is satisfied without declining any tile inventory.

The stationary additional-consumer bounds are 400,267,312 device and 193,267,824
host bytes, including the 1,038,352-byte map allowance actually admitted. The
existing 256-lane/9,216-batch Becke phase route is retained. Independent matched
receipts assert equal XC point and Becke primal/reverse visit counts in both
arms; this is not grid pruning or a reduced Becke domain. The DF response owner
remains outside the stated additional-consumer bound.

Final validation: 224 focused device-free regressions, 509 compiler modules with
zero dependency errors, and all six independent Slurm CUDA force tests pass.
Configured Ruff lint/format checks and `git diff --check` pass. This does not
claim that the entire current-master CUDA suite was executed.

A separate post-selection Nsight/cProfile diagnostic makes point evaluation the
largest remaining individual grid kernel (1.807984 s), followed by primal Becke
pairs (1.333425 s). The previously dominant specific NN GEMM kernel family falls
to 0.656749 s; cooperative AO geometry is 0.485265 s. These intrusive component
totals are not added to endpoint wall times. Retain the follow-up scheduling
experiment separately from this qualified policy change.

Raw receipts, scripts, library hashes, cache statistics and separate intrusive
profiles are retained locally under `.artifacts/pbe0-parity/` and on node n1 under
`/data/jzzeng/qc-pbe0-df-cold-20261009-4e20f7a7b/results/`. All real GPU work runs
through finite `srun` jobs, preserving assigned visibility. No Release/tag or
external evidence host is created.

Revisit the crossover after complete-endpoint evidence on different source/device
families, not an isolated projection microbenchmark. Any cutoff change, reuse of
a lower-order map, new composition domain, or extra resident provider requires
new independent numerical and complete-owner admission gates. Remaining point
evaluation, Becke phases and non-force preparation/SCF costs require separate
profiling; reduced AO projection work alone does not establish GPU4PySCF parity.
