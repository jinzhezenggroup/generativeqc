# Decision: qualify native force AO work at the actual public endpoint

Status: implemented
Date: 2026-11-17

## Problem

#1853's October explicit-consumer matrix proved map mathematics and lifecycle,
but explicitly did not prove native public force selection. The tracker and
current-state documentation still described the native controller as dense.
The current master controller instead already uses the shared native exact-jet
map owner for supported force bindings at its work/resource crossover. Repeating
the old explicit matrix, changing Python test profiles, or adding functional-name
dispatch would not establish which path a public native call executes.

## Decision

Keep the existing native scientific binding, four-/ten-jet capability, `2**35`
dense-work crossover, `1e-16` cutoff, 16-MiB optional peak reservation and 0.8
mean occupancy guard. Add a decline-only native map control,
`GENERATIVEQC_CUDA_FORCE_ACTIVE_AO=0`, for same-binary complete-endpoint dense
ablations. It is read when constructing a controller; it does not change a
retained owner's schedule or shrink the published conservative capacity.

Correct `indexed_grid` tracing to distinguish actual indexed FP64 execution from
a retained map bypassed by the separate FP32 grid route. Retained allocation or
a policy label alone must never establish sparse work.

Qualify public native E/F above the unmodified crossover. Independent GPU4PySCF
references reconverge the original and moved geometries, with all moving-grid
and fitted auxiliary response. The full-Fock reference adapter is installed
after density fitting so its bound potential belongs to the fitted engine,
not the pre-fitting Direct instance. Native calls forbid Python force
orchestration and CPU scientific fallback. Compare automatic selection with a
same-capacity dense ablation and a genuinely minimum-capacity native owner.

## Invariants

- Capability and full mandatory dense capacity remain independent of optional
  maps, functional names, Fock approximation and spin labels.
- Real AO-jet and projection work must decrease, not just reported eligibility.
  Warm geometry reuses masks; moved geometry discovers a new force-order map.
- Separate nonlocal quadrature retains its independent dense traversal; sharing
  the primary grid's map with it would be a scientific identity error.
- Unsupported jets, ECP, resource/occupancy refusals and optional allocation
  failure retain dense work. Scientific/driver/stale-state errors remain errors.
- The Python explicit/compatibility profile registry remains distinct. Native
  evidence does not promote its unqualified composite profiles.

## Evidence and limits

`tests/python/test_native_force_ao_capacity.py` executes the real shape authority
without CUDA: both spins, both supported jet orders, nonlocal compositions,
crossover boundaries, bounded storage refusal, unsupported jets and overflow.

`tests/python/test_native_cross_functional_force_ao_cuda.py` retains complete
native cold/warm/moved/moved-warm E/F, residuals, SCF/Fock counts and executed AO
work in JUnit properties. It covers semilocal, global-hybrid, meta-GGA and
range/nonlocal compositions, fitted and Direct controls, and minimum-capacity
fallback. The tests use finite Slurm allocations and source-matched native
builds. A small-fixture or first-call timing is not a fresh-process, multi-device
performance promotion; the production crossover is unchanged by this work.

The minimum-capacity test must account for automatic force-tile retirement.
An allowance one byte below the original automatic-tile minimum may validly
select the smaller bounded tile. First reject an absolutely insufficient
allowance, query the retired tile's capacity, then test that exact minimum
boundary. The calculator is configuration/cache ownership, not a closeable
context manager; prepared batches carry and release the native owner.

Large cross-functional fixtures also require a genuinely matched fitted
Hamiltonian. For cc-pVDZ-JKFIT on the 48-atom cluster the normalized LR metric
has 1856 dimensions but retains 869 modes under the native relative `1e-10`
cutoff. GPU4PySCF uses Cholesky or an absolute-cutoff fallback, not that model;
its dense WB97M-V baseline failed the energy gate before any AO selection.
Do not loosen the energy tolerance or tighten the native metric cutoff to
mask this mismatch. The final WB97M-V/DF fixture uses the retained orbital
basis as auxiliary and independently verifies full rank of both radial
metrics at both geometries, so the independent analytic-force oracle has
the same Hamiltonian. Ordinary DF cases retain the practical JKFIT auxiliary.
Rank-truncated LR response remains covered by its independent existing tests.

The reduced full-rank auxiliary is not a universally good SCF fixture. A large
water-plus-H doublet did not converge independently after 200 iterations even
with a 20-vector DIIS diagnostic (orbital-gradient norm about 0.339, RMS about
0.00152). A separated H2-plus-H doublet also failed. Do not loosen convergence
or share native densities to turn these into an oracle. Direct WB97M-V also
failed independent convergence for water-plus-H and water-plus-OH doublets.
The WB97M-V UKS fixture for both providers instead uses 61 separated fully
polarized H atoms (305 AOs,
multiplicity 62), crossing the same production work guard and also exercising
an empty beta channel. Other UKS cases retain water/H with both occupied spins
and oxygen d shells. The passing fixture and rejected diagnostics define the
evidence boundary; neither changes production admission.
The original mixed UKS JUnit files retain all five passing cases and the failed
Direct water/H reference in the diagnosis archive. The accepted five-case
receipt split is explicit and hash-bound; it does not relabel the mixed campaign
as a success or accept a failed/skipped testcase. Separate WB97M-V acceptance
does not qualify those rejected low-spin SCFs.

Five fresh-process pairs on each of RTX5090/PRO6000 retain 480 complete E/F
timings across r2SCAN/DF and WB97M-V/DF/Direct. Paired warm medians improve by
about 25/26%, 5/6% and 8/8%, respectively, on the tested RKS workloads. Some
RTX5090 cold pairs regress (up to about 2.42% for r2SCAN/DF). Retain those
negative observations; do not confuse fresh-owner cold with a purged compiler
or JIT artifact cache. The result supports the existing guard, not a universal
cold/device or release-packaging promotion. Full arrays, all endpoint samples,
semantic work and source/build identities belong to the benchmark receipt.

Final replay covers 28 cases / 336 complete native E/F calls across the seven
method/provider domains, both spins and both device models. Maximum independent
energy and force-component errors are `1.3583303370978683e-9` Eh and
`3.5500491435414006e-10` Eh/bohr; maximum native physical residual is
`3.0410681495460085e-12`. All 480 separate fresh-process endpoint calls pass
independent parity, actual sparse work and matched SCF/Fock-branch gates.
Host/resource/evidence/inventory checks pass 557 tests and six subtests; one
pre-existing test assuming an unbuilt PBE binding is explicitly excluded for
this intentionally PBE-enabled build. Sphinx warnings-as-errors also passes.

The existing explicit-consumer matrix and map lifecycle tests retain empty and
noncontiguous selection, stale-state rejection, zero-cache fallback and
independent numerical coverage; they are not replaced by source-shape checks.

## Rejected alternatives

- Do not implement another per-functional map cache or declare native selection
  from the Python selector's result.
- Do not force admission by lowering the production work/occupancy guards for
  native acceptance. Use workloads crossing the actual existing guard.
- Do not obtain a dense timing by switching to Python forces or disabling the
  geometry cache; either would confound the selected/dense comparison.
- Do not call a retained FP64 map selected when the force used FP32 grid work.

## Revisit when

Change a crossover only with complete source/device-matched endpoint evidence,
including cold/moved costs and actual selected/dense work. A larger AO jet domain
needs its own derivative-order numerical gates. Independent nonlocal sparse
selection requires a separate, lifetime-bound map and bounded capacity rather
than borrowing the primary quadrature's inventory.

## References

- #1853; prior explicit matrix #2881 and fitted map reuse #2905.
- `docs/developer/preao_force_domain.md` owns current execution contracts.
