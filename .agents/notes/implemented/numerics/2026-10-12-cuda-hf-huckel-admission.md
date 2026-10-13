# Decision: admit native Hückel to the CUDA RHF execution owners

Status: implemented
Date: 2026-10-12

The subsequent default/DF policy is recorded in
[the superseding note](2026-10-12-default-huckel-df.md). This note preserves the
earlier admission decision and its exact-HF measured evidence.

## Problem

Native Modified Hückel is independent of the target functional, but its public
admission domain currently follows the existing MINAO domain: exact restricted
CPU HF/KS and CUDA KS, all-electron H–Ar, FP64. Omitting CUDA HF is an incomplete
execution-owner integration, not evidence that Hückel cannot seed an HF target.
The user permits a default when stable complete-endpoint benefit is established;
the existing independent numerical and strictly >5% endpoint gates still apply.

## Decision

Explicit `InitialGuessSpec("huckel")` now admits exact, all-electron H–Ar, FP64,
closed-shell CUDA RHF energy and analytic-force endpoints. UHF, DF, ECP and
second-order domains remain independently unqualified. CUDA HF MINAO remains
unsupported; automatic CUDA HF selection remains Hcore.

Resident/bucketed CUDA HF borrows its existing target solver and idle DIIS
history buffers before ordinary iteration begins. Preparation downloads the
actual public-AO target overlap, constructs atomic RHF/minimal-subspace frames
using the explicitly native CPU production provider, and performs strict final
same-AO admission with the charged GPU solver. Successful frames require solver
info, finiteness, eigen-residual, orthogonality and metric occupation/trace gates;
ascending eigenpair presentation moves coefficient columns with eigenvalues.
No second molecular Fock owner, GPU arena or hidden reference retry is allocated.
Only an admitted cold density is copied into the target; it never passes through
warm-density normalization. Failed preparation leaves core untouched. A failed
publication deletes the partially written arena before propagating the error.

The independent CUDA Fock route already owns host SCF algebra and GPU J/K. Its
explicit Hückel policy uses the shared native scalar HF eigen owner consistently
for preparation and target iteration. This is not CPU reference admission for a
resident GPU target, nor a provider change after failed numerical validation.

## Retry, resources and force ownership

Warm/imported inputs take precedence and perform no new atomic preparation.
A failed seeded target has one ordinary Hcore retry. Fleet warm failures also
reset the policy before their single core retry, rather than composing multiple
preparation/target attempts. Original tolerances and iteration limits survive.
Allocation errors propagate. Release the original bucket and host candidate
before constructing the retry arena; never rerun successful neighbors.

Resident HF's scalar native Fock count was previously zero even with a valid
device precision census. Policy diagnostics now consume mixed, strict and
post-SCF counts from that existing census, including discarded and final core
attempts. The public legacy CUDA HF scalar count and residual queries remain
unqualified: callers use `result.precision` for target work, and qualification
independently rebuilds F(D) at the exported target density. Absent public fields
must not be interpreted as zero work or zero residual.

A Hückel density carries no matching full orbital frame. It therefore cannot
inherit the core coefficient buffer's force-ready seed provenance, even though
strict density admission succeeded. Used Hückel seeds retain ordinary force
canonicalization; its additional physical Fock/eigen work stays in the endpoint
and precision census. Assuming that every cold target necessarily takes enough
iterations to overwrite the mismatched seed frame would be an unsafe invariant.

## Numerical evidence

Frozen base `ab5282f74c0acf98c18ec05b343a5494de85cf49`, plus native Hückel,
eigenpair ordering and this CUDA HF integration, is qualified separately from
the earlier PBE0 campaign. Library SHA-256:
`d598040711420953ece33ecabeff61751c5c2651911290560d54a59959981886`.
Slurm job 7861 on node1 passed the native preliminary contract suite, 111
native/CPU Python tests and all 17 real-GPU CUDA HF cases. Host control-flow and
PBE0 benchmark regressions separately pass 127 tests with 17 GPU cases skipped.
The final host suite adds seven evidence-reader regressions: 134 pass and the
same 17 unallocated GPU cases skip.

The new GPU cases gate independent CPU PySCF energies, analytic forces, final
densities and physical F(D) commutators for STO-3G/def2-SVP and Cartesian/spherical
representations, both batch and standalone endpoints. They share exact raw
primitive coefficients rather than comparing rounded packaged tables, and
convert Cartesian component normalization explicitly without relaxing gates.
Other cases cover mixed imported-warm/cold neighbors, preparation caps/atomic
failure, exactly one target retry, zero hydrogen atomic Focks and bounded device
allocation. Checkpoint import's separate host staging is charged above the target
host plan; the warm-failure retry test keeps the device budget exact.

Complete-endpoint performance evidence is a separate fresh-process campaign;
numerical tests alone do not qualify automatic promotion.

## Endpoint evidence and default decision

Slurm 7869 collects three rotating fresh-process core/Hückel samples for each
48/96-atom exact spherical def2-SVP HF energy/force endpoint. Slurm 7873 adds a
3-atom scope control on the same frozen source/library. Assigned physical-GPU
UUID and process checks exclude concurrent CUDA consumers without changing
Slurm visibility or cancelling other jobs. All 18 native endpoints pass
independent GPU4PySCF energy/force/F(D) residual and paired-density gates.

| Atoms | Core median (s) | Hückel median (s) | Complete wall reduction |
| ---: | ---: | ---: | ---: |
| 3 | 1.427763 | 1.376158 | 3.61% |
| 48 | 11.586258 | 10.781192 | 6.95% |
| 96 | 40.352622 | 36.235778 | 10.20% |

All 48/96-atom pairs exceed 5%. Small-system pairs are only 2.31%, 3.81% and
0.006%, so broad automatic CUDA HF promotion fails the existing profitability
gate despite favorable larger cases. Keep explicit admission and the current
Hcore automatic default; do not overfit an atom-count whitelist. These HF
results do not replace the earlier PBE0 MINAO/Hückel comparison or qualify CPU
and other functional/backend defaults.

Preparation/admission medians are 0.002043 / 0.176473 / 1.260207 seconds. Each
Hückel sample has seven atomic Focks, and no hydrogen J/K. Complete target Fock
counts are 18 / 22 / 22, including one post-SCF force-canonicalization build,
versus core 19 / 24 / 25–26. All seeds are used without retry. The source,
records, density bytes, semantic work ledger and numerical maxima are retained
in the linked evidence directory. The large-job postprocessor's empty-output
reader failure is fixed without altering any measurement; the original measured
harness and unchanged measurement-function audit are retained.

## Rejected alternatives

A registry-only switch would bypass missing consumer contracts. Reusing PBE0
timings would not measure HF work or force ownership. A molecule-size whitelist
from the two water fixtures would overfit the screen. Hidden CPU admission or
unaccounted cached atomic work would change the production/performance boundary.

## Revisit when

Repeated complete-endpoint measurements exceed the existing strictly >5% gate
with stable numerical/work gates in a defensible admission domain, or a new
spin/approximation consumer has its own independent contract. A qualified seed
orbital-frame transport could remove force canonicalization, but density
admission alone cannot prove that frame's provenance.

## References

- [Native provider decision](2026-10-12-native-modified-huckel.md)
- [Current user contract](../../../../docs/user/initial_guesses.md)
- [Frozen qualification evidence](../../../../benchmarks/results/cuda-hf-huckel-20261012/)
