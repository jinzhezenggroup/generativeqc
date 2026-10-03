# Proposal: observe actual generated J/K admissions at the shared provider

Status: proposed; intrusive profiling seam, no screening/default promotion
Date: 2026-10-04

## Problem

The initial shared incremental-KS prototype exported full/delta build counts,
but no provider work. Its completed 48/96-atom campaigns pass the independent
E/F gates yet regress in warm replay. Initial warm OFF/ON medians are
17.860668/22.421234 s and 77.213533/123.139572 s. Two 48-atom ON warm calls retry
cold and take 143.253/148.156 s; returned-attempt diagnostics omit discarded work.
The later audited-full-RKS reuse source fe6aea98 restores one-build 48-atom warm
replay (17.854/17.851 s OFF/ON), not a win over the ordinary OFF endpoint.
All these campaigns are ordered/shared-cache and remain source-scoped.

Code inspection reveals another specific limitation: the generated pure-J
consumer uses geometry-only screening in `integral/production_emission.py`.
Its `stream_survives` Coulomb arm returns before reading density bounds; its
coarse system bound is 1.0. Native DDDD mirrors this distinction. Smaller delta
density cannot improve that particular admission gate. It does not follow that
all downstream primitive work is constant; actual observations are required.

## Decision under qualification

Expose the existing generated FP64 task-entry counters through optional borrowed
class-major arrays on the generated J and K owners. Keep channels separate and
normal execution null. The counters observe actual admitted streaming shell-task
dispatches after screening, including native DDDD, not capacity, rejected
candidates, AO quartets, primitive recurrences or FLOPs. Bounded higher-l fallback
classes are unobserved. A complete probe must explicitly require generated/native
streaming coverage rather than report missing classes as zero.

The observer owns zeroing, storage accounting and stream-ordered lifetime. No
production numeric allocation or public selector is added. Add a read-only
drained-iteration incremental diagnostic to CudaKsPlan so a native observer can
associate measured admissions with actual full/delta/final provider applications,
not a predicted refresh pattern. It refuses pre-begin and pending reads and does
not confer final-state eligibility. Public quartet-work completeness remains
false: this partial census still does not observe candidates/rejections.

Intrusive atomics, readbacks and extra synchronization invalidate clean timing.
Complete E/F campaigns stay separate. The new work probe solves fresh SCF from a
29,997-byte geometry/basis-only input, verified against the 96-atom README
geometry. It does not consume the old force census's final density or claim to
replay its historical work. This small new input can be retained independently
of the older unpublished 13.76 MB exact force-census fixture.

## Qualification and retained failed approaches

The separate integration tree incorporates master79418329e without mutating the
older measurement trees. Runtime source
`61c68d2c86b294949c1ae29694fef4ece3233a0a1f5fd0b33720bb1ab14d6264`, library
`f798540f30e109b76aa8b5c3c11905afe2e25e62e32cabd663978826e2a8cdca`:
39 focused host tests pass, 34 GPU-gated tests skip on the host. Job5673 passes
the complete native CUDA-KS suite, all seven public shape/RKS/UKS independent
E/F/warm/FD controls and isolated linear-kernel sanitizers. Its combined-HF
fixture fails before ordinary HF controls because the first observer test calls
the host direct API, which uses a separate generic path and never hits these
streaming counters. This failure is retained, not treated as zero work.

The first fixture repair also fails (5674): the resident provider correctly
rejects managed input/status memory. The final repair keeps that admission
contract, borrows real provider device scratch and allocates a device status
slot. Host generic J/K remains the independent numerical control. Job5677 then
passes the combined incremental-HF/observer suite and whole-test memcheck and
initcheck. These fixture-only repairs do not change the runtime source/library
above. The baseline-reproduced whole-HF synccheck failure remains separately open.

The standalone work probe's first compile misses `resource_cuda.cuh`; the corrected
build includes the resource-aware allocator rather than bypassing accounting.
No GPU result is attributed to the failed compilation. Jobs5675/5676 reveal a
second harness error: the small-fixture eager CPU MolecularGrid constructor is
not the production CUDA preparation route. Both have zero SCF iteration records
when cancelled after this source diagnosis. An attempted CPU stack attachment
inside a finite overlapping Slurm step is denied; no ptrace policy is changed.
The repaired probe calls the existing MolecularGrid::from_cuda factory used by
ks_molecular_grid, retaining the same grid specification and resident lifetime.
This is not a new production optimization. Complete work counts and independent
energy checks remain pending; no J/K work reduction or large-system speedup is
claimed from this API alone.
