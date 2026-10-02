# Decision: reuse CUDA AO radial factors across four and ten jets

Status: implemented (explicit qualification opt-in; not a production default)
Date: 2026-10-02

## Problem

The compiler's CUDA AO producer flattened `(jet, point, AO)`. PBE-family energy
collocation requests four jets and the spatial GridPlan stationary-force
consumer requests ten,
so each primitive exponential, displacement and squared radius was repeated
four or ten times. Native KS and the spatial/selected-AO runtime use this same
compiler-owned arithmetic. CPU AO reuse does not remove this CUDA work.
The separate native XC geometry producer `force_ao_kernel` in
`xc/geometry_cuda.py` is outside this change and retains its scalar traversal.

## Decision

Emit four explicitly named, fixed-extent kernels: four/ten jets in FP64/FP32
compute. Each thread owns one `(point, AO)` and scalar named accumulators for
exactly the requested extent. Move only the displacement, squared radius and
primitive radial factor outside the emitted jet updates. The one-jet and
20-jet/order-three paths retain the original scalar traversal; no universal
runtime-indexed 20-element accumulator is introduced.

A compiler-owned `scheduled_ao` chooses the kernel and launch domain. Radial
reuse is disabled by default. Explicit `ao_radial_reuse=True` in
`emit_grid_source` or the JIT `compile_cuda` helper enables it;
`tools/generate_grid_kernels.py --ao-radial-reuse` emits the same native opt-in
composition. Existing CMake/native and JIT callers retain scalar dispatch. Both
native XC launch sites (including nonlocal-potential assembly) and the spatial
AO runtime call it, retaining their stream, buffers, selected map and existing
error-publication ownership. Empty work returns without launching. Opted-in
native PBE compiled-resource classification matches `ao_radial_kernel_4` exactly;
LDA and all scalar-default builds match `ao_kernel` exactly. Other precision/extent
kernels cannot substitute
for a missing active resource record. The selector is recorded in the compiled
resource shape and emitted source identity. Resource validation requires the
same explicit `--ao-radial-reuse` flag and exact generated source, preventing
scalar evidence from qualifying an opted-in artifact or vice versa. The shared
production schedule assessor still requires the scalar-default resource shape
and rejects opted-in evidence; the latter is experimental qualification data,
not admission to the production profitability path.

## Invariants

- Each jet's primitive and Cartesian-expansion accumulation order is unchanged
- Each contribution remains `radial * coefficient * axis_x * axis_y * axis_z`
- Primitive underflow skips axis evaluation before overflow can occur
- Each output passes through the existing finite check
- Jet-major output layout and selected-AO labels remain unchanged
- FP32 subtracts double coordinates before narrowing local displacements
- Both precisions use the same existing scalar DAG and retain `axis_jet`'s
  noinline boundary; normalization and spherical coefficients remain native

## Evidence

`tests/python/test_cuda_ao_radial_reuse.py` host-executes the actual emitted
arithmetic and launcher with instrumented transport/libm stand-ins. It compares
against the retained scalar CUDA body and an independent long-double Leibniz
oracle through f and order three, including signed three-term spherical
expansions, selected/reordered columns, canaries, grid-stride/tail work, empty
points/AOs, translated coordinates, underflow and nonfinite publication.

Counted exponential calls establish that four/ten-jet schedules perform one
exponential per selected point/AO/primitive rather than four/ten. Axis-call and
output counts are unchanged. One/twenty-jet work remains unchanged. This proves
source work removal, not CUDA wall-time improvement.

Focused resource tests reject inactive AO substitutions. The optional nvcc test
compiles the emitted kernels and launch helper without opening a device and
requires a distinct PTXAS row for all six kernels. The allocated CUDA libcint
fixture test now exercises the production dispatcher in both precisions and
all 1/4/10/20 extents, including diffuse/tight and Cartesian/spherical f shells.

This environment has no nvcc or allocated GPU. CUDA compilation, PTXAS pressure,
device numerical parity, and complete endpoint timings are not measured here.
The optional compile test and 48 allocated CUDA cases therefore skip locally.

## Rejected alternatives and consequences

- A runtime 20-value accumulator would retain unnecessary values in PBE energy
  and force producers and risk local-memory/register pressure
- Axis reuse is deferred: it introduces additional live values and could alter
  multiplication association if folded into weighted coefficients
- Inlining the whole axis dispatch would undo its existing compiler-pressure
  boundary
- Fusing all 20 jets is deferred until the less-common order-three consumer has
  separate compiled-resource and complete-endpoint evidence

Opted-in 4/10 schedules reduce independent thread count and increase per-thread live
accumulators. Named scalar accumulators avoid dynamic array indexing but do not
prove spill-free compilation or a speedup. No scientific approximation,
precision-policy change, production-default promotion, or speedup claim is made.

## Revisit when

Before default promotion, collect PTXAS registers, spills and stack/local memory
on the target compiler,
then run allocated-device numerical tests and unchanged cold/warm/changed-
geometry PBE0 energy-plus-force endpoints at 24/48/96 atoms. Compare actual work
counts and fixed scientific settings. Revisit axis reuse, extent policy or the
retained scalar schedule only with that evidence, rather than microkernel time
alone.
