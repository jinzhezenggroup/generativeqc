# Decision: qualify homogeneous bounded full-range force passes

Status: proposed (opt-in prototype, not a promoted schedule)
Date: 2026-10-04

## Problem

Ordinary stationary PBE0 forces reach the shared independent J/K derivative
owner, then a mixed bounded shell queue. Its generic force contraction is a
noinline runtime angular dispatcher. Historical corrected-PBE0 class isolation
identified DPPP/DPPS as substantial costs, not DDDD alone. Existing fixed-order
workers in `direct_angular_force.cu` are not new work proposed here: their
topology-capacity queues differ from this bounded owner.

## Candidate and boundaries

`GENERATIVEQC_BOUNDED_FORCE_SCHEDULE=homogeneous` selects compiler-generated
disjoint passes: 0--3, 4, 5, 6, 7--12. The first retains scalar weighted sources;
4/5/6 invoke the qualified fixed-order contraction directly; the last retains
the high-order implementation. Compile-time guards exclude scalar drain from
warp-only workers and the generic drain from low-only workers. The existing
mixed path remains default. Full-range independent sources are the only new
consumer; this also applies to the full component of RSH decomposition, not its
separate LR traversal. It does not affect energy/SCF builds or enable incremental
J/K, DF, AO-density-product screening, or approximate integral math.

The same immutable shell domain is rescanned, with a stream-ordered, error-checked
reset of the existing cursor before each additional pass. Overflow `heads` are
not available cursor scratch. Force outputs are zeroed once and J/K share each
admitted derivative calculation. No resident quartet list is introduced.

## Work and memory

Let S denote the current screening traversal work and C_o the actual admitted
contraction work for order o. Mixed work is S + sum(C_o); the candidate is
5S + sum(C_o), plus four cursor resets and launches. Empty angular ranges are
not yet elided. This does not lower asymptotic complexity or the number of
integrals. Its hypothesis is improved execution of existing work by removing
unrelated recurrence code from fixed workers. Task counts are not FLOPs.

Algorithmic queue storage remains O(CTAs * 256); host pass metadata is constant.
Generated code size increases. Registers, local memory, and any runtime stack
reservation are separate device resource costs requiring compilation and actual
measurement; unchanged explicit allocation does not prove unchanged peak memory.
The high-order generic pass still calls the all-order retained dispatcher.

## Acceptance and rejection criteria

Host gates enumerate every through-f quartet exactly once, compile generated
membership, and inject failures into each cursor-reset/launch/download boundary.
They do not validate CUDA numerical results. Device gates must compare independent
J/K components against the original and an independent oracle for restricted and
unrestricted densities, pure J/K, spherical transforms, nonzero screening, and
indexed/non-indexed traversal; include high-order and coincident-center cases.
Use corrected PBE0 weights J=1 and K=-0.125, not the old doubled-K probe.

Then compare complete cold/warm/moved endpoints on the same frozen source,
binary and GPU, preserving all-repeat 1e-8 Eh / 1e-7 Eh/Bohr gates. Profile
executed class counts and kernel resources separately from endpoint timing.
Reject this schedule if repeated screening, code/stack costs or cold regressions
outweigh the fixed-worker gain. Do not promote based on static register counts,
microbenchmarks, or missing telemetry. No GPU execution on n3.

## Evidence scope

This prototype starts from master `97e55dfc9`; the already-running integrated
AO/paging/Becke qualification is separately frozen at `c87d698d4`. Do not mutate
that live source or splice its timings with this candidate. No numerical or
speedup result for this prototype is claimed by this note.
