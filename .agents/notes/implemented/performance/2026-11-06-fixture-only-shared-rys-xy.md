# Decision: fixture-only shared unweighted Rys x/y tables

Status: implemented default-off compiler fixture; synthetic CUDA gates passed; production selection rejected
Date: 2026-11-06

## Problem

Derivative-free DPPP/PPPP Rys3 component-lane Fock workers build identical
unweighted x/y tables in every retained component lane. The weighted z table
contains a component-dependent coefficient and must preserve its arithmetic.

## Decision

`CudaScheduleIR.fock_shared_rys_xy` is a strict boolean, default false, admitted
only for derivative-free DPPP/PPPP Rys3 Fock at the incumbent 192/96 threads.
Its true value enters schedule serialization and schedule symbol identity.
Production selections reject it; no manifest, registry, runtime selector, SCF
policy, or default is promoted. Explicit mixed, streaming and work-K requests
reject. UHF continues to instantiate the original private task.

One Python recurrence emitter supplies the private-axis body and shared fill.
The private helper remains noinline. A restricted task adds exactly 72/54
doubles (576/432 bytes) of CTA shared state. After geometry publication, six
fixed lanes fill one root/axis table each. Screened and padded component lanes
still participate in the unconditional publication and retirement barriers.
A uniform retained-component count of at most two selects the original private
axis calls. This is a fixed conservative boundary, not an algebraic optimum or
measured tuning result; no threshold sweep is included. The all-screened return is uniform. Primitive/root summation order,
weighted z, HRR, screening, coefficients and canonical scatter are preserved.
There is no global allocation or storage across primitive quartets.

## Rejected alternatives

Changing private helper inlining would confound resource behavior. Moving the
z weight outside recurrence would change floating-point association. Neither
change belongs to this capability. Larger shared/global caches, more classes,
and production integration await a separate causal design and authorization.

## Evidence and boundaries

Host gates execute emitted private/shared fills, HRR and production weighted-z
arithmetic against independent Libcint for every DPPP/PPPP Cartesian component,
signed unequal primitive contractions, coincident centers and reversed pairs.
Their tolerances remain atol=2e-12, rtol=2e-10. A one-off comparison against the
clean b681a1fc3 baseline confirmed byte-identical default-false DPPP/PPPP emitted
sources; this is not a permanent source fingerprint test.

`tools/generativeqc_validation/shared_rys_xy_fixture.py` prepares a single CUDA
executable source with both variants, the incumbent persistent mapping, full /
partial / one / two / zero retained masks, four primitive quartets per task,
one-CTA task reuse, ordinary launches and two graph replays. The 27-live
partial mask screens all six producer component lanes while still taking the
shared path, testing producer ownership independently of component retention. It checks every
full Fock matrix entry against independent Libcint contraction and candidate /
private parity at the existing class-fixture atol=2e-11, rtol=2e-10. It prints
register/local/shared attributes; ptxas and cuobjdump reports supply stack and
spill evidence. Preparation binds generated source and independent input hashes.
At the phase-A source freeze, CUDA compilation, execution, resource deltas and
sanitizer outcomes were pending and could not be inferred from host gates.

The subsequent source-bound H100 qualification of `b56c2c935aa63f68049a116e420aa5a5dcb529e8`
passed full independent Fock-matrix checks, ordinary/twice-replayed graphs and
memcheck/racecheck/synccheck. Registers remain 166/thread; local bytes remain
96/72 for DPPP/PPPP; static shared deltas are exactly +576/+432 bytes, with zero
persistent RHF spill loads/stores. The original first attempt failed for a
missing sanitizer path before compilation and remains failed. Its runner-only
repair changed no scientific source or input. Exact device, source, executable,
tool bindings, all reported numerical observations, resource rows and failure:
[retained phase-A evidence](../../../../benchmarks/results/h100-shared-rys-xy-phase-a-20261106/README.md).

This synthetic fixture is not an actual HF96 task replay or timing measurement.
Actual native task/density payload capture remains mandatory before class
performance claims or production integration. No new capture hook is included.
The full original30 cold E+analytic-force goal, independent E/F gates, three
same-GPU fresh-process pairs per case (each ratio >1 and median >1.05), and
warm <=5% regression remain unchanged and unfulfilled by this slice.

## Revisit when

The synthetic device gate is now passed. Separately capture bounded native HF96
task/density inputs and demonstrate actual-task savings before considering
production integration. Synthetic timings never qualify HF96.
