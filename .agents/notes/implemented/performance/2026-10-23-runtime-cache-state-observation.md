# Decision: observe fresh-process cache states without warming scientific plans

Status: implemented; sampled device campaign complete
Date: 2026-10-23

## Problem

Object-cold calls with persistent caches do not measure first-use compilation.
Repeated CBS calls in one Python process also retain imports and process-local
state. Neither is a substitute for the requested empty/primed fresh processes.
The runtime's artifact metadata retains the artifact producer's compilation
duration, which must not be mistaken for compilation in a later replay.

## Decision

The controller launches a separate Python interpreter for every complete E+F
endpoint. Empty replicates create distinct private roots; primed replicates
reuse only their case's completed, byte-verified root. Cache corruption stops
the experiment instead of triggering cleanup or silent re-priming.

The observer forwards original compiler/cache functions and their imported
aliases, preserving results and exceptions. It records actual adapter calls,
subprocess commands/results, artifact metadata and private live-server stats.
It does not change production implementation files or numerical operators.
An optional AOT loader hook is installed only when its owner is already
imported; importing scientific plans solely for observation could warm work
before the constructor timer. Missing AOT/native NVRTC coverage is explicit.

The process timer includes startup, verification, observation and export.
The E+F timer starts immediately before calculator construction and ends after
host forces are available. Nested compiler, server-status and statistics spans
remain separate and are never added to endpoint wall time. Observer overhead
is retained, rather than estimated and subtracted.

NVCC `-M` dependency identity is a probe, not a compiled runtime module.
sccache's CUDA/PTX/CUBIN counters are preserved as their original hierarchy;
they are not counts of internal NVCC passes. Artifact metadata compilation
durations remain historical provenance, separate from current driver spans.

## Rejected alternatives

- A same-object warm replay or same-process CBS loop would change the requested
  population and hide process-local first-use work.
- A global Python profiler would instrument unrelated lowering calls as well
  as compiler calls. Targeted forwarding limits the observation surface.
- Pre-importing AOT/scientific plan owners for more counters would change the
  first-use work being measured. Unavailable counters are preferable.
- Clearing a shared cache would affect other work and would not establish a
  reproducible private empty-state contract.

## Validation and boundaries

Host tests exercise no-clobber roots, priming integrity, fresh child failures,
exception/result forwarding, incomplete stats, shared-path rejection and the
original independent water/CBS gates. Real CPU-allocation probes retain GCC
compile/link plus artifact replay, and NVCC dependency/object/link calls with
actual scoped cache misses/hits. NVCC's sm120 code object is not executed by
that probe and does not establish device or E+F acceptance.

The independent CBS projection always uses the original receipt direction.
Regeneration only checks normalized seed-709 identity within a componentwise
norm-reduction roundoff bound; source, geometry, scientific protocol, receipt
hashes and original observable tolerances retain exact checks. This does not
provide independent full-vector CBS force qualification.

The separate pilot and all thirty formal processes passed on the same frozen
production SDK and physical GPU, including per-sample independent numerical
and cache checks. Raw recovery and independent CPU replay preserve their own
provenance. These sampled results do not qualify a whole method domain or
promote a default; host preparation alone remains insufficient.

## References

- Issue #2680
- `benchmarks/runtime_cache_state.py`
- `benchmarks/results/runtime-cache-state-20261023/README.md`
- `benchmarks/readme_defaults_qualification.py`
- `tools/render_readme_cbs.py`
