# Decision: consume the existing factorized ERI jets on CPU

Status: implemented (bounded consumer; complete Ni qualification pending)
Date: 2026-11-04

## Problem

The physical one-center translation elision removes exact zeros, but the next
NiH4(2-) probe still returned no HVP within 1,200 seconds. Its retained cache
contained 275 first-component C++ units and 273 shared objects, and had not
reached a second-integral cache. This trial also had an unconverged independent
reference, so it is a failed runtime diagnostic, not a scientific acceptance
receipt. The earlier 1,800-second failure is retained separately.

The compiler already has a factorized Hermite/Boys second-order emitter. It
differentiates the canonical pair, prefactor and Coulomb factor graphs before
native compilation and returns value, independent-slot gradient and gradient
JVP. Existing independent libcint tests include f/f/f/f, and distinguish full,
long and short radial domains. Re-emitting thousands of expanded component
programs is unnecessary for this CPU consumer.

## Decision

Add one host execution template for the existing factor jets. Runtime angular
and component selections reuse one content-verified artifact per cache/compiler
identity rather than creating a program for each component tile. Admit only
unit full-range s/p/d/f ERIs here; do not infer arbitrary SR/LR omega support
from the factor emitter's fixed-omega diagnostic interface.

The compiler adapter streams at most 128 primitive records and 64 components.
The native consumer publishes either raw value/first-derivative rows or a
fixed-weight HVP, transactionally with semantic primitive/component counters.
It owns no recurrence, density/spin factors, SCF, molecular rank-four tensor,
production oracle or finite differences. Public Cartesian normalization remains
in the existing weight pullback, exactly once. Physical atom scattering stays
in the shared stationary integral provider, not this independent-slot ABI.

Use it for CPU full-range first-integral sources and weighted second-integral
HVPs. Keep S/T/V, range-separated lowering and CUDA unchanged. For second order,
reserve the shell weights, normalization buffers and publication accumulator
before lending numeric capacity to the host consumer. If its fixed lease does
not fit, retain the original bounded component lowering; compilation, integrity
or arithmetic failures are not reasons to silently select another method.

## Resources and identity

Record pages, Python/native result tiles, selections, weights and directions
are conservatively charged by `required_bytes`. Generated scalar/jet stack
temporaries, compiler subprocess memory, shared-library mappings, caller
primitive metadata and Python object overhead are explicit exclusions. This
numeric lease is not a complete-endpoint RSS bound or a work/time bound.

The source and native headers participate in the existing runtime artifact
identity. Binary identity is checked before loading. Cache-miss compilation
uses the existing finite C++ adapter and verified compiler launcher; an artifact
hit does not invoke a compiler. Reuse the established compiler cache and retain
real launcher evidence rather than mistaking shared-server aggregate statistics
for hits from foreground JIT servers.

## Validation and remaining gates

The new independent contraction test varies exponents and signed coefficients
through multiple primitive pages, compares raw first derivatives and weighted
HVPs with libcint and multistep gradient differences through f/f/f/f, and tests
late invalid records, finite-input overflow, unchanged failure publication,
capacity and closed-owner guards. It is not Ni molecular qualification.

An independent Ni state protocol now disables incremental reference Fock
updates without using native density as an initial guess. With native energy/
density thresholds 1e-13/1e-11 and reference energy/orbital-gradient thresholds
1e-12/1e-10, one explicit-grid trial converged and matched density to 1.958e-10,
energy to 4.548e-12 Eh and overlap to 8.882e-16. The strict-incremental and Newton
failures are retained; this successful state alone does not qualify derivatives
or the default-grid/installed endpoint.

Further molecular work may still dominate, particularly ordered shell-quartet
repetition and repeated first-source passes. Measure the complete HVP/Hessian,
actual semantic work and memory before any performance claim or promotion.
Keep #2843 open until independent Ni raw-matrix/HVP, covariance, lifecycle and
installed/public acceptance gates are met. The separate through-f public CPU
KS force owner remains unresolved.

The expanded source-checkout regression passed 127 cases (eight JUnit
record-property warnings) in 2150.44 seconds. A separately staged compiler-only
wheel-layout test passed with the public runtime, PySCF and checkout tools
unavailable to its isolated child. This verifies asset packaging, not an
installed public Calculator endpoint. The retained receipts are
`/tmp/gqc-2843-2849/factorized-full-qualification.{log,xml}` and
`factorized-staged.{log,xml}` in the same local evidence directory.

A profiled 600-second Ni trial using the converged independent state still
returned no HVP (timeout status 124). Its retained profile shows repeated
ordered-quartet IR/signature construction and first-component evaluation;
removing the compilation barrier alone did not complete the molecular source.
`ni-hvp-factorized-profile.log` and `ni-factorized-profile.{prof,txt}` preserve
that failure. No compiler subprocess remained after the timeout. Do not use
the different timeout or profiled execution to claim an endpoint speedup.

## References

- Issues #2224 and #2843.
- `python/generativeqc_compiler/integral/factorized_second.py`.
- `python/generativeqc_compiler/integral/factorized_host.py`.
- `tests/python/test_factorized_host.py`.
- `2026-11-04-physical-one-center-derivative-elision.md` under performance notes.
