# Decision: reuse immutable TensorIR graph metadata

Status: implemented
Date: 2026-11-04

## Problem and evidence

After bounded stationary cotangents fixed the Ni integral-block admission
failure, the complete low-level explicit-grid HVP on eb051d0d1 passed the frozen
independent moving-gradient gate. Its assembly nevertheless took 1875.44 s,
including 350.59 s for Coulomb moving-weight/first-integral contraction. Memory
boundedness alone does not bound aggregate work or establish useful throughput.

A bounded synthetic replay of 2000 signed shell cotangents took 4.177 s under
cProfile. Repeated graph order/content hashing consumed 2.134 s cumulatively;
debug naming consumed 1.831 s and liveness 1.171 s, with overlapping attribution.
These are synthetic diagnostics, not a molecular speedup measurement. Receipts
are `/tmp/gqc-2843-2849/weight-replay-uncached-metadata.{txt,prof}`.
The same profiled replay with retained metadata takes 1.092 s with the exact
same output checksum (-40.29706614632586). This still is not an endpoint timing.
TensorIR legality, replay, interpreter, canonicalization, forward/reverse AD,
precision and independent prepared-cotangent tests pass 211 cases in 43.69 s.
Compiled CPU tensor/Boolean/where/bundle/preparation contracts pass 88 cases in
64.26 s. Shared stationary/compiler/gradient integration passes 129 cases in
51.28 s; complete global-hybrid and WB97M-V direct/full-rank-DF RKS/UKS molecular
files pass 52 cases in 344.92 s with eight JUnit warnings, using the retained
warm cache outside pytest retention. These 480 source-checkout cases include
no real-GPU execution. The compiler structure check covers 586 modules with no
dependency errors. Complete Ni HVP/raw-matrix replay is evaluated separately.

## Decision

`Program` already freezes outputs, definitions, provenance and all Node inputs
and attributes, and already retains its logical hash. Retain its deterministic
node order, live-node tuple and immutable debug-name map for the same lifetime.
The public name property still returns a detached dictionary so caller edits
cannot affect later validation, precision naming or execution. Each new Program
gets independent metadata; no numeric feed, tensor value, geometry, shell
weight, live owner or compiled artifact is reused by this change.

The interpreter still validates every action's feeds, dtypes, shapes, finite
values and explicit retained-byte budget. Serialization, equation identity,
duplicate-node occurrence naming, dead-definition inspection and execution
order remain unchanged. Compiler provenance and native cache checks are not
overridden. Tests preserve observable replay, changed-feed behavior, detached
names and output independence rather than pinning internal cache call counts.

## Limits and rejected alternatives

Do not cache evaluated tensors or whole-molecule rank-four cotangents to hide
the metadata overhead. Do not omit per-action numeric validation. This change
does not remove scalar second-source lowering or primitive execution work and
does not qualify the installed/public/default-grid or production-scale endpoint.
The old raw-matrix probe passed its first independently checked column but was
intentionally interrupted before completion to avoid repeating known metadata
work; its partial columns and failed/limited earlier attempts remain retained.
