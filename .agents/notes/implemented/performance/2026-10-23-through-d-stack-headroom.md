# Decision: reduce the proven through-d CUDA stack request

Status: implemented in the candidate; integrated endpoint qualification ongoing
Date: 2026-10-23

## Problem

The earlier [driver-memory investigation](2026-10-21-streamed-coulomb-and-driver-memory.md)
reduced a conservative 64 KiB request to 32 KiB for proven through-d sources.
An allocation interposer on the held-factor WB97M-V DF96 source still observes
about 14.58 GB of explicit arrays and only 9.99 GB free on a 33.67 GB device.
The difference limits optional range reuse. It is driver capacity, not another
hidden raw DF tensor or XC allocation.

## Decision and invariants

Request 16 KiB for the same proven through-d domains in both prepared KS/Direct
and ordinary HF. Keep the schedule's larger ceiling for f and unknown metadata.
Use the existing serialized read/raise transaction: a smaller request never
lowers another live owner's context-wide limit. Preserve allocation-failure
fallback, all compiled recurrences, FP64 arithmetic, finite execution limits,
grids, screening and physical/final-state acceptance gates.

The CUDA limit is a call-stack resource request. PTXAS local frame sizes are
retained as diagnostics, not equated with that limit or used to infer safety.
Numerical and sanitizer qualification must execute the actual value/force
consumers. A fresh context is necessary to measure the capacity reduction;
an already larger context limit remains larger by design.

## Evidence and scope

A fresh-process diagnostic interposer on the frozen stage-three precursor
reduced only the requested resource control. The optional exact range cache
then admitted 13,840,319,872 bytes, versus about 9.48 GB with the original
capacity-selected buckets. The same bounded selector retained its exact fallback
for omitted buckets; no hand-picked molecular mask was used.

WB97M-V DF96 completed cold/warm in 333.032802/45.076419 seconds with 28/1
iterations. Independent dense-reference errors were 3.0088e-6 Eh/atom,
1.9949e-5 maximum and 9.3917e-6 RMS Eh/Bohr. This traced single-process
observation is a diagnostic, not a repeated default performance claim.
The prior resource request, allocation/launch traces, failed interposer attempts,
compiled resource dump and original source/library identities remain in the
stage-three artifact directory.

Separate interposed through-d Direct streaming/replay tests and f fallback
tests pass, as does restricted/unrestricted fitted WB composition against CPU.
The final compiled candidate must retain its own source-matched clean endpoint
and sanitizer evidence. F/unknown behavior remains unchanged; no broader small
stack policy follows from this result.

## Revisit when

New recurrence owners, compilers, devices or larger angular domains require
renewed execution qualification. Never lower a context-wide limit to reclaim
memory while unrelated owners may still depend on it. If a new path needs more
stack, raise its request explicitly and retain the optional range-cache fallback.
