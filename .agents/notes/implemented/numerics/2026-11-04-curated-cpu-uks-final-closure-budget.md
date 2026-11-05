# Decision: retain the CPU UKS budget after a rejected physical closure

Status: implemented
Date: 2026-11-04

## Problem

PR #2991's original IH(-) force endpoint failed in required core-c CI before
its independent displaced-energy oracle. The exact source at
`75d6878487ff165035cf72a3e65a0dcf2cd2d67c` reproduces this on GNU 14.2,
Release CPU/OpenBLAS 0.3.34, with the declared spherical def2-SVP/ECP28,
Gaussian/VWN-RPA B3LYP, charge -1, multiplicity 2 and 16x6x12 atomic grid.
The unmodified library SHA-256 was
`41e130aeed8773ada0349d68dfca94047a6652e214905d97befdd212fa767758`.

The primary loop satisfied its gates at iteration 22. Four final physical
corrections then left energy change `5.68434e-13` above the unchanged `1e-13`
gate, despite density change `1.13990e-14` and residual `3.98547e-15`.
It returned failure after 27 physical Fock builds with 218 of the requested
240 primary iterations unused. A small residual does not make that failed
energy gate acceptable. The generic CPU point-program owner already supports
a bounded restart after this rejection; the curated owner stopped early.

## Decision

Allow every ordinary CPU UKS owner to resume from the rejected physical density
with cleared DIIS and only the remaining original SCF budget. Reuse the existing
controller, global history numbering, unavailable restart-baseline marker and
physical work accounting. Retain the four-correction limit per closure and all
energy, density, RMS/maximum residual and final-projector checks. CUDA/COSX
retains its existing behavior. This is not a new preliminary model, a warm-state
fallback, an oracle bridge or permission to consume an unconverged result.

The existing independent molecular force/energy test remains authoritative.
Its successful endpoint now also checks the returned budget/history and final
physical tolerances. Short-budget cold/repeated executions must still fail and
must not publish a warm seed. Generic automatic-program budget controls remain.

## Evidence and boundaries

The same cold IH(-) energy call now passes in 26 primary iterations and 34
physical Fock builds: energy change zero, density change `2.14790e-14` and
residual `4.15228e-15`. Every returned gate is unchanged. Seven exhausted-budget
and CPU eligibility controls pass. These observed counts describe this build,
not portable exact-iteration requirements or performance acceptance criteria.

The original failing full public IH(-) test passes with an independent force
error `5.92904e-12 Eh/Bohr`, density error `2.58816e-11`, and maximum normalized
independent reconverged-energy error `9.77374e-9 Eh/Bohr`, all within the original
gates. Existing native UKS and automatic CPU tests give 66 passes and 12 CUDA
skips. Four portable-replay cases initially lacked a compiler-cache launcher on
the local PATH; all four pass after exposing verified ccache 4.14.1. The repaired
CPU library SHA-256 is
`2541fc274add7a35c8da058f7c87e08584a6b0f580efc4887b1c1491a49a7a88`.

The earlier mixed Ni/I/H ECP quadrature failure remains unresolved and its guard
is untouched. Neither this repair nor a successful point endpoint qualifies
native displaced IH(-), looser/default CsH(-) controls, mixed/default-grid or
25–40-atom domains. No GPU execution or performance promotion follows.

This extends the curated-owner boundary of
`2026-10-12-automatic-cpu-final-projector-closure.md`; that note's original
generic-only decision remains a historical record.

## Revisit when

A shared final-state owner or compensated CPU energy convergence replaces this
bounded policy. Do not substitute looser gates, a larger hidden iteration budget,
fixture geometry changes, or exception-time reference energies.
