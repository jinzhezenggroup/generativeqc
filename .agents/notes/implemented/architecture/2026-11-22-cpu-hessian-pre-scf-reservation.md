# Decision: Size the CPU Hessian reservation before SCF

Status: implemented
Date: 2026-11-22

## Problem

The Python host-budget planner previously charged grid and response caps plus
result staging. It could admit a budget below the native scalar derivative's
reservation and spend the full SCF before discovering the shortage.

## Decision

The C/SDK query resolves the exact total MethodIR against linked electronic
closures and the native named correction catalog. It takes the normalized
orbital system and shares the mandatory reservation calculation with the
stationary CPU execution owner. Integral byte sizing is independent of the
primitive and quartet work census; the query needs the native AO count, not
the packed AO evaluator or grid. The first integral and ECP sources are
accounted for within the response cap; the scalar curvature sources are
additional. The full configured grid and response caps, not the observed
usage of a particular solution, are charged to the base reservation.

Python combines this floor with the existing SCF and correction peaks when
the caller supplies a shared host budget. The endpoint also screens explicit
derivative numerical/response caps this way. The preflight-only normalized
system is released before preparing the batch. Unbudgeted default execution
does not repeat basis expansion or run a preflight query. Whole-endpoint
integral/XC work census, final-state checks, and density-dependent admission
of optional panels and recycled groups remain with the execution owner.

## Rejected alternatives

Do not use the full integral planners before SCF and then repeat their quartet
census on execution. Do not copy the native shape formula into Python or
subtract unverified SCF releases from the shared host peak. The byte floor
is a necessary reservation, not a promise that a particular method, solver,
or explicit optional schedule will execute within all other caps.

## Evidence

Budget gates cover HVP and dense, bare LDA/PBE RKS, corrected PBE0 RKS and
corrected B3LYP UKS. Exact aggregate and explicit endpoint limits reproduce
the unbounded derivatives; one byte below either limit fails before batch
preparation. SDK tests query before preparing SCF, replay at the exact floor,
and verify that short execution limits do not modify caller buffers. This is
a CPU resource-authority increment for #2227 R2, not a new scientific domain.

Five interleaved fresh-process pairs per case on an AMD EPYC 7K62, pinned
with one BLAS/OpenMP thread, compare the immediate before/after complete
Python endpoints on distorted STO-3G water, including a geometry move.
Unbudgeted dense LDA/PBE/PBE0-D3(BJ) median changes were +0.12%, +0.03%,
and +2.08%; budgeted dense PBE and PBE0-D3(BJ) HVP changed by -0.06% and
-0.02%. Raw derivatives, total energies, and reported SCF, response, ERI and
XC work counts were identical for each pair. All cases ran ten SCF iterations
and twelve Fock builds; dense cases ran nine response directions and HVP ran
one. The matched library hashes, timings and work receipts are in
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/reservation-benchmark`.
