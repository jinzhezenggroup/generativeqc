# Decision: keep CPU Hessian recycle groups on the native accepted state

Status: implemented
Date: 2026-11-22

## Decision

The [linked bare CPU LDA/PBE Hessian](2026-11-22-native-cpu-semilocal-hessian.md)
now accepts explicit dense recycle groups of two to four RHS. The C/C++ SDK,
CLI and Python call the same stationary owner. This is sequential recycled
GMRES, not block GMRES: the shared native recycle space projects an initial
guess from previously converged solves, and the existing solver and physical
CPKS residual test determine convergence. The space resets between groups
and never survives a new geometry, SCF state or endpoint.

The caller's requested width is admitted before optional retained integral
and grid panels. A width that does not fit is an error, not permission to
switch schedules. The numeric bound includes prior solutions in the group,
one full GMRES solve, the old recycle space, and projection/replacement
scratch. The projected guess is released before the replacement; private
Arnoldi storage is not retained by completed solutions. Only the final raw
Hessian and detached work counters publish. Default independent columns,
HVP, explicit true block GMRES and unlinked widths/packages are unchanged.

## Evidence

The bounded GMRES group reproduces an analytic diagonal system, reuses a
measured repeated-RHS guess, and executes at exact but not one-byte-below
capacity. SDK tests repeat that boundary on actual LDA/PBE, verifying no
partial matrix publication. The SDK and CLI run with no interpreter/compiler
on `PATH`. An independent PySCF reconverged-gradient reference checks all
three requested group widths on the same distorted water fixture. This tests
both a singleton and a partial final group as well as complete groups.
The largest raw difference from reconverged gradients at `3e-4` Bohr is
`1.420e-7` Hartree/Bohr² (PBE). Eight selected CTests and 80 Python tests
with 25 subtests pass on the CPU build.

On an AMD EPYC 7K62 with one BLAS/OpenMP thread, three fresh-process samples
for each method at initial and moved geometries compared the old packaged
recycled endpoint (warm generated-artifact cache) to the new complete native
endpoint. Median LDA times were `95.71–95.84 s` versus `1.480–1.484 s`; PBE
times were `100.61–100.76 s` versus `1.941–1.944 s`. Every case ran 10 SCF
iterations, 12 Fock builds and nine directions. Native execution reported 80
Krylov iterations, 184 induced Fock actions and `23,004/32,886` first/curvature
ERI visits. The old route reported 189 response operator actions. Raw old/new
matrix agreement was `1.211e-12` Hartree/Bohr² or better. Five interleaved
scalar control pairs per method/geometry retained identical outputs and work
and met the 5% median non-regression gate. These are fixture-specific results;
the immutable comparators, raw timings and checker are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/recycled-benchmark`.

## Limits

This unmerged CPU slice on `cae2959e1` advances #2227 R2/R3/R5/R6 only for
linked bare all-electron Cartesian Direct FP64 LDA/PBE RKS. It does not
qualify CUDA, UKS, other methods, true block GMRES, broader resource authority
or other workflows.
