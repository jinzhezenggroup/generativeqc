# Decision: fuse canonical doubles denominators into their consumers

Status: implemented; qualification in progress
Date: 2026-10-05

## Problem

#1904 identifies one redundant `O(o²v²)` host/device tensor. Canonical orbital
denominators are separable, but the native molecular builder materialized and
uploaded every double gap. Response owners also retained the array despite
physical residual AD having no denominator dependence.

## Decision

Add an explicit canonical-spectrum representation alongside the original
supplied-array representation. Native canonical CUDA construction can omit d2
on both host and device, retaining occupied-then-virtual energies, level shift
and the physical denominator threshold. Singles remain explicit. No Fock
diagonal is treated as authoritative orbital provenance.

One scalar TensorIR expression owns the original pair-gap grouping for host
initialization/admission, CPU and CUDA Jacobi division, and Lambda diagonal
construction. The emitter recognizes only division by the declared Jacobi d2
input; arbitrary divisions and physical equations are unchanged. The two
generated core headers share one guarded definition. No standalone d2 kernel,
extra full reconstruction pass, or response denominator derivative is added.

## Invariants and bounds

The existing builder evaluates `(ei-ea)+(ej-eb)-2*shift`. Summing already shifted
singles changes rounding and is rejected even though it is algebraically equal.
Independent tests compare the original FP64 operation sequence bitwise and
contain samples that distinguish those two expressions.

Admission first checks every physical single gap for strict negativity,
finiteness and the unchanged absolute threshold. Round-to-nearest addition is
monotone: a sum of two negative admitted gaps cannot approach zero more closely
than either operand. The most negative single gap added to itself bounds the
negative endpoint of all doubles. Checking that endpoint before and after a
nonnegative shift excludes both kinds of overflow without visiting o²v² values.
This reduces canonical admission to O(ov); the physical minimum is a single gap.

The representation and all its numeric provenance enter the denominator
fingerprint. Explicit inputs may contain noncanonical denominators and retain
their existing admission/fallback semantics. The canonical path rejects mixed
representations and a stale singles array. No cache is introduced.

The primal host/device payload each replace `8*o²v²` bytes with `8*(o+v)` bytes,
before alignment and actual vector capacities. The native planner charges both
owners. Five scalar operations reconstruct a denominator at each consumed
Jacobi value. The iteration counter excludes host initialization and Lambda
diagonal setup; it is not a total endpoint FLOP count. Native CPU construction
and the internal explicit-CUDA selector retain bounded comparisons.

## Qualification

All compilation/compute remains on n2 through finite Slurm allocations with
verified ccache. Initial job2291 caught a missing input-role annotation in the
new scalar IR; this was repaired before successful code generation. Job2293
passed three admission/owner tests and 18 CPU solver cases, including all early
trajectory prefixes, independent determinant replay and exact/one-byte-short
canonical capacity admission (five CUDA-only cases skipped). Compiler, SCF,
cross-method ownership, promotion inventory and metadata checks passed.
Additional mixed/stale representation and independent noninteracting Lambda
coverage runs in job2295. The full CUDA build and complete endpoint gates are
pending. No GPU memory/timing result is represented here as measured until the
corresponding record exists. The CUDA ownership ledger remains unchanged for
handwritten scientific lines; runtime lines increase by 13 with no reclassification.

## Revisit

Retain explicit denominators in any measured domain where reconstruction loses.
An alternative rounding order, noncanonical provenance, mixed precision, or
nonnegative physical gap requires a separate scientific contract and gates.
