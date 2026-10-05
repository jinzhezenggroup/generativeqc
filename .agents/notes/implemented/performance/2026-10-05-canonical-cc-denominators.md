# Decision: fuse canonical doubles denominators into their consumers

Status: implemented and qualified; no endpoint speedup established
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
Job2295 additionally passes mixed/stale representation and independent
noninteracting Lambda coverage. CUDA build2292, 23 real-device solver cases and
18 Lambda/factor cases in2296 pass. Job2300 passes independent small-water
all-coordinate FD and failure-publication gates. The ownership ledger has no
new scientific lines; runtime lines increase by13 with no reclassification.

Complete energy2298 (two alternating pairs) and force2299 (one pair) establish
31,646,976 bytes less device allocation for ethane230, approximately one d2
tensor. Complete force capacity decreases63,296,152 bytes. Energy-run CCSD
medians147.256→146.944 s do not establish an endpoint win; full times vary with
unmodified RHF. Water CCSD's0.000719 s median regression is retained explicitly,
and the explicit selector remains available for this observed latency-losing
domain. Two repeats do not justify a universal size cutoff. This is a capacity
optimization; no bandwidth or whole-endpoint acceleration claim follows.

Large forces agree within4.371e-9 Eh/Bohr and pass independent two-coordinate,
two-step central-energy re-audit using existing2288 values. That audit does not
qualify all large coordinates or benzene264. Full raw and normalized evidence:
[`cc-derived-denominators-1904`](../../../../benchmarks/results/cc-derived-denominators-1904/README.md).

A follow-up device-free CUDA-admission sentinel exposed a stale test harness:
it extracted `validate_problem` without the newly called canonical validator.
The harness now compiles the actual canonical validator and shared generated
denominator expression, with ccache, before checking that invalid fields never
reach the allocation owner. This follow-up passes on n2 through Slurm and
changes no production source or endpoint results above.

## Revisit

Retain explicit denominators in any measured domain where reconstruction loses.
An alternative rounding order, noncanonical provenance, mixed precision, or
nonnegative physical gap requires a separate scientific contract and gates.
