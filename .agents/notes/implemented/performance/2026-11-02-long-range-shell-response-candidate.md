# Decision: share generated radial geometry with the LR shell response

Status: implemented, explicit candidate only
Date: 2026-11-02

## Problem

After the fitted LR source and occupied-factor lifetime repairs, the A100
96-atom WB97M-V endpoint completed with independent energy/force agreement but
remained slow. LSF 224641 spent approximately 127 seconds in the generic
three-center derivative. The enclosing `response_weights` trace includes that
work; treating it as an additional weight-GEMM cost would misidentify the
bottleneck. These are inclusive, intrusive diagnostic intervals.

## Decision

Carry the frozen radial omega through ordinary, signature-group and panel shell
views. The shell derivative uses the same generated polynomial radial geometry
as the generic derivative. No second LR recurrence is introduced. Nonzero omega
keeps polynomial execution because the existing shell Rys lowering implements
Coulomb radial geometry. Coulomb-only screening and mixed arithmetic are disabled
for LR. This is an operator capability restriction, independent of GPU model.

The LR shell route requires an explicit `GENERATIVEQC_DF_WEIGHTED_EXECUTION`
shell selector. Automatic LR response stays generic pending source-matched
production qualification. Fitted LR itself continues to require
`GENERATIVEQC_DF_RANGE_EXCHANGE=df`; this change does not replace the exact/mixed
Hamiltonian default.

## Independent controls and failed fixture

The native shell fixture covers Cartesian/spherical orbital and auxiliary f,
omega 0.03/0.3/3.0, pair policies, panel/group/packet schedules and tails. Its
independent CPU LR oracle only supports through f. The first fixture attempt
(Slurm 15274) incorrectly requested auxiliary g and failed at oracle admission,
before a GPU comparison. Keep this original failure. Actual README auxiliary g
is covered separately by the molecular libcint energy/force and displaced
geometry tests, including an executed-route trace for generic and shell paths.

Slurm 15269 passed 48 molecular cases with one skip on the preceding diagnostic
source. Slurm 15268 completed cold plus three warm endpoints: native cold
279.33 s and median warm 66.58 s, versus GPU4PySCF 373.44 s and 124.56 s. The
maximum paired force discrepancy was 2.46e-7 Eh/bohr. The old generic diagnostic
had approximately 213 s warm calls on that device. These `BUILD_TESTS=ON`, traced
results do not qualify the new master integration or its production latency.

## Invariants and revisit conditions

Keep the radial operator frozen with its source and preserve metric response,
normalization, center derivatives, final-weight contraction and bounded fallback.
Do not promote based on a component interval or an unmatched build. Revisit the
default only after clean production cold/warm endpoints, independent observable
gates and resource/lifetime controls pass on the target devices. A future LR Rys
implementation must share the compiler-owned radial contract and acquire its own
independent numerical controls.

## References

- [Fitted LR ownership rationale](2026-11-02-long-range-df-reuse.md)
- `benchmarks/results/wb97mv-fitted-range-20261102/`
