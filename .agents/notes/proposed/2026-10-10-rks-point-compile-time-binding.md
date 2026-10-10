# Proposal: bind restricted point algebra at the producer-owned kernel boundary

Status: proposed; static opportunity only, scientific qualification incomplete
Date: 2026-10-10

## Problem

GPU4PySCF's restricted XC specialization motivated a native four-direction
correlation prototype with one exchange evaluation. Its general point entry
checks exact equality of both spin densities and Cartesian gradients before
falling back to the existing full-spin implementation. Host parity is strong,
but source-level derivative width does not establish better CUDA execution.

The earlier five-direction compaction already produced identical machine code.
The new guarded restricted entry instead retains two large numerical bodies.
Merely supplying equal input values does not reliably remove its fallback:
IEEE NaN comparisons and compiler outlining remain observable control flow.

## Static evidence

Two finite CPU-only Slurm compilations on node1, jobs 7014 and 7018, used CUDA
12.9, sm120, FP64, O3 and `--fmad=false`. No device was allocated, imported or
executed. Both arms have the same input construction and nine-output ABI.
Retained source hashes match before and after each compilation.

Use **v2 within-arm comparisons only**: v1 compiles a cubin, whereas v2 compiles
a cacheable object with the verified ccache 4.5.1 launcher and the existing
`/data/jzzeng/ccache`. Its retained stats show one cache miss and no new
uncacheable invocation. Different compilation modes must not be paired as an
optimization comparison.

| v2 caller | Baseline non-NOP entry instructions | Candidate | Registers, baseline/candidate | Stack bytes, baseline/candidate |
| --- | ---: | ---: | --- | --- |
| General input, exact runtime guard | 7,133 | 18,530 | 138 / 138 | 0 / 0 |
| Equal spins constructed, guarded entry | 6,913 | 18,714 | 122 / 255 | 0 / 488 |
| Equal spins constructed, direct restricted core | 6,913 | 4,848 | 122 / 104 | 0 / 0 |

The last row removes 29.87% of non-NOP entry instructions. Its entry contains
127 call instructions versus 196 for the baseline. These are **static entry
counts**, not executed instructions, inclusive call-graph totals, profiler
evidence or complete endpoint timing. Registers belong to this point-only
probe, not the production geometry kernel. Static LOCAL=0 and ptxas spill
reports must not be interpreted as measured memory traffic.

## Numerical evidence and limitations

A fresh CPU diagnostic explicitly constructs both spins from the same density
and three gradient values in both arms, then compares the original full-spin
entry with the four-direction numerical core. PBE and PBE0 each retain 5,164
cases: 5,104 broad bound inputs plus all 60 original restricted reference inputs.
All nine outputs and validity status are bit-identical, including 90 invalid
cases per variant. No response layout or public API changes are made.

The independent original-formula gate is **still failed** in both arms: PBE
indices 8, 9, 13, 19, 25, 31, 37, 43 and 49; PBE0 indices 8 and 9. The unchanged
`5e-10*abs(reference)+1e-322` gate, inputs, oracle and failed outputs are retained.
Parity and static improvements do not turn these failures into qualification.
Do not launch the existing fail-fast GPU qualifier as if this issue were solved.

## Proposed ownership boundary

`src/dft/cuda_grid.cu` already has an owned `identical_spin_density` witness.
Only the single-spin device-density binding establishes it through
`split_restricted_density`; replacement paths invalidate it. The producer also
copies the first contraction panel rather than repeating its GEMM for the
second spin. This is stronger than a functional's `unpolarized` label.

However, the immutable v1 `GridTaskView` does not expose this witness. A future
integration must lend an explicit generation-checked producer proof and select
a separate restricted kernel/compiled entry. Keep the current general entry
as a bounded fallback for external densities, orbital sources, unsupported
consumers and missing/stale proofs. Do not extend the v1 view layout silently,
infer equality from labels, discard the second spin of an arbitrary view, or
inline both large bodies into the same point kernel.

The direct-core probe constructs equal inputs itself, so it does **not** prove
that the production consumer can use this dispatch boundary without additional
ownership and feature-identity validation.

## Promotion requirements

Resolve the independent numerical acceptance issue without dropping cases,
loosening gates or substituting Libxc tail cutoffs. Prove the producer/consumer
binding, lifetime, generation invalidation and fallback behavior. Then use a
small targeted real-device qualification and clean complete PBE0 energy/force
comparisons at 48 and 96 atoms before claiming a performance improvement or PR.
No native/AOT build, GPU execution or endpoint campaign occurred in this stage.

Latest inspected master is cd0eb059f (#2190), which changes ordered DIIS Gram
handling. Its point header remains byte-identical to the inspected baseline.
No old matrix was repeated for this merge, but future complete endpoint evidence
must distinguish its SCF policy/source from the retained #2185 binary.

## References

- `.artifacts/xc-restricted-stage/evidence/static/` (7014, cubin diagnostic)
- `.artifacts/xc-restricted-stage/evidence/static-v2/` (7018, cached-object diagnostic)
- `.artifacts/xc-restricted-stage/evidence/bound-host/diagnostic.json`
- `.artifacts/xc-restricted-stage/point-static.cu`
- `.artifacts/xc-restricted-stage/diagnose-bound-host.py`
- `.agents/notes/proposed/2026-10-10-restricted-point-oracle-boundaries.md`
- `.agents/notes/rejected/2026-10-10-compact-correlation-point-channels.md`
- `src/dft/cuda_grid.cu`
- `src/dft/grid_task_view.cuh`

## Subsequent scalar qualification

A new PBE0-only bound entry resolves its subnormal exchange denominator issue
without loosening gates or changing default general/response arithmetic. Its
CPU/GPU qualification is recorded in
`2026-10-10-pbe0-bound-subnormal-exchange.md`. The static v1/v2 source identities
and results in this note are historical controls, not evidence for a rebuilt
production owner. Producer-proof dispatch and complete endpoint timing remain
unimplemented and unqualified.
