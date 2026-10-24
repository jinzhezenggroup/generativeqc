# Decision: prove bounded XC cross tiles before eliding repeated overflow checks

Status: proposed for review; independent arithmetic and endpoint qualification complete
Date: 2026-10-24

## Problem

The FP64 XC potential preserves finite cancellation by retrying the original
grouped cross term whenever nested FMAs overflow. This safeguard is essential:
an incoming finite sum can overflow on the first leg before the second leg
cancels it. Checking every step also inhibits the ordinary tiled contraction.
Do not remove the safeguard or alter the scientific expression to gain speed.

## Candidate

At the existing shared-panel load barrier, vote whether all four operand panels
have magnitude at most 2^128. A tile satisfying that bound and an incoming
accumulator of magnitude at most 2^1020 can execute the identical ordered FMAs
without inspecting each result. Each product is bounded by 2^256, and all
supported tiles have at most 64 additions per thread. This leaves a large
finite margin, including rounding. The proof is renewed at each point tile.

Unbounded, unordered or nonfinite inputs and unbounded incoming accumulators
retain the original per-step retry. The ordinary branch does not rescale,
reassociate, screen, lower precision or change any product. Diagonal-jet
admission and FP32 execution retain their existing paths. There is no new
global storage or extra shared panel; the block-wide vote replaces the load
barrier and every thread still participates before output-lane masking.

## Evidence and qualification boundary

An isolated same-device probe extracts the actual emitted 16x16 body, comparing
the original with this bounded branch on seven active-AO/count shapes, one and
four jets, and 16 batches. All 28 observations pass an independent NumPy dense
bilinear gate of 2e-12 and are bitwise equal between schedules. Observed kernel
ratios range from about 1.3 to 3.2. This ordinary-input microbenchmark is not
an endpoint improvement or extreme-range qualification.

Retain the existing actual-generated-body numerical simulator, including
independent maps, tails, spin, scatter, diagonal-range and finite cross-leg
cancellation controls. Extend cancellation coverage across point tiles so a
large incoming accumulator is carried through an otherwise ordinary tile.
Real-device numerical, capture, sanitizer and complete cold/warm E/F gates
remain mandatory before promotion. Keep all scientific settings fixed.

The probe sources, allocation, identities and results are retained under
`qc-dft96-cold-stage23-20261024/.artifacts/xc-cross-guard/` and
`n1:/data/jzzeng/direct-dft96-cold-20261023/xc-cross-guard/`.

## Revisit conditions

Do not widen the operand or accumulator bounds without a new finite-range
proof and independent extreme controls. A different contraction schedule must
retain both the grouped retry and its original rounding behavior when this
proof is unavailable.

## Completed endpoint qualification

Frozen source `1ccc53419acd4c22d9496178a65fffd83f26d6b1` and core SHA-256
`4555f81fe4e277565cec634d642b41af99aa8c99ae108b0997ea78dec47e43db` pass 134
real-GPU arithmetic cases and the native/resource/capture/sanitizer controls.
All 352 complete E/F calls pass independent gates. Five alternating pairs show
cold gains of 7.966/8.265% (PBE DF/Direct), 4.526/6.270% (PBE0), and
4.319/4.774% (r2SCAN-3c), each above the unchanged 2%/MAD gate. Both arms use
the optional validation GEMM feature; other controls remain stock. These are
incremental complete-endpoint gains and do not establish all-method GPU4 parity.

[Retained evidence and offline verifier](../../../benchmarks/results/xc-bounded-cross-20261024/README.md)
preserve all populations, full forces, work counts and measured identities.
The PR/archive source is later than the frozen measured source. The original
local missing-library test failures are retained alongside the successful
source-matched native rerun. No diagonal or FP32 safeguard is removed.
