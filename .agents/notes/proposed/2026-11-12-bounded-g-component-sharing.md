# Decision: conservative exact sharing across g Cartesian alternatives

Status: proposed; bounded CI candidate, not CUDA-qualified
Date: 2026-11-12

## Problem and decision

PR #3057 baseline f9f0939ea5b262a278943c95ea637bb4f0acb724 emits a
16,190,014-byte full-g one-electron CUDA header. Existing Graph value numbering
shares within each Cartesian component but not between alternative components.
A large ptxas resident footprint was observed in a failing Release build. Source
size alone does not prove its cause or that this prototype fixes the failure.

Keep each component's scientific DAG and ordered arithmetic unchanged. Intern
its geometry-boundary lowering into a common target Graph. Use existing exact
value numbering and ScalarCEmitter.fork to emit only ancestors required by EVERY
alternative in a shell pair. Child cases inherit definitions without executing
inactive arithmetic. Compute the identical Boys argument/evaluator before its
consumers. Reject invalid indices before hoisted floating-point work. Preserve
g capability, FP64, Boys evaluation and strict gates. Leave <=f lowering intact.
No new generic optimizer, runtime backend or function partition is introduced.

## Bounded comparison and selection

Corrected operand-preserving variants were compared using emitted bytes and
selected-path textual definition-to-last-use scalar liveness:

| Sharing region | Header bytes | g-g attraction peak live | Maximum g-g same-path increase |
| --- | ---: | ---: | ---: |
| Baseline | 16,190,014 | 79 | 0 |
| Whole shell pair only (selected) | 14,904,718 | 80 | 2 |
| Nested leaf32 | 14,648,892 | 84 | 14 |
| Nested leaf16 | 14,427,776 | 88 | 23 |
| Nested leaf4 | 13,375,517 | 117 | 54 |

Choose whole-shell-pair sharing: 7.94% less generated text, no extra binary
branches, substantially smaller lifetime cost. Remove recursive region machinery
from the selected implementation. These are source estimates, excluding Boys
arrays, compiler scheduling and actual registers; no occupancy claim is made.
Across all g-containing helpers the selected variant's largest same-path
increase is four scalar temporaries. All variants retained selected-path scalar
assignment and Boys-call counts over 2,450 alternatives. Savings remove emitted duplication, not selected-path runtime
arithmetic. All 32 <=f helper bodies remain byte-identical.

## Validation, limits and rejected alternatives

The selected header is checked by paired original/candidate emitted-host probes
covering all shell pairs, Boys zero/20 boundaries, distant/translated geometry and
random g-g inputs. Independent Libcint tests execute the emitted header rather
than merely its interpreter, preserving incumbent strict tolerances. One-off test receipts
and measurement scripts were retained during review. Selected
results: 181 independent tests pass, 39,225 paired outputs are bitwise equal,
and invalid internal dispatch raises no floating exceptions. A separate review
probe checked another 56,700 finite outputs under six axis permutations, large
common translations, and diffuse/tight exponents; every paired result was
bitwise equal. Existing through-f coverage previously passed 69 tests; its generated artifact remains unchanged.
Host cache-miss diagnostics: original 21.15s / 1,437,116 KiB, selected 20.85s /
1,287,244 KiB; objects 13,789,208 / 12,334,760 bytes. The original repeat is a
cache hit, not another fresh timing sample. These single samples are diagnostics,
not a statistically established compilation-speed improvement.

Host GNU14 O0 diagnostics cannot establish CUDA12.9 Release compile resources or
GPU performance. CUDA compilation, device numerical gates, register/spill counts
and complete endpoint throughput remain unmeasured. The original build failure
is unresolved pending matching-toolchain/device evidence. Publication integrates
master fc9bbe6bea9b441f61a3bc0aab75ebb5a25f2960 and refreshes current catalog
identities with the official generator, without relabeling historical
qualifications. The integrated <=f and full-g generated headers are byte-identical
to the tested candidate artifacts. The PR remains Draft pending CUDA validation.

A shared scientific source Graph reordered n-ary arithmetic (paired differences
up to 5.7e-14), so source graphs remain per-component. Nested regions shrink text
further but increase lifetime estimates; retained as rejected local tradeoff
evidence, not a GPU-tuned winner. Hoisting expressions needed by only SOME cases
would execute speculative work and is not allowed. Mere function partitioning
does not remove repeated formulas.

## Why source bytes are not a compiler-memory prediction

The nine g-containing attraction helpers have 825 static Boys callsites in the
original header versus nine in this candidate. Each selected path still invokes
Boys once. Because the evaluator is force-inlined, this is a concrete reduction
in static duplication, but neither source-byte reduction nor callsite count
predicts final PTX size, optimizer memory, registers, or wall time. Measure the
matching CUDA Release build before claiming the failing compile is repaired.

Deeper nested sharing remains rejected for this candidate. Axis-permutation
canonicalization is a separate unimplemented opportunity: g-g has 42 joint-axis
permutation classes among 225 Cartesian pairs. That class count is not a measured
code-size saving. A future proposal must validate transformed geometry, FP64
ordering and actual device branch/register/complete-endpoint costs independently.

Reproduce the required arithmetic gate with
`pytest -q tests/python/test_one_electron_g_shared_values.py` using the repository's
verified compiler-cache fixture and PySCF/Libcint. This test does not substitute
for real CUDA compilation or device qualification.
