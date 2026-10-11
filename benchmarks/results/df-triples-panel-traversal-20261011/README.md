# Bounded FP64 triples panel traversal

One separate candidate-then-control complete ethane230 pair, o9/v221/Q488:
wall 48.491323059 -> 48.093351139 s, 0.8207% shorter (1.008275x); triples
4.135664454 -> 3.757277946 s (1.100708x). Both arms retain 19 observations/
evaluations and unchanged non-timing CC work. Original independent gates remain
total energy 1e-8 Eh, triples 1e-10 Eh and physical R1/R2 1e-10. Total/triples
energies are identical in this pair. Do not pool with #2232 or claim statistics.

Only strict FP64 energy with exactly three panel slots changes traversal.
Every tile stores at its original canonical address, preserving final reduction
order. Original W/energy kernels, LRU, six-cube/three-panel arena, resource
admission and finite checks remain. One/two-panel, non-FP64 and force/response
traversal remain original. Constant host storage, O(T) visits, no tile list.

Panels 152 -> 98; W evaluations stay 729, moment GEMMs 1458, total triples
GEMMs 1610 -> 1556, contraction summands 2610452107406 -> 2326012282334.
Every epilogue point and projected contribution remains. CC device numeric
capacity 8583749632 and complete endpoint host/device numeric capacity
8945677650 bytes remain unchanged; process peak RSS delta -1572864 bytes is
pair-specific, not a universal memory guarantee.

Actual compiled visitor coverage: occupied sizes 1..32, all canonical tile
slots. Six independent CPU energies retain identical canonical tile-array bits.
Candidate GPU matrix: 12 outputs over six fixtures, >256-point tail, one/two-
panel and generated-provider fallback, deterministic repeat, budget refusal
and all-equal nonfinite failure with unpublished sentinels. Memcheck/initcheck
each have two representative outputs and zero errors. Baseline fixture matrix
is reused because its e4e588... library and inputs are exactly unchanged.

`targeted-receipts.json.xz` adds only the previously uncovered three-panel
generated-provider fallback and a three-panel unsafe all-equal-seed request.
Both sanitizer tools pass both actions with zero errors; the generic action
reports exactly three slots/six panels and independent energy agreement, while
overflow preserves unpublished sentinels. No matrix or endpoint is repeated.

One native object rebuilds; all borrowed link inputs are checksum-verified
immutable. No existing ABI or scientific generated artifact changes. Baseline
dependency is the hash-bound df-triples-distinct-moments-20261011 publication;
its exact three generated kernels/header bytes are not redundantly copied.
The frozen library consumes #2171 HF and #2221/#2232 CC, not latest-master
HF/force. Master #2218 changes offline CPU XC AOT only. Broad performance and
production stages remain not-run. Passing suites are not repeated for unrelated
master commits or formatter-only changes.

raw-receipts.json.xz is stdlib lzma-compressed JSON of original UTF-8/base64
records, byte counts and SHA-256 hashes: all fixture/input/oracle outputs, raw
numerical/sanitizer/endpoint/build/cache/allocation receipts, exact compiled and
reviewed native/helper sources, borrowed object hashes and recipes. Reviewed
native source is an exact clang-format transformation of the qualified source,
bound separately rather than falsely claiming byte identity. The helper itself
is byte-identical. production-source.patch.gz reconstructs from the recorded
CC patch base f7f35bf45979c1e7d1c0e78395390d27eeb783d0; it does not claim a
clean full build of that tree. Reuse the prior #2232 baseline recipe and replace
one object, or perform a new full build without inferring matching whole-library
hashes. Use finite srun on node2, main/gpu:pro6000:1, assigned visibility intact
and verified compiler cache; run into fresh ignored directories, not this bundle.
