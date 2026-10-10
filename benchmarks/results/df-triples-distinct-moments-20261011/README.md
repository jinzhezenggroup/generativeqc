# Distinct occupied W moments, FP64 DF triples energies

One separate candidate-then-control complete ethane230 endpoint pair, o=9,
v=221,Q=488: wall 49.596829078 -> 48.494610759 s, 2.22% shorter (1.02273x).
Triples 5.155789883 -> 4.135521826 s (1.24671x). CCSD and physical replay
counters are unchanged; both arms use 19 observations/evaluations. Do not pool
this pair with #2221, or interpret it as statistical/universal qualification.

W evaluations 990 -> 729 (o^3), total triples GEMMs 2132 -> 1610, contraction
summands 3258407583236 -> 2610452107406. Every occupied tile, virtual point,
six projected energy contributions and finite check remains. Integral panel
builds remain 152. All-distinct tiles and non-FP64 W keep the original kernel;
forces/response retain full independent materialization. Six-cube layout,
panel admission, one-panel and generated-provider fallback remain unchanged.

Independent original gates: total energy 1e-8 Eh; triples 1e-10 Eh; physical
R1/R2 1e-10. Errors are 2.4301e-12, 9.2698e-14, 2.9439e-12 and 1.3072e-12.
Total/triples energies are unchanged in this pair. Numeric device capacity is
8583749632 bytes in both arms. Process peak RSS adds 1212416 bytes, not a
universal memory claim or peak GPU process-memory measurement.

Five independent CPU cases execute actual emitted bindings with unused slots
poisoned. Six real-GPU fixtures include a >256-point tail: baseline 7 outputs,
candidate 11 outputs cover independent energy/work, deterministic repeats,
one-panel and generated-provider fallback, one-byte-short refusal and all-equal
overflow with unpublished sentinels. Both representative sanitizer tools report
zero errors. The original sanitizer summary filename was reused, leaving final
initcheck numerical records and separate zero-error logs. A collector-only
repair reran only the two tiny memcheck actions to preserve a unique receipt.

The measured library freezes #2221 CC and #2171 HF consumers on the recorded
whole-library base. Only two objects rebuild; every borrowed link input is
checksum-verified immutable. The publication revision is the measured CC source
patch base 9b5b4849629f4ace4ffb7f00bf37dc4d5dd0a492, not an assertion that its
entire tree was built. Baseline/candidate library hashes and complete commands
are retained. No new #2215/#2217 HF, #2219 force/response, broad statistical
performance or production qualification is inferred. Parent #2221 merged as
5da28bdca: its tree is equivalent, consumed sources match, and current compiler
regeneration is byte-identical to all three qualified artifacts. Unrelated
master changes did not trigger another endpoint or passing GPU matrix.

production-source.patch.gz reconstructs the two modified files from the pinned
CC base. See source_reconstruction in validation.json.gz and the existing
df-cc-batched-replay-20261010 HF-integration receipts for the frozen baseline.
raw-receipts.json.xz is stdlib lzma-compressed JSON of lossless UTF-8/base64
records with original byte counts and hashes: fixture inputs/oracle energies,
all raw numerical/endpoint/sanitizer/build receipts, exact three modified
generated artifacts (including .hpp), native/generator source, reconstruction
recipes and immutable borrowed-object hashes. Unchanged unrelated shadow
headers are deliberately not a second evidence copy. Offline tests validate
publication admission, every receipt, original gates and exact work.

Generation needs compiler dependencies, not GPU/runtime/PySCF. Recover recipes
into a fresh ignored directory; do not run benchmarks in this retained bundle.
All real GPU execution uses finite srun on node2, main/gpu:pro6000:1, preserving
Slurm visibility and reusing verified ccache. No release/external archive needed.
