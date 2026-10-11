# Owner-local Lambda invariant frontier: cold ethane E+F

Measured baseline: merged master `7685c251ded6c986b3cb63336e3c1cd31c0cd739`.
Candidate: that exact base plus `measured-source.patch.gz`, **dirty**, not an
unmeasured clean commit. The exact Git base, reconstruction patch, archive/patch
and changed production-input identities, library/endpoint SHA256 identities,
device cohorts, toolchain/cache receipts and quantitative gates are retained.

Three hardware-stratified ABBA cohorts use twelve fresh processes, six per side,
with unmodified strict-FP64 controls and no timing instrumentation. Profiling and
sanitizer observations are separate. Both sides retain Q32, cadence 30, original
independent audit and fresh primal replay, 21 Lambda iterations, 22 physical
solver actions, CCSD replay and both complete physical nuclear-force passes.

| Complete native endpoint / phase | Baseline median s | Candidate median s | Reduction |
| --- | ---: | ---: | ---: |
| Cold ethane CCSD(T) E+F | 327.989404 | 318.739047 | 2.820% |
| Lambda | 88.312413 | 78.998265 | 10.547% |

All cohorts improve and the observed complete median saving exceeds three times
the combined median absolute deviations. These are descriptive scoped results,
not a universal profitability claim or a formal production-promotion envelope
with complete iteration histories. Numerical acceptance is the publication
decision; mathematical/runtime constraints and bounded fallbacks remain primary.

## Storage and independently predicted work

The existing purity/dependency proof still executes 103 invariant operations in
preparation. Only their 40 boundary values require persistent storage:
396,517,056 bytes rather than the previous 1,710,877,464-byte whole-core arena.
Transient preparation/dynamic storage is 578,431,456 bytes, contained by the
712,793,032-byte original core matrix scratch. Native admission additionally
checks its actually selected scratch and full host/device/provider ceilings;
it never enlarges mandatory scratch or reduces Q/audit/replay to admit retention.
The existing default requests this cache; explicit argument 25=`0` refuses it.

One preparation and 23 reused core actions replace 23 complete core evaluations.
An independently compiled host query of the actual generated core predicts each
of the 22 skipped preparations: 103 scheduled operations, 33 GEMM calls,
34,875,793,292 contraction terms, 34,851,307,725 GEMM summands, and 777,609,024
logical packing-output bytes. All five measured savings match these predictions
exactly. Summands are not hardware FLOPs and packing bytes are not DRAM traffic.
The separate CUDA trace verifies every invariant non-GEMM node launches once,
and every dynamic non-GEMM node launches 23 times. No occupancy/spill or hardware
counter claim follows from node-level tracing.

## Numerical and failure gates

Thirteen complete endpoints, including the separate diagnostic, retain passing
quantitative independent energy, qualified 24-component force-vector and four
independent coordinate/step finite-difference gates. Exact Lambda/Z residuals,
stationarity, CCSD replay and translation gates are not relaxed. Forty-nine
real-GPU tests cover cache/uncached bit patterns, budget refusal, changed same-
shape owner epochs, independent matrix audit and provider lifetime. Seven
bounded native cache/budget/owner tests pass memcheck with zero errors/leaks;
this is not a full-ethane sanitizer claim.

## Corrected predecessor integration

The merged force implementation has an independent corrected source-matched
campaign: clean `7f342546d887796e6a92a005ba033029ed73ce2a` versus clean
`590e076c4d3345b797ed2b2babc0d1778fd793d5`, whose production force changes are
identical to PR #2210's merged code. Twelve cold ABBA observations give
350.604799 → 328.097170 seconds, a 6.420%
reduction. Its numerical/default/fallback/memcheck gates and separate exact
force-domain trace pass. Unrelated later master changes are not silently
attributed to this predecessor snapshot.

The initial integration attempt reused one build directory across Git archives.
Preserved archive mtimes prevented rebuilding the changed dispatch object; the
unused new class-domain translation unit rebuilt, but its dispatcher did not.
Its null-gain data remains in the local audit as a failed source/build
qualification, not a valid candidate comparison. The corrected build refuses
existing staging and verifies actual dispatcher references before deployment.
The force qualification's first profile exceeded its finite Slurm allocation;
the incomplete trace is retained, not accepted. A separate finite allocation
completes the same-binary profile without pooling it into cold timings.

## Reproduction and retained scope

Use the repository's original ethane input and reference files under
`benchmarks/results/rhf-phase-values-auto-20261010/`; input identity is
`9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
On the recorded compiler/toolchain, create the named campaign, generate
`master-7685.tar` with `git archive` of the exact base, decompress the retained
patch as `master-prototype.patch`, and run `build.sh`. Initially empty source
and in-tree build directories are mandatory for both sides. The verified
ccache launcher and existing shared cache are reused; FAST_COMPILE is OFF.

Deploy the two immutable binaries and exact candidate source to an allowed
Slurm GPU node. Install `run-abba.sh`, `run-gpu.sh` and `run-qualification.sh`
as `run-abba-master.sh`, `run-gpu-master.sh` and `run-qualification-master.sh`
in the script's campaign directory. Run every real-GPU command via finite
`srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:35:00`
and preserve Slurm's `CUDA_VISIBLE_DEVICES`. Nodes n1/n2/n4/n5 are allowed;
n3 is excluded. The measured n2 driver mismatch means it is a CPU build host
only for this campaign. Never silently substitute FAST_COMPILE, a different
precision, reference outputs, GPU visibility overrides or source/binary receipts.

Raw runs, failed attempts, compile receipts and traces remain in the ignored
local audits `evidence-lambda-next-20261010` and
`evidence-force-pr-integration-v2-20261010`. They are not external release assets.
The original frozen force bundle and its original old-base measurements remain
unchanged; no new timing is relabeled as that earlier campaign.
