# Native CLI initial-guess qualification

Measured source: `824b94bdea0084a1728952030d6689382147d887`. Baseline: `40c31f7bd6130903ade0cd87ef60fb06c56d9d8b`.
The feature forwards explicit initial-guess choices to the existing native owner;
the CLI default remains core density. No target solver or scientific equation changed.

All 16 CPU/CUDA complete CLI case/order gates pass at the unchanged 1.05 ratio
limit. Maximum ratios: CPU 1.001454261198, CUDA
1.029778280651. Four water/STO-3G cases cover
Direct RHF E/F, DF RHF E/F, PBE energy and PBE E/F. Both ABBA/BAAB orders retain
21 samples after three warmups per block: 1,344 timed fresh-process endpoints
and 1,008 force publications. Timing includes startup, preparation, SCF, output
and teardown. Every published metadata/iteration field agrees; energy and force
parity is bounded by 1e-12. This is scoped default-path non-regression evidence.

Installed policy checks retain 24 CUDA plus two CPU CLI E/F publications against
independent PySCF 2.14.0 references, with gates 1e-8 Hartree and 1e-6 Hartree/Bohr.
Maximum errors are 2.1316e-14 and
3.0209e-08. CUDA covers base/moved H2,
Direct/DF RHF none/Huckel, Direct PBE MINAO/Huckel, and default/exact/one-byte-below
Huckel preparation caps. HF/LDA CUDA requests refuse explicitly. The installed
public C++ probe checks backend/oracle, diagnostic absence before execution,
budget skip, move and copied diagnostic lifetime after destruction; CPU/CUDA pass
with PATH=/nonexistent, and CUDA memcheck reports zero errors.

CPU Release qualification passes 119/119 CTests, 23 contract tests plus 25 subtests,
and the final CLI regression rerun. CUDA 12.9.1 sm_120 Release uses all configured
stationary artifact targets, fast compile OFF, and verified sccache 0.16.0. Native
preliminary contracts pass. All GPU executions used finite Slurm allocations on
n1/main/gpu:5090:1 and preserved scheduler visibility.

Job 14410 passed build/science/sanitizer before its timing runner incorrectly
required byte equality from repeated unchanged CUDA baseline results. The failed
log is retained. Diagnostic 14420 records three baseline results with roundoff
differences (about 1.4e-14 Hartree); job 14421 preserves exact metadata/iteration
checks with 1e-12 numerical parity. The 1.05 timing gate and all cases/sample counts
are unchanged. No production source or products changed between these jobs.

## Reproduction and limits

Decode `samples.json.gz` with Python gzip/json. It retains every timed output,
CUDA warmups, installed outputs/references, exact runner/probe inputs and selected
build/test/integrity receipts. `qualify-cli-cpu.py` and `qualify-cli-cuda-v2.py`
run the paired default endpoints. Adapt their explicit product/input paths;
use a finite srun allocation for the CUDA script. `reference.py` regenerates
independent references. `qualify-installed.py` checks installed CUDA policy
requests; `installed_sdk.cpp` compiles against installed public headers/library.
The worker retains build flags and commands. Compile through a verified cache.

Preparation caps do not establish whole-device/process admission. CLI output
provides iterations, not independently measured Fock/transfer/launch counts or
per-iteration residual traces. Full allocation/provider/RSS and cold compilation
costs are unavailable; the envelope leaves full performance promotion unqualified.
Fresh-process repeats are not prepared warm starts. R2 and the broader #2227
migration remain open.
