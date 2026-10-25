# Native Hückel capacity: CUDA and installed-client qualification

Measured implementation: `7e555f3841e3109dea82fa94d12de8ad78ff0353`.
Baseline: `40c31f7bd6130903ade0cd87ef60fb06c56d9d8b`.
Both were built from frozen clean archives. No scientific source changed during
qualification. Source archive and library integrity receipts are retained.

All eight CUDA ABBA/BAAB complete-endpoint gates pass (limit 1.05; maximum
candidate/baseline median ratio 0.925986535017).
Each block has 21 samples after three warmups. The 672 timed endpoints
include construction, preparation, cold/warm/moved execution and closure, with
2520 item publications and 1008 force arrays. Published energy, forces,
work/convergence signatures and planned peaks agree exactly across both products.
This is scoped non-regression evidence, not a universal speedup or complete
allocation/performance promotion. Unknown Fock counts remain unavailable.

Slurm job 14395 on n1/main/gpu:5090:1 used CUDA 12.9.1, sm_120 Release, fast
compile disabled and sccache 0.16.0. It passed all 161 selected CPU/CUDA tests,
the native preliminary contract, and six independent-oracle CUDA cases under
compute-sanitizer memcheck (zero errors). The first attempt, 14382, passed 155
tests but lacked the six PBE/PBE0 force cases' packaged stationary artifacts;
the second attempt built those configured targets without changing sources.

The installed public SDK probe ran with `PATH=/nonexistent` on CPU and CUDA:
24 E/F publications per backend cover Direct/DF RHF H2 at two geometries,
default/exact/one-byte-below preparation caps and repeated execution. These
single-calculation repeats are not advertised as warm starts. Exact cap used
the seed; one byte below skipped it with zero preliminary Fock work. The cap
covers preparation numeric payloads, not whole-device OOM or RSS admission.
Four installed CUDA CLI E/F cases completed in job 14403 after correcting the
runner's invalid auxiliary-basis flag on the Direct case. All 52 installed
publications pass independent PySCF gates (energy 1e-8, force 1e-6); maximum
errors are 1.9096e-14 Hartree and
1.1768e-14 Hartree/Bohr.

## Reproduction

`samples.json.gz` contains every timed publication, installed output, independent
reference, exact runner/probe source and selected integrity/test receipts. Decode
it with Python's `gzip` and `json`; `input_files` maps original filenames to
exact text. `reproduce-installed-reference.py OUTPUT.json` recomputes the reference
using PySCF alone. `installed_sdk.cpp` compiles against installed public headers
and the installed library, through a verified compiler cache.

`run-paired.py OUTPUT_DIR cuda CANDIDATE_CHECKOUT BASELINE_CHECKOUT` runs the
existing `benchmarks.native_resource_composition` endpoint with explicit CUDA
selection, both block orders and unchanged numerical/work/1.05 gates. Build the
native library, CLI, preliminary test and the configured stationary AOT manifest
targets first. The worker scripts retain the exact build flags, test commands
and installed execution commands. Adapt checkout/toolchain paths, and run every
GPU operation through a finite `srun` allocation preserving its visibility.

Whole-process allocation/provider memory, cold compilation cost and independent
per-iteration residual traces were not measured. The evidence envelope therefore
keeps full performance promotion unqualified. This does not complete #2227.
