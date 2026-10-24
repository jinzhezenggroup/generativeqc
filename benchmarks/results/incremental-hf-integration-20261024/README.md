# Incremental HF integration qualification

This receipt covers PR #2530 composed with master
`fc854745e14eb9a21734372a86403e2491a8e5d4`, preserving the newer qualified
RHF convergence owner and automatic/bounded-proposal KS policies. It is
correctness and compatibility qualification, not a new performance campaign
or an incremental-HF default promotion.

## Source-matched final results

`validation.json.xz` retains complete final logs, returned force vectors,
independent references, semantic work counts, frozen build/run scripts, tool
versions and SHA256 maps. The native source identity is
`78313c1312b65d83decf0480eb1474636ec441164760d0e0048bc780e9ebd8df`;
the library SHA256 is
`fe26ffa35c08fb6d7ad5edbad37d1806773a2c517ceed85dce3149f278c047fa`.
The build is Release, CUDA 12.9.1 / nvcc 12.9.86, sm120, scalar CPU linalg,
with shell/stationary AOT disabled and verified sccache 0.16.0. Compiler-cache
statistics are retained; shared totals include concurrent work.

- **373 unique host tests pass**: 306 controller/default/structure/ownership/
  evidence tests and 67 resource/API tests against the final native library.
  Formatting, structural and inventory checks also pass.
- Slurm **n1:13951** passes the selected native convergence, incremental HF,
  KS and Fock-provider suites plus the 67 resource/API tests.
- **n1:13952** passes all six Compute Sanitizer 2025.2.1.0 runs: memcheck and
  synccheck for the convergence/bounds, complete incremental HF and complete
  incremental KS executables. Every run reports **zero errors**.
- **n1:13954** passes convergence/bounds racecheck with **zero hazards, errors
  or warnings**.
- **n1:13950** passes all **18 independent energy-plus-force endpoints**:
  water RHF and OH UHF against live CPU PySCF 2.14.0, plus the retained
  384-AO case against its independent PySCF arrays, each with incremental
  OFF/on and cold/warm/moved states. Maximum errors are
  **1.1368683772161603e-11 Eh** and **1.7576797222557117e-10 Eh/Bohr**.

The incremental 384-AO cold/warm/moved solves take 18/1/16 iterations,
respectively: 4/1/4 full SCF builds, 14/0/12 delta builds and 2/0/2 final full
builds. All six large endpoints use one target attempt with no discarded target
iterations or warm fallback. Every incremental warm endpoint uses one full
SCF build, zero deltas and zero final full builds. Streaming quartet work is
unavailable, not zero. These are generic/non-AOT correctness diagnostics, not
an all-target build or a production performance qualification. No new endpoint
timing campaign was run; historical timing observations and failed cohorts
remain unchanged.

## Reproduced failure and repair

The composed source before the barrier repair has native identity
`1d503add2ba0f97691c308b4b2c85d09b229090b52a2f576133bc91aba546e88`
and library SHA256
`7c189b6b05365fe8f254bf62ef9be85d8b22d14bd6f2b5cf4bdf07c3383244c1`.
Restore only `src/scf/cuda/direct_density_bounds.cu` from the named master base
to reconstruct that source. Job **n1:13936** reproduces 96 complete-HF
synccheck errors at `+0x1250`; **n1:13937** passes all 18 independent endpoints
before repair, showing why endpoint numerics alone did not clear the blocker.

The receipt preserves the full failing log and SASS, passing plain/captured
standalone controls, and the failed attempt to add a warp join before the old
CTA barrier. The retained repair uses warp synchronization at both handoffs
for the exactly-one-warp HF launch and keeps CTA synchronization for wider
consumers. The complete HF executable provides the failing-to-passing regression;
the standalone probes alone did not reproduce the failure. No claim is made
that the old diagnostic was a false positive or that a particular compiler or
driver defect has been established.

## Recovered historical failure

`historical-sanitizer.json.xz` contains the complete original job n1:8706 log
as `raw_log`, with its SHA256. Encoding that string as UTF-8 reproduces all
321,307 original bytes, SHA256
`344a9bb60daffb90065fb8315f5793603627d83bb4a4c197da6c830818449a7d`.

The original memcheck runs report zero errors. The HF synccheck then reports
`Barrier error detected. Divergent thread(s) in warp.` in
`reduce_shell_pair_density_bounds_kernel<false>+0x1250`, beginning at
thread `(0,0,0)`, block `(0,0,0)`. Its final diagnostic is
`ERROR SUMMARY: 96 errors`; the target fails its baseline fixture and Slurm
reports exit code 99. The full log includes every coordinate and host stack.
This baseline used the patched library, not an independent clean-master binary.
The recovered log does not contain the exact command or historical tool version;
those fields are explicitly unknown. Source-level barrier reasoning alone does
not establish a false positive or discharge the historical failure.

## Reproduction

Build the native CUDA library and targets
`generativeqc_cuda_force_convergence_tests`,
`generativeqc_incremental_direct_jk_cuda_tests`, `generativeqc_ks_cuda_tests`,
and `generativeqc_fock_provider_tests` using a verified compiler cache. Run all
real-device commands in a finite Slurm allocation, preserving its assigned
`CUDA_VISIBLE_DEVICES`. The controller and incremental HF/KS tests cover
physical gating, both spins, resident warm reuse and canonical fallbacks.

Run `qualify.py` with this checkout's `python` directory on `PYTHONPATH` and
`GENERATIVEQC_LIBRARY` pointing to the built library. It compares small RHF/UHF
endpoints against live CPU PySCF and the retained 384-AO cold/warm/moved case
against the independent reference arrays in
[the original receipt](../incremental-hf-full-energy-refinement-20261015/README.md).
It retains returned force vectors, gates and semantic work counts. Energy/force
acceptance remains 1e-8 Eh / 1e-7 Eh/Bohr, with SCF energy/density/screening
controls 1e-12 / 1e-9 / 1e-12 and a 100-iteration bound. Reference densities
never seed the candidate. Historical performance results are unchanged.

See the [composition decision](../../../.agents/notes/implemented/compatibility/2026-10-24-incremental-hf-master-integration.md).
