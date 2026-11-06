# WB97M-V provisional-solver handoff

This bounded experiment addresses issue #2976: does native cold SCF oversolve
temporary operators, and does clearing history lose useful acceleration?
It is **not** a fresh GPU4PySCF timing comparison or an explanation of the
entire native 18 versus reference 10 reported-cycle gap.

Latest-master integration is pending; the clean paired qualification below is
complete and source-matched, but must not be relabeled as that later image.

## Decision

Enable spin-RMS progress for the existing qualified Standard FP64 full-Direct
WB97M-V provisional initializer/frozen-potential owner. Keep physical energy,
MO-gradient, canonical density/force checks, fresh energy epochs, CDIIS resets,
strict fallback, and all proposal/total budgets unchanged. Explicit
`GENERATIVEQC_VV10_RMS_HANDOFF=0` restores AO-maximum proposal control; explicit
`1` cannot bypass owner admission. Other precision/method/strict/fitted/chunk
owners retain their existing behavior. No GPU-name/size hardcoding, extra
resident storage, CPU/reference production work, grid weakening or Direct→DF.

## Clean paired qualification

Water96, WB97M-V, spherical def2-SVP, full Direct, Standard, FP64, native Huckel
guess, stock XC/nonlocal grids. Analytic moving-grid force response is enabled
by the endpoint harness in both arms. Energy tolerance 1e-9, density tolerance
1e-5, screening 1e-12 and DIIS history 8 are identical. Timings include
constructor, prepare, SCF, analytic forces and host export. Warm observations
restore the same engine-local cold density, not a changing sequence of seeds.
No DF_TRACE, scalar stage journal or experimental history selector is active;
force-pair selectors are unset. Each arm has one cold and three warm samples.

| GPU / real scheduled job | Arm | Cold/s | Warm median/s | Cold Focks | Provisional / target | Pair sweeps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A100 / LSF225862 | AO-max rollback | 111.882707 | 15.021796 | 18 | 13 / 5 | 4 |
| A100 / LSF225862 | Actual default | 108.185959 | 15.034534 | 16 | 11 / 5 | 4 |
| RTX5090 / Slurm16584 | AO-max rollback | 193.680082 | 14.299161 | 18 | 13 / 5 | 4 |
| RTX5090 / Slurm16584 | Actual default | 189.379635 | 14.340794 | 16 | 11 / 5 | 4 |

Cold improves 3.30% / 2.22%. Warm medians increase 0.085% / 0.291%; these
are noise-scale observations, **not** a claim of literally zero regression.
Every paired call passes the declared 1.05 ratio guard. Warm work remains
one Fock / one J / two K / zero pair sweeps. Cold J falls 18→16 and K 36→32.
All 32 independent repeat-pair observable comparisons pass unchanged gates.
No historical reference latency is used to rank these populations.

## Causal controls, not kernel-only timings

Actual A100 LSF225855 uses one fixed experimental image with an already-host
scalar journal; it adds no GPU readback and does not change MD-J routing.
Each completed arm has one cold plus one frozen-seed warm endpoint.

| Control | Cold/s | Focks | Provisional / target | Pair sweeps |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 111.947746 | 18 | 13 / 5 | 4 |
| RMS handoff only | 108.286345 | 16 | 11 / 5 | 4 |
| Unrebased proposal history only | 119.207417 | 23 | 19 / 4 | 3 |
| Both | 107.985260 | 16 | 12 / 4 | 4 |

The old initializer's AO-element maximum at steps 7/8/9 is
1.835e-4 / 1.418e-5 / 2.718e-6; step-7 spin RMS is already 4.954e-6.
The same provisional 1e-5 tolerance therefore hands off at 7 rather than 9.
The RMS arm's complete scopes are
`seed*7, target, frozen*2, target, frozen, target, frozen, target*2`.
Four frozen Focks and five physical targets remain: the causal net saving is
**two initializer Focks**, not eight physical iterations. All 16 independent
repeat-pair E/F checks for these four controls pass.

Keeping cross-operator Fock/error history only as bounded density proposals
makes the first frozen solve 4→10 Focks and total work 18→23. It is rejected.
This does not test or rule out an exact/secant-rebased history mapping; old
densities were not stored and old Fock/error pairs were not rebased. The
combined arm's single 0.3s advantage over RMS alone does not justify that risk.

The final physical tail is necessary under the unchanged policy: baseline
step 16 passes energy but fails MO stationarity; step 17 passes MO but its
energy change is about 1.214e-9; step 18 confirms energy on unchanged density.
Provisional spin RMS is not a replacement for that physical MO-gradient gate.

### Incomplete outer-off controls

Outer-off alone also disables automatic resident-LR source admission. It is
an integrated operator/resource-policy ablation, **not** a resource-matched
SCF sequence comparison. A100 reaches eight completed physical observations
before its 180s process cap; RTX5090 Slurm16517 reaches seven before 280s.
Neither has a complete cold E+F endpoint or accuracy qualification. The
subsequent Slurm16517 oracle test was never reached. Its completed baseline
is retained separately (cold 193.716044s, warm median 14.316116s, 18 Focks).

Latest-master integration additionally checks a bounded corrected control:
outer-off plus existing `GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1`. The 16GiB
default resident ceiling is unchanged; explicit source admission alone does
not establish identical realized resource bytes. Final results are retained
separately, not substituted for the failed original controls.

## Independent oracle and host checks

The static reference is the retained actual-A100 GPU4PySCF observable population:
`../direct-range-warm-anchor-20261105/publication.json.xz`, SHA256
`6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f`.
Every complete sample is checked against both reference samples with gates
1e-5 Eh/atom, max force 1e-4 and RMS force 3e-5 Eh/Bohr. This is **sampled
static default-grid parity**, not an independently grid-converged production
ranking. All full force arrays, not just error summaries, are retained.

Slurm16584 also runs `test_wb97mv_default_convergence_cuda.py`: both independent
GPU4PySCF original/moved-water tests pass, including default/explicit strict
and cold/warm energy plus analytic forces. The production image passes 23
host policy tests; the promotion inventory passes 136 tests and registers all
101 entries / 167 controls. Initial host failures (missing local native library;
missing n1 CCACHE environment) and their repaired runs remain in the payload.
These setup errors are not relabeled as scientific failures.

## Replay and retained evidence

From the repository root:

```bash
python benchmarks/results/wb97mv-stage-handoff-20261106/replay.py
```

`evidence.json.xz` contains raw endpoint JSON, full E/F arrays, scalar journals,
source/ELF/actual-device/scheduler identities, image/base/overlay receipts,
executed endpoint/build/job scripts, source patches, actual C++ compile command
lines, compiler-cache/configuration receipts, host and oracle JUnit/logs, failed
control exit receipts, and borrowed-tree restoration receipts. Authentication
and transfer logs remain private with byte-count/SHA256 receipts only; no
credentials, libraries or source tarballs are committed. Large local build logs
are hash-addressed while exact relevant compile invocations remain replayable.
`summary.json` is derived, not a substitute for the raw records.

The replay authenticates the independent reference, checks observable gates,
exact settings/FP64/physical-proof/work counts, clean paired guards, and the
causal provisional/physical distinction. Incomplete prefixes stay explicitly
unqualified and have no invented latency or converged cycle count.

Durable rationale:
`.agents/notes/implemented/performance/2026-11-06-vv10-provisional-rms-handoff.md`.
External memory: issue #2976.
