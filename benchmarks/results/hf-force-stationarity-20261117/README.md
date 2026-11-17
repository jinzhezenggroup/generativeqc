# Default HF force-stationarity qualification, 2026-11-17

**Latest comparator result:** stock GPU4PySCF also fails the `1e-7` force gate
on these cases; see the comparator section below. The stricter prototype is
not a justified default promotion merely because it passes that gate. The
earlier master-relative slowdown and all failed scientific samples remain
retained; no default change is shipped or gate retroactively marked passed.

## GPU4PySCF comparator: default accuracy versus strict accuracy

Finite RTX5090 Slurm jobs 17690 and 17691 measure GPU4PySCF 1.8.1, PySCF
2.14.0 and CuPy 13.6.0, using the exact retained Cartesian direct-RHF water
SVP/TZVP inputs and independently converged CPU force arrays. Stock GPU4PySCF
controls are unmodified: energy `1e-9`, gradient tolerance unset, direct-SCF
threshold `1e-13`, stock DIIS and 50 maximum cycles. The installed source uses
`sqrt(conv_tol)` when the gradient tolerance is unset. No reference density is
injected and no extra final cycle is artificially added.

Maximum force-component error, Hartree/Bohr, across all relevant fresh repeats:

| Basis / phase | Stock master GQC | Stock GPU4PySCF |
| --- | ---: | ---: |
| SVP / cold | 5.61053e-8 | 1.12874e-6 |
| SVP / moved | 4.38978e-8 | 5.93425e-7 |
| TZVP / cold | 6.75801e-7 | 1.29161e-7 |
| TZVP / moved | 4.48414e-8 | 5.67409e-7 |

The master column is the unchanged source-bound n1 cohort in
`performance.json.xz`, commit `8f4166f0c6a692045eaf568c814b868aa1057482`,
both profiles. The comparator column contains three fresh processes per basis,
including the initial pilot, with every result retained. In the original TZVP
cold case GPU4PySCF is about five times more accurate, but both miss `1e-7`;
on the moved case and SVP native master is more accurate. These are comparable
default error scales, **not equality on every endpoint or a universal bound**.

Every comparator energy passes `1e-9`; of its 30 force observations only three
pass `1e-7` (TZVP warm). Unchanged scientific gates therefore classify 9/36
total stock endpoints as valid, including six energy-only observations. An
explicit diagnostic changes only GPU4PySCF `conv_tol_grad` to `1e-9`: all 24
endpoints pass, maximum force error `2.28321e-10`. The two integrated native
profiles contribute 48 additional endpoints, all passing, maximum force error
`1.87033e-9`. This demonstrates convergence-limited force errors, not a reason
to call the failed stock results equal to strict results.

Complete synchronous E+force timings and semantic work are retained. Median
SVP moved time is 2.692 s stock GPU4PySCF, 9.598 s strict GPU4PySCF, 0.433 s
repair auto and 0.256 s repair portable. TZVP moved is respectively 8.058 s,
25.102 s, 1.570 s and 1.101 s. Stock comparator moved work is seven counted
cycles / eight actual potential evaluations; strict SVP/TZVP takes 31/29
cycles and 32/30 potential evaluations. Native repair takes 12 cycles / 13
Fock builds. Recorded `get_fock` Python calls are not equated to physical
potential evaluations. Warm runs use each engine's own fixed post-update
density; GPU4PySCF reruns stock SCF rather than a forced zero-cycle shortcut.

**These times are not an optimized-reference speed ranking.** This installation
has no cuTENSOR distribution and reports its CuPy contraction fallback. The
strict setting is a diagnostic, not a search for the fastest comparator that
passes `1e-7`. Cold initialization/cache effects, the pilot, all force arrays,
actual call counters, exact acquisition sources, installed source/library
identities, Slurm visibility and job receipts are in `gpu4pyscf.json.xz`.
Replay all 108 endpoints with:

```bash
python3 benchmarks/results/hf-force-stationarity-20261117/gpu4pyscf_replay.py
```

Following the user's request to assess comparator default accuracy, these data
do not justify imposing the slower strict prototype as a mandatory stock fix.
Keep PR #3108 Draft; do not merge a default slowdown to meet a stronger implicit
accuracy target than this comparator supplies. Any deliberately stronger force
contract needs explicit policy and independent qualification; these results do
not relax existing gates or erase the master-relative regression. This limited
study does not qualify UHF, DF, H100/A100, PRO6000 or larger systems.

Four additional unbound historical JSONs are whitespace-compacted solely to
retain this evidence within the unchanged cap, saving 46,467 bytes. Exact
numeric/string lexemes and failures are preserved; `storage-compaction.json`
records original Git byte identities and the additional saving.

This is independent scientific qualification for issue #3092, not a performance
ranking or a claim that every system satisfies a universal force-error bound.
The repair separates force admission from energy-only orbital-gradient stopping;
it does not change public defaults or add production reference-oracle work.

## Source and acquisition

- Native code commit: `6b52f067c581de21a6c06d11db08d3c7a6873ea4`.
- Base: `50482bc351756772660c45b3e8430fb388e9cf99`.
- Native source identity:
  `15f2cf4ebb59e16c3c3d31a50aaa3dcc508b4e7c07c539e63a82910d73e745c6`.
- Final device-test recipe:
  `tests/python/test_hf_default_force_stationarity_cuda.py`, SHA256
  `644f782b50ab124dc9d70564a24a06d5723b3f7386332607517a9ca1ad4ca370`.
  This is retained acquisition provenance, not a fingerprint requirement on
  future production refactors.
- CUDA 12.9.86, GCC 12, Release, no fast compilation, verified sccache 0.16.0.
  Build job 17644 and finalization job 17662 used n1 Slurm allocations.
  Both AOT-profile tests and all three compiled final-state/force-convergence
  tests passed. Complete CMake settings, cache statistics and actual verbose
  launcher command samples are retained. Hessian AOT leaves and stationary
  CPU-force AOT are disabled in these bounded qualification builds.
- Final n1 job 17670: RTX 5090, driver 580.95.05, SM120, `main/gpu:5090:1`.
- Final n2 job 5324: RTX PRO 6000 Blackwell, driver 595.99.02, SM120,
  `main/gpu:pro6000:1`. Both jobs have finite 25-minute limits and genuinely
  successful driver exits. Slurm visibility is preserved. Actual CUDA-driver
  UUIDs, rather than an assumed GPU index, bind observations to the inventory.
- For each profile, both cards execute the **same ELF bytes**. Auto SHA256:
  `3a474081c78a59b3c9c3cbe0f90d93b5412a575f15f0503348bdc2443e37af2e`;
  portable SHA256:
  `32828213de7d0122de423dbc913bba85ff4c1cdbe53ace21c663b79672d667a4`.
  n2 uses the recorded isolated host runtime, not a source rebuild. No A100,
  H100, different-SM, release-distribution startup or speedup claim is made.

`qualification.json.xz` retains raw references and observations, acquisition
scripts, build/source receipts, numerical work counts, device metadata and
failed-setup observations. Large ELF files and complete compile logs remain in
ignored local acquisition storage; no Release or external backup was published.
The aggregate evidence cap is unchanged. `storage-compaction.json` records
94,140 bytes reclaimed solely by removing insignificant JSON whitespace from
five historical records, preserving every numeric/string token and parsed value.
Original byte identities and their existing-Git recovery revision are retained;
no scientific observation, failed case or gate was deleted or changed.
Master reconciliation preserves its independent compressed-manifest migration
for one of those records. Two additional water records lose only whitespace;
another four records make room for the independent integrated and performance
acquisitions. The twenty currently minified paths save 258,449 bytes. The storage
receipt retains the original five-entry history, documents the superseding
transport, and preserves all original byte identities and numeric lexemes.

## Integrated scientific qualification

Master reconciliation `e9f3031ae7f324111ad5c84cf1549ac4a75f4bb6`, based on
`8f4166f0c6a692045eaf568c814b868aa1057482`, has native source identity
`1615b290a1e7d706a1e360036b410d490d73b1835d5ac77f1de1ef4d937d8d40`.
The bundle retains this as a separate `integrated_qualification`; the earlier
source-bound measurements above are unchanged, not relabeled.

Fresh Release/no-fast build and acquisition jobs 17672 (n1) and 5325 (n2)
successfully repeat the complete 96-case / 576-endpoint campaign, including
480 force endpoints, with the same frozen test recipe. Both cards execute
auto ELF SHA256 `c5b825c94b4a57ef157e05a0e019ba741e8e68596b76488467288d9d99df598b`
or portable ELF SHA256
`1a13249bb60e7f2cbd5934be5aef49b28f99c565780a49a24063b3ed6f0a2eed`.
Maximum absolute energy, force-component and translation errors are
`1.1220890883123502e-10`, `4.8487097220302824e-9`, and
`2.3581137043038325e-13`, respectively. All 32 original initial-TZVP cold/warm
observations pass, maximum force error `1.3859449820435854e-9`.
Both compiled profile tests and all three compiled numerical tests pass;
the complete host convergence/error/retry suite passes all 246 tests in
finite n1 job 17683. Replay validates both complete bundles independently.

**Performance nonregression is an additional blocking acceptance requirement.**
These scientific observations do not contain endpoint wall-clock measurements
and cannot establish it. In particular, the original cold solve increases
from 9 to 12 iterations. The repair must remain Draft until source-matched,
balanced fresh-process endpoint comparisons satisfy that requirement. Originally
correct master endpoints cannot be excluded just because the repair is slower;
incorrect master force results cannot enter a valid speed ranking. Neither
scientific gates nor public tolerances may be relaxed to obtain a speed claim.

## Performance acceptance: failed, do not merge

The retained `performance.json.xz` is a **negative result**, not an
accepted default promotion. Finite Slurm jobs 17686 (RTX 5090) and 5329
(RTX PRO 6000 Blackwell) each complete 96 fresh processes / 576 endpoints.
For every card/profile/case, ordering is ABBA then BAAB (A=master, B=repair),
giving four distinct fresh pairs. There are 1,152 endpoint observations in
total; all 576 repair endpoints pass the unchanged scientific gates.

The baseline is actual master `8f4166f0c6a692045eaf568c814b868aa1057482`, native
identity `1832792c9425c24a67a436fa117928011d1291fc43124e7e2c2edc25d2811101`.
Verified-cache Release/no-fast build job 17685 creates auto ELF SHA256
`a34600ce6f12a5308682974fc3a02e516fae25be954dae0b73fcd5bf1ad59a1f`
and portable ELF SHA256
`162c53b6834aa3800bd9193bee395c32701c546cd0587434b825be482a5f68aa`.
Both cards use those same baseline bytes and the same integrated repair bytes
identified above. Native probes must match independently hashed Python/source
trees; labels alone cannot establish a baseline.

Complete synchronous prepared execution includes native work and host result
export. Cold also includes preparation; first-use additionally includes public
package import and Calculator construction. Cleanup is recorded separately.
Reference inputs/checks, probes, ELF hashing and serialization are outside timing.
Actual iterations/Fock builds, optional native precision/quartet diagnostics,
warm-state flags, all energies and every force component are retained. Unexposed
force-evaluation counts remain unknown rather than fabricated. No kernel-only,
peak-memory, release-distribution startup or cross-card GPU-speed claim follows.

Representative **scientifically valid** moved-geometry endpoints, means in ms:

| Card | Profile | Endpoint | Master | Repair | Slowdown |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 5090 | auto | Direct RHF / SVP | 351.660 | 433.150 | 23.17% |
| RTX 5090 | portable | Direct RHF / SVP | 205.435 | 258.903 | 26.03% |
| RTX PRO 6000 | auto | Direct RHF / SVP | 347.345 | 435.566 | 25.40% |
| RTX PRO 6000 | portable | Direct RHF / SVP | 205.871 | 257.130 | 24.90% |
| RTX PRO 6000 | auto | Direct RHF / TZVP | 1300.553 | 1580.575 | 21.53% |
| RTX PRO 6000 | auto | Matched DF RHF / SVP | 36.352 | 40.482 | 11.36% |

For example, the PRO 6000 auto/SVP moved ratio is `1.25399`, paired-bootstrap
95% interval `[1.25147, 1.25751]` (20,000 resamples, seed 3092). Direct moved
work grows from 8 iterations / 10 Fock builds to 12 / 13; matched DF moved
work grows from 8 / 9 to 12 / 13. This is a real work-amplification regression,
not the invalid original TZVP cold result being ranked. Every other raw process
also enters replay; the table is not a selected acceptance population.

Of 84 profile/case/phase-plus-first-use groups per card, replay identifies
23/21 positive-lower-bound regressions, 46/36 inconclusive groups, 9/21
nonregression groups, and 6/6 scientifically invalid old groups on n1/n2,
respectively. Invalid old results remain failures, not speed wins. A confidence
interval crossing one is inconclusive, not a nonregression pass. No slowdown
allowance is introduced. **Both cards fail performance acceptance.**

The first baseline build attempt exposed a backdated-archive/Ninja timestamp
trap: unchanged inventory membership and older mtimes left candidate objects
and identity in place. This was detected from actual source/ELF probes before
interpreting timings. Allocation 5328 was stopped; its partial wrong-baseline
records remain under `rejected_setup`, outside the valid cohort. Refreshing the
four content-different native input mtimes and rebuilding through job 17685
produces the correct distinct identity. The compiler cache was not cleared,
disabled or given an invented key. This setup failure is not a production pass.

PR #3108 remains Draft: scientific repair works, but the strict-only prototype
does **not** satisfy the user's no-regression condition. Do not merge it or
close #3092 as fully resolved based only on the preceding scientific results.
Future optimization must retain the same public controls and scientific gates;
re-enabling energy-only force publication is not an acceptable speed fix.

Replay every retained endpoint and timing group without a GPU:

```bash
python benchmarks/results/hf-force-stationarity-20261117/performance_replay.py
```

`performance_acquire.py` preserves the actual acquisition recipe; the bundle
also retains the balanced driver, raw reference-input bytes, verified build
receipts and the rejected setup. This historical recipe is not a production
source-body fingerprint requirement on future refactors.

## Coverage and unchanged gates

For each card/profile, four fresh pytest processes execute six cases:
Cartesian Direct RHF water, Direct UHF OH and explicitly matched DF RHF water,
each with def2-SVP and def2-TZVP. Each case covers cold, frozen warm, moved,
moved warm, energy-only and energy-to-forces calls. Warm reuse is asserted.
Production imports of PySCF are forbidden; independent references run in
separate processes. Stock production controls are omitted and then checked as
energy `1e-9`, density `1e-5` and screening `1e-12`.

| Quantity | Unchanged gate | Maximum observed error |
| --- | --- | --- |
| Absolute energy, Eh | `1e-9` | `1.1220891e-10` |
| Maximum force component, Eh/Bohr | `1e-7` | `4.8487086e-9` |
| Maximum translation residual, Eh/Bohr | `1e-7` | `2.2470914e-13` |

All **96 cases / 576 endpoints**, including **480 E+force endpoints**, pass.
The original TZVP initial cold/warm case passes in all 32 observations, maximum
force error `1.3859373e-9`; initial cold SCF now takes 12 iterations rather than
9. The earlier-source negative regression remains a failure at `6.7580089e-7`.
Its receipt predates the final combined DF recipe and is labeled accordingly.

The DF oracle matches the explicit Cartesian auxiliary model and native default
relative metric cutoff `1e-10` in the unit-self-overlap auxiliary gauge. PySCF/CINT
independently constructs the integrals and spectral factors. Complete reference
forces use reconverged displaced energies at five-point steps 0.01/0.02 Bohr,
requiring unchanged retained rank and step agreement within `5e-8`, followed by
Richardson extrapolation. Raw displaced energies are retained and replayed.
This is not qualification of arbitrary/default auxiliary choices or fitting
error relative to direct HF.

Initially spherical/Cartesian-mismatched and full-rank DF oracle comparisons
remain separate failed setup observations, not relabeled production passes.
Likewise, the earlier host-cache-prerequisite failure retains its nonzero exit;
the full scientific job was rerun with verified sccache and exited zero.
The expanded host convergence/error/retry suite also passed all 246 tests in
finite n1 job 17671; force owners retain the same typed-error and arena-cleanup
guarantees without re-enabling relaxed convergence.

## Reproduction

Replay the raw scientific gates without loading a native library or GPU:

```bash
python benchmarks/results/hf-force-stationarity-20261117/replay.py
```

For a new source-matched Release library, run the device test through Slurm,
with matching Python/host-runtime dependencies available. Repeat for each ELF
and supported card; do not override scheduler-assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:10:00 \
  env -u GENERATIVEQC_CUDA_RHF_CONVERGENCE \
  GENERATIVEQC_RESOURCE_CUDA_TEST=1 GENERATIVEQC_PROFILE=off \
  GENERATIVEQC_LIBRARY=/absolute/path/to/libgenerativeqc.so \
  GENERATIVEQC_SCF_ORACLE_RECEIPTS=/absolute/path/to/new-receipts \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_hf_default_force_stationarity_cuda.py
```

Future source changes require new acquisition, not reassignment of these
historical source-bound observations. Historical failed profile-qualification
data elsewhere in the repository remains failed and unchanged.
