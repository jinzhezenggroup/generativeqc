# #2474: exact warm-seed occupied RI-K qualification

This record qualifies the opt-in candidate on issue source
`36574095876e4d12a3945b39cef72f6c77895f00` plus `measured-source.patch`.
Baseline and candidate use the **same** linked library, SHA-256
`b2d38025066fcece1afdb23f0dd1e14879208bc51fe7b2d490ff32e68507bc9c`;
only `GENERATIVEQC_CUDA_KS_WARM_FACTOR=0/1` changes. This is not a claim that the
older, independently dirty publishing checkout supports M06-2X DF.

## Scientific and execution boundary

- M06-2X, PBE0 and B3LYP RKS, strict FP64, spherical retained def2-SVP and explicit
  cc-pVDZ-JKFIT snapshots, unpruned moving `GridSpec(48,16,32)` in Bohr.
- Identical DF J **and** K Hamiltonians, relative metric threshold `1e-10`,
  energy/density tolerances `1e-12` / `1e-10`, screening `1e-12`, maximum 200
  iterations. No Direct substitution, convergence relaxation or skipped physical
  closure. GPU4PySCF references use full DF, grid and auxiliary-basis response,
  `conv_tol=1e-12`, gradient tolerance `1e-10`, screening `1e-14`.
- All accepted current-source device work runs on n1's Slurm-assigned RTX 5090,
  CUDA 12.9.1, driver 580.95.05, eight allocated CPU cores, preserved device
  visibility and finite time limits. Compilation uses verified sccache 0.16.0;
  actual launcher commands and before/after counters are retained.
- Each independent population retains cold, five frozen warm, moved geometry
  (last atom +0.01 Bohr in x), and five moved-warm **energy plus analytic-force**
  endpoints. Native cold includes Calculator construction, preparation and the
  complete host-returned E/F call; input loading and provenance hashing are outside
  the timer. Moved rebuilding stays inside execute.

## Clean complete-endpoint results

There are 216 accepted clean native endpoints: both arms, all three functionals,
3/48/96 atoms, all 12 calls. Nine independent reference populations retain 108
complete endpoints. Maximum clean errors are `3.638e-11 Eh` and
`1.635e-10 Eh/Bohr`, versus gates `1e-8` / `1e-7`. No failed physical endpoint is
removed from these populations. All samples and complete SCF histories are retained
in `samples.json` and `histories.json`; references and raw traces have hash receipts.

96-atom warm values are medians of **five complete calls**, not per-build times:

| Functional | Frozen warm baseline → candidate, s | Fock builds | Moved-warm baseline → candidate, s | Fock builds |
| --- | ---: | --- | ---: | --- |
| M06-2X | 36.562 → 21.403 | 3 → 1 | 36.538 → 21.563 | 3 → 1 |
| PBE0 | 22.821 → 12.501 | 5 → 1 | 19.343 → 12.506 | 3 → 1 |
| B3LYP | 20.498 → 13.206 | 3 → 1 | 20.476 → 13.213 | 3 → 1 |

Cold and moved controls each contain one complete call per arm, not a statistical
cold-speedup claim:

| Functional | Cold baseline → candidate, s | Builds | Moved baseline → candidate, s | Builds |
| --- | ---: | --- | ---: | --- |
| M06-2X | 149.582 → 149.942 | 18 → 18 | 128.606 → 128.884 | 15 → 15 |
| PBE0 | 68.655 → 68.590 | 18 → 18 | 60.314 → 60.486 | 15 → 15 |
| B3LYP | 68.872 → 68.895 | 16 → 16 | 63.932 → 63.974 | 15 → 15 |

At 48 atoms the incumbent ordered/tiled provider does **not** expose a complete
retained occupied projection. The candidate correctly keeps the dense first-K
fallback: all frozen and moved-warm calls remain one Fock build. No 48-atom speedup
is claimed, and no full U allocation or molecule-size whitelist is added to obtain
one. `summary.json` includes every cold/warm/moved control at all sizes.

## Additive first-build and full-endpoint attribution

For the original 96-atom M06-2X three-build branch, baseline starts from warm
density generation 18 but has no retained factor generation. Candidate starts from
the **same density-producing factor**, generation 18 / solve epoch 1, and restores
it for solve epoch 2. Frozen replays keep that seed generation unchanged. It is
not a final-Fock canonical frame substituted for the consumed physical density.
Provider scratch U is freshly recomputed, not borrowed across solves.

The first baseline energy change is `3.325224230570919e-12 Eh`, above the unchanged
`1e-12` gate; its first two builds fail that energy gate. Candidate's first
independently rebuilt physical F[D] has zero energy change against the incumbent
compensated electronic-energy anchor and passes the existing density/residual
gates. The controller still owns that anchor and its compensation. Subsequent
physical builds are removed **because the unchanged gates pass**, not skipped.

An additional ABBA experiment switches only the first-K policy on one exactly
frozen, factor-backed owner. Each geometry has five samples per arm; no sample is
discarded. `interleaved.json` retains order, complete E/F times, work counts and
the repository's noise assessment. Its seeded cold setup is not a cold baseline
comparison and does not replace the independent populations above.

The following **single representative traced warm-0 pair** is additive on the
host completion clock. These diagnostic times are not clean performance samples:

| Nonoverlapping scope | Baseline, s | Candidate, s |
| --- | ---: | ---: |
| KS begin + finish | 0.000879 | 0.000261 |
| KS J/K, including nested provider work | 5.137631 | 0.558756 |
| KS XC | 16.060799 | 5.352798 |
| Other KS work + final diagnostic wait | 0.224444 | 0.035738 |
| One-electron stationary response | 0.518474 | 0.518246 |
| DF force response, two calls | 2.512723 | 2.514682 |
| Untraced complete-endpoint remainder | 12.337876 | 12.316433 |
| Complete E/F call | 36.792827 | 21.296913 |

The remainder is explicit uninstrumented E/F/host work, **not a measured CPU-only
cost or an inferred XC-force breakdown**. On the alternative GPU-event clock,
first K changes from dense `4.009605 s` to occupied `0.548070 s`. The first KS
build changes `9.411188 → 5.947279 s`; the baseline's additional two physical KS
builds cost `6.001204 + 6.010354 s`. This separates the approximately 3.462-second
first-K contraction saving from the approximately 12.012-second convergence-branch
change. Do not add these GPU intervals to host costs, or add nested RI-K/projection
roots again to `ks_jk`. Per-call additive ledgers are in `work-ledger.json`.

Both arms retain exactly 175,688,520 state bytes, 92,808,608 XC bytes and
14,299,355,893 provider bytes on this M06-2X case. Numeric arena growth is zero.
Explicit owned KS synchronizations change from three to one; diagnostic profiler
fences/waits are separately retained, not called ordinary runtime synchronizations.
No full-process allocator peak or exhaustive CUDA-library API count is claimed.

## Matrix and fallback gates

`matrix-audits.json` contains actual first-warm dense-versus-occupied K/Fock
comparisons for all three 96-atom functionals, on both geometries and all five
frozen replays. Conservative error ceilings use units of `1e-16`; the strict
matrix gate is `2e-11`. Separate audit runs include an extra dense K, temporary
host matrices and an occupied replay restoring fresh U. Neither those times nor
their extra work enter the ordinary ledger or clean timing table.

The current-source linked focused PBE0 RKS/UKS regression passes: reported GPU K
error `2.22045e-16`, Fock error zero, independent CPU-DF error `6.66134e-16`. It
covers frozen seeds, stale tokens, intervening same-shape provider writers,
non-idempotent imports, failed-begin recovery, cold and changed geometry. The
retained `focused-test.cpp` fixes the inherited bare-PBE-plus-quarter-K fixture to
the valid PBE0 composition; it links the same measured library without mutating
its frozen source tree. Host executable guards additionally cover owner/source,
metric/rank, precision, spin and projection-capacity refusals. The latest focused
host/source-contract run has 206 passes and six skips; the preceding broader
selection has 222 passes and six skips.

## Decision and reproduction

Retain the verified optimization **opt-in**. The measured 96-atom complete warm
endpoint wins do not authorize a universal/default selector, new U residency,
relaxed numerical gates, or a claim about the unrelated older dirty checkout.
Cold/moved, UKS, explicit/mismatched density, truncated metrics, lower precision
and unavailable complete-projection capacity retain bounded dense fallbacks.

Use a **separate clean checkout** of the measured revision. Apply
`measured-source.patch`, copy `measured-runner.py` to
`benchmarks/ks_df_warm_factor.py`, and run `sha256sum -c source-files.sha256`
from that checkout. Do not reset the publishing checkout's unrelated dirty work.
Set an appropriate CUDA 12.9/toolchain/Python environment; `provenance.json` records
the measured versions and force-artifact identities. Then, from the restored
checkout, invoke `bash /absolute/path/to/reproduce.sh build`, followed by
`reference`, `clean`, and separately `trace` for each functional and 3/48/96 atoms
(traces only at 48/96). Run `audit` and `ab` at 96 atoms. The syntax is
`bash reproduce.sh STAGE FUNCTIONAL ATOMS`; `RAW_DIRECTORY` selects a fresh ignored
directory. Each GPU stage uses finite Slurm allocations; clean/trace arm pairs
share one allocation. Native force-manifest targets are required, not optional.
`PYTHONPATH=python:. python analyze.py --raw RAW_DIRECTORY --output REVIEW_DIRECTORY`
rechecks every source/library identity, endpoint gate, history count and additive
trace before deriving the compact records. Raw run directories remain ignored.

Initial infrastructure failures, the unsupported older-source M06-2X/B3LYP
attempts, and n4's incompatible host glibc are recorded in `provenance.json` and
retained local artifacts; they are not relabeled as accepted populations.
No Direct-versus-DF timing subtraction, iteration-normalized speedup, #2464 XC
credit, release asset, external upload, commit or default promotion is involved.

## Tracked storage representation

The narrative above describes the original qualification and expanded filenames.
For retention within the unchanged repository storage limit, JSON data records
and `focused-test.cpp` are stored as deterministic `.gz` files. Use `gzip -dc`
to restore their expanded bytes; `publication.json` binds the stored bytes and
the evidence record updates only its attachment storage paths, sizes and hashes.
All numerical values, endpoint populations, source/input bytes and the original
acceptance decision are unchanged. Reproduction writes fresh, uncompressed raw
outputs into the chosen ignored directory. The later guarded default policy is
documented separately in the superseding Agent Note, not asserted by this archive.

## Lossless source-patch storage

The original source patches are retained as XZ files. Before applying a patch,
restore its original filename with `xz -dc NAME.patch.xz > NAME.patch`.
The [storage inventory](../pr-merge-storage-20261028/manifest.json) records both
stored and original byte counts and SHA-256 values. Measurement identities,
all samples, and acceptance decisions are unchanged.
