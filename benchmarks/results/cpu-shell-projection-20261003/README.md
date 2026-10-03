# CPU shell projection and AO metadata evidence, publication package

Compact, non-lossless scalar summary and portable future benchmark replay. The three compressed historical payloads are byte-identical to v2. This README and source-aliases.json supersede only their old pending-source/publication wording; scientific results and limits are unchanged. Detailed raw files, full arrays and member indexes remain local; checksums neither retrieve nor recover them. Loss of the local store would lose omitted evidence. No remote write, build or native numerical rerun occurred while preparing this package.

## Primary current-main result

The completed 05:09:41–05:20:39 UTC three-arm cohort has 54 fresh processes and all 216 endpoints, with six arm permutations for each case. `evidence.json.xz` preserves every sample's unrounded time, energy, iterations, convergence, density RMS and available scalar diagnostics/RSS. Exact source/build identities are in its primary freeze. The copied preparation freeze remains correctly labeled prepared-not-run; the separate terminal cohort manifest establishes completion.

Baseline is public commit 30dece2594be0956c95fa07d265761cf7f42c277. Measured projection is 2bda89f39661b43711506b4e9ae09ed6e3adf7fd; metadata is 8ac08f0a19929e9916807c0f46ca15604345e4b1. Fetched public aliases are projection 39b854738b7329dd7728e29536e6fad69fb068c9 and metadata 1ca2baf3849a75160aa7843ec0f68b9034fc3e13 on branch evidence/cpu-projection-metadata-source-20261003 of https://github.com/jinzhezenggroup/generativeqc. Their complete trees equal the measured trees; source-aliases.json records the parent-chain proof. The aliases were not separately measured. Historical binary reproducibility and performance on later current main are not established.

Size labels mean AOs, not atoms: HF96 is four waters/12 atoms/96 spherical AOs, 100 Cartesian AOs; HF24 is one water/3 atoms/24 spherical AOs. `hf24cart` is one water/3 atoms/25 Cartesian AOs.

For the 12-atom/96-AO case, warm median times are 8.305469 → 7.285550 → 6.914398 seconds. Projection reduces 12.2801%; metadata is 5.0944% below projection; directly measured combined reduction is 16.7489%. Median cumulative process peak RSS is 1,498,968 → 718,936 → 718,544 KiB. All 216 timing endpoints took 16 iterations. Full Fock counts/histories/operator work/runtime provider telemetry remain unavailable; equal iterations and bitwise projection/metadata energy/density-RMS scalars do not establish full-array or complete-work parity. Reference/default HF behavior is a source/build contract, not observed native provider selection.

Losses remain explicit: metadata versus projection on 3-atom/24-AO water is slower by 10.4325% cold, 3.9355% warm and 2.4377% moved; combined versus baseline cold is 6.6485% slower. Projection's Cartesian control is 7.5075% slower cold and 2.4306% slower moved. Per-comparison loss counts and matched medians remain in the ledger; actual samples allow recomputation. No older or separate comparison gains are compounded.

Current-main science includes projection 67 native/seven focused Python controls, HF96 four prepared/two unbounded physical-export rows, and the separate 60-pass Cartesian DFT extension (20 CUDA deselected). Metadata has 67 native/47 focused Python controls and four prepared rows matching projection scalar metrics and retained density bytes. Prepared iterations are 16/4/15/18, not the timing cohort's all-16 sequence. These checks do not extend direct spherical/large-force or bounded-memory qualification.

## Historical results and review versions

The 432 older balanced endpoints and 108 intrusive startup samples remain separately attributed to the older bd/854 trees with their unpublished HF-policy ancestor. All 13 DFT failures, identical standard-control UHF24 torque failures, tighter-control distinctions, initial cache confounding and pilot/control losses are retained. New main endpoints do not repair those failures or transfer old binary/full-array claims to new owners.

The independent report included here is the appended current-main audit: SHA-256 7675a38d91a315705de148bd46e7b3880c8ed89fe5b7b3c94f446e93f1c9a79f. The prior report was 54880f5e0d47e0e260d0edde755081bf511e2b1ef998ce383ebc14c0736b864c. Its mutable source pathname changed; v1's index and exact staged old bytes remain untouched. Thus checking the old index against today's report pathname correctly fails. Versioned old/new report copies are preserved locally. Scientific/measurement source files remain frozen.

Warm means process-warm/fresh-owner, not density reuse. Cold means process-first, not cold OS page cache. RSS is cumulative, not phase-local. Source-derived work counts are not dynamic telemetry. No universal no-regression, default promotion, full HF96/PBE96 force/FD, derivative-Jet/physical-OOM, GPU or unique-density OH qualification follows.

## Decode and future replay

Repository build instructions require both `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` and `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`, including CPU-only builds. Add the CUDA launcher to the representative command in the preserved extracted recipe; verify generated compile commands use ccache.

Run `python3 decode.py --output NEW_DIRECTORY`. It verifies payload hashes/counts and expands evidence, review and the portable source bundle. It refuses existing output directories and does not execute bundled code. See the extracted `reproducer/RECIPE.md` and config example for explicit paths, source-tree/build preflight and optional future execution. The new driver records fresh actual build/library/harness identities and requires exact measured trees; it does not query private ancestors or assign old binary hashes to new builds. Its endpoint script is unchanged from the measured cohort. Mock/static checks and non-executing source/build/path preflight are the only new validation; the portable adapter has not run native calculations. validation.json.gz is the standard scientific envelope, and publication.json binds the full package. Formal performance/default promotion remains INCONCLUSIVE. SHA256SUMS covers every file except itself and publication.json; publication.json binds SHA256SUMS too. The decoder verifies all members before creating output and never executes recovered sources.

The verified public aliases resolve the source-availability prerequisite. Remaining fresh-replay prerequisites include local builds, installed documented dependencies/provider discovery, and an isolated timing lane. The unchanged extracted recipe's pending-alias wording is superseded by source-aliases.json. Full historical scientific replay additionally needs the original local arrays and harnesses. The three bundled hash-pinned oracle JSON receipts support comparator energy/input gates; their referenced density arrays are omitted.

All 433 newly scoped raw files (47,279,577 bytes) were indexed locally; only index/parent-manifest anchors are included here. V1's 818-member index remains local too. Package sizes are actual byte counts, not a storage request. This new package requires a fresh repository retention/review check on its actual integration base. Cleanup #1758 and its prospective headroom are not assumed landed here; no cap increase, extra cleanup or upload is part of packaging.

## Later current-main integration

The decoded `integration.json` (stored as `integration.json.xz`) records a separate current-main plus storage-prerequisite build,67 native/47 focused Python checks, four independent HF96 prepared states and two physical-reference exports. It is not a newly timed arm. Its source/library identity and raw-index checksum remain separate from the immutable measured cohort. The prepared lifecycle iterations are16/4/15/18, unlike the all-16 timing cohort.
