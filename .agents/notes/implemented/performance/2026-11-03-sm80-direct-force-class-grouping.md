# Decision: group SM80 Direct force classes after one bounded scan

Status: implemented

The architecture-specific policy is superseded by the
[portable packet-grouping revision](2026-11-03-portable-direct-force-class-grouping.md).
This note retains the original version's measurements; they do not qualify the
portable version's changed SM120 kernel.
Date: 2026-11-03

## Problem

After the guarded automatic-tile change, the default fully Direct water-96
endpoint still spends about 2.18 s (PBE0) or 4.21 s (r2SCAN-3c) in the scalar
low-order force consumer on A100. Its geometry-ordered queue can mix several
large class-specific device callees in one warp. Earlier angular partitioning
repeated candidate scans; an entry-only register cap failed the callee resource
contract. Neither is an acceptable replacement for the single bounded scan.

## Decision

Only the SM80 spatial low-order kernel groups already admitted tasks by their
canonical s/p/d shell class. A 21-bin shared histogram becomes exclusive queue
offsets; admitted lanes scatter into disjoint class spans after synchronization.
The existing 256-descriptor queue still receives at most one 128-thread packet.
The additional persistent storage is 84 shared bytes per CTA, not another
quartet inventory. No task is screened or admitted twice.

The existing scalar/warp drain, generated roots, source coefficients, spin and
seed handling, primitive order, AO tiling, precision, force scatter, grids,
solver controls and order-six/seven/eight ownership are unchanged. Atomics may
accumulate tasks in a different order, so scientific acceptance uses explicit
observable gates rather than cross-run bitwise force identity.

The compile-time guard is exactly `__CUDA_ARCH__ == 800`. Other architectures
retain their qualified queue policy. For the initial pre-integration pair,
the complete SM120 CUDA fatbin and resource inventory are byte-identical
between the two Release packages; that pair's fatbin SHA256 is
`b8f569785421640d4887746f5e4e6d17d95db958f254d7871a0a5b3cefcfe050`.
This is a code-preservation check, not a substitute for real 5090 endpoint runs.

Reduced within-warp class mixing is the performance hypothesis. No hardware
branch/occupancy-counter or register-reduction claim is made.

## Identities and protocol

The baseline is PR #2945 at `1c9f17444ae73641aead033a288b54537bba55c5`.
Scientific source identities are:

- Baseline: `f8be7954bd8cc20a41b2b41ba704e76ab4d17b569f28e10b8c0cc3da109fdb1c`.
- Candidate: `7029d2b62165f51a5ce71cdb5791f97918f45bb7d79646971d7c9d9c0595b4fd`.

Both architectures use full normal Release builds, fast compile OFF, tests OFF
in the production image, stationary AOT `auto`, retained compiler caches and
the same main/auxiliary native-library layout. Endpoint owners verify their
source inventory, package hashes and the source identity returned by the loaded
ELF. PySCF/GPU4PySCF imports are blocked in native timed endpoints.

Before device execution, `acceptance.json` declares three cold owners and nine
frozen-own-cold-density warm calls per image/method, alternating image order,
six methods, a maximum cold/warm median ratio of 1.05 on both GPUs, and at least
2% warm improvement for the two primary A100 methods. Cold includes calculator
construction, preparation, SCF, analytic forces and host export. Diagnostics
run separately and are excluded from timing populations.

Matched-phase gates are energy/atom `1e-10`, force maximum `2e-9`, RMS `1e-9`.
Independent default-grid gates are `1e-5`, `1e-4`, `3e-5`, respectively. Original
GPU4PySCF receipts from LSF 224875/224903 are retained without changing producer
identities; all native samples are checked against all applicable reference
samples. This does not publish a new cross-engine performance ranking.

## Initial screen, not final qualification

LSF 225002 and Slurm 15718 each complete three methods, one fresh cold owner and
three frozen-density warm calls per image, followed by separate diagnostics.

| A100 Direct water-96 | Baseline cold / warm (s) | Candidate cold / warm (s) |
| --- | ---: | ---: |
| PBE0 | 14.7421 / 5.3394 | 14.5650 / 5.1332 |
| r2SCAN-3c | 19.0208 / 10.3985 | 18.6170 / 9.9942 |
| PBE | 7.9272 / 3.0225 | 7.9111 / 2.9548 |

The primary warm reductions are 3.86% and 3.89%. The separately traced low
consumer decreases from about 2.18 to 1.97 s and 4.21 to 3.78 s; these are
diagnostic CUDA-event times, not complete endpoint times. Both primary census
receipts are exactly equal, including traversal, screening, admission,
execution, AO-component and primitive counters. Solver/Fock/application counts,
grid sizes and tile choices are also equal; unavailable counters remain null.

The 5090 initial screen stays within 0.8% cold and 0.1% warm. Both screens pass
matched-phase and independent default-grid observable gates. Fresh Libcint
checks cover pure-J RKS, hybrid RKS/UKS, and Cartesian/spherical outputs with
the spatial schedule explicitly enabled. A permanent compiled host test checks
empty/partial/full queue permutations; a permanent real-GPU oracle wrapper
forces the spatial owner even on small systems.

## Build provenance failures and rejected alternative

An initial one-source-storage candidate is removed before device measurement:
the full-range owner already disables the unused exchange traversal and resets
both public channels. It cannot improve these endpoints.

LSF build 224998 produces the correct baseline, but its candidate CUDA object
is stale: extracting a candidate tar with an older mtime lets Ninja rebuild only
the API identity. A loaded identity alone does not prove CUDA recompilation.
The stale-image screen 225000 is cancelled and excluded. Repair build 225001
touches the changed inputs, visibly compiles the CUDA translation unit, checks
that its object digest changes, and repackages the candidate before 225002.
The stale package, cancelled receipts and every failure log remain retained.

## First full qualification, before properties integration

LSF 225005 completes the six-method, three-round timing matrix, diagnostics
and moved/restored phases, but its later 129-atom baseline owner exceeds the
300-second wrapper budget. Its incomplete receipt and exit 124 remain retained
and are explicitly excluded from acceptance, not converted into a passing
owner. LSF 225011 repeats both 129-atom images with 900 seconds per owner:
baseline cold/warm are 203.996/181.410 s, candidate 203.857/181.241 s. This
changes the execution budget only, not scientific or solver controls.

LSF 225013 completes the other four method censuses, the 48-atom fallback
and explicit 32-GiB resource owners before a sanitizer startup failure. An
audit of all completed owners across these three jobs passes: 54 owners,
27 matched cold/warm pairs and two independent-reference moved-geometry
pairs. All six complete force census populations are exactly equal; null
unavailable counters remain null. The incomplete 225005 owner stays listed
separately in the audit output.

Slurm 15719 completes its full matrix, 15721 supplies the remaining censuses,
and 15720 runs the six permanent forced-spatial independent GPU cases. All
48 5090 owners and 24 matched pairs pass, with six equal census populations.
Both architectures retain three cold and nine warm samples per image/method
(144 clean complete E+F calls per GPU), not only the initial screen.

| Method | A100 candidate/baseline cold | A100 warm | 5090 cold | 5090 warm |
| --- | ---: | ---: | ---: | ---: |
| PBE0 | 0.987358 | 0.955767 | 1.004970 | 1.001979 |
| r2SCAN-3c | 0.977790 | 0.961917 | 0.999470 | 0.997376 |
| PBE | 0.996995 | 0.978626 | 0.992570 | 1.000058 |
| B3LYP | 0.986885 | 0.962850 | 0.999967 | 1.000529 |
| r2SCAN | 0.996714 | 0.979777 | 1.000549 | 1.000186 |
| M06-2x | 0.988086 | 0.966813 | 0.999105 | 0.999930 |

All predeclared timing, matched-phase, independent-observable and semantic-work
gates pass for these populations. These measurements belong only to the
`f8be...`/`7029...` source identities above, not the later integrated sources.
The known restored-versus-own-cold strict cross-trajectory limitation remains
reported separately; matched-phase image parity and independent physical gates
are not substitutes for that distinct diagnostic.

## Integrated qualification

While qualification ran, another actor fast-forwarded PR #2945 through
`63c898ede6b2eedc0ad5d47e00863e5bb41d8c2c`, merging electronic properties
PR #2938. Preserve that commit rather than force-pushing over it. The local
worktree fast-forwards to the same parent and keeps this optimization as a
separate change. Integrated scientific source identities are:

- Baseline: `d66089ee1c670a134d182b660cf11c7cab1e5ba7412a2cf48fdf97affbc31bd9`.
- Candidate: `b9c7832c8895c707ae7a7f48dcede637ef68a5438bf8e7665ae4d2f16056903d`.

Do not relabel earlier measurements. Full fresh Release SM80/SM120 builds
and the same predeclared six-method, three-round endpoint population qualify
the integrated source separately. LSF build 225014 visibly recompiles the
changed CUDA translation unit for both arms; its package manifests retain
normal Release, SM80 real code, fast compile OFF and production tests OFF.

PR #2945 merges on November 3, 2026 at 15:33:27 Asia/Shanghai as
`6affc9f92b6781ffc2072d5fc8696d5ae54e3747`. Its complete Git tree is exactly
the qualified parent's tree, `bf4d55115177fbbb3a32304620f3e1addded520b`.
The follow-up branch starts from that merged master without changing either
scientific source identity; the original measurement parent stays recorded
as `63c898...`, not relabelled as a newly measured commit.

The shared SCF structural checker classifies the new prefix helper as an
independent leaf. Only its exact header is admitted by Direct consumers;
this does not relax provider, solver or numerical ownership boundaries.

### Integrated A100 endpoints

LSF 225015 completes three cold owners and nine warm calls per image/method,
all six methods, plus separate complete diagnostics for every method. LSF
225044 completes moved/restored geometry and the 48/129-atom and 32-GiB
controls before its unfiltered vendor-API diagnostic described below. All
completed owners pass: 54 owners, 27 matched cold/warm pairs and two moved
pairs, with six exactly equal complete force-work census populations.

| Method | Baseline cold / warm (s) | Candidate cold / warm (s) | Cold / warm SCF iterations and Fock builds |
| --- | ---: | ---: | ---: |
| PBE0 | 14.882939 / 5.347296 | 14.649481 / 5.136151 | 9 / 1 |
| r2SCAN-3c | 19.179297 / 10.412776 | 18.785761 / 10.013885 | 10 / 1 |
| PBE | 7.949843 / 3.024289 | 7.857687 / 2.956632 | 10 / 1 |
| B3LYP | 15.022868 / 5.448115 | 14.844793 / 5.174198 | 9 / 1 |
| r2SCAN | 8.644846 / 3.312693 | 8.638528 / 3.247529 | 9 / 1 |
| M06-2x | 17.210695 / 5.590986 | 16.977102 / 5.368496 | 10 / 1 |

Both iteration and Fock-build counts equal the last column, in both images
and all three rounds. PBE0/r2SCAN-3c warm complete-endpoint reductions are
3.95%/3.83%; the other methods reduce warm time by 1.97–5.03%. All cold/warm
median ratios pass 1.05, and both primary warm ratios pass 0.98.

Complete per-force work is unchanged in cold and warm phases:

| Method | Candidates | Enumerated | Admitted | Primitive-pair products |
| --- | ---: | ---: | ---: | ---: |
| PBE0 | 161862824 | 161280352 | 92223688 | 695763928 |
| r2SCAN-3c | 521098784 | 512690844 | 243123674 | 1444681762 |

Aggregate executed-shell and AO-component totals remain unknown where any
class receipt is null; do not replace them with partial sums. Every available
per-class execution counter, traversal and admission counter is compared
exactly, including the other four methods. Independent default-grid and
strict matched-phase observable gates retain their original thresholds.

### Integrated RTX 5090 qualification

The fresh integrated Release pair has byte-identical complete CUDA fatbins,
resource inventories and core-library host `.text`. The fatbin SHA256 is
`f1f9c5a3c3069c3a4902f87827de104bc23740dfd31788cc41b7b6def5ef3aa0`;
the host-text SHA256 is
`076c0237926d31737e6fd12e9d574388d1c9204f6e4839b278cd3efcbc02f195`.
This compares the integrated pair with itself, not with pre-properties images.

Slurm 15722 initially requests only 30 minutes. The earlier full qualification
already needed 30 minutes 48 seconds with fewer diagnostic owners; increasing
the running allocation is denied. Cancel 15722 before evaluating its timing
ratios, retain its entire partial population, and restart the same frozen
protocol as Slurm 15723 with a finite one-hour limit. No scientific control
changes. A cancelled shell's EXIT trap can write zero, as 15722 does: that is
not completion or acceptance. Scheduler cancellation and the incomplete
three-round population keep this archive excluded.

Slurm 15723 completes the unchanged protocol in 36 minutes 31 seconds,
`COMPLETED`, exit 0:0. Its 144 clean complete E+F calls contain three cold
owners and nine warm calls per image/method. All 48 owners and 24 matched
pairs pass strict observables and independent default-grid gates, and all
six force census populations match exactly. Fresh forced-spatial independent
oracle tests pass six cases per image on both GPUs.

| Method | Baseline cold / warm (s) | Candidate cold / warm (s) | Candidate/baseline cold / warm |
| --- | ---: | ---: | ---: |
| PBE0 | 19.123748 / 7.103968 | 19.044681 / 7.114728 | 0.995866 / 1.001515 |
| r2SCAN-3c | 31.260175 / 13.953886 | 31.239335 / 13.995267 | 0.999333 / 1.002966 |
| PBE | 13.451679 / 5.158884 | 13.463261 / 5.160599 | 1.000861 / 1.000332 |
| B3LYP | 19.473812 / 7.322862 | 19.464715 / 7.314692 | 0.999533 / 0.998884 |
| r2SCAN | 15.694132 / 6.226734 | 15.687961 / 6.221344 | 0.999607 / 0.999134 |
| M06-2x | 23.218653 / 7.714154 | 23.160151 / 7.724964 | 0.997480 / 1.001401 |

Every cold/warm ratio passes the predeclared 1.05 gate; the largest observed
increase is 0.297% warm. This is no material regression in the stated water-96
population, not a universal zero-overhead claim. Complete byte preservation
and measured endpoints are separate checks, and both pass.

### Scoped memory/race validation, with vendor diagnostics retained

Compute Sanitizer 12.9 launch helpers require a newer glibc than the ECNU
host. Merely wrapping the parent misses its child launchers; exporting private
glibc to system-loader children fails. Rewriting the main vendor executable
also fails before CUDA. The working setup retains the main tool and injection
libraries byte-identically, repairs only interpreters of a task-owned copy of
the two vendor child helpers, and invokes the original parent with the private
loader. Original/patched tool hashes, version, interpreter paths and all failed
startup attempts remain retained. The shared vendor installation is untouched.

Unfiltered memcheck subsequently reports five `cuCtxGetLimit` errors with
`CUDA_ERROR_INVALID_VALUE` from `libcublasLt.so.12`/`cublasLtCtxInit`; the
independent force test still passes. LSF 225083 first repeats **both** integrated
images without suppression and verifies exactly those five reports per image,
with no other report kinds. These unfiltered runs remain exit 99 diagnostics.

Only after that paired check, a per-experiment XML exclusion matches kind
`Api`, exact API `cuCtxGetLimit`, exact error `CUDA_ERROR_INVALID_VALUE` and
result 1. It contains no memory or race suppression, no kernel filter, and no
blanket disabling of API error reporting. Do not install it as a generic CI
suppression or claim the unfiltered process is clean. This bounds the known
vendor diagnostic for this CUDA-library/driver combination rather than fixing
or hiding it in production.

Both integrated images then complete memcheck with zero unsuppressed errors
and racecheck with zero hazards, errors or warnings. All four runs pass the
independent forced-spatial pure-J Cartesian test, require CUDA initialization,
and produce three valid complete censuses containing 2619 spatial pages.
Their baseline/candidate work records match exactly. The other RKS/UKS and
representation cases remain covered by the separate six-case oracle matrices.
This is scoped Direct-queue safety evidence, not a clean-vendor-API claim.

The integrated host/catalog suite passes 271 tests. The checked Libxc catalog
updates only current checked source/native identities; historical measurements,
qualification and `stale=True` remain unchanged. Both ownership checkers classify
the prefix helper as generic runtime scheduling, not new scientific mathematics.

Ignored local evidence is under
`/data/jzzeng/direct-autotune-20261102/.artifacts/direct-density-20261103/`.
The directory name is historical; this experiment does not change density
contraction mathematics. Reproduce using its source/package manifests,
`acceptance.json`, `reference-reuse-manifest.json`, `screen-a100.sh`,
`screen-5090.sh`, `qualify-a100.sh`, `qualify-5090.sh` and `analyze.py`.

Screen archive hashes:

- A100 225002: `18f2ec96873899c8933df641c25fedea52ef728468467b906fd9d14a7423a8a3`.
- 5090 15718: `0ed8fdba8aa887f547cf75b03c4ddda5b84355057242549f95d28b2bbd8a6e66`.
- Repair 225001: `ea2ebd8735f267e962716ca72ec77a412dd2de970b2fe2819c08a98644f3ffaf`.

First full qualification archives:

- A100 225005: `3f2d7e623c9b92cbf453193a71ee670ae9ebe9f42e0905362dfbf7916d323e35`.
- A100 225011: `df959e8318fd454e6d8ec762899fbc8165e228e9e702a3066b16d57268e22375`.
- A100 225013: `eba8bd047f9717e5fa87964971491a7d0020f80e06d5d428448701c97747859c`.
- 5090 15719: `02c439d9b08e75dcbc303f3e89e2f087bd1b829efcdb867be499a3b98683b921`.
- 5090 15721: `685036bcebdad54cdaedc0982230e26a49393e20d8b440cc3bc97580277de5e0`.

Integrated evidence is retained separately under
`/data/jzzeng/direct-autotune-20261102/.artifacts/direct-integrated-20261103/`:

- Build 225014: `7aeb37652a04650293f56d572e981f5ef861f42a6a42dd44c6d4efde25d16086`.
- A100 225015: `d8f12ad1417534782ea6afc6f3b6119beefb08045092a260d6d3f21ce1986642`.
- A100 controls 225044: `30fa247dd5c894e31b2a1b3c4144b12913ab656ffe5086d0034ebca6e683ac7d`.
- Scoped safety 225083: `3a70d92d123e66a0b7182ab6de7c510aadd527b98559446a69f95e52bab7d89f`.
- Cancelled 5090 15722: `1e41780ec7d6c0320b509350574f4ee5e683c04e2541f1215fce5e0f013d57ea`.
- Complete 5090 15723: `29129dda2f018cfcfa5374f4a06274203c5c6b12b54ce62f0dc804498569f0b6`.

Reproduce with the frozen source/package manifests, `acceptance.json`, the
unchanged reference-reuse manifest, `qualify-integrated-a100.sh`,
`qualify-integrated-5090.sh`, `safety-integrated-a100-v5.sh`,
`sanitizer-integrated-a100-final.sh`, `verify-sanitizer-api.py`, `analyze.py`,
`audit.py`, `gate-performance.py` and `audit-sanitizer-final.py`. Preserve the
failed and incomplete receipts alongside any rerun instead of overwriting them.

Do not expand the architecture guard or replace the retained queue fallback
without fresh endpoint, independent-oracle and work-count qualification.
The earlier restored-versus-own-cold diagnostic and the stock 192-atom
nonconvergent baseline remain limitations, not newly passing controls. See
`2026-11-03-a100-direct-automatic-tiles.md` for their original evidence. This
change does not claim grid-converged accuracy or a new cross-engine ranking.
