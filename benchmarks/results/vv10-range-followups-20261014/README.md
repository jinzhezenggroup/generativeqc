# VV10 range reuse and retained convergence experiments

This follows #2401/#2473 and supports #2484, #2485 and #2490. Only the automatic 4 GiB campaign replaces the default direct WB97M-V points at 3/12/48 atoms. #2426 remains open: the default 96-atom population is still incomplete.

All populations retain FP64, def2-SVP spherical shells, the 48×16×32 moving grid, 100 SCF iterations, the original 1800-second whole-point deadline, cold + five warm + moved + five moved-warm calls, and independent E/F gates of 1e-8 Eh / 1e-7 Eh/Bohr. Every warm value below is the median of all five calls.

| Campaign | Atoms | Calls | Cold / warm / moved / moved-warm (s) | Outcome |
| --- | ---: | ---: | --- | --- |
| range-ks-qualified | 3 | 12 | 7.021 / 0.425 / 5.326 / 0.425 | complete; all gates pass |
| range-ks-qualified | 12 | 12 | 30.624 / 1.792 / 19.459 / 1.794 | complete; all gates pass |
| range-ks-qualified | 48 | 12 | 255.074 / 15.738 / 214.900 / 15.668 | complete; all gates pass |
| range-ks-qualified | 96 | 6 | 1339.858 / 60.623 / — / — | exit 124; incomplete |
| range-paired-baseline | 3 | 12 | 7.738 / 0.427 / 5.784 / 0.424 | complete; all gates pass |
| range-paired-baseline | 12 | 12 | 34.633 / 1.786 / 21.103 / 1.791 | complete; all gates pass |
| range-replay-partial-qualified | 3 | 12 | 7.563 / 0.412 / 5.777 / 0.411 | complete; all gates pass |
| range-replay-partial-qualified | 12 | 12 | 35.147 / 1.791 / 21.721 / 1.794 | complete; all gates pass |
| range-replay-partial-qualified | 48 | 12 | 304.876 / 12.857 / 243.112 / 12.859 | complete; all gates pass |
| range-replay-partial-qualified | 96 | 6 | 1018.641 / 50.542 / — / — | exit 124; incomplete |
| range-replay-expanded-qualified | 96 | 6 | 950.947 / 49.353 / — / — | exit 124; incomplete |
| range-replay-full-qualified | 3 | 12 | 7.026 / 0.387 / 5.330 / 0.386 | complete; all gates pass |
| range-replay-full-qualified | 12 | 12 | 30.409 / 1.652 / 19.388 / 1.668 | complete; all gates pass |
| range-replay-full-qualified | 48 | 12 | 253.488 / 11.408 / 212.392 / 11.392 | complete; all gates pass |
| inexact-inner-launch-failure | 3 | 1 | — / — / — / — | exit 1; incomplete |
| inexact-inner-launch-failure | 12 | 1 | — / — / — / — | exit 1; incomplete |
| inexact-inner-qualified | 3 | 12 | 71.922 / 0.432 / 6.571 / 0.433 | complete; all gates pass |
| inexact-inner-qualified | 12 | 12 | 31.892 / 1.796 / 23.666 / 1.745 | complete; all gates pass |
| inexact-inner-qualified | 96 | 6 | 947.696 / 50.152 / — / — | exit 124; incomplete |
| outer-secant-qualified | 3 | 12 | 7.410 / 0.406 / 5.593 / 0.404 | complete; all gates pass |
| outer-secant-qualified | 12 | 12 | 31.613 / 1.709 / 20.355 / 1.720 | complete; all gates pass |
| outer-secant-qualified | 96 | 6 | 1056.500 / 51.026 / — / — | exit 124; incomplete |
| range-default-qualified | 3 | 12 | 7.048 / 0.386 / 5.339 / 0.386 | complete; all gates pass |
| range-default-qualified | 12 | 12 | 30.652 / 1.663 / 19.502 / 1.666 | complete; all gates pass |
| range-default-qualified | 48 | 12 | 387.507 / 15.480 / 289.643 / 15.459 | complete; all gates pass |
| range-four-gib-qualified | 3 | 12 | 7.059 / 0.387 / 5.341 / 0.386 | complete; all gates pass |
| range-four-gib-qualified | 12 | 12 | 30.388 / 1.651 / 19.602 / 1.668 | complete; all gates pass |
| range-four-gib-qualified | 48 | 12 | 253.595 / 11.406 / 212.176 / 11.392 | complete; all gates pass |
| target-tail-qualified | 3 | 12 | 41.074 / 0.414 / 6.387 / 0.412 | complete; all gates pass |
| target-tail-qualified | 12 | 12 | 32.018 / 1.718 / 27.520 / 1.789 | complete; all gates pass |

The original full-cache request at 96 atoms required 14,985,773,728 bytes and was refused by the complete resource allowance. Partial mask `0xffffff8` used 12,435,799,200 bytes and reduced cold time, but its population still timed out. Expanded mask `0xffffbfc` retained 13,728,372,608 bytes without force eviction (1,112,145,920 bytes free before the 1,036,363,408-byte cold force request), but also timed out during moved/0 at the original deadline. Inexact-inner and outer-secant experiments also failed to complete the original 96-atom population. They remain explicit, separate observations.

The first automatic 2 GiB cap refused the 48-atom 3,609,071,056-byte inventory. The corrected 4 GiB cap admits that already independently qualified inventory while keeping the 96-atom full source outside automatic selection. Warm calls retain only audited same-geometry values; policy changes and memory pressure restore the ordinary provider. No complete endpoint is cached.

The authentic-target DIIS tail passes 24 scheduled device cases at its recorded source but increases expensive changing-density pair sweeps from four to six. Its 12-atom moved endpoint regresses to 27.520 s; no large timing trial or default promotion is inferred. Its observed first cold call, including setup, is retained unchanged.

`campaigns.json.xz` retains the exact raw JSON text, full force vectors, independent cold/moved oracles, metadata and final outcomes, including the initial missing-ccache launch failures. Diagnostic oracle durations never become reference timing samples. `qualification.json.xz` retains source/build records, verified consumer hashes and verbatim compact test/sanitizer outcome excerpts with full-log hashes. Full build/test stdout and native binaries remain in ignored local qualification storage.

Each campaign binds its own frozen Git source and library in `manifest.json`. Test snapshots, PR heads and integrated benchmark snapshots are distinct. All GPU work used finite Slurm allocations; benchmark timing uses RTX 5090 on n4/n5, while some independent qualifications use n2 RTX PRO 6000. No visibility override or deadline extension was used.

Run `PYTHONPATH=.:python python benchmarks/results/vv10-range-followups-20261014/verify.py` to recompute every retained E/F error, audit phase counts and exact hashes, and check that only the automatic campaign enters the default plot. The parent publication verifier additionally replays the figures byte for byte.
