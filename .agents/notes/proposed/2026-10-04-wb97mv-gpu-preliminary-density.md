# Experiment: GPU semilocal preliminary density for WB97M-V

Status: private complete-endpoint experiment; no public API/default promotion
Date: 2026-10-04

## Motivation and boundary

The [cold-work investigation](2026-10-03-wb97mv-cold-scf-work.md) found that
same-functional coarse-grid preparation still costs about 69 seconds at 24
atoms. Loosening its energy/density thresholds leaves the strict physical
residual gate and its 19 iterations unchanged. The next experiment changes
the preliminary operator instead: GPU LDA or PBE has neither exact exchange
nor VV10 pair work. The final target remains full-grid FP64 WB97M-V.

This is a private driver using the existing native KS density restore
contract, not an extension of CPU-only `InitialGuessSpec` or the public HF
`initialize_from` API. `KsPreparedBatch::restore_warm_states` checks density
dimensions, coordinates, diagnostics and the source AO metric/spin convention.
Only the converged same-basis density is imported; target Fock, DIIS,
functional, grid, convergence and energy baseline are not imported. The
target constructs fresh physical work and retains its ordinary bounded retry.

## Fixed comparison

Run no preliminary solve, `lda-rks`, and `pbe-rks` sequentially on one RTX 5090.
Both sources use spherical def2-SVP, the target nuclei/charge/spin, grid
16×8×16, FP64, DIIS 8, at most 64 iterations and energy/density controls
1e-6/1e-4. The source must actually converge before its density can be used.
An unfinished source leaves the core guess intact; non-convergence is the only
source failure status allowed to continue. Resource/runtime errors propagate.

The target keeps grid 48×16×32, energy/density/screening controls
1e-11/1e-9/1e-12, DIIS 8 and maximum 180 iterations. SCF/force AO maps and
indexed forces are enabled for every target variant. Native experimental SCF
AO maps currently admit WB97M-V only: the private single-threaded driver
temporarily disables that opt-in for the separate LDA/PBE owner, restores it
on every exit and retains the already-prepared target's map. This is not a
general mechanism for selecting per-owner policies concurrently.

Complete cold includes source construction, preparation, solve, synchronized
density export/import and destruction as well as all target preparation and
the first complete energy/analytic-force call. Actual source XC submissions,
source iterations and available Fock-build diagnostics are retained; an absent
Fock-build counter remains absent. The driver has a 16 MiB density/coordinate
transfer bound. It does not establish joint public host/device-budget admission.

After a 3-atom pilot passes all 15 cold/priming/warm pairs, the same allocation
continues the 24-atom comparison. Every pair must pass independent 1e-8 Eh /
1e-7 Eh/Bohr gates, finite/shape/convergence checks, one-iteration warm replay,
actual target AO selection and on-GPU reference XC. Every accepted preliminary
variant must also prove successful source convergence and inclusion of its
complete cost in the cold timer. No source result is compared to the reference
as though it were a WB97M-V answer.

## Provenance and attempts

Source 49f0f9bc078a3f714ccc4fa1b394f638c784bacf includes master dc6ea9940.
That master update changes evidence consumers only; all 1355 native build inputs
remain identical to the GPU-qualified master-9c/VWN composition:

- Source identity: `ebdf07921464440e085b2925a1bd061ba9090788485d1cab01e3cada7122814c`
- Library SHA-256: `5a1b86cef1c0e978118d3024dc36861ebe8cd326dee8a6fe944a6b34000a5461`

Each job verifies source/library and measurement-script hashes before execution.
No old timings are relabeled to this composition. All source and build receipts,
scripts, raw results and unsuccessful attempts remain in ignored
`.artifacts/semilocal-seed-20261004/`.

Slurm 5634 stops because LDA preparation inherited the WB97M-V-only map flag.
Slurm 5637 then completes the no-seed/LDA pilot endpoints but stops on the
invalid shorthand `pbe`; the supported selector is `pbe-rks`. An attempted
retry, 5639, is cancelled after local setup failed due to an omitted Python
search path. These are driver/setup failures, not numerical method rejections;
their outputs are preserved separately and do not qualify the full campaign.
The corrected finite n1 allocation is Slurm 5640. It restores the source map
policy explicitly, validates provider names before deployment and propagates
unexpected source failure statuses. Full comparison results remain pending.

## Completed small pilot

Slurm 5640 completes all three 3-atom variants and passes the independent
15-pair verifier before continuing to 24 atoms. Maximum errors are below
8.669e-13 Eh / 1.177e-9 Eh/Bohr; every reference XC component stays on GPU
and every priming/warm replay takes one iteration. The target gates and full
grid are unchanged.

| Preliminary provider | Native complete cold | Paired reference complete cold | Target iterations / XC submissions | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 9.404071 s | 9.744874 s | 15 / 15 | none |
| LDA | 8.536679 s | 9.598415 s | 12 / 12 | 15 / 0.656032 s |
| PBE | 8.156632 s | 9.645183 s | 11 / 11 | 15 / 0.681369 s |

Both sources actually converge and are imported. Their XC submission counts
are 15 each, their Fock-build counters remain unavailable, and the complete
source cost is included in the respective cold preparation totals. The small
pilot supports testing the mechanism at 24 atoms; it does not establish a
large-system benefit or justify a default/API promotion. The 24-atom sequence
continues in the same allocation and will retain all three controls.

## Promotion gates

Require larger and displaced-system complete endpoints before claiming a
useful target domain. A production extension would need an explicit CUDA
preliminary provider contract, total resource admission, diagnostics and
failure/warm-priority tests; a profitable private density bridge alone does
not satisfy those requirements. Keep the unseeded path and all target gates.
