# Bounded automatic resident PBE0 final-state validation

This supersedes the all-off selection decision, not the immutable measurements
in [the original publication](../ks-resident-final-validation-20261010/README.md).
An unset `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION` now requests resident
proof for full-precision exact-direct PBE0/RKS with at least 384 AOs. Existing
resource, library-handle, identity and unpublished-W/total-D scratch-lease gates
still apply. `0` remains a durable opt-out; `1` retains experimental admission.
No numerical acceptance gate is relaxed and failures are not hidden by retry.

The algorithm is unchanged: reuse resident inputs/products, remove two redundant
full-AO products per canonical nonempty spin (11 becomes 9), retain the occupied
density reconstruction and legacy dense exports. This is not a zero-transfer
path or a GPU4PySCF timing contest. Admitted proof adds 112 bytes and two drains;
published stationary-weight leases retain the existing fallback.

## Source-matched complete endpoints

Measured clean source: `5a9b107ca8b9a654d45d3dc15fac45e962323f3b`, tree
`9f30746049fb1d4626b8dda6e5bce50a9e69aaf2`. Source archive SHA256:
`6a19ffd1a7a45558f06a7846307d21aceae18759ea5df6a3816155befa8ea716`.
Native library SHA256:
`4836e6b16249be1cca0be92024d21784e37e283b98930fa286400d4ccaa05901`.
Stationary AOT library SHA256:
`969d92da3008b710c287c673d778194485443ba2517f5b0b334d99bdf89192b5`.

Both arms use this same native binary: baseline selector is explicitly `0`,
candidate selector is genuinely **unset**, not an explicit `1`. Five
interleaved pairs per original/moved geometry measure complete synchronized
PBE0 energy-plus-force host return, with one iteration and one full-density
Fock per replay on an RTX 5090, sm_120, CUDA 12.9.

| Atoms / AOs | Geometry | Reference median (s) | Unset median (s) | Reduction |
| --- | --- | ---: | ---: | ---: |
| 48 / 384 | warm | 4.936970990 | 4.819580037 | 2.378% |
| 48 / 384 | moved-warm | 4.975060266 | 4.815071207 | 3.216% |
| 96 / 768 | warm | 14.577087622 | 13.690183643 | 6.084% |
| 96 / 768 | moved-warm | 14.566437654 | 13.634650297 | 6.397% |

All four pass the unchanged descriptive gate
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`. All 40 complete
records independently pass 1e-8 Eh / 1e-7 Eh/Bohr gates. Maximum E/F errors are
6.593837e-12 / 2.470735e-11 at 48 atoms and
8.185452e-12 / 3.745027e-11 at 96 atoms.

The analyzer authenticates all eight retained NPZ checkpoints per size,
unchanged owner-local density/geometry seeds, available semantic-work parity,
bounded resource counters, actual route admission and repeated-weight fallback.
Capacity counters are not executed screened J/K quartet counts. The measured
domain is warm replay and moved-geometry replay, not cold SCF convergence.

## Failed and incomplete attempts remain retained

- Job 7266: both sizes failed setup admission because the first matcher compared
  the native signed exchange coefficient against positive 0.25. No timed samples.
  Restricted total-D Fock convention requires `-alpha/2 = -0.125`; the corrected
  matcher is shared with executable host regression tests.
- Job 7271: the complete 48-atom cohort remains qualified. Its 96-atom record
  contains 17 partial samples and is **not** qualified or pooled with job 7276.
  Slurm cancelled the step at `2026-10-11T02:33:41` for its 15-minute limit;
  authoritative srun exit 143 overrides its erroneous EXIT-trap receipt of 0.
  Every retained partial numerical sample passes its separate analysis.
- Job 7276: reruns **only 96 atoms**, with a finite 20-minute limit and explicit
  TERM/INT exit traps. Srun, driver and final trap exit zero; all 20 records and
  final checkpoint checks complete. The copied `job.txt` scope label says 48-96,
  but the exact retained loop is `for atoms in 96`; 48 was not retested.
- Earlier below-gate/slow samples and the disclosed seven-AO broader KS admission
  assertion remain in the original publication, unchanged. There is no new
  all-green broader KS-suite claim.

## Qualification and reproduction boundaries

Native CPU build job 7270 verifies ccache 4.5.1 and preserves before/after stats,
actual compiler commands, full source manifest and pre/post artifact checks.
GPU jobs 7271/7276 run through Slurm on node1, main, gpu:5090:1, preserving
assigned visibility 1 for job 7271 and 3 for job 7276. Slurm accounting storage
is disabled; original terminal and per-driver receipts are retained without
inventing accounting records.

At publication, master `20baf5826a76661de7ed0ea7f06dfefb41574f2a` was inspected
and automatic merge-tree was clean. Neighboring DF/RSH, CLI, HVP, CPU-AOT and
architecture-capability changes do not replace this measured source. Source
inspection does not transfer timing qualification to latest master, and does
not justify blanket GPU retesting.

`validation.json.gz` is a standard numerical/endpoint evidence envelope with
independent analyses, quantitative errors, failed-attempt descriptions and
hash-bound pointers to the complete sample/reference receipts, without storing
those same vectors twice. Its formal performance/production statuses remain `not-run`:
the four scoped timing observations are not fabricated equation/IR/schedule
identities, measured memory peaks or universal production qualification.
The AO lower bound avoids promoting small owners; larger owners use the same
fixed schedule and bounded packets, but are not newly measured. UKS, fitted
providers, other compositions and other hardware are not claimed measured.

`raw-receipts.json.xz` losslessly stores exact UTF-8 receipts, SHA256/byte sizes,
selected build/cache receipts, complete and partial sample JSON, references and
recipes from all three default-promotion attempts. Large full build logs, compiler
command listings and the source manifest remain ignored locally; their exact
hashes/byte sizes and an actual executed cached KS compiler command are retained
in the envelope. Retention migration metadata is whitespace-compacted only,
with parsed content checked unchanged, to retain the existing 64-MiB budget.
The actual NPZ files remain ignored locally
and were independently hash/blob-checked; digests do not reconstruct their bytes.
Extract the recipes under their stored relative paths, recreate the clean source
and native build, and adapt only the site-local paths. GPU replay must use finite
`srun`, preserve Slurm visibility and avoid node n3. The 96-only recipe is:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash default-promotion-v3/run-endpoint.sh
```

The independent analyzer is `default-promotion-v2/analyze-default.py`; invoke
it separately for each retained complete size. Do not reinterpret the incomplete
96 record from job 7271 as a third qualified cohort.
