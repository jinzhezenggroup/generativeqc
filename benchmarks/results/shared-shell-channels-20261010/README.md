# Shared shell derivative channels: complete endpoint qualification

Date: 2026-10-10. This is numerical qualification and a **negative performance
result**, not promotion of a faster default. Sharing evaluator/recurrence calls
does not establish less arithmetic, unchanged registers, or a faster endpoint.

## Matched experiment

- Native control is frozen master `2daa0aceaf6d140389ac5a183ead8b0ac3d3cd4a`;
  candidate differs only by the reconstruction patch in this bundle. No oracle
  density or CPU integral/gradient evaluation enters production execution.
- Both use Release, sm_120, CUDA 12.9.86, pinned `/usr/bin/g++`, verified ccache
  4.5.1, fast-compile OFF, and identical nonempty 22-entry shell registries.
  Stationary PBE0 SPD libraries/manifests are built explicitly; scientific
  contract validation is not bypassed. Binary and input identities are retained.
- Public PBE0/def2-SVP spherical energy **and all analytic force coordinates**:
  nested 3-atom/24-AO and 12-atom/96-AO water clusters. Explicit moving grid
  48x16x32, no density fitting, energy/density/screening tolerances
  1e-12/1e-10/1e-12. Independent GPU4PySCF uses the same offline basis/grid.
- `separate` explicitly requests both full-range J/K channels; `combined` is
  the retained single-source/default control. s/p/d endpoints exercise the
  admitted low/order-four/order-five owners, not the fsss molecular endpoint.
- Final Slurm job 6943, node1, main/gpu:5090:1, finite 40-minute limit; scheduler
  visibility is preserved. Each point/mode uses fresh processes in ABBA order,
  two observations per side for complete cold, fixed-density warm, moved and
  moved-warm endpoints. Cold includes preparation; every timer includes host
  force return. JSON serialization is outside timing.

## Final observed medians

Seconds are baseline -> candidate; these two-repeat medians are not statistical
speedup claims. Complete trajectories and unfavorable repeats remain in samples.

| AO | Reduction | Cold | Warm | Moved | Moved-warm |
| --: | :-- | :-- | :-- | :-- | :-- |
| 24 | separate | 3.88377 -> 3.78423 | 0.16909 -> 0.19576 | 1.36905 -> 1.32960 | 0.16735 -> 0.19461 |
| 96 | separate | 7.87881 -> 7.90265 | 0.87322 -> 0.91700 | 4.54740 -> 4.60747 | 0.87946 -> 0.92261 |
| 24 | combined | 3.88432 -> 3.76670 | 0.16232 -> 0.16163 | 1.35655 -> 1.28842 | 0.16133 -> 0.15652 |
| 96 | combined | 7.85751 -> 7.92230 | 0.87010 -> 0.85896 | 4.53334 -> 4.54582 | 0.86818 -> 0.86953 |

The shared dual-source **warm endpoint regresses 15.8% at 24 AO and 5.0% at
96 AO**; moved-warm regresses 16.3%/4.9%. Cold observations do not reverse that
decision, especially where Combined controls show similar drift. The proposal
remains a draft rather than a merge-ready performance improvement.

All 64 final native endpoint samples pass independent energy <=1e-8 Hartree and
force <=1e-7 Hartree/Bohr gates. Iterations, Fock builds, warm-use and fallback
flags match across every repeat in each phase. Full public residual histories,
solve-local AO work and raw force work remain retained. Native post-screen
force primitive/recurrence counts are unavailable and are not invented from
descriptor counts or capacity bounds. The separate host census is not an
endpoint census.

Linked production resource inventories retain every matched row, not only
global maxima: raw-component sharing increases some per-thread stack frames,
especially order-four/five. A common global maximum cannot establish unchanged
occupancy or spill cost. GPU allocator/NVML peak and comparable cache-miss
compilation cost were not measured; no memory/compilation-budget promotion is
claimed. Source generation and native force compilation remain expensive.

## Retained first experiment and optimization

Job 6939 is retained in `samples.json.gz`, with its source patch and linked
resource receipts. Its shared dual-source warm endpoint was 0.17044 -> 0.19628 s
(24 AO) and 0.87293 -> 0.93801 s (96 AO), also slower despite passing numerical
gates and matching solver/Fock work. It was not excluded for being unfavorable.

The final emitter removes redundant per-component finite tests. In strict FP64,
an active nonfinite contribution cannot regain a finite running sum, so the
final transactional channel audit preserves rejection/replay without testing
each component nine times. Exact-zero/inactive guards remain. A source-shape
test and forced-nonfinite/zero/publication tests protect this behavior. Separate
allocations mean cross-campaign timing differences are not controlled causal
evidence for this optimization.

Preflight failures 6935-6938 are explicitly recorded in provenance. None
produced a complete force endpoint: normalization-hook setup, reference CUDA
headers, a missing stationary AOT target, then rejection of an older AOT
contract. Both current manifests were rebuilt; no identity check was relaxed.

Final host core requalification: 13 tests, including all low/order-four/five
independent displaced-value and native-adapter gates. Adjacent regression
qualification: 536 tests. Compiler/SCF ownership: 509/224 modules, zero dependency
errors. Actual generated CUDA roots for all 17 classes pass 1,224 comparisons,
maximum error 5.552e-16, in finite Slurm job 6944; Compute Sanitizer reports zero
errors. This correctness probe is not a production timing claim.

## Reproduction and retention

`measured-source.patch.gz` reconstructs the final scientific delta from the
frozen base. `initial-source.patch.gz` reconstructs the initial candidate.
`samples.json.gz` contains both complete campaigns, including independent
references; `resources.json.gz` contains linked final-binary inventories.
`evidence.json` declares numerical acceptance and explicitly withholds formal
performance promotion. Detailed transient build/runtime logs remain ignored
under `.artifacts/shared-shell-performance/` and the retained node1 campaign.

Extract the frozen base into baseline/candidate directories, apply the selected
source patch to candidate, and unpack the compressed reproduction scripts.
`build.sh` builds both native libraries with matching flags; `build-aux.sh`
builds both `generativeqc_stationary_pbe0_rks_spd_manifest` targets. The optional
`rebuild-candidate.sh` reproduces the final emitter-only update. Adjust only
deployment paths/toolchain locations, preserving the scientific inputs/flags.
Run `run.sh` through finite srun with the recorded main/5090 request, then use
`analyze.py <run-directory> --output <summary.json>` to audit every phase/repeat.
An old auxiliary binary is not a substitute for a matching current manifest.

Future promotion requires a schedule/materialization strategy that preserves
cross-channel reuse without the observed dual-source endpoint penalty, plus
adequate repetitions, complete semantic-work/peak-memory and compilation
qualification. Fewer evaluator calls alone is insufficient.
