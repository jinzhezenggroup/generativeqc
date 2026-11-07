# HF density-difference convergence and 96-atom qualification

The numerical/convergence repair passes independent complete energy/force
qualification. **HF incremental J/K remains opt-in.** Standard cold-start
improvement at 96 atoms is below the shared performance protocol's >2% floor;
strict and displaced benefits do not replace that question. This publication
rejects default/performance promotion, not the scientific repair.

## Source and protocol

Final delivered source: base `ee9e454f33778489a1b5c3aa5ed95b6d6b44d182` plus
the losslessly encoded `source.patch.xz`, native identity
`8cecc4e053d209fdcf37d9f7323757f2885018aa899590fc20235b94d64d7e18`,
library SHA256
`67bf62291385eccecc5185bd4364d458fdfecbfb6dd69135673bd2d18aa84e47`.
Slurm `16001` uses an RTX 5090, Release/sm_120, CUDA 12.9, g++ 12.4,
fast-compile OFF and verified sccache 0.16.0. The last source-matched
incremental build takes six seconds; that is **not** a clean full-build cost.
Allocator peaks and complete per-iteration native energy/residual histories
are unavailable; no formal performance/production pass is asserted.

The full-precision samples include 12/24/48/96 atoms, **96/192/384/768 AOs**,
spherical all-electron RHF/def2-SVP, screening `1e-12`, refresh interval eight,
six independent sequences per explicit OFF/ON arm. Order is AB/BA/BA/AB,
repeated as needed. The earlier actual-unset default candidate has four
sequences per arm and a different source/library identity; it is retained
separately and is not substituted for the final implementation.

Cold includes calculator construction, preparation and complete host E/F.
Common imports/library metadata and tiny-water bootstrap are excluded. Cleanup
and bootstrap diagnostics are retained in `cleanup.json.xz`. Warm and moved each
independently replay their own arm's frozen cold density; neither receives
an oracle density. The moved fixture displaces the last atom's x coordinate
by `0.02 Bohr`. These are application-object cold endpoints, not process/JIT
startup measurements.

## Final 96-atom observations

Median seconds, **OFF / ON**, six samples per side:

| Controls | Cold E/F | Frozen warm E/F | Displaced E/F |
| --- | ---: | ---: | ---: |
| Standard | 16.764247 / 16.610585 | 2.876159 / 2.877016 | 14.062550 / 11.513996 |
| Strict | 23.679828 / 21.332947 | 2.879646 / 2.880033 | 22.490326 / 16.186528 |

Standard cold latency decreases about **0.92%**, displaced about **18.12%**;
strict cold about **9.91%**, displaced about **28.03%**. Warm differences are
noise-level. These are descriptive same-engine endpoint observations, not a
GPU4PySCF speed ranking or a formal performance-promotion pass.

Standard cold uses 11 target Focks in both arms: OFF 11 full, ON four full
plus seven deltas, without post-SCF Focks. Standard moved uses seven target
Focks: OFF seven full, ON three full plus four deltas; both also perform two
post-SCF full builds under the unchanged moved-seed fallback. Frozen warm
uses one full target Fock and no delta/post-SCF build. Initial-guess work is
retained separately in every sample. Quartet work is **unavailable**, not zero,
when the resident/streamed routes do not populate that census.

`summary.json.xz` retains medians, ranges, relative MADs, full sample counts and
work examples for every cohort/profile/phase. The `qualified` fields in raw
reports mean complete populations passing E/F gates, not performance promotion.

## Scientific gates and references

Public Standard controls remain E/D `1e-9/1e-5`; strict remains `1e-12/1e-9`.
Standard independent gates are energy `2.5e-6 Eh/atom`, maximum force
`2.5e-5` and RMS force `7.5e-6 Eh/Bohr`. Strict gates are absolute energy
`1e-8 Eh` and maximum/RMS force `1e-7 Eh/Bohr`. All **288** timed E/F
endpoints pass. Maximum observed errors across Standard populations are
`3.36514e-11 Eh`, `6.33211e-7` maximum and `2.66119e-7 Eh/Bohr` RMS;
strict maxima are `3.13776e-11 Eh`, `1.81070e-10` maximum and
`7.19527e-11 Eh/Bohr` RMS.

The smaller fixtures reuse the independently computed tight full-density CPU
PySCF arrays pinned by `reference-reuse.json`; historical native timing/winner
flags are not reused. The 96-atom reference is newly computed full-density
GPU4PySCF, with E/gradient/screening `1e-12/1e-11/1e-14`. Original/displaced
fresh full-Fock orbital gradient norms are `9.24822e-12` and `8.89741e-12`.
Both reference E/F vectors and reproduction inputs remain offline in this
publication. Reference software is never a production dependency.

The final library also passes all three focused native suites, four live
CPU-reference composition cases covering explicit OFF/ON/unset and resident
OFF/auto under Standard/strict controls, 61 host tests, and both native
compute-sanitizer memcheck suites with zero errors.

## Reproduction and storage

`publication.json` is the checked inventory. Its decision is rejected
performance promotion; its evidence record records passing numerical/endpoint
stages and explicit missing performance evidence. Compressed samples retain
all returned force vectors, observables, timing and semantic work, not just
force hashes. Build/test streams remain ignored; compact checksums and outcomes
are in `receipts.json`. Failed setup/resource attempts are kept separate and
never become accepted timing observations.

Use the standard publication reader to inspect a role without relying on its
compression suffix:

```python
from pathlib import Path
from tools.generativeqc_validation.record import load_publication_record

root = Path("benchmarks/results/hf-incremental-composition-20261105")
evidence = load_publication_record(root)
summary = load_publication_record(root, role="summary", name="summary.json")
```

Build the patched source normally with a verified compiler-cache launcher and
set `GENERATIVEQC_LIBRARY` to that build. Run every GPU command through a finite
Slurm allocation without overriding its assigned visibility. For the 96-atom
Standard arm pair:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 python benchmarks/hf_incremental_direct_jk.py \
  --source-revision ee9e454f33778489a1b5c3aa5ed95b6d6b44d182 \
  --reference benchmarks/results/hf-incremental-composition-20261105/references/atoms96.json.xz \
  --reference-scope 'Retained independent full-density GPU4PySCF original/displaced E/F' \
  --sizes 768 --repeats 6 --output .artifacts/benchmarks/hf-incremental-96atoms
```

Add `--strict` for the strict profile. For smaller cohorts use the pinned
`benchmarks/results/incremental-hf-full-energy-refinement-20261015/validation.json.xz`
and `--sizes 96 192 384`. The actual-unset candidate can be reconstructed by
decoding/applying `default-selection.patch.xz` after `source.patch.xz` and using the preserved
`reproduction/default-candidate.py`. Its reconstructed inventory is checked to
match `6122b317aad07cb06d44e159414ce8fcd24a77c1f587fb9107e37c205275c2d3`.
It is a rejected diagnostic, not delivered production behavior.
