# Budget-admitted PBE0 CUDA force scheduling

This publication qualifies complete PBE0/def2-SVP RKS energy plus analytic-force
endpoints, not an isolated kernel. It uses the resource/finite-grid analysis from
#2029 and the frozen source-trial tooling from #2031. The promoted schedule keeps
512 MiB device / 256 MiB host budgets, prefers an admitted 512-point force tile,
uses 128 cooperative geometry threads and retains four Becke pair-panel rows.
The scratch ceiling is 16 MiB, not an unconditional allocation. No new GPU-model,
SM-version, atom/AO-count or grid-size whitelist is introduced.

## Frozen #2036-base qualification

Base: `21f6314f5a812e9e83a41b3c9921549d6e944778`. Candidate bytes are reconstructed
by `source.patch`; they are not inferred from current master or this PR's HEAD.
Finite, exclusive n1 Slurm job 6292 uses the RTX 5090, CUDA 12.9, FP64, unchanged
scientific grids/cutoffs/convergence, and five fresh interleaved pairs per size.
Intrusive profiles, prewarming and older pilots are excluded from the timing
population. Each run retains cold, warm, moved and moved-warm endpoints, complete
vectors and actual native SCF histories. Priming used the same frozen sources.

| atoms | phase | control median / s | candidate median / s | paired median gain | positive pairs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 48 | warm | 8.928828 | 8.416576 | 5.609640% | 5/5 |
| 48 | moved-warm | 8.921205 | 8.409540 | 5.735386% | 5/5 |
| 96 | warm | 24.426597 | 23.201106 | 5.038910% | 5/5 |
| 96 | moved-warm | 24.483404 | 23.237256 | 5.070176% | 5/5 |

The statistic is the median of `(control - candidate) / control` for paired
complete endpoints, not the ratio of the displayed marginal medians. Every warm
phase has one actual SCF iteration / Fock build, matched SCF AO work and unchanged
force point/pair/quartet/Becke inventories. All phases pass independent gates of
1e-8 Eh and 1e-7 Eh/Bohr; observed maxima are 1.064109e-10 Eh and
3.022205e-11 Eh/Bohr. The original source also passes 42 selected device tests,
with zero memcheck errors and zero racecheck hazards/errors/warnings.

Cold and moved timings are **diagnostic only**. An earlier all-phase equal-work
campaign was rejected because unchanged controls naturally varied in SCF
iterations. A new warm-focused protocol was declared before job 6292: five new
pairs, >2% median benefit and at least 4/5 positive pairs for both warm phases.
No samples are cherry-picked from that earlier campaign; no solver tolerance or
iteration limit is modified to force equal work.

## Evidence and limitations

`bundle.json.xz` retains exact raw UTF-8 streams with byte counts and SHA-256,
independent references, source/harness checks, native/runtime binary identities,
compiler-cache receipts, actual launch assessments and rejected experiments.
`manifest.json` authenticates every stream. `qualification.json` is independently
recomputed, not trusted as a passing flag. `evidence.json.gz` is the shared numerical
validation envelope; `publication.json` binds the selected files.

Failed/malformed device prerequisites are retained honestly: n4's driver-library
mismatch produced no successful GPU endpoint; the initial n1 matrix lacked ten
packaged artifacts, which were then built before a successful full rerun. The
fixed-1024 dense-AO loss and the unfrozen 1024/four-row pilot are not promotion
evidence. The stopped point-parallel XC-preparation and stable-owner-grouping
routes are not reopened.

Allocator bounds are checked against the unchanged complete-owner budgets, but
global concurrent allocator peak and deployment/first-install compilation cost
are not measured. The shared publication accepts **numerical** scope; it does not
assert generic performance-envelope closure or benefits on unmeasured devices.
The resource estimates are analytical occupancy bounds, not achieved occupancy,
bandwidth, or a replacement for endpoint measurements.

## Offline reproduction

Only Python and NumPy are required; no CUDA/native library, PySCF or cluster is
needed to authenticate the publication and recompute independent whole vectors,
actual warm solver work, byte bounds, source/binary populations and timing gates:

```bash
python benchmarks/results/pbe0-cuda-schedule-20261006/verify.py \
  benchmarks/results/pbe0-cuda-schedule-20261006 \
  --output .artifacts/recomputed-cuda-schedule.json
```

The publication tests deliberately edit force vectors, warm iteration counts,
grid visits and runtime binary hashes, then regenerate storage checksums. The
offline verifier must still reject these records.

## Fresh GPU reproduction

Restore the exact base, apply `source.patch`, and authenticate the bundled full
candidate source manifest. For the control, restore the same base and copy only
the shared benchmark receipt helper/source-hash axes and endpoint-helper test;
authenticate the separate full control manifest. Do not transplant a binary from
another toolchain, libc or driver installation. The bundle retains the exact
trial-preparation, build, environment and interleaved scripts; local paths need
adaptation without changing the scientific protocol.

Use a compatible CUDA target/toolchain and verified sccache/ccache; preserve the
cache, record its statistics, and retain source/binary before/after checks. Every
real-device command must use a finite Slurm allocation and preserve its device
visibility. The n1 measurement recipe is hardware-specific, not a production
selector:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --exclusive --time=02:15:00 bash .artifacts/campaign.sh adaptive512
```

The current architectural contract and rejected-route rationale are in
`docs/developer/stationary_cuda_scheduling.md` and
`.agents/notes/implemented/performance/2026-10-06-pbe0-budget-admitted-cuda-schedule.md`.

## Separate upstream integration

The production integration is pinned to
`25675d88d64bfec13a8b7100eebcabef2a83d534`, which includes #2033's default Combined
two-electron reduction and #2038's shared incremental Direct-J/K policy. Its
population is separate from the #2036 results; old timings and binaries are never
reattributed to this source. `integration-source.patch` restores the new measured
candidate; apply only its benchmark hunks for the control. Both full trees are
independently reconstructed and authenticated.

Both native production trees were built in finite Slurm job 6293 through verified
ccache. Its first device matrix reports 38 passes and four `KeyError:
'two_electron'` failures: the semilocal component oracle still addressed the old
J output name. The test now explicitly compares Combined's two-electron output to
the independent J reference (LDA/PBE have zero exact exchange), preserving every
vector/tolerance gate. No production source is changed for this adaptation.
The failed matrix/source/build receipts remain in the integration bundle.

Integration uses a new full device/sanitizer gate and fresh, discarded priming
before five new interleaved pairs per size. The same >2% / 4-of-5 matched-work warm
gates apply, with no reuse of original timing samples. Exact streams are in
`integration-bundle.json.xz`, with `integration-manifest.json` and independently
recomputed `integration-qualification.json`. They are separately bounded files,
not an external archive or a raised retention cap.

The new population passes in finite Slurm job 6294, including all 42 device tests,
zero memcheck errors and zero racecheck hazards/errors/warnings. All qualified
warm endpoints retain one SCF iteration / Fock build and matching semantic work.
Independent error maxima across both modes and all phases are 1.045919e-10 Eh
and 3.028999e-11 Eh/Bohr.

| atoms | phase | control median / s | candidate median / s | paired median gain | positive pairs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 48 | warm | 8.420617 | 7.937067 | 5.698574% | 5/5 |
| 48 | moved-warm | 8.429935 | 7.932589 | 5.885798% | 5/5 |
| 96 | warm | 23.383137 | 22.101151 | 5.421262% | 5/5 |
| 96 | moved-warm | 23.355537 | 22.129023 | 5.283901% | 5/5 |

The published reconstruction matches every measured production file in the PR;
later documentation, publication consumers and capacity-pin updates are outside
the timed population. The original launch-resource assessments remain diagnostic
for their original source, not an achieved-occupancy measurement for integration.

```bash
python benchmarks/results/pbe0-cuda-schedule-20261006/verify.py \
  benchmarks/results/pbe0-cuda-schedule-20261006 --integration \
  --output .artifacts/recomputed-integrated-cuda-schedule.json
```
