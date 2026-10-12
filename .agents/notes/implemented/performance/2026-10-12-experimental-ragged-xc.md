# Decision: bounded, default-off Ragged XC scheduling

Status: implemented, experimental; not promoted to automatic execution
Date: 2026-10-12

Superseded for ordinary automatic scheduling by
`2026-10-12-ragged-xc-default.md`. This note retains the original experiment,
including rejected serial-scatter evidence; its measurements are not a claim
about a later master revision or all-workload promotion.

## Problem

Master `e29bd57cd1b65ec702a234ef0af208265104b90a` admits batched mapped
contractions only when every nonempty tile in its original group has 32–128
active AOs. A large domain rejects its entire group even though the existing
shared-memory contraction bodies already traverse arbitrarily many AO blocks.
The rejected ACE performance experiment does not justify further approximate
SCF work or weaker scientific acceptance gates.

## Decision

Add `GENERATIVEQC_CUDA_XC_RAGGED_BATCH=1` as an explicit, default-off scheduling
experiment. Extend admission through 1024 active AOs, retaining the original
32-point minimum and every tile's independent selected AO map. Launch density,
panel and triangular potential contractions over the actual group maximum;
do not allocate a full-basis AO tensor or copy a local density matrix.

The existing additional byte allowance charges the sum of maximum AO, work,
potential and point slots, plus all immutable descriptors. The planner can
shrink the group width, fall back to point-only batching, or retain one-tile
execution on an optional device allocation failure. Larger domains, response,
unsupported precision, tiny tails and selected optional providers keep their
incumbent paths. The public resource ledger still grants no unreserved optional
arena. Inactive groups launch legal nonzero grid extents and publish zero
features and their original point totals.

Large groups partition global AO rows into shared-schedule-width strips. Each
CTA owns the authoritative upper-triangular entries of its strip and their
symmetric mirrors for one spin. Sorted maps allow two lower-bound lookups to
restrict local rows; owners do not scan or replicate the full local matrix.
Each owner visits the same tiles and additions in the original order. Point
totals remain with exactly one owner. There is no additional numerical scratch,
new descriptor storage or global floating-point atomic. Small groups and shapes
outside the row-owner launch domain retain the single-CTA scatter.

Preparation receipts report actual admitted groups, nonempty tiles, **large
tiles above 128 AOs**, byte allowance use and selected AO extent. These are
prepared-domain counts, not cumulative SCF evaluation counts.

## Invariants

- Traverse every selected density column, not just diagonal AO blocks.
- Traverse every upper-triangular potential block and publish both symmetric
  entries. AO domain selection and scientific summands do not change.
- Preserve per-entry tile accumulation order and avoid floating-point atomics.
- Keep exact J/K, SCF thresholds, DIIS, precision, grid and analytical force
  execution unchanged; disable ACE in both benchmark arms.
- Keep response and optional-provider qualification independent of this flag.
- Do not report coverage or a fixed-density kernel win as a complete E+F win.

## Validation and evidence

The final focused and adjacent executable host suites pass 380 tests. Host planning
probes cover 129, 257, 513 and 1024 AOs, reject 1025 AOs, account every retained
slot, exercise exact-budget boundaries, and preserve explicit fallbacks.

The native Slurm suite checks independent CPU masked bilinears for all curated
functional/spin families and graph replay. Additional 129/257/513/1024/1025-AO
RKS/UKS probes compare complete energy and potential arrays bitwise to the
incumbent, with nonidentity overlapping maps, nonzero cross-block density
entries, density changes, canaries and the upper-bound fallback.

Changing a compiler scheduling emitter invalidates the stationary-force AOT
compiler-closure identity even when its generated scientific source is
unchanged. An initial complete endpoint correctly failed closed on that stale
asset. Rebuild the official `generativeqc_stationary_pbe0_rks_manifest` and
`generativeqc_stationary_pbe0_rks_spd_manifest` targets; never patch expected
hashes or bypass artifact validation. The trajectory harness now records
failed endpoints with absent forces instead of hiding the actual status behind
a `.tolist()` exception.

`benchmarks.pbe0_ragged` reuses the independent-reference trajectory harness
without enabling ACE or overriding the ordinary SCF chunk controller. It
requires five interleaved clean trajectories unless explicitly feasibility-only;
diagnostic trajectories are separate. Every cold/warm/moved/moved-warm endpoint
retains the independent `1e-8 Eh` energy and `1e-7 Eh/bohr` all-component force
gates. Raw working evidence is under `.artifacts/ragged-20261012/`.

The first, deliberately serial-scatter version passes every independent E+F
gate but is slower: one clean 96-atom trajectory measures cold 66.623 → 71.992 s,
warm 13.230 → 13.674 s, moved 58.059 → 63.028 s and moved-warm
13.293 → 13.664 s. SCF iterations remain 17/1/12/1. This is rejected feasibility
evidence, not performance qualification. Slurm job 2925's separate diagnostics
show 8096/8448 nonempty tiles admitted versus 672/8448 on the incumbent;
7248 original-geometry large domains and 7249 moved domains enter contractions.
Both routes select a maximum 536 AOs. The candidate retains 24,989,968
additional bytes versus 25,311,448, with group widths 4 versus 8.

Slurm job 2926's Nsight Systems fixed-density diagnostic identifies ordered
scatter as the largest individual kernel family: 9.141 s, 23.3% of combined
kernel time across both arms and both geometries, including grid preparation.
The diagnostic XC medians are about 1.207 versus 1.598 s for the original
geometry. These diagnostic/kernel measurements justify unique output owners;
they do not establish complete-endpoint gains. Preserve this rejected source
and raw evidence under `.artifacts/ragged-20261012/serial/`.

### Unique-owner complete-endpoint qualification

Slurm job 2927 passes the native independent-oracle and boundary suites, Compute
Sanitizer memcheck (zero errors) and full-suite racecheck (zero hazards, errors
and warnings). Its separate feasibility trajectories are retained under
`.artifacts/ragged-20261012/owner/`.

Slurm job 2928 measures five interleaved clean trajectories per arm and separate
diagnostic trajectories on the RTX PRO 6000 Blackwell. Both arms use strict
FP64 PBE0 RKS, 96 atoms, 768 spherical AOs, the original grid/screening/tolerances,
ACE disabled and ordinary SCF control. Cold endpoint time includes preparation;
all endpoints include analytical forces. All 48 records pass the independent
same-grid energy and every-component force gates. Final native SHA-256:
`ddf568ac6296139fd9163dbde2dc8269142246a1b5822974e5a95e8ab99762b0`.

| Phase | Incumbent median (s) | Unique-owner median (s) | Time reduction |
|---|---:|---:|---:|
| cold | 65.235161 | 61.892507 | 5.124% |
| warm | 13.282428 | 13.059074 | 1.682% |
| moved | 58.268003 | 55.839363 | 4.168% |
| moved-warm | 13.270648 | 13.085955 | 1.392% |

The repository's descriptive MAD/noise gate qualifies cold and moved, but **not
warm or moved-warm**, whose changes fall below its 2% floor. Do not call this an
all-workload performance promotion. A moved candidate in repeat 4 takes 13 SCF
iterations versus 12 and is 1.22% slower in that pair; the incumbent also takes
13 moved iterations in repeat 0. Both arms total 156 SCF iterations and 161
Fock builds across the five clean trajectories. Raw times include these
variations rather than deleting samples or normalizing away executed work.

Diagnostic J/K/XC counts match: 18/18/17 cold, 1/1/1 warm, 12/12/12 moved and
1/1/1 moved-warm. The final receipts retain the same 8096 admitted nonempty
domains and 7248/7249 large domains as the serial candidate. Combined J/K event
timings are not separated into invented K-only timings. The diagnostic cold
XC intervals total 20.636 versus 17.297 s for 17 calls; these remain diagnostic
measurements, separate from the endpoint table.

Maximum independent errors over qualification records are `1.1824e-11 Eh` and
`3.8062e-11 Eh/bohr`. Clean paired native force differences remain below
`2.0224e-11 Eh/bohr`. Source hashes and the loaded native binary match the final
checkout. Raw records, journals, descriptive assessment and source/build
manifest are retained under `.artifacts/ragged-20261012/final/`.

## Rejected alternatives

- Simply widen the admission guard while leaving 128-wide launch grids: this
  silently drops large-domain density/potential blocks.
- Merge AO maps or omit cross-block pairs: these change the scientific domain.
- Restore full-basis AO tensors or unbounded per-group potential storage: these
  defeat the explicit residency contract.
- Enable the candidate by default because coverage increases: complete endpoint
  qualification, resource qualification and default promotion remain separate.
- Retain one CTA per spin for large mapped matrices: measured serialization
  outweighs the first version's launch-count savings.

## Revisit when

Promote only with reproducible complete-endpoint gains and an accounted public
resource reservation. Retune row-owner width or group buckets only against
source-matched complete endpoints, preserving unique output ownership and
original tile order.

## References

- `docs/developer/xc_native_cuda.md`
- `benchmarks/pbe0_ragged.py`
- `tests/native/dft_point_batch_cases.cuh`
- `tests/python/test_xc_compact_batches.py`
