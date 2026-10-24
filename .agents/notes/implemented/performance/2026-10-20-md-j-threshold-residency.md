# Decision: reuse primitive J thresholds and qualify sm_120 residency hints

Status: implemented; complete endpoint qualification recorded below
Date: 2026-10-20

## Problem and source correspondence

This follows [actual-target omission budgets](2026-10-20-md-j-target-budget.md)
and the [GPU4PySCF source audit](2026-10-19-gpu4pyscf-cold-source-audit.md).
The parent is `b394818dbb02895ae4b68629ad53a1addd746c3b` (PR #2628).
Fresh n4:1209 complete fixed-density J medians are 1.156280 s for the
96-atom r2SCAN-3c basis and 0.357326 s for the PBE0 basis. Pinned GPU4PySCF
MD with released 1.8.1 frontend/helpers takes 0.627771/0.347119 s.

Further reading of upstream `md_contract_j.cu` and `scf/j_engine.py` identified
three distinct opportunities. Upstream caches Hermite index tables in shared
memory; native generated contractions read the exact table from device memory.
Upstream prepares hierarchical q/qD screening outside its innermost work; native
still divided each target's allowance again in every admitted tile. Finally,
native low-angular kernels used 76–114 registers despite small shared footprints,
while shared-memory-limited shapes could benefit from more registers. The latter
is a native scheduling issue, not a missing upstream line to copy: upstream also
uses different unrolled kernels, task grouping and omission contracts.

## Decision and invariants

Compute each primitive's downward-rounded source threshold once per replay in
the existing hierarchy pass. Its `minimum` field is recipe-specific: reference
nodes retain maxima of public-pair Schwarz minima, while primitive nodes retain
**minimum target source thresholds**. Primitive qD nodes retain maximum represented
source norms. Inactive/padded thresholds are infinity and inactive sources zero.
No storage is added. A coarse tile can be rejected only if neither maximum source
norm reaches the opposite minimum threshold. This encloses every leaf decision.
Target-prefix admission uses that leaf's actual target allowance; source-prefix
admission retains the conservative global maximum target multiplicity.

The allowance remains downward-rounded epsilon/(m*N), where m is the output
shell pair's actual primitive multiplicity, N the full source census and epsilon
is min(screening,1e-12). If a coarse direction is rejected, all its nonzero sources
fall below all its target thresholds; summing at most m*N omitted products retains
the same per-public-J-entry absolute screening cap. Zero tolerance, zero norms,
unknown bounds, subset windows and changed-density replay retain their explicit
conservative behavior. This cap does not bound FP64 accumulation error.

For **sm_120 only**, derive a launch-bounds hint from the kernel's shared storage
within a 96-KiB scheduling envelope and cap the target at eight resident blocks.
The envelope leaves room in the measured 100-KiB SM shared pool. It is a compiler
register-allocation hint, not a memory reservation, device-capacity limit, or new
tile shape. Other architectures retain their compiler default. Lower-angular
shapes can reduce registers to admit more blocks; shared-limited shapes can use
more registers without losing useful residency. Do not replace this with a global
register cap or infer performance from nominal occupancy alone.

Retain the repaired outward triangle norm of the actual represented Hermite
source. Do not reinstate the invalid pre-underflow Q*D cap. Generated recurrence,
precision, target error budgets, density/projection mappings, reference screening,
K/derivative ownership and bounded capacity fallbacks remain unchanged. There is
no production oracle, replay allocation, host synchronization or pointer-based
reuse assumption.

## Controlled experiments

All use finite Slurm RTX 5090 allocations on n4, five clean complete fixed-D J
samples, and separate work/profile calls. Timings include density upload,
evaluation, public-AO mapping and host publication; preparation is separate.
The input public-AO MINAO density is identical in all arms; no SCF is timed here.

| Arm | Job | r2 J (s) | PBE0 J (s) | Decision |
| --- | ---: | ---: | ---: | --- |
| Parent | 1209 | 1.156280 | 0.357326 | control |
| Shared Hermite index cache | 1210 | 1.149752 | 0.355031 | reject extra code/storage for <1% |
| Threshold hierarchy | 1211 | 1.125481 | 0.349070 | retain |
| Threshold hierarchy + exact outward Gram norm | 1212 | 1.115640 | 0.346611 | reject extra numerical complexity for <1% incremental gain |
| Threshold hierarchy + residency hint | 1213 | 1.085448 | 0.334371 | qualify final source |

The rejected Gram experiment used an independently checked, outward-rounded
sparse signed metric of the represented source, with the existing triangle norm
as fallback. Its 181 same-parity terms were checked against an 80-digit oracle,
including subnormal, diffuse and tight cases. It reduced r2 roots by only 0.80%
and summands by 1.28%, after adding numerical complexity. Neither its generated
table nor its production arithmetic is retained. This is distinct from the
invalid historical Q*D cap and does not justify bringing that cap back.

The threshold and residency arms preserve the parent root, direction and
Hermite-summand counts exactly. r2 source tiles fall 7,453,609 → 7,036,821 and
screened tiles 3,272,721 → 3,165,722; PBE0 source tiles fall
2,328,885 → 2,245,613 and screened tiles 1,040,763 → 1,036,964.
Root/direction/summand counts are, respectively:

- r2: 2,003,901,808 / 3,766,588,912 / 108,322,370,192.
- PBE0: 811,922,712 / 1,522,688,792 / 22,695,676,456.

Residency-arm r2 profiles report zero local bytes per thread for all 15 classes.
For example, (0,1) falls from 110 to 62 registers; (1,2) rises from 86 to 126,
with its shared footprint still limiting residency. The benefit is an allocation
tradeoff, not simply fewer registers everywhere. Final qualification uses the
architecture-guarded, cleaned source, not the experimental all-architecture hint.

## Qualification

The measured candidate source identity is
`f35e6751f6ece0a40d8ffe7b078a37e815452cdb1b2f0e8ea73c047c2cd7a73a`.
Exact binary/force-library identities, all endpoint rows, fixed-J samples,
per-class register/shared/local storage, work counts and tests are retained in
[the qualification record](../../../../benchmarks/results/md-j-threshold-residency-20261020/qualification.json.xz).
Raw local artifacts and scripts are under `.artifacts/j-followup-20261020/`;
immutable remote roots are `/data/jzzeng/qc-j-next-*-20261020`.

The protocol is 96 atoms, direct/noDF, spherical FP64, r2SCAN-3c/def2-mTZVPP
(1088 AO) or PBE0/def2-SVP (768 AO), 48×16×32 unpruned moving grid
(2,359,296 points), E/D convergence 1e-12/1e-10 and screening 1e-12.
Complete r2 endpoints include analytic forces and D4/gCP. Independent gates are
1e-8 Eh and 1e-7 Eh/bohr. This is a matched-grid implementation diagnostic,
not a grid-converged production ranking. Complete cold includes preparation;
complete energy/force time is never divided by SCF iterations to infer J cost.

Final n4:1214 endpoints retain every clean sample:

| Complete endpoint | Parent (s) | Candidate (s) | Fock builds, parent/candidate |
| --- | ---: | ---: | ---: |
| r2 cold, including preparation | 132.776465 | 127.193875 | 25 / 24 |
| r2 warm, median of five | 24.750075 | 24.698179 | 1 / 1 |
| r2 moved geometry | 81.897380 | 80.874539 | 13 / 13 |
| r2 moved-warm, median of five | 24.761415 | 24.724518 | 1 / 1 |
| PBE0 cold, including preparation | 61.467269 | 60.943116 | 17 / 17 |
| PBE0 warm, median of five | 10.344571 | 10.275292 | 1 / 1 |

The r2 cold difference includes one fewer Fock build, so its 4.20% reduction
must not be attributed wholly to J. Cold/moved are single observations; warm
ranges and all records remain in the receipt. Separate cold profiles each
capture **24 actual Fock intervals**. Median J falls 1.170395 → 1.090155 s
(6.86%), while XC stays 2.191585 → 2.191731 s. Median J+XC interval falls
3.388549 → 3.308414 s (2.36%). Summed J falls 28.229660 → 26.295634 s,
a 1.934026-s reduction. No endpoint/iteration quotient is used. All four
preparation/cold trace streams have zero dropped events or callback errors.

Fresh fixed-density comparison uses the final cleaned/guarded source, n4:1215:

| 96-atom complete J | Parent (s) | Candidate (s) | Pinned GPU4 MD (s) |
| --- | ---: | ---: | ---: |
| r2SCAN-3c basis | 1.156280 | 1.080505 | 0.629548 |
| PBE0 basis | 0.357326 | 0.333043 | 0.348568 |

J decreases 6.55%/6.80%. Against upstream, r2 is still **71.63% slower**;
PBE0 is 4.45% faster. This does not establish r2 J parity. Upstream is the MD
target pinned at `9af748a71ca8f4881c604332c2f0cc1632e033ac` with released
1.8.1 frontend/helpers, not a fully pinned package. Omission contracts differ.
Final matrices differ from parent by at most 2.85e-14/3.56e-14, and upstream by
1.614e-10/1.346e-10; all pass atol=1e-9, rtol=1e-10. Final roots, directions,
summands and owner device bytes equal the parent. All 15 final native angular
classes report zero local bytes per thread in both basis cases.

All 36 clean E/F records pass the declared gates. Maximum clean energy error is
9.55e-12 Eh; maximum force error is 4.55e-11 Eh/bohr. Host validation has
792 math/executed-kernel tests and 12 compiled scheduling-contract tests.
New hierarchy cases cover 33 leaves, mixed target multiplicities/systems,
neutral padding, zero/positive cutoffs, dense/zero/sparse/unknown source norms,
changed-density replay and full/partial/empty windows; independent long-double
Cauchy products may never be lost by prefix or coarse rejection. n2:3497 passes
120 MD CUDA tests and all four Compute Sanitizer tools with zero errors. One
fully pinned GPU4PySCF package test is deselected because only the explicitly
identified MD-target/released-helper comparator is available.

PR #2628 merged during qualification. This change was rebased onto master
`e35aacb4507701ae38b12fdefbc0dcb2b552a3d8`. The five MD-J engine/contract,
owner/header and compiler-emitter files are byte-identical to the measured
candidate (hashes in the receipt); intervening changes include ordered Gram host
admission, public precision names and native CPU user-XC expressions. Rebuilt
source identity is
`cc3c3533e00688e911aed5f193695fb65a40de3e77ca4e3eaf6b14f28d9efaf7`.
Post-rebase n2:3498 passes all 120 MD CUDA tests and independent complete
cold/warm r2SCAN-3c and PBE0 E/F gates (four records). These PRO 6000 times
are separate correctness gates, not pooled with RTX 5090 timings.

The maintained CUDA ownership report records +40/-27 handwritten scientific
code lines (net +13) and no runtime/ownership reclassification. Growth is the
native threshold reduction and resource scheduling hint; generated recurrence
and contraction capability is unchanged. Existing fallback recipes remain.
The full-build generated-output report encounters an existing overlapping RCCSDT
family claim; the PR's maintained-source delta uses the checked source ledger
without generated-build inventory, since this change emits no new math.

## Evidence retention

The existing 64-MiB tracked-results cap had only 45 bytes free. Losslessly compact
whitespace in `rtx5090-a5473bf-water-tetramer-def2-svp-spherical-b4.json.gz`, saving
60,933 bytes (130,067 → 69,134). Parsed JSON is identical, its public README link
is unchanged, and no checksum/publication manifest binds the old file bytes.
The original bytes remain in parent Git history. Old/new SHA256 and the equality
check are recorded with this qualification. No measurement is removed and the
retention cap is unchanged.

## Revisit

r2 J still trails GPU4PySCF. Different omission contracts and execution domains
prevent attributing the entire remaining gap to any one source-level difference;
they do not prove the gap unavoidable. Revisit work-domain changes or a rejected
cache/norm only with a measured bottleneck and independent complete-endpoint
qualification. Other architectures need their own register/spill and endpoint
evidence before adopting the sm_120 hint.
