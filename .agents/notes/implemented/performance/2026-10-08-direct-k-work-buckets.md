# Decision: experimental primitive-work-aware exact-K buckets

Status: implemented, opt-in; not promoted
Date: 2026-10-08

The opt-in/default-selection decision is superseded by the
[2026-10-09 work-default decision](2026-10-09-direct-k-work-default.md).
The original resource decisions, qualification and cold-timing caveats below
remain historical evidence.

## Problem

Cross-chunk fill optimizes task occupancy, not the primitive-product work of
independent tasks sharing a warp. The existing primitive schedule sorts only a
2W lookahead window, adds serial insertion-sort overhead, and cannot retain
separate contraction groups across an entire bra row. PBE0 cold qualification
must include high-angular classes rather than assume psss/psps dominate.

## Decision

Freeze `GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=work` on the raw-K owner. Emit separate
work kernels for covered packed/subgroup angular classes. Keep the ordinary
kernel's queue footprint and the default fill selection unchanged. The compiler
owns grouping; integral producers, screening, retained precision tags and raw-K
scatter remain shared with the incumbent.

Keep one bra resident and retain eight post-admission queues. Ket work bins are
1, 2–3, 4–7 and 8+, split by whether both shells have multiple primitives. Fixed
bra work makes these also primitive-product bins. Angular class is the existing
kernel boundary. The saturated bin never truncates primitive work and makes no
arbitrary custom-contraction work-ratio promise.

Every queue owns a 2W arena. Scan a chunk only when all queue counts are below W;
at most W newly admitted tasks can then enter any bin. Drain full bins in
descending work order before scanning again. Flush each partial bin independently
at the original row tail. Reset all counts between bra/system rows. Memory is
O(W), with a fixed factor, and never materializes the quartet domain.

Use blocking full-warp or full-CTA collectives through publication, selection,
task preparation, consumption and overlap-safe residual compaction. Preserve the
double bra-claim barrier required by empty rows. Whole-CTA classes, native dddd,
unsupported classes and explicitly selected Rys/block lowerings retain their own
qualified workers; there is no retry into partially accumulated matrices.

## Rejected alternatives

- Global quartet sorting/materialization: breaks the bounded streaming owner.
- Sorting unscreened pair order: breaks monotonic Schwarz-tail screening.
- Assuming high-l dominates one or two classes: requires measured evidence.
- Thirty-two exact contraction/work bins: unneeded shared-memory and partial-tail
  cost; compile qualification exceeds the 48 KiB static limit for dpss. Generating
  unused work variants for optional K-block lowerings also exceeds that limit.
  Eight bins and exclusion of optional lowering variants retain a bounded
  ordinary static-shared-memory launch, without opt-in dynamic shared memory.

## Evidence

Concurrent C++20 lanes run the actual emitted packed psss, subgroup dppp and mixed
dsds workers against independent nested-loop admission, precision and bucket
censuses. Cases cover sparse/dense/empty work, W boundaries, multiple original
chunks, canonical duplicates, inactive neighboring systems and saturated long
contractions. Barrier-completion inspection independently checks that no consumed
batch mixes bucket labels. Each bra executes exactly the sum of per-bin
`ceil(survivors / W)` batches. The prepared parser freezes work selection and
keeps fill default and non-K topology incumbent.

The baseline complete 96-atom/768-AO PBE0 cold trace runs on n1 through finite
Slurm scheduling, RTX 5090, CUDA 12.9.1, Release FP64, def2-SVP spherical,
48x16x32 grid, screening 1e-12, fresh prepared owner/density. Persistent compiler
and runtime caches are reused. Preparation plus first synchronized energy/force
execution is the endpoint; diagnostic trace timing is not clean wall evidence.

The reducer validates 27 paired J-then-K passes of all 21 angular classes,
including native dddd. Total K value-kernel device time is 44.28994 s. Classes
containing d account for 60.2952%; psss/psps together account for 14.3994%. The
largest K classes are ddds (9.31%), ppps (8.71%) and dpps (8.20%); the top three
sum to 26.2133%. Thus high-angular work matters, but no small group accounts for
most K time. These are cumulative device sums, not shares of the complete cold
endpoint, and the trace does not supply primitive-work counts. The independent
energy/force errors are 7.64e-11 Hartree and 2.27e-11 Hartree/Bohr.

Raw trace and complete endpoint records are retained under the ignored n1
qualification directories. Machine-readable compact evidence and further GPU
qualification are recorded with this change; no performance/default promotion
is authorized by a host test or isolated device measurement.

Subsequent real-GPU qualification passes the complete native independent J/K
provider suite, through-f values and Cartesian order-two/CPU finite-difference
checks. Sixteen public restricted/unrestricted, symmetric/nonsymmetric and
scaled/zero/restored matrix checks have max error 4.6863e-13. Work-mode memcheck
and synccheck both report zero errors.

The frozen full-scale independent MINAO initial-density diagnostic reduces K
wall medians by 22.82% at 384 AOs and 17.36% at 768 AOs. Generated streaming
admissions are exactly identical per class: totals 23,480,495 and 81,907,624.
Native dddd/fallback admissions are not observed. Full-size matrix errors stay
below 1.8902e-11 against the retained 1e-8 gate. This uses a freshly compiled
qualification-only SCF device-K seam, not host Fock/CPU one-electron assembly;
the private handle layout is checked against the current C API. An initial
standalone host-Fock diagnostic did not complete one large-case sample within
eight minutes and is not used as timing evidence. Raw-K qualification explicitly
requests derivative order zero, retains the existing device seam and charges
neither source preparation nor unrelated host assembly to K wall.

Two clean ABBA samples per mode observe 384-AO cold E+F medians 103.37/91.66 s
and 768-AO medians 203.49/198.74 s for fill/work. These observations do not
establish a stable cold gain: Fock counts differ (26/23 versus 23/23 at 384;
24/28 versus 28/25 at 768). Work remains opt-in and fill stays default. The
separate work cold trace validates 25 paired class passes and actual execution
of 19 work kernels, retaining native dddd and whole-CTA ddpp. See the
[compact evidence](../../../../benchmarks/results/pbe0-k-work-20261008/README.md).

## Revisit when

Independent matrix/force and sanitizer gates plus clean matched-trajectory
complete cold endpoints establish a repeatable benefit. Refine the saturated
bin or per-class work schedule only with resource, primitive-work and
complete-endpoint evidence, not a psss/psps-only microbenchmark.
