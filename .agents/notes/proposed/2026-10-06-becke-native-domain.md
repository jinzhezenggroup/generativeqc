# Decision: authenticate a bounded dynamic Becke domain before native dispatch

Status: proposed — implemented in the draft branch, partial qualification passed;
not promoted and not full issue acceptance.
Date: 2026-10-06

## Problem

The complete canonical graphs introduced in `5b021012f` authenticate exact atom
counts. An N=3 or N=128 graph is not a witness for another actual count, while
rebuilding a large exact graph at each native owner/geometry/tile would hide
substantial compiler work in force execution. The current dense coefficient
schedule also loses every isolated median estimate; integrating it must not
silently replace the measured phased default.

## Decision

Share the canonical expression builder between exact-N graphs and a bounded
active-prefix family. The latter has integer `atoms` and `owner` leaves with
no tangents. Every inactive pair contributes one to **both** incident products,
and every inactive atom product contributes zero to normalization. Guard ratio
operands before division so padding's zero separations remain unobserved even
in an eager evaluator. Differentiate the entire reachable objective using the
existing Graph AD. Match both actual roots, not supplied metadata.

The family carries an authenticated v3 operation identity and an explicit
1–128 atom bound. Exact-N v2 and legacy scalar-only v1 identities remain
unchanged. Both prototype planners now check the operation's actual domain;
neither a fixed graph nor a changed bound can authenticate a dynamic primitive.

Recognition and emitter revalidation occur during finite source generation.
The shared stationary emitter propagates its actual switch iteration count,
emits the authenticated coefficient schedule and a native bound constant,
and retains the existing ordinary kernels. Native allocation, center rebinding,
capability checks, stream order, launches and source publication remain in
`src/dft/stationary_gradient_cuda.cuh`.

The primitive is opt-in through the private owner argument or
`GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE=coefficients`; the default is `off`.
This does not recognize a functional or molecule name. Native admission requires
the already admitted phased storage, cached centers, actual atom bound and both
kernel thread caps. It allocates nothing new, evicts no owner and falls back
without revoking the generic/ordinary route. Older artifacts lack the optional
ABI and retain their bounded route. Actual requested/selected flags, batches
and reverse pair visits are separately reported; a plan is not execution proof.

Composite planning charges **both** concurrent source owners. The dry capacity
qualifier binds the new policy, constructor, resource helper, native admission,
metrics and dispatch; it reports the actual request and uses the same request
in its resource query. Its previous fail-closed guards remain, with explicit
mutation tests for accidental default promotion, a detached atom bound,
weakened kernel caps and fabricated work reporting.

## Evidence and cost

Independent host gates compare every active prefix of small bounded families
against exact canonical objectives/JVPs, including mixed vectorized counts,
zero padding separations and arbitrary padding tangents. Their geometric JVP
projections also match separate generic and coefficient force consumers.
Resource gates preserve concurrency at the byte boundary and charge both
composite owners for either spin count.

A fresh local default-iteration bound-128 graph has **635,655 nodes**. Construction
takes **10.356756849 s**, actual-root recognition **12.060217018 s**, and process
peak RSS is **648,320 KiB**. This is real compiler overhead, not GPU performance.
The emitter also revalidates the operation; source-generation and complete
cold endpoint measurements must account for that work. It never belongs in
per-point/tile/seed or moved-geometry timing as repeated recognition.

Graph identity:
`389ba8af8fc7ce291546a4b52e6ceb7c4c45b0f9d2f7d674a2147e12ed97dd9e`.
Operation identity:
`54a2495eba36e0cf9ae296828afcaf58bf0dee24844bcf08a02cee59c977bf32`.

The final combined host cohort passes **973 tests**, including the actual
bound-128 graph at 48/96/128 atoms against a 60-digit independent direct-product
Decimal geometry difference. Compiler structure checks cover **467 modules**
with **zero dependency errors**. These host results include the later capacity
guards and Decimal follow-ups, not additional device coverage.

Frozen n1 Slurm job **6139** exits 0, with a finite 25-minute
`main/gpu:5090:1` allocation, preserved device visibility and mandatory ccache
4.5.1. It builds the default-iteration PBE0 RKS shared owner, passes **48 GPU
gates with no skips**, and passes one external-source primitive case under
each of memcheck/racecheck/initcheck/synccheck. Each reports zero errors;
racecheck additionally reports zero hazards/warnings. Native requested versus
selected flags, reverse visits, batches and actual phase bytes are asserted.
The ccache statistics show **one additional miss, not a hit**. Fresh complete
probe source generation takes **18.33 s** on n1, including recognition/emitter
validation; that is a different host/scope from the local graph-only timings.

Reviewed partial evidence is published with `tools/evidence.py publish` at
`benchmarks/results/becke-native-domain-20261006/publication.json`. It retains
the **actual frozen dirty GPU patch**, checksums, compact receipts and a finite
Slurm reproduction recipe. The later host-only capacity/test edits do not
change any compiler, native or GPU-test input; the frozen snapshot comparison
explicitly verifies that fact. Actual maximum force differences were not
recorded, so the publication does not invent a quantitative error record and
its numerical acceptance decision remains **inconclusive**.

The formatting bot subsequently stripped one trailing space from the published
Slurm receipt and broke both its manifest and attachment checksum. Restore the
original frozen bytes rather than recompute those hashes. The trailing-whitespace
hook now excludes only reconstruction patches and `receipts.txt` under reviewed
result bundles; evidence/attachment checks continue to enforce their exact bytes.
No compiler, native or device-test input changes in this correction.

The shared-owner gate uses synthetic AO/XC inputs. It is not an independent
molecular oracle or a complete endpoint, and does not establish full physical
RKS/UKS/composite acceptance. No new performance measurement is claimed.

## Outstanding acceptance

Keep #1894 open. Final matched-source complete-consumer qualification, independent
physical RKS/UKS/moved forces, per-phase work/traffic/launch/device timing and
paired complete 48/96-atom PBE0 E+F cold/five-warm/moved/five-moved-warm remain
required. Preserve actual solver histories and builds; do not normalize solver
work or reinterpret either losing prototype as an endpoint regression/win.

## References

- #1894 and draft PR #1996.
- `2026-10-06-becke-whole-partition-ir.md`: exact-domain predecessor.
- `2026-10-06-becke-dense-coefficients.md`: retained losing dataflow.
