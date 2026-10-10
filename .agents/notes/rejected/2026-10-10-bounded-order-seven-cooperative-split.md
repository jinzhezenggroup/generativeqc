# Decision: isolate the existing cooperative order-seven force consumer

Status: rejected as a production schedule; independent device gates pass
Date: 2026-10-10

## Problem

The implicit one-electron prototype improves 96-atom PBE0 endpoints but fails
the unchanged 48-atom warm promotion gate. Its intrusive matched-owner trace
already reduces Hcore from 153.667 to 18.989 ms at 48 atoms and from 1060.791
to 92.085 ms at 96 atoms. Further SP-domain derivative/Boys specialization is
not implemented: the remaining one-electron cost is small compared with the
mixed two-electron force worker (about 1.5 s and 4.2 s per profiled warm call).
Do not spend a new numerical/Boys-range qualification campaign chasing that
small region without better causal evidence.

An existing diagnostic angular partition identifies order seven as a material
cost: on the same frozen native library, its generic consumer takes 1700.65 ms
at 96 atoms and the existing cooperative consumer takes 915.912 ms. Both admit
exactly 29,328 shell quartets and 19,115,136 primitive-AO quartets. All three
complete calls per diagnostic owner pass independent retained-reference energy
and force checks. This is intrusive angular-route evidence, not an achieved
speedup of the ordinary bounded endpoint or a default-promotion campaign.

## Rejected scheduling decision

Reuse the existing compiler-owned cooperative derivative algebra in the
ordinary full-range bounded route. Its existing materialized-dddd split guard
proves a maximum shell angular momentum of two, resident primitive-pair data,
compatible derivative recurrence and the original 256-lane launch shape.
With the existing cooperative selector enabled, partition the canonical queue
into generic orders zero through six, cooperative order seven and pure
materialized dddd. Negative angular tag -3 excludes seven/eight; the old -2
control still excludes only eight. Keep exact screening after ownership.

The cooperative worker uses its existing 256-lane/shared-workspace contract and
omits the generic private AD frame by compile-time specialization. Preserve
Combined/Separate coefficients, masks, profile pointers, physical orientation,
same-stream resets and sticky launch-error observations. Any failure terminates
before subsequent owners. No new retained allocation or scientific recurrence
is introduced. The existing explicit cooperative disable keeps the old two-way
control; do not ignore a caller's selection. Native preparation selectors use
`enabled`, which is true when unset and false for `0`/`none`; this prototype
changes dispatch, not those selector defaults.

## Invariants and fallbacks

Missing caches, materialized disable, unproved/f-containing bases, nonstandard
launch shapes and reachable/convolution derivative modes keep the complete
existing fallback. Order six is not promoted to cooperation: prior qualification
found that schedule slower. SR/LR/RSH and Fock routes remain unchanged.
Production does not depend on CPU/PySCF reference work. Weighted summation may
change order, so independent numerical acceptance remains mandatory.

Three outer domain traversals replace two only in the admitted cooperative
case. Memory bounds alone do not justify that extra work: retain outer scan,
launch and per-class admitted counts alongside complete endpoint evidence.
Primitive-AO counts are admitted upper bounds, not executed recurrence counts
or measured hardware traffic. Do not sum `worker_ms` once per class: all classes
within a launch share the same recorded worker interval.

## Evidence and remaining gates

The compiled actual host dispatcher checks both spins, Combined/Separate,
all resource/recurrence/basis/shape guard states, both cooperative selections,
workspace maxima and fault injection at every launch/peek/reset boundary.
The actual ownership helper checks all s/p/d/f physical shell assignments,
repeated shells and reordered pairs: complement/seven/eight own every quartet
exactly once. The adjacent angular composition checks retain their semantics.

Raw diagnostic evidence is retained under local
`.artifacts/shell-nucleus-stage/evidence/{profile,angular}/` and n1
`/data/jzzeng/qc-next-hotspot-20261009-125a4e33f/evidence/shell-nucleus-v3/`.
The matched profile's exact class census agrees arm-by-arm: 48-atom totals
23,220,006 shell / 320,981,006 AO / 1,511,432,246 primitive-AO quartets; 96-atom
totals 92,233,228 / 1,263,186,780 / 5,944,643,268. The angular controls also
match that 96-atom census class-by-class. Main-library source identity is
checked; the frozen ordinary bounded source before this patch is byte-identical
to the review checkout. The one-electron native library is
`111b9692d05609931523ec2b62538e168e030c8266d4e7ef993c24bfe7da668d`.

The frozen native/math basis remains #2166 v7 composition on `125a4e33f`, not
a timed whole-master build. Review checkout master advanced to `880b46ef3`
through unrelated CC Lambda #2163. No full test matrix is rerun for that change.
The candidate needs live independent J'/K'/Combined/Separate gates, a checked
actual three-way class census, unchanged semantic/resource bounds and robust
48/96 warm/moved-warm complete timing before any PR. Keep every sample and the
strict >max(2%, twice-summed-relative-MAD) gate.

During the corrected native build, master advanced again to `4183fa70b` (#2153):
Direct K selectors and KS electronic-energy low-word differences changed, but
the four edited force sources, one-electron emitter and derivative cache guards
did not. Preserve the frozen ablation and existing build; do not relabel them
as measurements of the new K/convergence composition. Focused source, seed,
convergence and selector integration remains required before publication, not
a duplicate whole-method qualification merely because master advanced.

## References

- `../implemented/performance/2026-10-10-ks-implicit-cooperative-pairs.md`
- `../implemented/performance/2026-10-10-order-seven-class-major-force.md`
- `../implemented/performance/2026-10-09-direct-force-dddd-queue-split.md`
- `../implemented/performance/2026-10-06-cooperative-weighted-shell-force.md`
- `../../../docs/developer/direct_pair_recurrence.md`

## Outcome: third full-domain traversal is not work-bounded

The final native build (Slurm 6889) completed. Its corrected, explicitly enabled
resource gate (6895) executed all six Cartesian/spherical RKS/UKS/disabled-K
independent libcint cases, with no skips. Diagnostic 6897 completed fourteen
complete calls at each size; independent energy/force and exact per-class
shell/AO/primitive-AO admission checks passed. Maximum errors were
6.1391e-12 Eh / 2.4716e-11 Eh/Bohr at 48 atoms and
9.5497e-12 Eh / 3.7736e-11 Eh/Bohr at 96 atoms. Both owners' before/after
seed bytes were stable, but seeds differ across owners; this is not a matched
single-owner promotion population.

Intrusive two-electron launch intervals (baseline/candidate milliseconds) were
1681.921/1752.816 and 1778.507/1749.079 at 48 warm/moved, and
4629.416/4632.014 and 4641.222/4661.813 at 96 warm/moved. The extra traversal
raises outer logical pages from 1,033,088 to 1,549,632 at 48 and from
4,476,288 to 6,714,432 at 96. Saved recurrence work is cancelled by producer
work. No clean promotion cohort was started; repeating full timing without
changing that producer would not be justified.

The next proposal is a bounded class-major dd x dp producer borrowing existing
resident topology, with the unchanged cooperative algebra and the original
two-way fallback when coherent topology is unavailable. Preserve these V1
receipts under `.artifacts/force7-stage/evidence/` and remote
`evidence/force7-v1/`; do not relabel intrusive intervals as endpoint timing.
