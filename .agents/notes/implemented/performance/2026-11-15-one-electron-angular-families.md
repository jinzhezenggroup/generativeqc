# Decision: retain through-f one-electron CUDA families beside through-g

Status: implemented; native/device endpoint qualification pending
Date: 2026-11-15

## Problem

Expanding the generic one-electron value and gradient programs from f to g also
expanded the programs used by ordinary lower-angular bases. On the original
SM120 board, after restoring linked LAPACKE, three 96-atom cold endpoints still
missed the unchanged GPU4PySCF comparison by about 1.1--1.6%. PBE0 Direct had
recovered. These are retained negative observations, not an accepted full matrix.

Corrected-source diagnostic job 1584 compared native identity `18a4dd49...`
with the earlier `9d305b13...` image. HF DF's first one-electron response took
about 515 ms versus 161 ms; later responses were about 98 ms versus 94 ms.
SM120 image metadata recorded 255 registers for the expanded value kernels,
versus 168--196 previously. Gradient stack frames grew from 1152--1200 bytes to
3456--3488 bytes. Intrusive traces and image resources motivated this bounded
specialization; they do not themselves establish complete-endpoint recovery.

## Decision

Emit a through-f family from the same canonical scalar emitters and output
policy, with an independent namespace and header guards. Keep the complete
through-g family. The compiler supplies typed derivative bindings; native
traversal retains one shared implementation of normalized AO/primitive loops,
weights, signs, nuclear partitioning and gradient accumulation.

Select the smaller family only from a producer-validated maximum angular
momentum across the physical shells. Cartesian g has a single expansion term:
neither expansion stride nor AO dimension proves the angular domain. Unknown
views conservatively retain through-g. Matrix exporters use their actual packed
shell metadata, compact gradient owners use their physical system, and prepared
Direct J/K owners inherit their constructor's explicit through-f admission.

No solver controls, precision, screening, physical work, convergence checks,
geometry/basis data or benchmark timing boundaries change. The CUDA view adds
only scalar admission metadata; it does not own buffers or extend their lifetime.
The general `DeviceBatch` ABI is unchanged.

## Validation boundaries

Sixty-eight cached native host cases pass against independent Libcint value
blocks and normalized/raw/spherical derivative blocks. The emitted f family and
generic g values coexist without namespace or guard collisions. An earlier host
fixture compiled all large f/g programs together and reached its 180-second
compile limit without a numerical result; that attempt remains retained. The
corrected fixture uses bounded separate translation units, and through-f emission
does not require unrelated through-g derivative definitions.

The canonical derivative emitter supplies the same typed binding at its own
declared angular bound. Actual CUDA compilation, existing Cartesian/spherical
f/g arbitrary-weight tests at their unchanged independent gates, resource and
transactional controls, and fresh complete E+F timings remain required. Host
arithmetic is not device qualification. A paper figure requires the full fresh
three-process cohort; partial, slower or failed observations must remain visible.

## Rejected alternatives and revisit conditions

Changing SM120 native-only versus native-plus-PTX packaging did not recover the
two controlled cold comparisons. Do not repeat that as an assumed remedy.
Removing g support, inferring its absence from packing stride, changing numerical
targets, or moving initialization outside the existing cold timer would change
the claim rather than qualify this schedule.

Revisit the specialization if complete-consumer measurements do not recover its
cost, if emitted program/resource boundaries change, or if a broader shell
domain is admitted. Preserve the generic fallback and independently qualify each
new mathematical domain instead of extending a shape heuristic.
