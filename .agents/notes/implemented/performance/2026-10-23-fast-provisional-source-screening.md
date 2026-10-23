# Decision: reduce Fast proposal-only integral source work

Status: implemented
Date: 2026-10-23

## Baseline and hypothesis

This experiment starts at master
`ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`, after #2692 and #2696. The
earlier native cold DF factor handoff is not an additional optimization over
this baseline. Existing open DF raw-J, charge-partition, Schwarz and occupied-XC
work is outside this experiment.

Fast full-range hybrid cold solves build a phase-local FP32 integral source and
reuse it for bounded mixed proposals. The FP64 closing pair uses the primary
provider after that source has been audited and retired. Historical measurements
show that removing the entire mixed cache is negative; they do not measure the
new candidate or establish a current-baseline gain.

Test a coarser `1e-8` Schwarz predicate for this proposal source, including its
cached and uncached consumers. Reducing retained rows should reduce source
generation, storage and repeated contraction work. Any extra strict iterations
could erase those savings, so complete energy/force endpoints decide admission.
This is a proposal heuristic, not an accumulated-error bound.

## Boundaries

The qualified default enables automatic selection at 64–768 AOs
and 2–96 atoms. `GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING=0` keeps the
incumbent predicate; `1` permits other sizes without bypassing the other guards.
Only already-admitted relaxed Fast full-range restricted hybrids on the default
grid, through d shells and with scalar iteration ownership can use it.
DF, pure J, range/nonlocal, strict stopping controls, unscreened
inputs and public resource-ledger owners retain their original route. Existing
optional cache capacity refusal also retains the original streamed provider.

The primary provider, strict final screening, consecutive fresh physical Focks,
canonical validation and force derivatives remain unchanged. A coarser lease
cannot survive precision handoff or supply a strict warm/force consumer. No
additional retained matrix or oracle work is introduced in production.

## Qualification protocol

Keep exploratory pilots separate from qualification. Freeze the final candidate
before measuring five fresh alternating off/default process pairs at each of
12/48/96 atoms, PBE0/def2-SVP spherical water. Each process includes constructor,
preparation, a complete cold E/F call and two frozen warm E/F calls from its own
native cold density. Retain every sample, physical Fock/iteration count and
precision receipt. Reduce warm repeats within each process before comparing
process medians. Instrumented source-work traces are separate from clean timing.

The cold 96-atom gain must exceed max(2%, twice the sum of relative median
absolute deviations). No sampled size/phase may regress by more than the same
noise floor. Small gains below that floor are not performance claims. Check
changed geometry, frozen moved replay and reset with PBE0/B3LYP/M06-2X; retain
disabled/unavailable residency and explicit strict/DF controls as fallbacks.

All endpoints must pass independent full-density GPU4PySCF references with
unchanged gates: `2.5e-6 Eh/atom`, `2.5e-5 Eh/Bohr` maximum force and
`7.5e-6 Eh/Bohr` RMS force. References supply no production density. This is
stock-grid solver qualification, not grid convergence or universal method
ranking. Record source/binary/AOT identities, compiler-cache verification and
finite Slurm allocations. Promote only after all gates pass; otherwise retain
the negative evidence and the incumbent default.

## Pilot observations and build correction

The first copied candidate sources retained modification timestamps older than
the freshly built baseline objects. Ninja rebuilt only the build identity and
relinked; the candidate selector and counter were absent from that binary. Its
`pilot/` records are invalid mechanism evidence and do not establish a gain.
After touching the changed build inputs, the rebuild log includes the actual
runtime translation unit; binary markers and diagnostic work confirm execution.
Subsequent source synchronization does not preserve stale modification times.

The repaired explicit-opt-in `pilot-v2/` population uses library SHA256
`17f8a1c53414c610dde2b2c67306b1d07577bbb7c51d7e2519069c97825708dc`.
Two alternating PBE0/96 pairs give cold medians 27.767705 / 25.115376 seconds
(off/on), both with nine Focks. Cached integral values decline from
1,476,473,708 to 1,035,084,412. Every E/F endpoint passes its independent gates.
Three native control suites and four public PBE0/B3LYP lifecycle cases pass.
These pilots ran alongside independent setup/test work and remain exploratory,
separate from the final default candidate's clean qualification.

## Final qualification and retained policy

The final default candidate passes the predeclared protocol: 50 processes and
180 complete energy/force calls. Five-pair cold medians improve by 6.373% at
48 atoms and 9.096% at 96 atoms. The 12-atom cold median is 0.264% slower; warm
changes range from -0.303% to +0.667% improvement. These small differences are
below the predeclared noise floor and are not performance claims. All 25 paired
controls retain identical phase, iteration and physical Fock counts.

Maximum independent errors are 1.84083e-10 Eh/atom, 1.13717e-5 maximum force
and 3.38396e-6 RMS force in Eh/Bohr. Original, moved, frozen, reset, Standard/DF
exclusions and source-capacity refusal pass. The benchmark now freezes warm
updates after a moved solve as well as cold, and records the source controls.
Three native GPU suites and eight public GPU cases also pass on the final
library, including public resource-ledger refusal and malformed selection.

Retain this bounded automatic policy and explicit 0 rollback. The selector is
snapshotted during preparation; changing it requires a new owner. Extend the
size/model/resource domain only with new independent complete-endpoint evidence.
Do not turn a proposal source or its timing into a final screening guarantee.

The [receipt](../../../../benchmarks/results/fast-source-screening-20261023/README.md)
retains all numerical arrays, source identities and reproducible gates. Final
core SHA256 is `378e026ff3ea23ff48987147df8aa1795d881728e3fd1657c52389a00bb3fb90`.
The recorded production inputs match the branch; later note/receipt edits do
not change the measured runtime.
