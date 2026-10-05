# Decision: distinguish executed AO/grid schedules from mask admission

Status: implemented, device-qualified instrumentation; complete P0-C optimization pending
Date: 2026-10-05

## Problem

Issue #1893's P0-C lane requires an actual force AO/XC work census. Existing
active-map statistics describe selected domains but do not prove the number of
AO evaluation passes, density-submatrix gathers, or local projections executed.
The previously qualified atom-indexed gather changes cold trajectories and is
not a promoted default. Its mechanism evidence does not complete this lane.

## Decision

Count schedules at their native launch sites in the existing indexed `CudaGrid`
owner. No second mask representation, density transformation, GPU allocation,
screening threshold, reduction formula, or provider selection is introduced.

- Record deriv=0/1/2/3 passes and active AO×point counts separately. A deriv=2
  pass includes lower-order jets; do not report it as another deriv=1 pass.
- Count discovery separately from evaluation. Empty maps still contribute to
  evaluation points and the dense comparison domain, but launch no AO pass.
- Count actual density gathers, projected matrices/FMA pairs, output values,
  identical-spin copies, map uploads, and scatter elements.
- Explicitly identify MO-feature tiles; density-gather/projection fields do not
  pretend to cover the separately scheduled orbital-factor route.
- Expose counters through an additive optional ABI. An older library reports
  their absence rather than invented zeros.
- Opt-in stage timing measures AO evaluation, density gather, projection
  (including the restricted-spin panel copy), and feature construction using
  the owner's existing CUDA events. It deliberately adds fences and is not a
  clean endpoint timer. Normal deferred leases retain their error-publication
  and synchronization policy.
- Subtract cumulative counters as well as timers on prepared force replays.

Traffic fields describe explicit logical writes/copies, not profiler-measured
DRAM transactions. Discovery and evaluation panel sizes are work, not an extra
allocation; the existing bounded arena remains the storage authority.

## Evidence

The focused initial host suite passes 14 tests. The expanded host suite passes
103 tests and skips one opt-in CUDA test. It executes the native counter helper,
checks every ctypes field offset/size, tests all derivative orders and overflow,
checks empty-map accounting and prepared-replay deltas, and preserves the
existing generated geometry and indexed-TensorIR gates. CUDA/native test targets
build with verified ccache launchers. Compiler ownership checks cover 440
modules with zero dependency errors.

The local n3 GPU node is drained for maintenance. The user permits choosing any
available node among n1/n2/n4/n5; this is not a requirement to use every node.
Select a suitable available allocation and avoid redundant coverage jobs.
Portable source/binary snapshots and finite Slurm jobs retain per-job source
identities and binary hashes. n2's PRO 6000 results must not be pooled with
RTX 5090 endpoint measurements.

n2 Slurm 2388 completed successfully on PRO 6000: 14 selected real-device grid
tests, the native local-AO independent CPU E/V/feature/nonlocal/capture/resource
gates, and a 12-test warm census replay all pass. The first two attempts failed
before device execution because of missing harness `ptxas` and the node's
GCC-12/G++-11 mismatch; retain them as failed harness evidence, not numerical
failures. The retry explicitly records G++-11 as NVCC's host compiler.

n2 Slurm 2390 also completed successfully: the 12-test memcheck replay reports
zero errors, and the 12-test racecheck replay reports zero hazards, errors, or
warnings. These are PRO 6000 correctness/sanitizer gates, not RTX 5090 endpoint
performance evidence.

n4's first job failed because kernel 580.173.02 and installed user libraries
580.178.04 disagree. The matching `libnvidia-compute-580` Ubuntu snapshot package
is extracted under the ignored job artifacts and selected only through the
job's library path. No system driver, kernel module, scheduler state, or device
visibility is modified.

### Executed 96-atom force work

n1 Slurm 6024 completed on RTX 5090. Every one of its 12 native E+F calls was
independently compared with all six reference calls on the identical geometry:
72 pairings pass, with maximum energy error 1.060e-10 Eh and maximum force error
2.171e-11 Eh/bohr. Actual native Fock-build counts are cold 27, moved 15, and one
for every warm/moved-warm replay. This is intrusive mechanism profiling, not a
clean endpoint timing campaign or a speedup claim.

The first warm replay executes:

- 2,359,296 points in 9,216 tiles, including 768 empty tiles;
- 8,448 deriv=2 AO passes and no distinct deriv=1 pass;
- 388,141,056 active AO×point visits against 1,811,939,328 dense visits;
- 3,881,410,560 AO-jet values written, exactly ten per active visit;
- 8,448 density gathers totaling 633,258,592 elements;
- 33,792 matrix projections and 324,228,399,104 projection FMA pairs;
- 12,420,513,792 bytes of identical-spin panel copies and 12,129,408 map-upload
  bytes, with no scatter or MO-feature route.

The RKS route already computes the identical spin projection only once. Its
four projected matrices per nonempty tile are not eight GEMMs; the second spin
panel is explicitly copied. The copy is a measured duplication candidate, but
logical copied bytes alone cannot establish an endpoint benefit from removing
it. Do not claim a factor-two projection-FLOP reduction from this census.

Opt-in stage times for that replay are AO 832.415 ms, density gather 37.493 ms,
projection including copy 900.610 ms, and features 128.929 ms. These fences alter
the schedule; none is substituted into a clean additive endpoint ledger.

All 12 force censuses reconcile with their existing map owner's work: AO visits,
ten-jet materialization, four times the selected AO-square work for projection,
density gathers, and map uploads. Warm discovery is zero; cold and moved map
discovery are accounted for separately. The frozen scientific source identity
is `3ccb0334edefcc46f2e6b99fc4ff2fb0b9be6ce8e4cb7c74a29e6c40661a29ea`.
Raw receipts and the independent host-only verification are retained under
`.artifacts/n1-profile-6024/.artifacts/endpoint-6024-96-profile/`.

### Clean endpoint collection

n4 Slurm 630 has completed its clean 48-atom subcampaign. All 72 same-geometry
pairings pass: maximum energy error 8.868e-12 Eh and maximum force error
2.750e-11 Eh/bohr. Five-call warm medians are native 9.051 s and reference
6.077 s; moved-warm medians are native 9.066 s and reference 6.070 s. Actual
native Fock builds are cold 25, moved 12, and one per warm replay. All 12 census
records also reconcile, and all intrusive stage timers remain zero.

The same job completed its 96-atom subcampaign and is COMPLETED with exit 0.
Its 72 same-geometry pairings pass with maximum energy error 1.060e-10 Eh and
maximum force error 3.338e-11 Eh/bohr. Native warm/moved-warm medians are
25.132/25.135 s; reference medians are 10.313/13.858 s. Actual native Fock builds
are cold 26, moved 12, and one per warm replay. Both sizes' executed-work
censuses reconcile. These are clean native/reference observations, not evidence
of improvement over an uninstrumented same-node native baseline. Receipts are
retained under `.artifacts/n4-clean-630/endpoint-630-48-clean/` and
`.artifacts/n4-clean-630/endpoint-630-96-clean/`.

n5 Slurm 1450 was still PENDING when inspected on 2026-10-05 at 20:33 +08:00.
The user clarified that the four nodes are alternatives, not required coverage.
After checking job identity and pending state, only our redundant sanitizer job
1450 was cancelled; Slurm confirms CANCELLED with zero runtime. Its already
completed n2 equivalents retain the sanitizer evidence. No competing user
allocation was modified. Future GPU work selects a suitable available node
rather than allocating all four merely for coverage.

## Alignment with the new Direct analysis

Issue #1892's 2026-10-05 12:01:16 UTC analysis, based on #1965 Slurm 6018,
refines P0-B into a barrier/divergence/local-state throughput problem, rather
than a register-count target. The reported generic force kernel has 16.67%
achieved occupancy, about 44% barrier stalls, and 17.91 active threads per warp.
Local load/store counters are requests, not bytes and not proof of spilling.

The actionable Direct direction remains single screen/classify, homogeneous
shell-class/source queues, generated consumers, and a residual generic
fallback. Bounded 128/64-thread variants are experiments, not defaults justified
by theoretical occupancy. Retain the rejected repeated angular scans and
require complete 48/96-atom E+F and actual semantic-work qualification.

This does not transfer the Direct barrier diagnosis to the AO/XC owner. P0-C
continues independently: distinguish admitted map work from actual AO/projection
work, separately measure atom gathers and owner reductions, and remove only
demonstrated duplicate work. Do not tighten AO masks or alter cold trajectories
as a speculative proxy for the Direct findings. The signed-response reuse note
in #1972 also does not authorize importing SCF density screening into response.

## Remaining P0-C work

This instrumentation is not completion of the block-sparse layout objective.
The first shared indexed-domain/compiler-program contract and ordinary-force
producer binding are now implemented; see
`../architecture/2026-10-05-indexed-ao-grid-domains.md`. SCF producer integration
and complete consumer qualification still remain. Also required: separate
atom-gather and point-owner reduction timing/bytes;
same-geometry native/reference local-domain comparison; demonstrated duplication
removal; RKS/UKS fixed-density E/V and force gates; profitable dense fallback;
and complete 48/96-atom E+F improvement without prohibited cold/moved regressions.
Do not promote an optimization from these counters or intrusive stage times.

## Completed same-geometry reconciliation

n1 Slurm 6071 completed its intrusive 96-atom reference census with exit 0.
The comparator reconciles it against both completed v4/control campaigns from
6070, passing 24 same-geometry numerical pairings for each variant. The exact
protocol differences are five versus one replay, and `warm_definition`'s
description derived from that replay count. Validate that derivation before
excluding these two execution-count fields; retain equality of every scientific
field, both exact geometries and the basis checksum. Do not silently relax
grid, convergence, precision or numerical acceptance gates. Receipts explicitly
record the unequal replay counts and exclude all observer timings.

For geometry-zero warm force calls (768 physical AO, 2,359,296 grid points),
reference publishes 576 deriv=1 plus 576 deriv=2 panels, 580,878,336 AO-point
values in each derivative domain, and 8,132,296,704 jet values including padded
local rows. Both native variants execute 8,448 deriv=2 panels with no separate
deriv=1 evaluation, 388,141,056 active AO-point values, and 3,881,410,560 jets.
This is not evidence that the native force gap is mostly duplicate collocation
or inactive AO evaluation: native already does less of those recorded domains.

The qualified v4 restricted witness reduces actual force density gathers from
633,258,592 to 316,629,296 values. Native projection FMA-pair work remains
324,228,399,104 in both variants; 12,420,513,792 bytes of identical-spin panel
copies also remain. Neither logical bytes nor reference's padded square-domain
sum is measured DRAM traffic. The clean 6070 endpoints, not this intrusive
observer, determine whether the reduced gather work is profitable. They do not
show material 48/96 improvement and include a 96 cold 29-to-30-Fock regression.

Reconciliation receipts and input hashes are retained under
`.artifacts/n1-reference-census-6071-96/`; clean campaign receipts remain under
`.artifacts/n1-layout-native-6070/`. The subsequent native CSR binding is
described in `../architecture/2026-10-05-native-csr-indexed-grid-binding.md`;
these v4 observations must not be relabeled as its device qualification.

## Actual warm public-force kernel census

n1 Slurm 6090, v6 identity
`6fb2d9b652e10a78ce921699b95c6a812242adc5ca3da3a649f7541959af7750`,
captures the first geometry-zero warm public-force invocation. Exit 0; all 72
same-geometry reference pairings pass (maximum E/force errors 1.055e-10 /
3.275e-11). Preserve its source/binary/harness hashes, raw `.nsys-rep`, SQLite,
CSV reports and audited `kernel-domain-analysis.json` in
`.artifacts/n1-force-kernel-profile-6090/`.

Aggregate kernel durations: combined AO/XC point setup, AO gradient construction,
atom gather and point-owner reduction 2.675511 s / 9,216 kernels; final lane
reduction 0.097822 s / 9,216; density gather 0.014520 s / 8,448; AO evaluation
0.808566 s / 8,448. Device-to-device transfers aggregate 0.009351 s over 8,449
copies. Removing the restricted panel copy cannot plausibly recover seconds
from this observation alone. The generic Direct-force kernel remains a separate
#1892 consumer (6.555168 s); this is not P0-C gather work.

Do not rename the 2.675511 s combined kernel as pure atom-gather time. The trace
does not separate its point setup, AO arithmetic, atom gather and owner reduction.
Likewise `geometry_reduce` is the all-source lane publication, not exclusively
the point-owner reduction. Summed kernel times are not endpoint wall time and
may overlap; default Nsight device-event completion tracing adds overhead and
can add dependencies. Internal preparation/eigensolver work appears within the
public-force capture, so the preceding external energy call is excluded but
pure SCF-free force attribution is not established.

The source domains include 37,261,541,376 ordered atom-membership tests,
1,164,423,168 owner-reduction scalar additions, 9,315,385,344 logical shared
gradient write bytes, and 16,307,453,952 logical lane-partial read bytes. These
are schedule-domain counts, not machine instructions or measured DRAM bytes.
The existing cooperative gather still scans every active row for each atom.
See `../../proposed/2026-10-05-stable-grid-owner-gather.md` for a bounded,
order-preserving scheduling candidate; it is not implemented or qualified.
