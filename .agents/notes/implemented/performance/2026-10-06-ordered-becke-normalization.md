# Decision: Parallelize Becke products without reassociating their denominator

Status: implemented
Date: 2026-10-06

## Problem

The phased stationary force path normalizes each point with one thread scanning
the atom domain. A 256-point production tile launches only two 128-thread
blocks. At 96 atoms, normalization remains under-parallelized even after the
pair and log phases have been specialized. Previous coefficient reverse/gather
and fusion experiments in #1894 have negative evidence; they are not a reason
to replace the scientifically validated phase schedule wholesale.

## Decision

Generate an eight-point by sixteen-atom-lane normalization block. Atom lanes
parallelize the frozen maximum, scaled products and product adjoints. One lane
per point sums the products in the original ascending atom order and calls the
canonical ratio AD. Both schedules share the scalar product/adjoint helpers;
the serial control retains its original interleaved product/sum loop.

The compiler owns emission and mathematical recognition. The native owner owns
resource admission and launches. Actual CUDA device/kernel thread and static
shared-memory limits, optional phased workspace and the 128-atom cap determine
admission. A miss keeps serial normalization, and existing generic/optional-memory
fallbacks remain bounded. No extra retained device allocation is introduced.

Qualification can select a schedule only before topology. Legacy artifacts
remain usable with the default selection; an explicit request fails before
allocation if the versioned configuration ABI is absent. Telemetry counters
are cumulative: benchmark consumers must difference them for each force call.

## Rejected Alternatives

- A parallel denominator tree changes rounding/order and needs broader
  numerical qualification; it is unnecessary for this scheduling win.
- Floating-point scatter atomics would change deterministic reduction policy.
- The initial 32-point/four-atom-lane prototype exposes less parallelism. Its
  earlier sanitizer receipts are not final eight-by-sixteen qualification.
- Pair-coefficient reverse/gather remains separately opt-in and default off;
  its losing evidence must not be reinterpreted as a normalization result.
- GPU4PySCF's explicit point/derivative-atom grid was useful scheduling context,
  not a formula to copy. The inspected source is `gen_grids.cu` at
  `d99e5569b469da905909ed15924692118d19990b`; our zero/log semantics and ratio
  derivative remain canonical AD owned. The endpoint oracle uses installed
  GPU4PySCF 1.8.1, a separately recorded version.

## Evidence

Slurm job 2500 on n2/main, `gpu:pro6000:1`, retains five interleaved complete
PBE0/def2-SVP energy-plus-force pairs per size/geometry. Both arms use one public
frozen native post-solve snapshot and perform one actual SCF iteration/Fock
build per timed replay. Whole gradient owners are replaced and primed outside
timing; every prime and setup call is retained. The grid has 48 radial and
16-by-32 angular points per atom. Screening/SCF/accuracy policies are unchanged.

| atoms | replay | serial median (s) | cooperative median (s) | point estimate |
| ---: | --- | ---: | ---: | ---: |
| 96 | warm | 23.572726 | 22.878456 | 2.945% faster |
| 96 | moved-warm | 23.582680 | 22.907833 | 2.862% faster |
| 48 | warm | 8.663366 | 8.518470 | 1.673% faster |
| 48 | moved-warm | 8.663816 | 8.530103 | 1.543% faster |

Both 96-atom comparisons pass the shared descriptive improvement gate: greater
than 2% and greater than the relative-MAD noise floor. The 48-atom point estimates
do not pass the 2% benefit gate; they support no material regression, not a
claimed significant win. Maximum independent energy/force errors across all
setup, prime and timed calls are respectively `1.033e-10` hartree and
`3.125e-11` hartree/bohr, within unchanged `1e-8`/`1e-7` gates.

At 96 atoms, each force retains 9,216 phased batches, 2,359,296 points,
226,492,416 normalization atom entries and 10,758,389,760 primal and reverse
pair visits each. Seven phase launches per batch remain unchanged. Logical
traffic models are not hardware transactions; the schedule removes no pair work.

The measured source is base `83109befe5b97a5542b30e169da74c81d04719e6` plus
the retained dirty reconstruction patch, core identity
`b0e32abf342aa53877f2a350801a1dfdc1bc74d341df0cebe82a738d74429d2f`.
The source archive was reconstructed and all consumer hashes independently
verified before publication. The remote archive has no Git metadata; its null
Git fields are retained rather than converted into a fictitious clean checkout.

Integration is already based on `80296618e2003427b220331e1c0d4dad6752391e`.
The normalization compiler, scalar helper and native owner files are identical
between the measurement and integration cohorts. Separate integration receipts
authenticate its numerical/owner/sanitizer gates; they are not a new endpoint
performance measurement. The user explicitly accepted the frozen old baseline
instead of chasing subsequent master changes.

Final Slurm job 2505 passes 49 normalization/interface/fallback tests, all four
sanitizers on 39 isolated CUDA cases with zero errors/hazards/warnings, 144
synthetic shared-owner cases, and five CPU snapshot/lifetime cases. Five of the
49 cases are independent three-atom RKS/UKS/finite-difference checks of the
deliberately retained small-domain fallback, not admitted cooperative physics.
The compiled cooperative kernel uses 56 registers, 9,424 static shared bytes
and no stack/spills. These resource attributes are not endpoint peak memory.
Current host/compiler validation passes 574 tests with 44 real-GPU skips and
checks 480 compiler modules with zero dependency errors.

## Consequences And Limits

This is a narrow #1894 scheduling lane, not closure of the dedicated-primitive
or #1893 shared-domain workstreams. No 5090 measurement is claimed: the local
device was unavailable. Full 96-atom UKS, matched cold/moved A/B, intrusive phase
profiles, whole-core compilation cost and concurrent endpoint memory peak remain
unmeasured. The publication accepts numerical qualification and retains the
positive endpoint comparisons without fabricating generic promotion-envelope
memory/compile gates. Compilations use the existing ccache; cache hit counters
are shared diagnostics, not attributed compilation-speed evidence.

Revisit the denominator ordering only with an explicit new numerical contract.
Revisit the schedule when pair-domain work, tile width, admitted atom domain or
device resource limits change. Use `tests/python/test_becke_normalize_evidence.py`
for offline revalidation of raw vectors, solver histories, counters and medians.
