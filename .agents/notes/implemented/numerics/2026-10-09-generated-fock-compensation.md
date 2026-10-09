# Decision: retain compensated RHF accuracy in generated Fock workers

Status: implemented; matched sm_120 real-device and complete-endpoint gates passed
Date: 2026-10-09

## Problem

The exported restricted bounded RHF frame reserves a Cartesian correction plane
to recover unordered FP64 atomic rounding. The previous generated launch ABI
could not carry that plane, so its presence routed every shell class through the
bounded generic recurrence. This avoided mixing corrected and uncorrected sums
but discarded the selected generated schedules even for already qualified
classes. The retained 230-AO ethane/aug-cc-pVTZ DF-RCCSD(T) endpoint exercises this
route; it is not evidence that changing screening would be equivalent.

## Decision

Share the runtime `CompensatedOutput` value between native and generated
launchers and kernels. Its host/device declaration is separate from the
device-only atomic implementation. The compiler-owned canonical scatter is
polymorphic in its output; it still owns every operator/spin/symmetry coefficient.
Packed, subgroup, component-lane, mixed and optional alternative workers carry
the same two pointers. Pointer-only callers construct a null correction.

Bounded pages, retry waves and streaming passes receive the exported reference's
existing plane. Uncovered classes retain their compensated generic recurrence;
class masks remain disjoint. Clear/fold ownership and the allocation's budget,
Cartesian offsets, spherical projection and final physical rebuild are unchanged.
The null-plane branch issues an ordinary atomic without recovering an unused
previous sum. Compensation remains non-bitwise-reproducible.

This changes the private generated launch ABI. Source call sites remain
compatible, but every generated artifact and native registry must be rebuilt
together. Compiler source/signature identities change normally; no cache key is
overridden to reuse an incompatible binary.

Runtime/device includes are emitted globally, before any generated profile
namespace. Fragment-local copies are stripped when composing a production shard.
The first nonempty AOT build exposed that including the runtime header inside a
profile namespace creates a different, shadowing `generativeqc` namespace; the
global-prelude rule and a one-include/before-namespace test prevent its return.

## Rejected alternatives

- Removing the correction would reintroduce the exported-frame error identified
  by the earlier signed-response qualification.
- Adding a plane pointer to each quartet would grow the protected 192-byte task
  ABI and every bounded descriptor arena for a value invariant across the launch.
- A global device pointer would violate stream/owner isolation and graph replay.
- Skipping all generated classes whenever a plane exists is no longer necessary
  once their scatter can consume that same plane.

## Invariants

- No new recurrence, operator coefficient, mixed-precision policy, screening
  threshold, oracle-dependent production work or numerical gate.
- Exactly one clear and fold per Fock build, including final physical export.
- Generic fallback for uncovered classes and the retained native DDDD source.
- Existing correction capacity accounting, cold/warm lifecycle and task ABI.
- No speedup claim from isolated Fock timing or a changed SCF iteration count.

## Evidence

Compiler/scatter/schedule/production tests, signature checks and task ABI checks
pass locally. A real-GPU analytic ssss test starts a nonzero-offset sum at 2^60,
launches the actual generated worker, then cancels 2^60. The surviving RHF value
must match pi^(5/2)/8, with untouched sum/correction sentinels. Existing shuffled
dyadic cancellation and nonfinite tests now exercise the launch value by value.
The optional Rys/Libcint tests now express both null and non-null planes; their
opt-in GPU execution is not claimed by this qualification.

The focused Python union passes 169 tests with two skips. Compiler structure
checks cover 494 modules; SCF structure covers 224 modules. Electronic boundaries,
the 339-file CUDA ownership inventory, Ruff, clang-format and whitespace checks
pass. The actual generated analytic gate and eight Cartesian/spherical
source-handoff cases pass on n2 in Slurm job 2768.

Qualification uses a detached worktree at
`4385f72751b829883407c01106186917c344317b` and a separate source mirror on n2.
Reproduction and raw evidence live in
`/home/jzzeng/codes/qc-branch-audit-20260922/evidence-generated-fock-20261009/`.
Full CUDA compilation uses verified ccache; GPU execution uses finite Slurm
`main` allocations with `gpu:pro6000:1`, preserving assigned device visibility.
The complete ethane endpoint retains the old input, strict tolerances, force
publication and all algorithm selectors. Numerical and timing receipts below
come from complete execution, not from compilation.

### Inventory-controlled comparison

The retained cold endpoint was built with `portable_cuda`, whose generated
inventory is empty despite CUDA and AOT being enabled. An initial same-allocation
portable pair (Slurm job 2762) therefore cannot qualify this generated-route
change: native times were 676.050690205 / 673.527390598 seconds, with RHF
129.660816914 / 129.193255715 seconds. Both preserve the numerical audits, but
the generated analytic gate did not execute. The complete evidence is retained
and excluded for the explicit empty-inventory reason, not because of its timing
or solver trajectory. No improvement against the historical portable cohort is
attributed to generated routing.

The proper comparison rebuilds both sides with the same nonempty `sm_120`
inventory and the same new sink ABI. Its control restores only the old three
compensation-triggered routing decisions in `cuda_rhf.cpp`; the candidate keeps
generated classes enabled. This isolates routing, rather than comparing old
portable master with a differently configured AOT candidate. Modified-file
digests agree between local and remote candidate sources, and differ only in
`cuda_rhf.cpp` for the control. Generated compile commands retain ccache and
`compute_120` / `sm_120`, without fast math.

The paired complete endpoint uses fresh processes, identical input/controls,
and one allocated GPU. It checks identical compiled inventory and unchanged
binary/input digests, requires the actual generated analytic success message,
and retains progress journals and GNU time measurements. Its independent
accuracy audit uses conservative triangle bounds from retained oracle error
receipts, because those receipts do not contain signed oracle values; no fresh
PySCF or finite-difference run is claimed. The unchanged energy gate is 1e-8 Eh;
all 24 forces must meet the conservative 3e-7 Eh/Bohr absolute bound. Independent
finite differences still cover only C0-z/H1-x at 1e-4 and 3e-5 Bohr. Reference
energy/density tolerances stay 1e-12 / 1e-11, and independent replay, Lambda, Z
and stationarity checks stay at 1e-10.

### Complete endpoint result

Slurm job 2768 completed both fresh-process cold DF-RCCSD(T) energy-plus-force
endpoints for ethane/aug-cc-pVTZ with aug-cc-pVTZ-RI, 230 AO and 488 auxiliary
functions. Both compiled inventories contain 22 shell-class entries and have
SHA256 `a3d0588e0ddc8523d36b61a37569699b3df766e3643a1b31683b60ac56c9c936`.
All binary/input identity and unchanged acceptance checks pass.

| Measurement | Legacy compensated route | Generated compensated route |
| --- | ---: | ---: |
| RHF seconds | 129.497453410 | 92.776048241 |
| Complete native endpoint seconds | 676.322367701 | 646.882652883 |
| Fresh-process wall seconds | 676.66 | 647.24 |
| Total energy, Eh | -79.71851664319274 | -79.71851664319261 |
| RHF iterations | 19 | 19 |
| Native numeric capacity, bytes | 7,107,919,137 | 7,107,919,137 |
| Process MaxRSS, KiB | 3,661,692 | 4,049,696 |

This single pair observes a 28.3569% RHF reduction and a 4.3529% complete native
endpoint reduction. It is not a statistically qualified speedup: execution is
sequential, and concurrent unrelated host compilation was observed on n2. No
profiling instrumentation or fresh oracle computation is included in these
endpoint times. Process MaxRSS increases by approximately 0.37 GiB despite the
equal planned numeric capacity; this run does not measure a GPU allocator peak.

The candidate's conservative independent energy-error bound is 2.3306e-12 Eh.
Its all-component retained-force error bound is 1.4499e-9 Eh/Bohr; the maximum
two-coordinate independent FD gate ratio is 0.104257. The maximum candidate
replay/Lambda/Z/stationarity residual is 6.1153e-13; stationarity is 5.0149e-13.
The largest candidate/control force difference is 4.1954e-10 Eh/Bohr. No numerical
threshold is relaxed and no failed solver attempt is discarded.

Available semantic work counters agree: RHF 19, CCSD 20 iterations and 38
evaluations, Lambda 21 iterations/42 actions, Z 12 iterations/13 actions,
J/K 17 actions; GEMM and contraction ledgers also agree. Orbital derivative
census is unavailable for this consumer: its zero quartet/jet counters remain
unmeasured sentinels, not zero-work claims. Clean endpoint receipts do not provide
a complete RHF quartet census for comparing the different kernel schedules.

`evidence-generated-fock-20261009/summary.json`, `report.md`, `aot-runs-2768/`,
`aot-build/`, `build-aot.sh`, `run-aot.sh` and `analyze.py` retain the measurements,
provenance, exact routing delta, rebuild and audit procedure. The final patch
includes the new untracked header, ownership entry, tests and note; no Git commit
or GitHub submission is made by this qualification.

## PR integration qualification

PR preparation rebases the change onto master
`cdb2131a47aaeb003bbc85fc76cf172848b8044b`. Its newer optional work-bucket Fock
launchers must carry the same two-plane value in the native declaration, stub
and compiler-emitted extern declaration. A regression test checks all emitted
work-bucket declarations; the standalone host scheduler census includes the
real sink declaration and mocks that ABI, not an obsolete raw pointer.

The rebased sm_120 production ssss shard (including its work-bucket variant),
complete 22-class registry and fallback stub compile with verified ccache on n2.
Compiler structure checks now cover 504 modules; electronic boundaries cover
107 shared native modules. The rebased focused compiler/scatter/schedule/registry
suite passes 133 tests with two skips, including the corrected host scheduler
fixture. SCF structure and the 339-file ownership inventory
still pass. This is ABI/integration validation, not a new complete-endpoint GPU
qualification. The paired timings and independent error bounds above remain
observations on the frozen 4385f727 source cohort, not measurements on the
rebased master with its other independent performance changes.

## Standalone validation integration

The standalone f-shell release gate, schedule tuner and benchmark CLI also
consume generated Fock code. They must resolve the runtime sink headers through
the source include directory (or the same bundled wheel assets). The f-shell
object cache and CI restore key include both runtime headers; the numerical gate
rejects an object if those dependencies change after compilation. Generated CUDA
bytes alone no longer identify the complete compilation input. Compiler logs are
retained with the f-shell CI artifact so failed compiles remain diagnosable.

The numerical fixture driver declares Fock outputs as `CompensatedOutput` and
passes the address of an actual `{sum, nullptr}` value to `cudaLaunchKernel`.
Force outputs retain their pointer ABI. Passing `&result.pointer` for Fock would
make CUDA copy the adjacent buffer element count as the correction pointer.

The host regression compiles all eight driver declarations against signatures
extracted from actual emitted kernels and intercepts direct/persistent launch
arguments. Restoring either the old declarations or old argument packing fails
this regression. Standalone checkout and simulated-wheel commands additionally
pass real host preprocessing, and both runtime-header changes invalidate the
cached object in the cache regression. These are host integration checks, not
new GPU numerical or performance qualification; release CUDA validation remains
the unchanged CUDA 12.9 CI gate.

## Consequences and revisit conditions

The correction no longer changes class scheduling. Atomic residual recovery
still costs additional work, and a larger fraction of generic high-angular
classes may limit complete-endpoint benefit. Revisit if independent frame/force
gates fail, owner isolation changes, or complete matched endpoints regress.

This routing-only follow-up supersedes the generated-ABI limitation in
[the earlier signed-response decision](2026-10-07-compensated-signed-jk-response.md);
its compensation algorithm, response policy and independent acceptance remain.
