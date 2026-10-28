# Shared precision admission qualification

This scoped #2227 R4/R6 migration moves budget-Fock work admission, strict warm
returned-state proof, and Standard fallback decisions into the native query.
It does not transfer model/gradient qualification, solver defaults, candidate or
auditor execution, observable comparison, or independent warm-seed ownership.
R3 is excluded. Neither the parent issue nor the full precision lifecycle is closed.

## Source and execution identities

Baseline: `162ea8332179af8e2974562e30ae92f284bcc4f8` (#2857).
Candidate production: `7adf42cea`, including `c465b71d8`'s precision policy.
Subsequent changes correct baseline-reproduced test expectations and retain this
evidence; they do not change production source. `identities.json` binds both CPU
and CUDA products and the unchanged packaged stationary AOT libraries.

Release builds use GCC 11.4, CUDA 12.9.86/sm_120 and verified sccache 0.16.0.
Before/after statistics describe the shared cache, not exclusive task hits.
Real GPU execution uses finite Slurm jobs on node1/main/RTX 5090 and preserves
the assigned visibility. No GPU work runs on node3. Raw profiler databases,
build logs and installed test binaries remain in ignored qualification artifacts.

## Complete endpoint comparison

Four configurations (RHF, UHF, PBE RKS, PBE0 RKS) request energy or energy/forces.
The scalar is H2; the fleet is H2/water, with neutral singlets except H2+ in the
UHF configuration. CPU uses STO-3G and GridSpec(12,6,12); CUDA uses spherical
def2-SVP and GridSpec(24,12,24). All use Fast with initial_guess=None.

Two source-isolated persistent processes prime their ordinary model/artifact
caches. Each measured protocol creates fresh computational owners and cold
initial densities; static Calculator construction and process imports are outside
this scope. Scalar timing includes preparation, execution, publication and
cleanup. Fleet timing includes preparation/cold, two frozen replays, moved
coordinates, two moved replays, teardown and a separately retained protocol total.
These small-grid measurements are not a basis/grid-limit or general-method claim.

Eight alternating ABBA/BAAB groups retain **1,792 complete scalar/batch calls per
backend**, combined across baseline/candidate. Every sample preserves E/F, actual
iterations/Fock/grid/J-K work, policy disposition, solve stages, operator census
and gradient-work receipts. Gates are identical work/policy signatures, energy
within 2e-9 Eh, forces within 2e-8 Eh/bohr, and every median time ratio <=1.05.

- CPU maximum ratio: **1.0149881714488609**; exact energies and forces.
- CUDA maximum ratio: **1.0074556733868567**; maximum E difference
  **1.1368683772161603e-13 Eh**, force difference **1.3167245072054357e-13 Eh/bohr**.
- CPU/GPU raw samples are retained losslessly in the two `*-samples.json.xz`
  files. CUDA timing/profiling is Slurm 14833. A separate targeted validation
  job 14834 briefly used the node's other assigned GPU during this campaign;
  this is scheduled shared-node evidence, not exclusive-node timing.

`qualify.py` reproduces the matched comparison; use finite srun for CUDA.
`endpoint_worker.py --once` is the complete 56-call untimed profiler matrix.

## Native work, science and lifetime gates

The current production build passes 128 CTests and 304 focused Python tests plus
25 subtests. The installed public-header SDK query executes with Python/compiler
absent from PATH. The GPU-linked SDK also passes inside Slurm. Host ABI and Python
transport tests cover exact-state binding, required model operators, unknown
counts, budget neighbours, descriptor failures, integer-width wrapping, output
transactionality and authoritative native errors.

GPU qualification covers Direct/DF HF/KS, cold/frozen/moved prepared fleets,
independent audit histories, numerical failure isolation, range-cache invalidation,
all five standard semilocal/hybrid models, WB97M-V and r2SCAN-3c. Independent
references use PySCF 2.14.0, GPU4PySCF 1.9.0 with Libxc 0.9.0/CuPy 14.2.0,
and dftd4 4.3.1 plus the separate gCP reference. Existing numerical gates remain
unchanged. `native_policy_guard.py` poisons all three missing-protocol compatibility
predicates during actual GPU tests, proving that the native policy is consumed.

Slurm 14831 passes 53 tests and exposes three existing expectation failures.
The unchanged baseline reproduces all three in 14832: two tests omitted the
already supported strict-warm status, and one omitted the completed Standard
recovery after a moved proposal failure. The repaired assertions preserve every
independent E/F gate and native-force ownership guard, and additionally verify
strict counts, returned-state Fock/audit binding or the failed-candidate/FP64
recovery solve record. All three pass in 14834. Their original failures are
retained; neither a tolerance nor a numerical model is removed.

Nsight reports identical complete-matrix GPU work for both engines: **94,042
kernels**, **149 graph API calls**, **5,344 explicit dynamic allocations**, and
**118,503,293-byte simultaneously live explicit peak**. Both end at zero dynamic
live bytes. H2D is 5,658 copies / 17,760,192 bytes; D2H 5,959 / 16,613,368; D2D
2,706 / 292,245,568. Static allocations (1,823,689 bytes) and opaque vendor/driver
memory are excluded from the dynamic peak. This is an ownership migration, not
a reduced-work claim. Detailed CSV summaries are inside `receipts.json.xz`.

## Retained unsuccessful setup attempts

The initial CPU UHF water-cation force preflight fails on both independent baseline
and candidate, including a 256-iteration probe. It is not promoted as a qualified
endpoint; the matched matrix uses the unrestricted H2+/neutral-water fleet above.
The first remote suite lacked pinned Libxc inputs, a retained auxiliary basis and
reference dependencies; the second also exposed missing packaged force artifacts
and a reference cuSOLVER search path. Full failed logs are retained. Successful
runs use complete source/provenance inputs, verified current stationary binaries,
and isolated reference dependencies with the actual CUDA toolkit path. Production
code was not changed to conceal these missing inputs or retry Python policy.

## Raw receipt recovery

To remain within the unchanged 80 MiB repository evidence budget, the three
raw `cpu-samples.json.xz`, `cuda-samples.json.xz` and `receipts.json.xz` payloads
are retained in Git commit `8d6225b4fa49a82fdaca1abc1d56927008dde76c` at their
original paths in this directory. `identities.json` records exact hashes and
recovery paths. Use `git show COMMIT:PATH` to restore them. All failures, samples
and original source identities remain unchanged; no new external archive exists.
