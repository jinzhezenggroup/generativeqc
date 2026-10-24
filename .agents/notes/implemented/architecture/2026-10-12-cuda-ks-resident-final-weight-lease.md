# Decision: validate CUDA stationary D/W without a detached matrix frame

Status: implemented
Date: 2026-10-12

## Problem

The native Direct integral derivative provider already consumed resident spin
densities and total D/W, but acquiring W implicitly required `read_final_state`.
That exported D/F/C/epsilon and, outside the measured device-product default,
performed dense host validation and weighted reconstruction. Merely allowing a
CUDA force bit in the prepared CPU force controller would additionally run CPU
XC derivatives. Neither route constitutes the source-complete native CUDA
cutover requested by #2227.

## Decision

Factor the backend-neutral determinant product acceptance and KS scalar
component-energy checks out of the detached validation path. The resident owner
authenticates its exact token and physical F[D] generation, canonicalizes on its
ordinary device eigensolver, computes the existing validation products, and
supplies bounded scalar evidence to those same policies. The quadratic HF
validation energy is never substituted for the KS component energy.

Reuse the four dead iteration matrices and charge bounded reduction packets in
the common state-size query. Tiny owners keep their existing native matrix
kernel; extending its square-padded product with a right transpose and leading
reduction rank avoids imposing a new cuBLAS allowance. Larger owners reuse their
already prepared handle, with the native route retained when it is unavailable.
No force equation or density/XC functional is copied.

After every gate passes, the existing weighted projector and UKS spin-sum
kernels stage total D/W. A device finite scan and scalar status drain precede
publication. The published scratch cannot be reused by another proof while its
token remains current; repeated leases, or a lease after a validated weighted
detached read, borrow it unchanged. Failure drains submitted work and revokes
final eligibility without mutating the separate last-good warm density.

The actual uncached prepared Direct integral-derivative consumer calls this
lease instead of exporting a detached frame. Cached consumers retain their
existing verified-frame contract. DF retains its explicitly host-staged response
contract rather than claiming an all-device cutover without implementing it.

## Rejected alternatives

- Enable native CUDA forces by changing the CPU force predicate: this still
  invokes CPU XC and does not provide a complete device source controller.
- Force the experimental/default-limited device-validation environment selector
  for every energy endpoint: it changes an evidence-bound energy policy and still
  exports a detached frame. The explicit resident consumer requires its own
  admission, not a wider default promotion.
- Copy acceptance thresholds into the CUDA adapter: diagnostics are backend
  evidence; the shared policy remains the sole acceptance authority.
- Allocate another dense validation/W arena or add cuBLAS to tiny molecules:
  phase-local scratch and the retained native product already suffice.
- Treat native numeric slab receipts as complete endpoint memory receipts:
  provider/context internals, host storage and complete force consumers remain
  distinct obligations under R2.

## Invariants

- Token/owner/epoch/model/device/generation checks precede any launch or transfer.
- Canonical products, solver status and all existing absolute/scaled gates remain
  mandatory; symmetry/finiteness scans cover S/H/D as well as F without a host
  detached scan.
- W comes from the accepted C/epsilon/occupation frame, including empty beta;
  it is not a lagged SCF projector or a handwritten stationary equation.
- A resident proof charges scalar packets, not matrix exports or detached reads.
- No public C ABI, force bit, complete native CUDA force claim, or energy-only
  default policy changes in this prerequisite.

## Evidence

`test_ks_final_state.cpp` compares scalar acceptance with the existing detached
analytic fixture, exercises metadata and numerical failures, and preserves the
inclusive commutator boundary. `test_resident_final_validation.cpp` compares both
native and library products with independently constructed nonidentity-metric
frames, including empty beta, layout orientation, operator asymmetry, canaries,
solver/generation/capture failures and nonfinite staged weights.

`test_ks_resident_weights.cpp` exercises PBE/PBE0 RKS/UKS and empty beta. It checks
zero D/F/C/epsilon export during preparation, exact scalar packet accounting,
idempotent pointer-preserving leases, stale/capture rejection, device-density
corruption, real warm recovery, and exact/below numeric arena gates without new
device allocations during proof. Its prepared Direct derivative tests compare
the uncached resident source route with the existing detached/cached route.
These source comparisons are not an independent complete molecular-force oracle.

Host qualification logs live under `.artifacts/2227/cuda-resident-host-*`:
86/86 CTest and 40/40 independent CPU UKS force tests using pinned PySCF 2.14.0.
The CPU oracle confirms no regression of the shared acceptance policy; it does
not establish CUDA force qualification. Real-device qualification is separate
and must run through finite Slurm allocations on an allowed node.

### Real-device proof and admission fixture

An n1 `main` allocation with `--gres=gpu:5090:1` and an explicit finite
limit passes the resident-product, resident-weight, determinant final-state and
KS final-state tests (4/4, no skips). Visibility remains Slurm-owned. The proof
drains 116 bytes for RKS and 228 bytes for UKS: one 112-byte product packet per
spin and one four-byte staged-weight status, with no D/F/C/epsilon export.
These counts describe the explicit proof, not the complete force endpoint.

Exact/below arena tests must measure the incumbent under the same active public
ledger contract. An unbound owner can retain optional point panels that are
deliberately unavailable under that ledger. Using its larger capacity as the
bounded owner's exact limit tests different storage policies and is not an
admission proof. The ledger-bound PBE RKS H2 and UKS H/H3 numeric arenas measure
488568, 274872 and 746216 bytes; PBE0 measures 488600, 274888 and 746360 bytes.
These exclude complete RSS, provider/context internals and downstream owners.

The first matching passing library hashes to
`d7a1956dd88b0e685e9a70cdc71a6fa6bacc4d4d8297a54834399c14588d081a`;
the raw log is `.artifacts/2227/cuda-resident-n1-passing-results.log`.
Later source changes require their own library checksum and requalification.

### Broad-suite baseline discrimination

The broader `generativeqc_ks_cuda_tests` fails while materializing the
DF-J/long-range-K fixture in `run_density_fitted_range_exchange_case`.
An exception trace locates the rejected detached frame; the existing
`valid_model` policy excludes that composition. This change neither modifies
that model policy nor treats the failed suite as passed.

Restoring every changed tracked file from
`864462c5ba59a827595f0e84e32dd25388d5ea52`, rebuilding the native library and
rerunning the same suite through finite n1 Slurm reproduces the rejection.
The clean baseline library hashes to
`9bfac942b4a29cfb2fab746ac9b71f077f89c145575a35be8c05d17530ee073f`.
Evidence is retained in `.artifacts/2227/model-baseline-rebuilt-{build,gpu}.log`
and `model-baseline-rebuilt-library.sha256`. The earlier restoration attempt
reported "no work to do" because archive timestamps predated the objects; it
is not baseline execution evidence. Deployment must refresh source mtimes
before relying on an incremental scientific rebuild.

Concurrent work advances the branch to
`98c52ec045ab4fd1c48604b7c0941a768ed06024` by fast-forward, preserving its
13 commits and the uncommitted resident lease. The integrated host build passes
87/87 CTest after regenerating its stale CLI MethodIR catalog, 97 Python
oracle/contract tests with eight GPU opt-in skips, Sphinx warnings-as-errors
and the 525-module compiler ownership audit. The 40 independent CPU UKS force
cases run with pinned PySCF 2.14.0, without oracle skips. This is regression
evidence, not completion of #2227 or CUDA force qualification.

The extended square-product launcher keeps its default right orientation and
complete reduction rank. Host CUDA doubles must match its actual signature;
the matrix-owner mock additionally asserts those unchanged defaults. The
integrated resource/matrix suite passes 384 tests with seven GPU opt-in skips,
and the remaining source-contract suite passes 59 tests. Logs are
`.artifacts/2227/cuda-resident-integrated-resource-tests-fixed.log` and
`cuda-resident-integrated-source-contracts.log`.

Separate 18-AO closed-shell dimer fixtures exercise a real library-admitted
detached device proof before publication, then weighted and unweighted detached
reads after publication. The device-proof count must stay unchanged and W must
remain bitwise identical, followed by a pointer-preserving zero-work lease.
Both PBE and PBE0 pass through finite n1 Slurm. Only the test process opts into
the experimental detached selector; the measured 384-AO default remains
unchanged. This protects the existing scratch-exclusion guard rather than
promoting a new energy or force domain.

Final expanded real-device qualification runs the four CTest targets without
skips (7.78 seconds), including both 18-AO fixtures, and the independent
incremental KS physical-state/fresh-solve gates. Each large fixture records
two device proofs before the rereads and zero proofs from published rereads.
The library hashes to
`33dd383ae4697ddd389cac93fb8a6d608dfcf2848baf2e2dc9eb2de0fefb71b8`;
the broadened suite still reproduces the documented model rejection with that
same library. Logs, launcher versions, cache statistics and deployment-source
checks are retained under `.artifacts/2227/cuda-resident-final-gpu/`. This
qualifies the uncommitted scientific working tree atop `98c52ec0`, not a new
committed PR head or a complete native CUDA force endpoint.

## Consequences and revisit conditions

Small owners retain a bounded packet reservation even for energy-only execution.
This makes the explicit resident consumer predictable without changing the
energy-only numerical path. Keep the DF host contract until its actual response
owner becomes resident. The complete generated XC/AO/Becke force controller,
installed compiler/Python-free endpoint evidence, endpoint allocation receipts
and complete default-regression acceptance remain required for #2227 R1–R6.

The later [private native source-controller decision](2026-10-12-native-cuda-stationary-source-controller.md)
implements the generated binding and complete-source composition portion. It
does not promote public forces or complete the remaining cross-cutting gates.
