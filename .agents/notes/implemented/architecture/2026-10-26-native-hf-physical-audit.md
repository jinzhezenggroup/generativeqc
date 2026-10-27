# Decision: native fixed-density HF physical audit

Status: implemented
Date: 2026-10-26

## Problem

The progressive controller still constructs NativeAO/FockPlan owners, evaluates
the final target Fock, constructs a second overlap owner, and forms FDS-SDF with
NumPy. The scalar acceptance migration in #2814 does not move this scientific
orchestration. Native stage scheduling alone cannot close #2227 R4/R6.

## Decision

The implementation adds a bounded CPU fixed-density audit using the existing PreparedFockPlan,
Fock assembly and reference mean-field residual/energy implementations. A
synchronous native scope owns preparation, one Fock build, overlap, residual
reduction and cleanup. It never performs a new SCF solve or substitutes a DIIS
residual. Python transports the requested model, immutable final density and
typed receipts; it no longer imports NativeAO or evaluates matrix products for
the admitted CPU domain.

The audit uses the existing CPU HF host-fleet capacity owner plus the incumbent audit
workspace reservation before constructing a Fock provider. Keep a separate
metadata-only query and exact/below-limit tests. These are conservative numeric
bounds, not process RSS or complete allocator/provider accounting. Reuse the
existing Python resource admission as an additional frontend gate until its
remaining authority is separately retired under R2.

The initial domain is all-electron CPU RHF/UHF, Direct/DF, with the existing
supported Cartesian/spherical topology and unchanged FP64 model controls.
Missing native protocol and unsupported domains retain named compatibility.
Native execution failure must never retry Python or relax the requested budget.
CUDA audit migration requires its own device-resident scientific qualification;
this CPU step must not introduce hidden host work after CUDA admission.

## Qualification contract

- Independent PySCF fixed-density energies, Focks/overlap and FDS-SDF maximum/RMS,
  including perturbed density and changed geometry, RHF/UHF and Direct/DF.
- Transactional invalid/count/nonfinite/alias errors, capacity overflow,
  exact/one-byte-below numeric admission and cleanup; native SDK without Python.
- Complete progressive endpoint parity and unchanged published iteration/Fock
  work, with Python physical-audit compatibility forbidden in the admitted path.
- Pilot cost estimate before full ABBA/BAAB nonregression sampling. Preserve
  the original 1.05 bound, every sample, science gates and source identity.

## Rejected shortcuts

Moving only the NumPy matrix multiplication leaves the scientific lifecycle
under Python. Reusing a stored SCF residual would replace the independent
fixed-density check. Copying the commutator or HF energy equations into another
implementation would violate the shared-science ownership boundary.

## Evidence

The independently based candidate was frozen against
`02562ba426c1d6d9458b1f3b7af3bc9f84d907c4`, patch SHA-256
`58b11f20aa893a4da2a373cd4e7fed35d6cb60fe2ce6d31550ca5933596f1f2e`.
The complete publication is
[`benchmarks/results/native-hf-physical-audit-20261026`](../../../../benchmarks/results/native-hf-physical-audit-20261026/README.md).
It retains all samples, scripts, patch, pilot estimate and independent reference.

121 CTests and 81 Python tests plus 25 subtests passed, along with Sphinx,
API/ownership checks and the native physical-audit SDK executable with
`PATH=/nonexistent`. Independent fixed-density tests cover RHF/UHF x Direct/DF
and Cartesian/spherical Ne+H2 with d functions, perturbed density and moved
coordinates. ABI gates include immutable inputs, transactional outputs,
alias/nonfinite/count errors, exact/below capacity, and execution refusal without
Python retry.

The complete progressive endpoint campaign used four RHF/UHF x Direct/DF cases,
two different geometries, 21 samples plus three warmups per block, ABBA and BAAB,
and the original 1.05 nonregression gate. All eight timing/work decisions pass;
maximum candidate/baseline ratio is 0.9906245353813529. All 1536 independent
energy/force/density publications pass the 1e-8/1e-6/1e-7 gates; maximum errors
are 2.731148640577885e-14, 8.287537323070637e-12 and 1.5955570198400437e-11.
The pilot estimated 85.84 seconds; the completed campaign took 83.22 seconds.
This qualifies bounded nonregression, not a significant-speedup claim.

The independent PySCF oracle initially compared differently normalized
Cartesian d+ AOs. The correction is a representation transformation, with
`scale = 1/sqrt(diag(S_pyscf))`, `D_pyscf = D_native * scale_i * scale_j`,
`F_native = F_pyscf * scale_i * scale_j` and the same transformation for overlap.
Do not loosen numerical gates or change production equations to hide that
normalization difference. Initial failed oracle/setup logs are retained.

## Consequences and remaining scope

The CPU audit owns preparation and cleanup in one synchronous native scope and
reuses shared scientific implementations. Its capacity is a conservative numeric
envelope, not RSS or complete provider accounting. The additional Python
preflight remains intentionally until the resource authority cutover is qualified.

This step does not close general progressive/force-aware control, CUDA physical
audit, resource authority or #2227. It is independent of #2814's scalar policy
migration. Revisit compatibility when each missing backend/domain has its own
independent scientific, resource-lifetime and complete-endpoint acceptance.
