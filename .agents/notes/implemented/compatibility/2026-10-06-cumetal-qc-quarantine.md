# Decision: quarantine the pinned upstream CuMetal shared-owner QC lane

Status: implemented
Date: 2026-10-06

## Problem and evidence

CuMetal `e87f368060cf09a45b148b4f8892470c74093ebd` with `cumetal-ir` and
`fast48` cannot compile the shared SCF `initialize_state_kernel`: the typed
PTX backend has no definition for `__nv_longlong_as_double`. The one-electron
kernel also reaches an unsupported `ptx.branchtargets` normalization boundary.
The shared production calls are in `src/scf/cuda_rhf.cpp`
(`launch_initialize_state_kernel` and `launch_generated_one_electron_values`);
the state kernel is in `src/scf/cuda/scf_state_kernels.cu`.

The bounded diagnostic in
[run 37414311819](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37414311819)
reported a skipped first RHF endpoint after approximately 166 seconds. This
identifies upstream compilation blockers, not a completed scientific result.
[The retained investigation](https://github.com/jinzhezenggroup/generativeqc/pull/1997#issuecomment-6009823623)
records both failures. Later endpoint groups were not independently run to
failure: their common SCF/integral owners motivate a conservative lane quarantine.

This supersedes the routine acceptance policy in
[the timeout observation decision](2026-10-06-cumetal-timeout-observation.md)
after the user explicitly requested temporarily skipping affected tests pending
an upstream fix. The diagnostic source, regression tests, and exact-input
fixtures remain available for reproducing the evidence.

## Decision and invariants

Use a separate exact allowlist in `manifests/cumetal_qc_quarantine.json`, selected
only by an explicit opt-in in the routine and full CuMetal QC workflow steps.
It covers the four routine and fourteen full endpoint groups, all using the
affected shared owners. Never generate the quarantine from the current selection:
new endpoint groups must execute under the existing strict acceptance checks.

Require the exact provider commit, typed backend, precision mode, and CuMetal
environment before applying any skip. A changed or missing contract fails
closed, forcing a new qualification/review. Report each skipped group in JUnit
and the job summary with the reason and **QC NOT QUALIFIED**. Do not invent
GPU provenance, passing testcases, or numerical coverage for these groups.

Retain the native production build, normal runtime CTest and Apple-GPU dispatch
smoke checks. Leave every numerical tolerance, independent oracle, runtime
capability check, NVIDIA lane, and other required gate untouched. Unquarantined
groups and the runner without the opt-in continue to reject unexpected skips,
failed/empty results, missing per-case GPU provenance, and exhausted budgets.

## Rejected alternatives

- Catch-all xfail or accepting arbitrary CUDA-unavailable skips would hide new
  defects and could silently remove NVIDIA coverage
- Raising timeouts cannot repair unsupported lowering or establish correctness
- Passing compilation, legacy lowering, and runtime smoke cannot replace real
  endpoint numerical and GPU provenance evidence
- Altering production kernels or upstream code is unnecessary for this temporary
  repository-local CI decision

## Revisit and restoration

Any provider/backend/precision change must revisit the quarantine; do not merely
move its pin forward. Once upstream implements both shared-owner lowering paths,
run routine and full selections with the opt-in unset on the Apple GPU. Every
actual endpoint must pass its existing oracle/numerical assertions and per-case
dispatch checks with zero skips and unchanged budgets. Retain that evidence and
then remove the workflow opt-in. Isolated PTX compilation is necessary diagnostic
evidence, not sufficient restoration evidence. `fast48` also retains the prior
binary32 libdevice limitation; quarantine does not establish IEEE-FP64 support.

## Verification

`tests/python/test_cumetal_qc_quarantine.py` checks the exact pin-bound scope,
truthful skipped-group reports, strict opt-out, and enforcement of all ordinary
error/provenance conditions for groups outside the quarantine. Existing CuMetal
coverage and diagnostic regression tests remain part of the local test set.
