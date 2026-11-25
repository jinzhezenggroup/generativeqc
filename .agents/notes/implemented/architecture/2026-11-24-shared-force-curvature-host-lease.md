# Decision: Model forces in the same prepared host lease as curvature

Status: implemented
Date: 2026-11-24

The fixed-chemistry saddle task needs one native primal state for forces and
curvature. Its SDK building block must not run forces after creating an
energy-only Hessian lease: those derivatives need additional verified frame,
integral source and XC workspace.

An explicit reservation flag selects the existing Direct CPU force inventory.
The prepared per-item capability gate still decides whether native analytic
forces are available. An SDK caller pays for the added workspace before SCF,
uses the same retained state for the Hessian/HVP, and cannot fall back to an
unbudgeted force. Existing batches without the flag retain the energy-only
contract. The combined reservation currently excludes ECP forces and orbital
export; scalar ECP setup capacity alone does not prove those lifetimes.

The SDK regression checks force and HVP parity, an exact capacity success,
one-byte-short refusal and transactional pool release. The end-to-end saddle
adapter still needs separate root, warm seed and optimizer capacity accounting.

References: #2227 (R2); `src/api/ks_cpu_df_host_budget.hpp`;
`tests/native/test_ks_hessian_sdk.cpp`.
