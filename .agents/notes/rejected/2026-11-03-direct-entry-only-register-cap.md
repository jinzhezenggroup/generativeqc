# Decision: do not cap only the SM80 Direct force entry kernel

Status: rejected
Date: 2026-11-03

## Problem

The previous Direct investigation retained 255 registers for the shared low-order
force consumer even after angular partitioning. Test whether three 128-thread
blocks per SM can reduce register pressure without repeating candidate scans.

## Decision and evidence

Reject the entry-only change from `__launch_bounds__(128, 1)` to a three-block
SM80 bound. LSF build 224944 fails in PTXAS: the entry is limited to 168 registers
but calls out-of-line `contract_two_electron_force_low_order_sources_impl`
instantiations with 255 registers, including order-four and order-five roots.
Changing the caller alone cannot satisfy that call-graph resource contract.

This is a compile failure, not a numerical or performance result. No GPU endpoint
ran for the candidate. Its scientific source identity is
`6d1b68ade951a6873fce283e1e9cf0a9da80e286ed46486b1fc29087f249b40c`.
The original orbit source `13f12658...` builds successfully in LSF 224937 after
pinning CMake to Python 3.12; its original launch bounds are restored.

The complete compiler diagnostic and cache receipts are retained locally under
`/home/jzzeng/codes/qc-gencode/.artifacts/a100-direct-20261103/build-224944/`;
archive SHA256 is
`0ad784e0b123812a9f411f282d6f8949ff13c948e71f0cebe187790e5bc3755e`.

## Invariants and revisit conditions

Keep the single bounded quartet scan, FP64, source coefficients, spin/seed
fallbacks and shared scientific recurrence owners. Revisit register reduction
only with a callee-aware compiler/resource plan, independent derivative oracles
and full endpoint/work-count qualification. Repeating angular scans or merely
reducing the entry's launch bound does not solve the measured problem.

See the prior
[orbit/tuning investigation](../implemented/performance/2026-11-02-direct-force-orbit-and-tuning.md).
