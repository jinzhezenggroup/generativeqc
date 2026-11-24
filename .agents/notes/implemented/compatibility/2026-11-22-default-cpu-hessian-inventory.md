# Decision: default to the linked CPU PBE0 Hessian inventory

Status: implemented
Date: 2026-11-22

## Decision

Fresh host-Hessian builds use `manifests/host_hessian_aot_catalog.json`.
This promotes canonical all-electron Direct FP64 PBE0 RKS and named two-body
PBE0-D3(BJ) RKS to the same native scalar Hessian/HVP owner as B3LYP. It
supersedes the default-selection part of the
[opt-in decision](../numerics/2026-11-22-scaled-pbe0-cpu-hessian.md), not its
restrictions on the model or execution context.

An existing build keeps its cached selection. B3LYP-only remains an explicit
inventory and is checked independently. When the library lacks the PBE0
binding, the CLI refuses the product; Python retains the prior PBE0 route.
The Python whole-calculation `ResourceBudget` case also retains its previous
owner until the native CPU aggregate can compose and enforce that budget.
This preserves existing primal admission, not a whole-derivative budget
guarantee. An admitted native failure never retries in Python.

## Evidence and limits

The opt-in qualification covers independent moving-grid gradient differences
for water and ammonia, the final D/W and J/K energy contractions, corrected
force differences, and installed CLI/SDK/Python parity for both products.
Installed first use of corrected def2-SVP water takes 1.48 s for HVP and
2.70 s for dense in separate processes with no compiler discovery path and
an unused, initially nonexistent artifact cache. The SCF, primitive,
direction, induced-XC and correction counts match the corresponding warm
native calls.

B3LYP-D3(BJ) RKS def2-SVP water and UKS STO-3G H₃ controls retained exact
outputs and identical work counts across the installed B3LYP-only and
catalog builds. Five warm samples per endpoint at initial and moved
geometries give median shifts of +0.63–0.95% and +1.27–2.61%, respectively.
These are fixture-scoped comparisons, not a heavy-element, GPU or general
performance qualification. The source-bound #2227 delivery ledger records
master, not this working tree's promoted inventory.

The fresh default CPU build passes seven selected CTests covering both
derivative products and the CLI runtime contract. Installed SDK/CLI/Python
parity and eight linked PBE0 Python tests also pass against that build. The
fresh explicit B3LYP-only build passes three selected CTests, including the
parity test's absent-PBE0 CLI refusal; a corrected HeH⁺ Python smoke test
confirms compatible dense/HVP results without the linked PBE0 binding.

The whole-calculation budget decision is superseded for linked CPU endpoints
by [native CPU Hessian resource composition](../architecture/2026-11-22-cpu-hessian-resource-composition.md).
