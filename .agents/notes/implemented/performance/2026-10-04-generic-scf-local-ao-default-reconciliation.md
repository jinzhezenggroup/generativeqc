# Decision: preserve automatic SCF local AO selection through the shared XC capability

Status: implemented in the #1847 reconciliation candidate; publication pending
Date: 2026-10-04

## Problem

PR #1847's head `80e6e1fa0ad323ee7cf9a91cba39f870f1e25ceb` promoted
automatic SCF AO selection but retained exact RKS-PBE0/WB97M-V/provider checks
and a whole-schedule FP64 veto. Master `cb4924ddc92b103463c87937931495461a6870cb`
contains #1865's generic XC capability owner and shared precision intersection.
The admission block conflicts; resolving in favor of the old method whitelist
would regress the merged ownership boundary.

## Decision

Preserve the requested default-on behavior for every eligible device-fused XC
layout, using the same compiler-emitted point capability and physical-layout
constraints as explicit local AO selection. Do not introduce a second method,
spin, exchange-provider or nonlocal-composition whitelist in SCF. The enclosing
KS composition still has to pass its independent admission and validation.

`GENERATIVEQC_CUDA_KS_ACTIVE_AO=0` disables discovery. `=1` explicitly requests
it and fails closed when the layout/execution schedule cannot support it. An
unset switch automatically requests discovery only for eligible layouts.
Unsupported generated point programs retain dense execution; a generated
program gains automatic eligibility only when its owner exposes that capability.

This deliberately extends the default to capability-supported domains beyond
the frozen PBE0/WB97M-V endpoint campaigns. The user's generic-default direction
accepts supplementary evidence rather than requiring a new GPU campaign solely
because a domain lacks a fresh measurement. Expanded-domain performance remains
unmeasured. Capability and host tests do not establish cutoff error bounds,
complete-trajectory equivalence, profitability or current-head GPU execution.
Concrete numerical, resource or work regressions still warrant a scoped fix or
guard at the responsible owner; none is established by this host reconciliation.

## Invariants

- Retain cutoff `1e-16`, physical/non-response FP64 AO-layout restrictions,
  geometry/grid ownership and the fixed 64 MiB host peak admission bound
- Retain worst-case map resource charging and optional-device-allocation dense
  fallback; mean active counts never reduce capacity bounds
- Copy the selected physical layout back from the XC owner before iterations
- Preserve the shared precision schedule: local density stays FP64 while an
  independently admitted Coulomb J directive may remain lowered
- Preserve strict refinement, exact exchange and all other component policies
- Keep force AO selection and Becke behavior independent

## Evidence and limits

The bounded host capsule compiles the production admission/resource block,
XC layout and local-map capability/resource functions, the compiler-emitted
point-program selector and the shared precision resolver. Coverage includes
LDA/PBE/r2SCAN/B3LYP/WB97M-V with RKS/UKS, explicit/default/disabled policy,
strict/AUTO arithmetic, non-exact scaled PBE and nonlocal composition facts,
unsupported layouts/programs, zero dimensions, invalid switches, selected
layout propagation and host-budget dense fallback. Existing component-census
and point-selector suites remain part of the check. These are host control-flow
and arithmetic-contract checks, with UBSan; no CUDA kernel or SCF endpoint ran.

The sparse scaled-PBE independent full-AO oracle cases, CTest registration and
both retained JSON receipts from #1847 are unchanged. The
[original qualification note](2026-10-04-pbe0-scf-local-ao-admission.md) and its
frozen source/binary/device/settings retain their original scope, including
the moved-geometry iteration regression and adjacent-suite failures. This
reconciliation does not relabel those measurements as current-master results,
qualify every new default domain, or claim combined speedups.

## Rejected alternatives

- Restoring exact-PBE0/WB97M-V admission would duplicate scientific ownership
  and reject legal UKS, scaled-PBE and independent-J configurations
- Requiring the whole schedule to be strict FP64 would veto unrelated qualified
  Coulomb J lowering instead of constraining only local density contraction
- Silently dropping automatic selection would remove #1847's requested default
- Interpreting capability or host success as universal profitability would
  overstate the available endpoint evidence

## References

- [PR #1847 review](https://github.com/jinzhezenggroup/generativeqc/pull/1847#pullrequestreview-5407832167)
- [Merged capability owner #1865](https://github.com/jinzhezenggroup/generativeqc/pull/1865)
