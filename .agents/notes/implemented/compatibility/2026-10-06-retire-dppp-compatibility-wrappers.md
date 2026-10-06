# Decision: retire DPPP compatibility wrappers and the PPPS 1110 alias

Status: implemented
Date: 2026-10-06

## Problem

The compiler still exported a historical DPPP-specific planning/evaluation layer
(`DpppFusedPlan`, `build_dppp_fused_plan`, `dppp_components`,
`emit_dppp_fused_cuda`, and `evaluate_dppp_fused_component`) even though the
general fused-shell APIs own the same science and scheduling. Repository search
found no production callers; the remaining callers were codegen tests.

The obsolete name `emit_ppps_1110_resident_bra_cuda` was only an alias of
`emit_ppps_resident_bra_rys3_cuda`. It had no repository caller and no
documentation contract.

## Decision

Migrate DPPP tests to `DPPP_SPEC.components`, `build_fused_shell_plan`,
`emit_shell_class_fused_cuda`, and `evaluate_fused_shell_component`, then
remove the historical DPPP wrappers and their package/facade exports.

Remove the unused PPPS 1110 alias while retaining the live
`emit_ppps_resident_bra_rys3_cuda` compatibility adapter used by production
emission and benchmark/correctness tooling.

## Invariants

- DPPP production mathematics and CUDA emission continue through the generic
  fused-shell owner; no numerical formula, schedule default, or kernel body is
  intentionally changed.
- The resident PPPS Rys3 adapter remains available under its current semantic
  name.
- Compatibility facades retain only names with a remaining repository or
  documented compatibility owner.
- A test prevents the retired symbols from being re-exported accidentally.

## Evidence

Before this change, repository-wide symbol searches found the DPPP wrappers only
in their definitions, compatibility re-exports, and codegen tests. The PPPS 1110
alias appeared only in its alias definition and re-export lists. The affected
DPPP tests now exercise the generic APIs directly with the same component,
Coulomb-state, block-thread, generated-source and gradient assertions.

No CUDA runtime or endpoint performance claim is made by this cleanup.

## References

Related compiler ownership rules: `python/generativeqc_compiler/AGENTS.md`.
Related dead-owner cleanup: #2032.

Agent: ChatGPT
Model: GPT-5.6 Sol
