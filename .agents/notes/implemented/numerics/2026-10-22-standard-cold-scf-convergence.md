# Decision: qualify Standard solver stopping against complete observables

Status: implemented
Date: 2026-10-22

## Problem

Cold local HF/KS spent extra iterations tightening residuals after energies and
forces were already adequate. Delta-D stopping was already enabled; incremental
J/K is a different option. GPU4PySCF defaults to energy change `1e-9` and an
orbital-gradient norm of `sqrt(1e-9)`. Native default energy/density controls
were `1e-10`/`1e-8`, with still stricter AO commutator and canonical-density
checks. These norms cannot be equated by assigning the same numeric threshold.
The earlier [strict stopping proof](../performance/2026-10-22-final-state-aware-ks-stopping.md)
removed only a small conservative tail.

## Decision

The Python CUDA default and Standard/Fast fallback use energy `1e-9`, density
`1e-5`, and unchanged `1e-12` screening on all-electron RHF and complete canonical
restricted PBE/PBE0/B3LYP/r2SCAN/M06-2X identities. KS requires the default
Treutler/Lebedev/NWChem grid and device-fused schedule. r2SCAN-3c inherits the
r2SCAN electronic policy, retaining its canonical basis and corrections.
CPU, unrestricted, correlated/CBS, other models/grids, and older libraries
keep legacy controls. Explicit tighter controls survive resolution. The private
version query adds no public descriptor fields or C++ default changes.

Eligible native owners cap the physical maximum AO residual at `1e-5`.
KS additionally tests complete resident canonical products **before** accepting
convergence or publishing a warm seed. Failed proofs continue the intact DIIS
proposal and invalidate borrowed orbital scratch, including fitted exchange.
Force export repeats its token-bound proof with consistent limits. Trace,
idempotency, orthogonality and component-energy consistency retain their gates.
Either tighter public E/D setting restores the strict physical residual ceiling.
Unsupported owner/resource/precision conditions retain strict stationarity.

## Acceptance and the earlier M06-2X restriction

This is a change to the solver contract, not proof of total model accuracy.
Independent tight GPU4PySCF references use identical stock grids and fitted
operators, full grid response, and no shared initial density. Solver error must
fit within one quarter of the common endpoint budget: `2.5e-6 Eh/atom`, maximum
force `2.5e-5 Eh/Bohr`, RMS force `7.5e-6 Eh/Bohr`. The complete dense-reference
comparison separately keeps its original `1e-5` / `1e-4` / `3e-5` gates.

This supersedes the blanket M06-2X admission hold in the
[earlier strict-probe decision](2026-10-22-restrict-early-final-promotion.md)
only for the independently qualified Standard solver contract. Its strict
optional probe remains unchanged. The new evidence includes water, displaced
water, methane and ammonia, with original/displaced cold and frozen warm calls.
It does not rely on agreement with a native rollback alone or a size whitelist.
The previously failed M06-2X/3 default-grid and HF-DF direct-reference populations
were rerun and **still fail**; they remain retained and unranked. Their errors
are not cured by additional SCF iterations and are not renamed as passes.

## Rejected shortcuts

- Relaxing only the ordinary stopping predicate let PBE/96 return from SCF and
  then fail canonical force validation. The pre-stop proof is mandatory.
- Reusing the new thresholds while retaining an old strict force-export gate
  moves work or failure to a later consumer; all consumers need coherent limits.
- Deleting the old `1e-6` force oracle test would lose strict-control coverage.
  That test now explicitly requests its original `1e-10`/`1e-8` controls;
  a separate default-policy test enforces the stated solver budget. Initial
  default runs exceeded `1e-6` by about `1e-6`; their failed logs are retained.
- An extra FP64 fallback audit is not required by Standard. Fast fallback still
  completes one solve; explicit accuracy targets retain their requested audits.

## Evidence and limits

[Source-bound receipts and replay](../../../../benchmarks/results/cold-scf-defaults-20261022/README.md)
retain 23 targeted cold/three-warm populations, including four accuracy failures;
20 independent same-model cases (80 complete native E/F calls); 32 strict-control
and UKS compatibility tests; native CUDA residual/NaN/inactive-lane boundary
checks; and the original failed test run. All real GPU work
used finite Slurm allocations on n1 RTX 5090. No README campaign or CBS result
was updated. Exact source reconstruction, compiler cache stats and 25 library
identities accompany the dirty-source measurements; the base SHA alone is not
presented as their measured source.

At 96 atoms, Direct HF/PBE/PBE0/B3LYP/r2SCAN/r2SCAN-3c/M06-2X need
11/11/9/9/9/10/10 target iterations; the existing stock GPU4PySCF receipts have
10/11/9/10/10/11/10. DF needs 11/10/9/9/9/10/10. HF retains explicit final-Fock
work counts beyond iteration counts. These counts do not establish universal
endpoint speed leadership. The nonlocal omega-B97M-V policy is unchanged:
its provisional stages and final target iterations must remain distinguishable.

Requalify complete same-model E/F, dense-reference status, warm/final-state
lifetime and actual work before extending to other owners, spins, grids or
precision domains. A residual tolerance is never a universal force-error bound.
