# Decision: align native WB97M-V defaults toward stock GPU4PySCF

Status: proposed; exact-source device qualification pending
Date: 2026-11-05

## Problem

Issue #2987 identified unlike stopping contracts in the retained fitted-LR
capsule. Native WB97M-V used E/density `1e-10/1e-8`, physical AO RMS and maximum
`1e-9`, whereas GPU4PySCF 1.9 used E `1e-9` and MO-gradient norm `sqrt(1e-9)`.
Its density change was not a stopping gate. The source-bound offline replay
still passes all 96 sampled E/F comparisons; this does not make the stopping
contracts equivalent or explain all 28 versus 10 cycles (23 native cycles use
provisional semilocal/frozen-VV10 operators).

The user's clarification explicitly chooses **native defaults toward the
reference**, not a stricter reference. The earlier exploratory reference-audit
patch was discarded before submission. No reference defaults are changed.

## Decision

Extend the existing native default-controls query to canonical default-grid
restricted WB97M-V. The CUDA owner independently authenticates its actual
semilocal, full/LR exchange, VV10 and grid composition against the native catalog.
Its Standard SCF candidate requires E change below `1e-9` and physical orbital
gradient below `sqrt(1e-9)`. It does not add a density-change candidate gate.

For D=2 C_occ C_occ^T and X^T S X=I, the Frobenius norm of X^T(FDS-SDF)X is
sqrt(2) times ||2 C_vir^T F C_occ||. Transform the *current physical* residual in
existing scratch and reduce it on device; no CPU/reference work, extra resident
matrix, host synchronization or GPU-name dispatch is introduced. A separate
orthonormal MO evaluation tests this relation in a nonorthogonal AO basis.

Matching candidate thresholds does not replace native publication safeguards.
Fresh full-target energy history, complete canonical proof, electron trace,
idempotency and force-token validation remain required. The final Standard
density/stationarity ceiling remains `1e-5`. Rejected proofs continue SCF.
The nested VV10 confirmation/inner thresholds must use that same Standard
ceiling instead of silently retaining the old `1e-9` oversolve. Explicit tighter
energy or density controls restore the old strict policy, including VV10 inner
thresholds. Fast, CPU, UKS, custom model/grid, chunks and missing resource cases
retain their own bounded policies.

## Rejected alternatives

- Tightening GPU4PySCF: opposite to the user's requested default direction.
- Copying only E/density numbers: WB97M-V was excluded from Standard admission,
  and its physical gates and frozen-inner tolerances would remain strict.
- Equating AO RMS/maximum with MO gradient: wrong representation/normalization.
- Removing final canonical/force proofs or accepting provisional VV10: unsafe.

## Evidence and qualification boundary

- Audit native base: `0a214362fa9d762cf0a3a6be821d7cb5d185f257`.
- Inspected reference: GPU4PySCF 1.9 `2fdc1951670a8e173dce063e697727bbc9f97616`,
  `gpu4pyscf/scf/hf.py`, physical gradient and `_kernel` stopping predicates.
- Frozen replay: six pairs, all 96 independent cross-engine E/F comparisons pass.
- Slurm 15916, physical RTX 5090: the exact changed CUDA diagnostic unit passes
  an independent host-sum versus device MO-gradient reduction, preservation by
  physical diagnostics, and isolation from strict/unsupported diagnostics.
  This standalone unit is not a full production image or an endpoint timing.
- Exact-source CUDA build and targeted native default/explicit-strict E+F
  comparisons remain pending; no new speedup or default qualification claimed.
- An initial host test attempt used an older runtime missing
  `generativeqc_precision_execution_run_v1`: 24 pass/18 fixture errors. The native
  default-query adapter and predicate probe passed; complete current-runtime
  tests must be rerun rather than treating this installation mismatch as a pass.

## References

Issues #2976 and #2987. Separate strict fixed-anchor draft #2986 is unchanged.
