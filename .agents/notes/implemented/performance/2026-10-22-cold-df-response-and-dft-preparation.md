# Decision: corrected DF response and bounded cold preparation

Status: implemented
Date: 2026-10-22

## Problem

Equal SCF iteration counts did not imply equal cold/warm work. On master
`afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`, HF-DF water-96 performs eleven SCF
cycles followed by ten physical final Fock evaluations and nine density
corrections. The final determinant cannot borrow the old SCF projection.
Automatic response consequently used bounded fitted AO panels, repeating a
large projection for every response panel. Final device Fock evaluations were
also missing from the public work count. These were execution/reporting issues,
not permission to skip physical final-state checks.

HF and DFT DF owners additionally launch raw three-center generation separately
for every AO row even though retained B and its scratch use packed pair panels.
DFT's first force also enumerated every ordered Cartesian derivative tuple to
discover an inventory that already identifies exact center permutations.

## Decision and invariants

- Automatic restricted, single-system, full-rank packed-single response can
  reconstruct a corrected density's occupied factor using the existing native
  eigensolver and strict reconstruction checks. Owner, geometry/model, epoch,
  occupations and paired generation advances must still match. The reconstructed
  factor projects immutable B once; it never borrows stale final U. Unchanged
  determinants still require the original one-shot projection lease.
- Reconstruction rejection retains bounded fitted panels. Optional allocation
  failure may clear only an allocation error. A terminal device/numerical error
  propagates. After occupied-bridge OOM, the bounded retry withholds both the
  consumed projection and final-state authorization, preventing recursive
  reconstruction/re-admission. Explicit diagnostic requests retain their gates.
  Streamed automatic selection is unchanged.
- Materialization uses the existing packed raw source API to submit contiguous
  lower-pair panels across row boundaries. The same charged scratch, every raw
  value, truncated-metric fallback and two factor-ordered whitening products
  remain. Dense storage and all scientific tolerances are unchanged.
- Derivative inventory planning enumerates unordered AO pairs and unordered
  electron-pair pairs before the existing axis canonicalization. Exhaustive
  ordered-consumer comparison preserves all 362 sorted requests, including 313
  ERI representatives, and their AOT numbering. Runtime ordered bindings and
  emitted mathematics are unchanged. Compiler-source identity still changes;
  matching AOT manifests must be regenerated.
- Public RHF work includes device finalization Fock evaluations once. The explicit
  host/reference callback already counts itself and must not be counted again.

No grids, convergence criteria, initial guesses, auxiliary bases, precision
budgets or force/grid responses were relaxed. The M06-2X default-grid and HF-DF
default-auxiliary approximation questions remain outside this change.

## Endpoint evidence

Two independently built owners share the README driver and frozen inputs, using
stock native Standard/Fast, cold constructor+prepare+SCF+forces+host export and
two frozen-density warm calls. All work ran through finite Slurm allocations on
node1 RTX 5090 GPUs. This is a targeted six-case qualification, not a replacement
README campaign or a new GPU4PySCF ranking.

| 96 atoms, Standard | Master cold / median warm (s) | Candidate cold / median warm (s) |
| --- | ---: | ---: |
| HF DF | 100.340 / 4.924 | 42.684 / 4.854 |
| PBE DF | 41.121 / 6.905 | 40.165 / 6.989 |
| PBE0 DF | 54.038 / 11.643 | 52.302 / 11.452 |
| r2SCAN DF | 43.465 / 7.963 | 41.359 / 7.919 |
| M06-2X DF | 56.343 / 11.781 | 54.770 / 11.769 |
| PBE0 Direct | 30.625 / 8.459 | 30.350 / 8.518 |

Fast HF-DF cold is 100.584 -> 42.621 s; the other five Fast cold changes are
56.443 -> 54.960 (M06-2X DF), 38.502 -> 37.393 (PBE DF), 54.169 -> 52.348
(PBE0 DF), 39.661 -> 38.415 (r2SCAN DF), and 29.856 -> 29.365 s (PBE0 Direct).
Warm measurements fluctuate in both directions; this change does not claim a
general warm improvement. All 24 populations / 72 complete calls converge.
Actual settings agree exactly between each pair. Maximum paired energy and force
changes are 1.82e-12 Eh and 1.72e-11 Eh/Bohr, respectively. Those paired differences
establish preservation, not the absolute default-grid/default-fitting accuracy.

Separate instrumented Standard runs establish the causal work change:

- HF-DF cold force response: 59.207 -> 4.266 s. Corrected-factor reconstruction
  takes 0.015 s and accepts all nine correction generations. Warm still reuses
  the exact final fitted projection. Physical finalization remains unchanged;
  the corrected public count is 21 cold and 3 warm, versus 11 and 2 SCF cycles.
- Raw generation submissions: 782 -> 15 for both HF-DF and PBE0-DF. Whitening
  still uses 15 panels / 30 GEMMs. Materialization is 22.034 -> 21.003 s for HF
  and 22.161 -> 21.128 s for PBE0. Most of this remaining cost is real integral
  arithmetic and metric transformation.
- A fresh local process builds the identical derivative inventory in 0.090 s
  versus 0.569 s. This is metadata timing, not an endpoint speedup claim.

Master core SHA-256:
`32fb18a47fa20c9b73001e3cde019578addcdd8289e8e802a47fef91b9b0f0a7`.
Measured candidate core SHA-256:
`14d11eee67356bc121b7b612a4669371afe3ef3ae99c69da838ddba652f5ada7`.
Raw receipts, source reconstruction patch, runner scripts, compiler-cache stats,
build logs and failed qualification attempts are retained under
`n1:/data/jzzeng/cold-endpoint-fixes-20261022/`; local copies are under the
implementation checkout's ignored `.artifacts/` directory. Clean endpoint and
synchronizing trace populations are separate.

## Qualification and retained failures

Host executable tests cover consumed leases, reconstruction rejection, stale
epochs/generations, repeated corrected response, provenance/reconstruction/bridge
OOM, bounded retry, and terminal device/numerical failures. Exhaustive derivative
inventory/dispatch checks preserve independent center/axis semantics. The initial
broad automatic-factor prototype failed consumed-lease and OOM tests; it was
replaced before any endpoint qualification.

Device tests compare independent PySCF/GPU4PySCF energies and complete forces,
including cold/warm/moved geometry, corrected auto and explicit response,
Cartesian/spherical bases, RHF/UHF, batched owners, storage replacement, truncated
resources and streamed fallback. HF gates remain 1e-9 Eh / 1e-8 Eh/Bohr for the
corrected-state fixtures; PBE0 gates remain 1e-8 Eh / 3e-7 Eh/Bohr.

Final qualification passes 247 host/compiler tests, 62 real-GPU tests, all
pre-commit checks, and the corrected packed-single/cc-pVDZ-JKFIT auto response
under Compute Sanitizer memcheck (zero errors, including geometry replacement).
After the final explanatory-comment edit, the rebuilt core is
`a7fe1d09bdd07fce2265d6b03a213db2867bd6e370d00b5ccbf4cfa1781e62fb`;
all 62 GPU tests and memcheck use this image. Additional complete HF-DF and
PBE0-DF Standard cold/warm populations are retained under `results/final-clean`.

Two old fixture assumptions also fail on the unmodified master: automatic
storage at small sizes need not be packed, and independently rebuilt final
densities need not produce identical warm iteration counts. Updated fixtures
retain observable and work gates, explicitly exercise packed storage, and choose
budgets that actually exercise streamed fallback. Failed trial budgets and their
OOM results remain in the logs; no acceptance threshold was weakened.

## Rejected alternatives and remaining work

Do not retain an old occupied projection after density correction, skip the final
physical projector, infer a stationary frame from a loose residual, or force all
semilocal DF owners into streaming merely to improve cold at warm's expense.
Do not replace factor-ordered whitening with one explicit inverse-root product:
the current ordering protects weak metric directions from cancellation.

The corrected HF-DF cold endpoint still contains ten final Fock evaluations, and
DF materialization remains about 21 seconds at this size. Direct iterative J/K
and DFT XC/response arithmetic are not eliminated by these fixes. Further work
needs independent qualification and complete endpoint evidence; the remaining
gap is not grounds for a universal speed-lead claim.
