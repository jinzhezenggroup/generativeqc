# Decision: qualify indexed FP64 matrix products for large restricted GGA panels

Status: implemented
Date: 2026-11-08

## Problem

The full non-CBS A100 README audit has56/64 cold point wins, not64/64.
PBE0/B3LYP DF96 cold were9.179725/9.175714s versus8.447658/8.736263s.
Named-CUPTI228300 attributes about0.607s to XC density products and0.475s
to local potentials in PBE0 DF cold9.188s. Target iteration count is already
nine, matching the reference; this is an operator issue, not extra DIIS steps.

## Decision

Reuse the existing indexed strict-FP64 density and potential GEMM owners for
restricted one-jet PBE-family/B3LYP panels. PBE0 shares the PBE semilocal family.
Keep the same512-AO/2^19-point profitability threshold and FP64 matrix hardware
gate (compute major≥8, positive single/double throughput ratio≤2). No product
name, CPU oracle, DF substitution or lower precision selects the schedule.

Preserve the original generated indexed contraction on resource refusal,
public-ledger/reference-budget paths, lower-precision schedules, unsupported
capabilities and nonqualified functional/spin domains. Both optional providers
keep their existing128MiB unpromised-capacity admission. This is not a full
AO-grid cache. UKS and other one-jet methods do not gain a new default.
`GENERATIVEQC_CUDA_XC_GGA_GEMM=0` rolls back both GGA providers; unset/1
cannot force execution past any gate. The density-specific rollback remains
independent. The existing meta-GGA behavior is unchanged.

## Evidence and limits

`benchmarks/results/gga-indexed-gemm-20261108/` retains70 fresh populations,
280 complete constructor+prepare+SCF+analytic-force+host-export endpoints,
all FP64 vectors, early losses, semantic work, image/build/scheduler/transfer
receipts and explicit independent dense/CPU reuse. Offline replay recomputes
quarter-gate reference stability and CPU/GPU spots, all cold/warm comparisons,
unchanged native solve work and per-warm regression gates. Experimental
uncommitted source suffixes remain original; no fabricated commit relabels them.
The reused independent geometries, orbital/auxiliary bases, charge/spin,
representation, original producer/driver contract and package environments
remain explicit. No historical GPU timings or densities enter the new solve.

Experimental source`73bf0922e8a334da061cb1525b444dc74efe16d4d92fdf33209ba96dda80084e`,
ELF`3bc4644e120ec35ac39a2ec8ee56784b8fc6e991c95c9645d0f5f48a8181ad85`:
screen228334, two-order six-case cohort228356, RTX5090 protection16747.
All12 cohort cold pairs improve4.86–11.57%, with no>3% warm regression.
PBE uses10 target Fock applications; PBE0/B3LYP use9; every warm uses1.
Preliminary, target, J/K, precision and grid work counts do not decrease.

The GGA-only final default source is
`843a2dae51b7a8539e8732b30e627f8950e8ebb6a8c6849db4cf6a73557aac63`,
normal SM80-real+SM120-real Release ELF
`e25c91dfbcb19f69ef4e9fe5c85c0636f5c8761f62a4f797cb393bfcac303e3f`.
CPU build16752/deployment228379/A100228380/RTX5090 Slurm16753 complete.
Only`src/dft/cuda_ks.cpp` changes native identity; the dormant HF prototypes
in the earlier experimental image are explicitly excluded. Borrowed build
sources/runtime are restored and authenticated; compiler caching is retained.

The inherited evidence checkout had insufficient space for this new capsule.
Rather than raise its85MiB limit or retire any historical owner, the H100
fixed30 packet now uses the already-supported nested exact-text/FP64 codec.
Its original44,542,496-byte container, all1,398 members/1,060 distinct blobs,
negative results and decisions reconstruct exactly. Only storage identities
and its reader change; former compressed identities remain explicit. This
saves460,848 bytes and needs no new archive host or Git recovery.

Final A100 default versus rollback cold seconds:

|Method/mode|Rollback|Default|Independent GPU4PySCF|
|---|---:|---:|---:|
|PBE Direct|7.325541|6.537924|8.691721|
|PBE0 Direct|13.323914|12.596131|11.259502|
|B3LYP Direct|13.464607|12.871374|11.961932|
|PBE DF|6.631756|5.854885|7.678227|
|PBE0 DF|9.134051|8.499987|8.485574|
|B3LYP DF|9.225342|8.486766|8.865632|

B3LYP DF becomes a point win in all measured screens. PBE0 DF is approximately
tied: the first screen loses0.52%, the cohort wins narrowly, and final default
loses0.17%. Do not claim statistical superiority, ignore that first loss or
rewrite the frozen64-case baseline. Hybrid Direct, HF and some warm endpoints
remain behind. Final six-case default cold improvements are4.41–11.72%; the
largest RTX5090 cold fluctuation is+0.55%, and every warm remains within3%.
The hardware gate keeps the5090 arithmetic schedule unchanged.

## Rejected alternatives and future work

Do not loosen grid, convergence or observable gates to manufacture wins.
Do not extend the default to all one-jet methods, UKS or all hardware without
new endpoint qualification. Preserve the independent scalar, changed-density,
AO-map, lifetime and bounded-refusal coverage of the shared owners.

HF is a separate K/work and convergence problem. Its fixed-coarse-delta and
384-AO expansion experiments are rejected in
`../../rejected/2026-11-08-hf-fixed-coarse-delta.md`; their negative results
are retained, not shipped as another opt-in. See issue2976 and PR3041 for
the original audit and ongoing unresolved work.

## Post-measurement master integration

The final fetch discovered master`7704dd93dec18db0cf506d78ce1c76f0d6d40fe1`
(#3039), which landed during publication. Merge it rather than omit the user's
requested synchronization. It adds shared Gram/dense-provider adapters and
migrates RHF/occupied-response ownership; the GGA policy and the existing
`submit_column_gemm` implementation are unchanged. It does not authorize
relabelling the pre-merge source/ELF above as a newly executed combined image.
Those static GPU timings remain bound to843a2dae/e25c91df, with focused host
provider/lifetime/admission and evidence replay used for integration. No new
64-case or post-merge GPU qualification is claimed. Regenerate the stale
Libxc source inventory after the merge while retaining its original measured
receipts, stale status and zero qualified-product claims.

## Evidence storage after the next master merge

Master `d52d2ae2a` adds a separate H100 component capsule. Keeping that evidence
and this PR's original capsules exceeds the unchanged85MiB aggregate retention
budget by66663 bytes. Do not raise the budget or remove negative observations.
Regroup only this PR's two exact-FP64 predictor streams by scientific case and
engine using the already supported codec, then use stronger ordinary XZ storage.
This saves94676 bytes while preserving the388 README and333 GGA original texts,
every FP64 word, the experiment metadata and the source-bound acceptance rules.
Both `storage-receipt.json` files retain the previous compressed identities.
The acquisition identities, independent failures and frozen timing conclusions
do not change. This requires neither implicit Git/network recovery nor a new
codec, dependency or external publication. Revisit storage, not science, when
future source-bound evidence again approaches the inherited budget.
