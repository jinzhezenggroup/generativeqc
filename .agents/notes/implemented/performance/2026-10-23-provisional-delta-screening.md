# Decision: coarser screening for bounded delta proposals

Status: implemented in the opt-in incremental owner
Date: 2026-10-23

## Problem and evidence

The [bounded proposal controller](2026-10-23-bounded-incremental-cold-proposals.md)
removed redundant refreshes but saved only about 4.5% at water-96. Fresh profiling
found 8.81 s in its nine J/K actions, including 6.82 s of K. Six delta actions
still used the final operator's very tight screening cutoff.

## Decision

A distinct provisional Fock facade permits the first eight eligible delta
actions per solve to request a 1e-9 screening cutoff. Admission follows the
existing Standard complete-proof owner and producer capabilities, without an
extra functional whitelist. The MD-J and generated full-range K producers receive
an explicit per-call cutoff; no shared prepared metadata is mutated. Existing
producer error caps still apply, including MD-J's 1e-12 cap. Unsupported producers
return before touching outputs and the caller uses its ordinary operator.

This is a convergence heuristic, not a screened-error certificate. Full anchors,
warm replay, two consecutive closing full-density energies, physical/canonical
validation and forces keep the original cutoff and tolerances. Unscreened
requests stay unscreened. After eight deltas across all refresh epochs, further
proposals use the original cutoff. Explicit strict controls remain conservative.
`GENERATIVEQC_KS_INCREMENTAL_COARSE_SCREENING=0` retains the prior proposal
schedule. The global incremental mode remains off by default.

## Validation and limits

Three fresh-process rotated-order complete E/F rounds on RTX 5090, PBE0/def2-SVP,
spherical AOs and stock grids give master/new cold medians of 3.29684/3.13103,
13.13335/11.73880 and 30.37949/27.19843 s at 12/48/96 atoms. The same round's
previous candidate took 29.01222 s at 96 atoms. All samples, including slow pilot
warm calls and the prior candidate's 48-atom outlier, remain in raw evidence.
Warm remains one full Fock; master/new warm median changes are -0.59%, +0.19%,
and -0.73%. This does not establish a warm speedup or universal zero overhead.

A separate final profiling run reports 6.86 s J/K, including 4.84 s K, and six
coarse actions. Total cold Focks remain 10/9/9, with six deltas. Actual K quartet
censuses are unavailable; timings are not relabeled as work counts.

All 134 retained molecular endpoints (including pilots/diagnostics) pass tighter
independent full-density GPU4PySCF E/F references. Coverage includes PBE, PBE0,
B3LYP, r2SCAN, M06-2X, moved/reset/warm, strict controls and the opt-out. The
reference suppresses both RKS density and potential history and includes moving
grid response. Gates remain 2.5e-6 Eh/atom, 2.5e-5 maximum and 7.5e-6 RMS force
error. This qualifies sampled default-grid solver accuracy, not grid convergence.
Seven CUDA tests include signed-delta J/K against independent ERIs, invalid
cutoff rejection, unsupported-producer fallback and unchanged ordinary actions
after provisional calls. 222 host/controller/inventory tests pass.

The [compact receipt](../../../../benchmarks/results/incremental-cold-proposals-20261023/coarse-screening.json)
records all timing populations, errors and source/binary identity. Native builds
use the original measured base plus the candidate changes; the PR is integrated
onto bbd41a427. Full source snapshots, references, failures and raw logs remain in
`../evidence-delta-d-expand-20261023/`. All GPU execution used finite Slurm jobs;
compilation reused verified ccache 4.5.1.

## Rejected alternatives and revisit conditions

Do not mutate the prepared screening identity, scale density to hide a changed
cutoff, use proposal energies for acceptance, or relax the final convergence
gates. Broader producers or a different cutoff require new complete endpoint
and independent accuracy evidence; smaller isolated K timings are insufficient.
