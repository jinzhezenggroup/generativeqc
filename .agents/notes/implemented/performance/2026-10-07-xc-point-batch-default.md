# Decision: resource-guarded XC point batching by default

Status: implemented
Date: 2026-10-07

## Problem and superseded policy

The initial #2073 scheduling route remained opt-in while endpoint evidence was
collected. Independent E/V and complete cold/warm/moved PBE0 E/F gates pass, and
the measured route reduces complete endpoint time as well as point submissions.
The user explicitly requested default enablement after reviewing these results.
This supersedes only the opt-in decision in `2026-10-07-xc-point-batches.md`, not
its measured source/binary identities or its outstanding evidence disclosures.

## Decision

Ordinary native KS requests 32 independent original XC tiles with an additional
32-MiB per-owner allowance. This is a request, not a promise of 32 resident tiles:
the compiler halves the requested count until the actual indexed AO panels and
feature/total slots fit. The measured 96-atom maps admit eight tiles and retain
21,823,488 additional bytes. The shared numeric resource ledger may still reject
that allocation, leaving the incumbent available without extra scientific work.

`GENERATIVEQC_CUDA_XC_BATCH_TILES=0` or `=1` explicitly selects the old route.
`GENERATIVEQC_CUDA_XC_BATCH_BYTES=0` also disables optional residency without
requiring a tile override. Response, mixed arithmetic, single-tile and
resource-rejected domains retain the one-tile executor. Changed geometry creates
and qualifies a new owner; the force tile schedule is not promoted or changed.

Compiled-resource envelopes now include both the reachable batched point kernel
and its unbatched allocation-failure fallback. Missing evidence for either fails
closed. Explicitly disabled and single-tile regions exclude the unreachable
batch kernel; the selector participates in the evidence binding identity.
The default-promotion inventory audits both the 32-tile request and 32-MiB cap.
Benchmark arms continue to set their policies explicitly on creation and moved
owner rebuilds, so promotion cannot silently enable batching in a baseline.

## Invariants and rejected alternatives

- Canonical FP64 point algebra, original indexed AO maps, AO/jet/contraction work
  and ordered Vxc/scalar accumulation remain unchanged.
- Retain finite optional memory admission, allocation rejection fallback,
  capture-safe owner preparation and explicit opt-out.
- Do not replace resource admission with a device-model or molecule-size list.
- Do not set an unbounded cap or assume the requested count is admitted.
- Do not infer full-endpoint speedup from the eightfold submission reduction.
- CPU/PySCF remains an independent qualification oracle, not a production path.

## Evidence and limitations

The promotion's host/compiler/benchmark-orchestration suite passes all 284
tests. This includes unset-environment default admission, explicit zero/one-tile
opt-out, a zero-byte cap without a tile override, mixed-precision fallback,
exact admitted-byte accounting, reachable batch/fallback pressure envelopes,
missing-kernel rejection and per-arm policy restoration across moved owners.
Ruff/clang-format, compiler structure, CUDA ownership, native-complexity,
default-promotion inventory and evidence-retention checks pass.

The retained measurements pin implementation
`282dabad0cfaad612d93eb7789f55d3296f03d05` against master
`2b0feff1d5577e58f3a44fbecbc1974be92df4e3`; the initial note records all binary and
generated-source identities. Its explicitly enabled 32-tile/32-MiB route is the
same compiler plan now requested by default. Fixed-density E/V is bitwise
identical; independent native gates include both spins, spherical/cartesian AO
bases, all curated point consumers, ragged/empty/full maps, tails, captured changed
densities, scaled PBE and constrained-ledger teardown. Compute Sanitizer reports
zero errors and zero leaked bytes.

| Complete E+F scope | Samples per arm | Baseline median s | Batched median s |
| --- | ---: | ---: | ---: |
| 12-atom fresh-process cold | 3 | 91.631693 | 88.869089 |
| 96-atom fresh-process cold | 2 | 217.112518 | 205.049690 |
| 48-atom frozen warm | 5 | 7.966658 | 7.516457 |
| 48-atom displaced frozen warm | 5 | 8.000996 | 7.540783 |

These are bounded positive observations, not universal profitability or a
statistical confidence interval. All 12-atom cold trajectories use 23 Fock builds;
96-atom baseline uses `[25, 27]`, batched `[27, 28]`, so cumulative cold work is
not identical. Warm observations use one Fock build each but separately converged
snapshots, not identical density bytes. Per-build selected AO work is unchanged.
Every measured complete E/F observation passes the independent scientific gates
at unchanged precision, basis/grid/convergence/screening and full moving-grid
response. Native/reference screening differs (`1e-12`/`1e-14`), not equal work.

The extra arena count is not a full-process peak-memory measurement. New NCU
counters remain permission-blocked, and #2072 source-only/composed arms are not
qualified. Those diagnostics do not become performance claims merely because
the bounded scheduling route is now enabled by default. Additional devices,
workload shapes and composition should refine the profitability/resource guard
without removing fallback or weakening scientific gates.

## Promoted follow-up validation

#2081's actual merge tree `c8d71bb2b` retained the earlier opt-in policy, despite
its branch later advancing to a default-enablement commit. PR #2089 therefore
applies the promotion separately on master `7dc9d7944`, together with the native
density-provider ledger fixture repair. That fixture must measure incumbent-only
storage: including optional default point panels in its baseline otherwise leaves
headroom for the density cache whose allocation it intends to reject. A scoped
zero-byte cap affects only baseline measurement and restores the caller's policy
before every real default/admission/warm/moved/constrained-ledger case.

The clean promoted source is
`8dda81a75ed5b61192a79981c20ffb725f207c07`. It was rebuilt in Release for sm_120
using verified ccache and tested through finite Slurm job 6437 on node1/n1,
RTX 5090, preserving assigned `CUDA_VISIBLE_DEVICES=1`. Independent native point,
local-AO and scaled-PBE E/V gates pass, as do all 28 default density-provider
RKS/UKS/cold/warm/moved/resource cases. Compute Sanitizer reports zero errors and
zero leaked bytes. Current host/compiler/default-policy/orchestration tests pass
all 284 cases; compiler structure checks 493 modules and CUDA ownership 333 files.

Native library SHA-256:
`fd0169069726c44a47b2a5468e549b65f19ff6bd1535e10843c6b55c8aa92a1e`.
Native KS test SHA-256:
`aa157202887d256b779069d1139d7fd9d80a2f4fb86d019b64d15f869a95d2e6`.
The XC test remains `a061e3906fe36a691ec6c0ab0db6ee59646618e5af6c0e225e63cfbf098feeaa`
and generated grid source remains
`6ec2fddb973af34294a3e7ec361b93e2b80f99a506b365d978522ad790f8b6a2`.
Later evidence-only commits do not relabel these measured binaries.

| Fresh-process smoke scope | Complete E+F s | Fock builds | Energy error Eh | Max force error Eh/bohr |
| --- | ---: | ---: | ---: | ---: |
| 12 atoms, unset batch controls | 44.830536 | 23 | 4.434e-12 | 3.003e-11 |
| 12 atoms, zero-byte opt-out | 35.447133 | 23 | 4.320e-12 | 2.999e-11 |
| 96 atoms, unset batch controls | 185.608714 | 25 | 7.685e-11 | 2.491e-11 |

These three smokes all pass the independent complete E/F gates. They are **not**
a new clean interleaved performance population: one observation per scope, no
exclusive-node reservation, and compilation/source-cache history was not balanced.
The 12-atom default observation is slower than its opt-out observation; do not
hide that or interpret this single sequential pair as a profitability proof.
Only the earlier explicitly enabled scheduling ablation supplies the retained
performance evidence, with its original source and trajectory caveats.

The broader KS executable stopped at its final-state-identity gate on earlier
pinned builds. The same failure reproduces on the older tree with batching
disabled and on the promoted tree with explicit opt-out; no full-suite pass or
unrelated final-state fix is claimed here. The targeted default-policy cases and
independent complete PBE0 endpoint smokes are recorded separately. Raw current
results remain in `results/default-policy-current/` under the original local
qualification root; earlier failures/pre-promotion runs are retained separately.

## References

- #2073; PR #2081; default-promotion owner #1598; source-specialization sibling #2072.
- `2026-10-07-xc-point-batches.md`
- `docs/developer/xc_native_cuda.md`
- `manifests/maintenance/default_promotion_inventory.json`
