# Decision: default to resource-guarded compact XC contractions

Status: implemented
Date: 2026-10-08

## Problem and superseded policy

The initial compact-contraction implementation retained an explicit opt-in until
complete SCF energy plus analytic-force endpoints could establish profitability.
The user requested full E+F validation, default enablement if beneficial, and a PR.
This supersedes only that opt-in policy in
`2026-10-08-xc-compact-contraction-batches.md`; its scientific ownership, maps,
precision, deterministic scatter and explicit resource fallback remain unchanged.

## Decision

Ordinary native KS now requests compact contractions by default, still within
the existing additional 32-MiB / 32-original-tile point-batch allowance. Each
eligible group retains its original point spans and CSR labels; every nonempty
tile has 32–128 active AOs and at least 32 points. Other groups retain point-only
execution. Response, mixed arithmetic, optional providers, allocation rejection
and constrained public resource ledgers retain their existing bounded fallback.
`GENERATIVEQC_CUDA_XC_COMPACT_BATCH=0` preserves a complete point-only opt-out.
There is no molecule-name, atom-count, functional-name or device whitelist.

The compiled-resource selector now normalizes an automatic compact request to
the reachable multi-point-batch domain. It requires all five new stages **and**
the incumbent fallback. One-tile/disabled point batching stays incumbent-only;
explicit false selects the diagnostic incumbent envelope. Missing default batch
pressure fails closed rather than qualifying automatic execution with old data.

## Complete E+F evidence

All comparisons use the same Release sm_120 native binary,
`129cdaa7ac48b0d2c6cf672f646d40b44ae303d7fec6deb8b09f84651c1641c4`,
on allowed n1/node1 RTX 5090 through finite Slurm job **6641**, `main`,
`gpu:5090:1`, 16 CPU workers, `00:45:00`, assigned visibility `1` preserved.
Build job **6638** used verified ccache 4.5.1 C++/CUDA launchers, retained
before/after statistics and built the complete library and stationary PBE0 AOT
manifests. Both measured arms explicitly request 32 point tiles / 32 MiB,
256-point original AO maps and fixed 256-point force policy; only compact
contractions differ. No reference work enters the timed native production path.

| Complete SCF E+F scope | Samples/arm | Point-only median s | Compact median s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 12-atom fresh-process cold | 5 | 10.111890 | 9.461689 | 6.43% |
| 12-atom frozen warm | 5 | 1.031201 | 0.988964 | 4.10% |
| 12-atom displaced frozen warm | 5 | 1.035516 | 0.995790 | 3.84% |
| 48-atom frozen warm | 5 | 6.869269 | 6.877481 | -0.12% (within noise) |
| 48-atom displaced frozen warm | 5 | 6.906259 | 6.922295 | -0.23% (within noise) |

The 12-atom scopes pass the existing >2%-and-robust-MAD descriptive gate.
The 48-atom scopes do **not** pass a profitability gate: their tiny slowdowns are
within the 2% floor and are retained, not renamed speedups. Default promotion is
supported by the qualified bounded small-AO domain, not universal profitability.
All ten clean cold trajectories use exactly 16 Fock builds. Every warm/moved-warm
sample uses one physical iteration and one Fock build with no warm fallback.
Per-build selected AO visits/squares and moving-grid Becke work are unchanged.
Warm snapshots are independently converged, not claimed identical density bytes.

Cold timing includes preparation plus the first full E+F execution in fresh
processes/owners/densities, excluding imports/context/Calculator construction.
Persistent compilation/runtime caches are reused; both arms are primed outside
the five interleaved clean pairs. Warm/moved setup and priming are retained
separately and never counted as clean performance samples. The 48-atom setup
histories are not an interleaved cold population and supply no cold speed claim.

Every endpoint passes independent GPU4PySCF moving-grid E/F gates, energy
`1e-8 Eh` and maximum force `1e-7 Eh/bohr`, at unchanged precision, basis,
quadrature, convergence and screening. An independent consumer recomputes gates,
work invariants and timing assessments rather than trusting saved error fields.
The existing independent 12-atom cold oracle and 48-atom cold/moved oracle are
retained; a fresh 12-atom cold/moved oracle was generated for the warm campaign.

Separate instrumented cold execution, excluded from timing populations, confirms
384 executions **of each** compact density/features/panels/local-Vxc/scatter
kernel: 24 admitted groups × 16 XC evaluations. This is actual CUDA launch
evidence from Nsight Systems, not an inference from the requested control.
Native E/V, changed-density/canary/replay and both Compute Sanitizer gates from
job 6635 remain the scientific/device foundation; default-policy composition is
validated separately after rebuilding the promoted source.

Raw source/binary/input identities, full vectors, solver histories, resource
diagnostics, profiler launch census and assessments remain in ignored local
`build/xc-compact-qualification/scf-campaign-6641/` and remote
`/data/jzzeng/qc-compact-xc-20261008-e162/results/scf-campaign-6641/`.
Full-build provenance is retained under corresponding `scf-build-6638/` paths.
Retained file SHA256 authentication anchors:

- `pairs-12.json`: `fe42d8af67e0e9ad6f37cc4798d5893b893a610f8b6c350ebc89f6564eb5fb5b`.
- `pairs-48.json`: `3d1f0cfe987b7e426e824bb1e1b37f97d66fabe103895ba65c68648cce5908e2`.
- `independent-assessment.json`: `15b8143bbcd3f958456b8e582088482a17fe9f73e59b2357e36156d26476e853`.
- `kernel-census.csv`: `f24c91ae0eaf7a52a1349b159d599dc4ff36f419a3d7bc220c13bbc51e47ec8e`.

## Promoted follow-up validation

The promoted source was rebuilt with verified ccache through finite Slurm job
**6652**, node1/n1, `main`, `gpu:5090:1`, `00:25:00`, preserving assigned
visibility `2`. Native point-batch, local-AO and scaled-PBE independent E/V gates
pass. Compute Sanitizer memcheck reports zero errors and leaked allocations;
racecheck reports zero hazards/errors/warnings. The complete promoted library
SHA256 is `85ec4b6274e3f62c02ba3dcef6fa6571b6d3833d9fc689eb43dc87a2a606ee86`;
the native XC test remains
`446c0ea772d70a654a0393828119737a83430ae0a0269e2db202557544ea52f0`.
All 412 current host/compiler/selector/orchestration tests pass. Compiler
structure checks 495 modules; CUDA ownership checks 338 files and reports
zero handwritten scientific CUDA growth, with 48 additional runtime lines.

| Promoted fresh-process smoke | Full E+F s | Fock builds | Energy error Eh | Max force error Eh/bohr |
| --- | ---: | ---: | ---: | ---: |
| 12 atoms, unset XC controls | 9.492872 | 16 | 3.070e-12 | 2.126e-11 |
| 12 atoms, compact explicitly zero | 10.148889 | 16 | 3.070e-12 | 2.123e-11 |
| 96 atoms, unset XC controls | 112.890271 | 17 | 7.276e-12 | 3.776e-11 |

These are composition/scientific smokes, **not** a new interleaved performance
population or a 96-atom speedup claim. Default/opt-out and the larger fallback
domain all preserve the independent energy/force gates. Raw current evidence,
cache statistics and complete cuobjdump resource usage are retained under local
`build/xc-compact-qualification/default-qualify-6652/` and the corresponding
remote `results/default-qualify-6652/` directory.

Two failed preliminary qualification runs are retained, not hidden or counted
as successful performance samples:

- `--density-provider` stops at its route-5 admission assertion with compact
  both enabled and explicitly disabled. Existing master commit `f316421f5c`
  deliberately disallows AO selection under a live public ledger, including an
  explicit request; this fixture still assumes route 5 obtains local maps and
  therefore rejects the now-legal dense provider. This unrelated fixture is
  unchanged here, and no complete density-provider-suite pass is claimed.
- After the compiled-resource contract changes, rebuilding only the primary
  library leaves optional stationary AOT manifests stale. Job 6648's first
  force publication correctly fails closed with `contract_identity identity
  mismatch`. Regenerate both PBE0 stationary manifest targets against the new
  compiler contract; do not bypass checks, relabel artifacts or clear caches.
  Job 6652 rebuilds these targets and passes complete default/opt-out E+F.

## Rejected alternatives and revisit conditions

- Do not promote based on the earlier ~34% fixed-density XC-endpoint improvement.
  Only the complete E+F populations above justify this default.
- Do not change scientific maps, union AO domains, relax convergence or hide
  the larger-system noise/slowdown observations to manufacture a speed claim.
- Do not use force/SCF method or molecule IDs to select favorable measurements.
- Larger active spaces need a separately bounded scatter design and complete
  independent numerical/resource/profitability evidence. Additional devices and
  methods should refine the generic domain without removing fallback or opt-out.

## References

- PR #2089; issues #2073 and #1598
- `2026-10-08-xc-compact-contraction-batches.md`
- `docs/developer/xc_native_cuda.md`
- `benchmarks/pbe0_xc_tile_pairs.py --atoms 12 --point-batch-tiles 32 --compact-xc-batches`
