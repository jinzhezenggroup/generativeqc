# Frozen K pair arithmetic and compensated energy qualification

Qualified base: `dee3d522b71df9fb55c91478df3c44b7af454488` (PR #2153).
The frozen candidate retains smaller sm120 `ddds`/`dppp` value-pair caches and
the existing compensated energy-trace low words through CUDA KS convergence.
All predeclared complete endpoint, prepared-K, numerical and holdout gates pass.

These measurements belong to the frozen candidate at `9db85cf78451134d037513f73b583751e1c84ce7`.
A later source review found an incidental narrowing of the mixed `ddds`
coefficient products: changing pair orientation triggered an indentation-based
`double` to `float` rewrite. The compatibility repair explicitly preserves the
previous double-promoted products through `mixed_pair_products_fp64`, enabled
only for the tuned `sm_120` `ddds` Fock schedule. Other mixed schedules retain
their existing arithmetic. Host evaluator regressions verify this repair;
these frozen GPU receipts do not qualify a rebuilt repaired binary.
The pre-integration compatibility repair has source identity
`6d517c95ad467049382bee31d10630150fd706a6e594988423f7d48564c09b8e`;
the frozen candidate identity remains `d087c507d3e02b69b224a01d3dbb6bac478f152670452fc7a3710e38d152f9aa`.
The later integration with master `ec71ef7fea6f705a3623b45c7bfb51105c43337a`
also carries the compensated generated-Fock ABI through task/work dispatch.
Its source identity is
`afa64bf70ddb196e6450ed3f5f8260c69cee2a6137c26e38d55a93eb9a022163`.
The [integration audit](../../../.agents/notes/implemented/compatibility/2026-10-09-rys-task-compensated-fock-integration.md)
records exact regenerated artifacts and host ABI checks. Neither these frozen
receipts nor the separate master measurements qualify that combined binary.

The measured `ddds` helper contains `#pragma unroll 1`. Its Fock metadata is now
corrected to rolled loops; the repair does not introduce unrolling. The strict
FP64 helper remains byte-identical to the frozen candidate, but that alone
does not establish unchanged timings after recompiling its CUDA translation unit.

## Cause and bounded changes

The preceding two-class cache trial was rejected: 48-atom Focks changed from
`[19,19,19]` to `[19,20,20]`. Both extra-build rows fail the strict energy gate
at iteration 19 with delta `1.3642420526593924e-12 Eh`, while density/residual
histories nearly match the baseline. The unchanged threshold is `1e-12 Eh`.

Existing electronic traces use compensation but previously rounded each trace
into a large component, formed a large total including nuclear energy, and
subtracted rounded totals. Independent actual-kernel/Decimal regressions pin
the loss of low words, which can produce false passes as well as false vetoes.
The candidate retains both words through the electronic sum/difference; the
geometry-constant nuclear term cancels exactly. The same helper serves device
chunks and host-controlled CUDA solves, including compatible warm baselines.

FP64 products/storage, returned components/total energies, all physical gates,
DIIS, iteration budgets, screening, force equations and selector/queue ownership
remain unchanged. Two doubles per private scalar and one per device control
add bounded payload, accounted through existing `sizeof`-based storage/transfers;
there is no additional matrix sweep, allocation object or kernel launch.

Provider/atomic and product-rounding noise are not eliminated or proved to have
one unique cause. The final 48-atom **priming sanity** still takes 20 Focks and
records a compensated iteration-19 delta `1.2369186011243525e-12 Eh`. Keep that
row outside the predeclared promotion population; do not claim deterministic
trajectories or that every extra build is removed.

In the dense emitted-subset `ddds` probe, expensive order-four calls drop
**64 to 16**, while four order-two calls and 64 Coulomb products remain.
This is source work, not executed hardware FLOPs or guaranteed pruned work.
`dppp` retains eight/four calls and 32 products, but caches four rather than
eight entries. Seven probes execute actual emitted loops for all Cartesian
components; real scientific validation separately uses independent references.

## Complete Cold E+F gate

| Atoms / AOs | Baseline mean / median | Candidate mean / median | Mean / median time reduction |
| --- | ---: | ---: | ---: |
| 96 / 768 | 102.388464 / 105.571264 s | 99.333829 / 96.292204 s | **2.98% / 8.79%** |
| 48 / 384 | 42.130438 / 42.079572 s | 41.242694 / 41.233890 s | **2.11% / 2.01%** |

Actual Focks: 96 baseline `[19,17,19,17,19]`, candidate `[17,19,19,17,17]`;
48 both `[19,19,19]`. Retain every valid slower observation. Ratios include
actual changed SCF work, not solely kernel arithmetic; small populations do
not establish statistical significance or universal speedup.

Predeclared alternating policies use five fresh workers/policy at 96 atoms,
three at 48. Both **mean and median >=1%** gates pass at both sizes. Cold
includes construction/context/preparation, SCF, analytic forces, host return
and teardown, excluding imports/input decoding. Equally reused persistent
artifact/compiler caches mean execution-Cold, not cache-empty compilation.

Strict FP64 PBE0 RKS, spherical def2-SVP, grid 48x16x32, screening `1e-12`,
E/D thresholds `1e-12/1e-10`, eight CPU threads for complete E+F (one for
prepared-K), same reference/geometry/protocol and matching Python/native source
per policy. Reference densities/forces are external oracles, not production
work or a claim of identical stopping policies between engines.

Four independent warm/moved owners retain 32 rows. Every moved-warm resupplies
moved coordinates. Warm mean/median gains are 0.91%/0.96%; moved-warm are
0.81%/0.84%; both policies use one Fock. Moved-first gains 2.19% with 13 Focks
for both policies. All <=3% regression and numerical gates pass. First-sanity
holdout/priming rows are not substituted into the Cold speed gate.

## Prepared-K and kernel diagnostic

144 clean prepared-K wall samples: three fresh processes/policy, three repeats,
two sizes, four density scales. Timing includes upload, transformations, K,
projection, export and synchronization, excluding plan/context preparation.
384-AO full-density mean/median reductions are **8.14%/7.97%**; 768-AO are
**7.80%/7.67%**. All sparse regression gates pass with exact per-class admission
equality. Dense totals are 23,480,495/81,907,624 generated quartets, not primitive,
root or native-fallback counts.

Separate 768-AO single-launch `ddds` diagnostic: **143.638638 to 60.513141 ms**
(57.87% less time). Both policies report 168 registers/thread, 29,320 shared
bytes and zero local bytes. These static reports do not measure executed
spills, occupancy or FLOPs. Whole-TU compilation can alter otherwise unchanged
kernel diagnostics; do not infer complete endpoint speed from one capture.

## Scientific qualification and limits

- 320 public/prepared K cases against Libcint: max error `6.058e-15`.
- 864 actual production task matrices pass; all six task/union memcheck,
  racecheck and synccheck logs have zero errors/hazards/warnings.
- Native through-f values/derivatives and Cartesian order-two CPU finite
  differences pass. Only that routing census uses command-scoped
  `GENERATIVEQC_DISABLE_MD_J=1`; clean performance uses default MD-J.
- 71 actual host/GPU energy/trace tests pass; energy memcheck has zero errors.
  Additional host arithmetic/energy probes: 33 pass, 45 GPU resource skips.
- Focused unchanged native cases pass independent CPU method/state gates for
  LDA/PBE/r2SCAN RKS/UKS, PBE0, RSH and public WB97M-V, including mixed/FP64,
  warm/seed/chunk/replay/failed-state behavior.
- Profile/codegen/recurrence/task: 246 pass; compiler structure checks 504
  modules with zero errors. All 22 non-Fock definitions, 21 streaming queue
  bodies and tested portable profiles retain their definitions.

All 16 Cold and 32 holdout rows pass independent E/F accuracy, with maxima
**9.095e-12 Eh / 4.209e-11 Eh/Bohr**. Precision/stopping thresholds and external
acceptance gates are not relaxed.

**No complete stock native-suite pass is claimed.** Its ledger/local-map
route-5 assertion fails on the unchanged frozen baseline. A diagnostic-only
copy excluding that check also reproduces the legacy WB97M-V nonlocal
final-model rejection on that baseline. Both original failures, the diagnostic
copies and the separate focused harness are retained. Production tests,
admission/model policy and unrelated code are not modified to mask them.

## Reproducibility and retained negatives

Baseline scientific identity / library SHA256:
`cb788b62c65e98f375f29a3fbbf35cf58c61a0c329461d4667a51c7794d6eb7e` /
`1426e40973b32359dca90d0cfa106eb62d1052eddb07af50ce12c4b86f079376`.
Candidate:
`d087c507d3e02b69b224a01d3dbb6bac478f152670452fc7a3710e38d152f9aa` /
`5c9ac43c8c3edf32e1cb85fc8ee872a84a3794277b70019f26aec903265fcba0`.
CUDA 12.9.1/GCC12, verified ccache 4.5.1 with the existing shared cache, RTX5090
on node4, finite exclusive Slurm jobs 688 (clean K/profiles), 694 (completed
scientific records) and 695 (complete endpoints). Job 694's later priming
attempt correctly rejects a one-thread/eight-thread reference-protocol mismatch
before any complete observation. The corrected driver restores the policy's
own environment and reuses only completed same-binary scientific records.
No source identity or reference-protocol check is bypassed.

`summary.json.xz` retains the fixed gate; `receipts.json.xz` retains all declared
timings/work/accuracy; `diagnosis.json.xz` retains original energy-veto traces and
initial unformatted diagnostics. `negative-populations.json.xz` retains all three
preceding failed populations and identifies their full receipts/archives by
checksum, not only their fast K samples.

`ignored-artifacts.json` identifies full raw forces/histories, reproduction
scripts/inputs/source patches and frozen binaries by exact local location and
checksum. Initial canary/legacy/protocol failures are preserved, not relabeled
as completed benchmarks. No Release, release tag or external backup is made.
Every real-GPU reproduction requires finite `srun` with scheduler visibility
preserved and each binary's matching source/PYTHONPATH.

The four `.json.xz` files losslessly retain the original JSON bytes, including
all observations and failures, under the unchanged tracked-evidence size policy.
`storage.json` pins compressed and decoded sizes/SHA256 values. From the
repository root, decode to ignored local storage for inspection:

```bash
mkdir -p .artifacts/performance/ks-energy-expansion-20261009
for name in summary receipts diagnosis negative-populations; do
  xz -dc "benchmarks/results/ks-energy-expansion-20261009/$name.json.xz" \
    > ".artifacts/performance/ks-energy-expansion-20261009/$name.json"
done
sha256sum .artifacts/performance/ks-energy-expansion-20261009/*.json
```

See [the decision note](../../../.agents/notes/implemented/numerics/2026-10-09-ks-energy-delta-low-words.md).
