# Exact Direct analytic-force recurrence/contraction reuse

## Frozen sources and implementation

This is a source-matched qualification, not PR #2538's historical baseline.
Baseline `6d8192be87c0f9a347239f78622f870c4c3d6231` composes master snapshot
`9a92cad27`, #2531 `01ed327f4` and #2532 `fb37f6d1c`, including its later
Direct/1088 resident-final-validation hold. Candidate deployment is
`028249106afe0d2c4ed4431c3f678cc5b996d692`; production changes are `cca25d003`
and `31d95f0d6`. Subsequent `498cd2571` changes only host-test fixture selection.
The native inventories (1745 inputs) and loaded images are identity-checked.
Both builds are official Release/sm_120, CUDA 12.9.1, strict FP64, portable=0,
fast_compile=0, with verified ccache 4.5.1 CXX/CUDA launchers and existing caches.

PR #2541 is stacked on the frozen composition. During qualification master
advanced through #2531, #2528, #2535 and #2529 to `74d5f4227` (audited October
16, 2026, 02:34 +08); #2532 remains open at the frozen head. These results do
not qualify that subsequent master composition. Restack after #2532 resolves
and requalify material changes; no merge is performed.

- Order six preweights eight `(-2*rho)^n F_n` roots once per admitted spatial
  box and borrows prepared coordinate powers for low-degree states. The existing
  maximum 64-double box remains, with eight bounded primitive-local roots.
  Nonspatial and order-four/five consumers retain their incumbent arithmetic.
- Order seven uses the existing compiler-owned axis convolution before its
  shared three-axis Coulomb contraction, instead of expanding six pair indices.
- Exact J/K, FP64, signs, independent-center translation, screening, source
  weights, RKS/UKS, spherical/Cartesian packing, admission, bounded storage,
  active lanes, shared recurrence barriers and fallbacks remain unchanged.
  No functional, molecule, AO-count or GPU-model production policy is added.

## Complete endpoint observations

All warm entries are medians of five calls. Each population retains cold,
five warm, moved and five moved-warm calls under the original 1800-second
process limit. All **96 primary calls complete**; no population is incomplete.
Cold includes calculator preparation and the first synchronized E+F endpoint;
existing compilation caches are retained. Intrusive profiles are separate.

| Case | Endpoint | Baseline (s) | Candidate (s) | Reduction |
| --- | --- | ---: | ---: | ---: |
| r²SCAN-3c, 48 atoms | cold | 76.637 | 70.499 | 8.01% |
| r²SCAN-3c, 48 atoms | warm | 21.558 | 15.286 | 29.09% |
| r²SCAN-3c, 48 atoms | moved | 60.232 | 51.185 | 15.02% |
| r²SCAN-3c, 48 atoms | moved-warm | 21.566 | 15.358 | 28.79% |
| r²SCAN-3c, 96 atoms | cold | 411.131 | 386.093 | 6.09% |
| r²SCAN-3c, 96 atoms | warm | 90.752 | 65.555 | 27.76% |
| r²SCAN-3c, 96 atoms | moved | 260.226 | 234.750 | 9.79% |
| r²SCAN-3c, 96 atoms | moved-warm | 90.920 | 65.669 | 27.77% |
| PBE0, 48 atoms | cold | 23.704 | 23.538 | 0.70% |
| PBE0, 48 atoms | warm | 4.596 | 4.303 | 6.37% |
| PBE0, 48 atoms | moved | 22.537 | 22.274 | 1.17% |
| PBE0, 48 atoms | moved-warm | 4.612 | 4.321 | 6.31% |
| HF, 48 atoms | cold | 10.901 | 10.923 | -0.21% |
| HF, 48 atoms | warm | 0.742 | 0.743 | -0.16% |
| HF, 48 atoms | moved | 7.198 | 7.196 | 0.03% |
| HF, 48 atoms | moved-warm | 0.742 | 0.744 | -0.18% |

The primary cases use canonical def2-mTZVPP, 544/1088 spherical AOs, unchanged
moving 48×16×32 Becke grids per atom, screening 1e-12, energy convergence
1e-12 and density convergence 1e-10. Controls use Direct HF/PBE0 with def2-SVP.
Finite n1/n4/n5 Slurm allocations preserve scheduler-assigned visibility.

| Case | Cold iterations/builds, baseline→candidate | Moved iterations/builds, baseline→candidate |
| --- | --- | --- |
| r²SCAN-3c, 48 atoms | 19/19 → 19/19 | 14/14 → 13/13 |
| r²SCAN-3c, 96 atoms | 24/24 → 24/24 | 13/13 → 13/13 |
| PBE0, 48 atoms | 17/18 → 17/18 | 12/12 → 12/12 |
| HF, 48 atoms | 21/22 → 21/22 | 16/17 → 16/17 |

Every warm/moved-warm call executes one iteration and one physical Fock build.
The 48-atom moved reduction includes its changed trajectory; nothing is
normalized away. HF counts use returned valid native strict/mixed/post-SCF
counters after timing, not estimates from iterations.

Every endpoint passes independent 1e-8 Eh / maximum 1e-7 Eh/Bohr gates.
Maximum errors across both arms are **3.86535e-11 Eh / 6.01639e-11 Eh/Bohr**.
Some historical oracles are explicitly accuracy-only reconstructions, not
measured timing populations. Both complete independent geometries are required
before timing even when their historical status needs `--allow-missing-reference`.
The historical GPU4PySCF 26.59-second warm figure is not newly measured here;
the candidate remains approximately **2.47×** slower than that historical target.

## Profile and executed work

| Diagnostic device kernel, 96 atoms | Baseline (s) | Candidate (s) | Reduction |
| --- | ---: | ---: | ---: |
| Generated low | 4.626643 | 4.626063 | 0.01% |
| Order-six spatial J′ | 21.754343 | 17.441551 | 19.82% |
| Order-seven cooperative J′ | 28.034387 | 7.305988 | 73.94% |
| Order-eight materialized J′ | 5.772215 | 5.771981 | 0.00% |

`profile-summary.json.xz` retains mutually exclusive host owners and their
reconciled complete driver wall ledger: SCF J/K; XC/AO/grid; stationary J′;
other force/Becke response; one-electron preparation; final-state snapshot,
decode and validation; corrections; other ownership; and outside-public-execute
capture/driver synchronization. Kernel device times are a separate, nonadditive
view: never add them to synchronous host waits or parent intervals. Exclusive
J′ dispatch/wait is 60.196200→35.154146 s. Complete intrusive diagnostic time
is 95.189178→70.208878 s, not a clean timing-population sample.

Entered work is unchanged: order-six primitive products **2,400,504,912**;
order-seven primitive products/Coulomb preparations **1,250,544**;
order-eight **95,976**. Order-seven bra/ket preparations are
2,606,140/3,147,016 and component contractions 1,937,135,584 in both arms.
Unexposed order-six preparation counters stay null rather than being inferred.

| Compiled entered source-algebra scope | Baseline | Candidate |
| --- | ---: | ---: |
| All 28 order-six boxes, products | 16573 | 15984 |
| All 28 order-six boxes, additions | 2300 | 2956 |
| 648 dddp components, scalar/jet products | 194184 | 62244 |
| 648 dddp components, scalar/jet additions | 36288 | 27504 |
| 648 dddp components, coefficient reads | 127008 | 13608 |
| 648 dddp components, root reads | 36288 | 18432 |

These are entered source operations, including neutral operations a machine
compiler may fold, **not GPU instructions or FLOPs**. Spatial counts are an
unweighted 28-box sum, not endpoint totals; cooperative counts describe one
response view. The increased spatial additions are not suppressed. Nsight
Compute 2025.2.1 fails with `ERR_NVGPUCTRPERM` in finite n4 job 1008, so no
hardware instruction/FLOP count substitutes for the missing measurement.

Order six retains 255 registers/thread and 128-thread CTAs. Order seven changes
222→214 registers/thread, retaining 256-thread CTAs and 416 static plus 42984
dynamic shared bytes. These are resource metadata, not achieved occupancy,
measured traffic or full process peak-memory measurements.

## Scientific checks and rejected work

- 318 focused host tests pass: independent 70-digit Wick sums over all 28 boxes
  and response faces, including zero coordinates/rho; independent correlated-
  Gaussian quadrature and raised/lowered derivative checks. Existing independent
  finite differences remain. Overlapping host-test records are not summed.
- Six actual-linked-runtime independent Libcint GPU cases pass RKS/UKS,
  Cartesian/spherical and separate/combined J′/K′ with order-seven/eight work.
  Additional retained-AD/independent symmetry-orbit GPU checks pass.
- Compute Sanitizer memcheck and synccheck report zero errors through Slurm.
  Full process peak memory and global interleaved performance promotion with
  complete compile/memory budgets are not claimed.
- A cross-root scalar DAG is accurate but regresses the primitive probe
  ~0.428→0.443 ms despite registers 216→168. Its patch/test are archived, not
  enabled. Accepted isolated probes are ~0.428→0.308 ms for 32768 order-six
  primitives and ~285.51→75.38 ms for 512 dddp quartets, with derivative
  differences 3.61e-16/5.56e-17. These are not complete-endpoint speedups.
- Prior failed scheduling/page-width/cooperative-six trials are not repeated or
  restored. Incomplete-reference and portable-build preflights return no timed
  endpoints. The original 48-atom startup-SHA ambiguity is resolved by a clean
  repeat; the original population and different moved trajectory remain retained.

## Reproduction and storage

This compact bundle preserves **every primary energy/force array and both
independent geometry oracles losslessly** in standard `numeric-arrays.npy.xz`.
`numeric-index.json.xz` supplies shapes, offsets, all scientific protocols and
per-call geometry/timing/count/gate metadata. Run `python verify-arrays.py`
to recompute all 96 gates. `--restore-references` also reconstructs reference
JSON under `references/none/METHOD/ATOMS/`; decompress `def2-svp.json.gz`.
No rounding, delta encoding or custom numeric codec is used.

Other JSON records use ordinary XZ; source snippets/wrappers use gzip.
`endpoint-summary.json.xz` retains every phase timing/count and source/reference
digest. `provenance.json.xz` binds loaded images, source identities, scheduler,
runtime environment and raw local receipts. Detailed native documents with DFT
iteration histories, full source inventories, cache statistics, Nsight databases,
build logs and original supplemental force arrays remain hash-bound under
ignored `.artifacts/direct-reuse/` and remote
`/data/jzzeng/gqc-direct-reuse-20261016` on n1/n4/n5. Public resource rows bind
full kernel names by SHA-256; raw Nsight summaries remain local. This avoids
duplicated diagnostics without increasing the existing evidence storage cap or
altering unrelated historical evidence. No Release, tag or external backup is
created. The publication uses existing schemas and accepts the numerical scope,
not a formal universal performance promotion.

`build.sh` takes `SOURCE_ROOT` for each frozen checkout. Place compiled libraries
and manifests under each arm's `runtime/`, source exports under `source/`, and
references/basis under the campaign root. The runner accepts
`GQC_DIRECT_REUSE_ROOT`. Included n4/n1/n5 Python wrappers document the cluster
glibc/CUDA environment; adapt paths and match retained dependency versions.
The n1/n5 `oracle-python` directory contains only dftd4/cffi/pycparser, not
shadow GenerativeQC/PySCF production work. Never override CUDA visibility.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=28G --time=00:35:00 \
  bash run-gpu.sh candidate r2scan-3c 96 endpoint
```

Use `profile` for the separate intrusive capture, `tests` for Libcint checks,
and `counted` for HF's returned native work counters. The source-algebra
counters consume the actual generated baseline/candidate headers; compilation
must use the verified compiler cache. These replay paths are not reference
oracles in production.

## Default and remaining work

Both improvements are automatically enabled in their existing admitted
generated consumers on this branch, without a new opt-in. Master is unchanged;
PR #2541 remains draft/dependency-blocked on #2532/restacking. The next force
optimization is compiler-owned primitive-pair preparation reuse across Cartesian
components with convolved derivative contractions, not CTA/page tuning. Old
cooperative-six trials were removed and are not runtime tuning flags.
