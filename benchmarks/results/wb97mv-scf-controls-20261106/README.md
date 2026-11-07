# Rejected WB97M-V SCF controls, 2026-11-06

One bounded 2x2 mechanism cohort on an actual A100, followed by the
Huckel/raw-versus-orthogonal 5090 guard. These are complete cold
constructor/preparation/SCF/analytic-force/host-export endpoints for the same
96-atom water geometry, full Direct WB97M-V/def2-SVP, spherical basis and
unchanged Standard scientific controls. This is **negative evidence, not a
performance improvement or a fresh GPU4PySCF ranking**.

## Measured endpoints

| Device | Native initial guess | CDIIS error metric | Cold E+F (s) | Warm E+F (s) |
| --- | --- | --- | ---: | ---: |
| A100 | Huckel | raw AO | 111.897483 | 15.095748 |
| A100 | MINAO | raw AO | 116.359662 | 15.108940 |
| A100 | Huckel | orthonormal | 113.102631 | 15.067734 |
| A100 | MINAO | orthonormal | 116.629770 | 15.120110 |
| RTX5090 | Huckel | raw AO | 194.035070 | 14.353750 |
| RTX5090 | Huckel | orthonormal | 194.495628 | 14.349936 |

A100 warm values are one frozen-seed call per arm; 5090 values are medians of
three frozen-seed calls. There is one cold sample per arm, not a statistical
estimate of a small slowdown. No parameter sweep or second full timing cohort
was run. In particular, a few hundredths of a second of warm variation does
not establish a solver win.

Every cold arm performs **18 molecular iterations / 18 Focks**, partitioned
identically into nine semilocal/RSH initializer Focks, one full target, four
frozen-VV10 inner Focks, and four full-target tail Focks. That is **13
provisional + 5 full-target Focks, 4 VV10 pair sweeps and 1 pair-cache hit**.
Every warm sample performs one physical Fock and no pair sweep. MINAO is
actually used in both MINAO arms, not a fallback to Huckel; its A100 native
preparation costs about 2.55 s versus about 0.15 s for Huckel. Huckel's seven
preliminary atomic Focks are separately reported, not extra molecular cycles.

The prototype forms `X.T @ (FDS-SDF) @ X` in existing resident scratch only
for CDIIS history. It leaves the physical AO residual, publication gates,
grids, precision, screening, Direct operators and VV10 outer strategy intact.
No CPU/PySCF/reference computation supplies a native endpoint. The prototype,
selector and temporary tests are **removed from production**; both defaults
remain unchanged. The exact prototype patch and executed test sources are
retained inside the compressed evidence, not shipped as a permanent opt-in.

## Provenance and numerical qualification

- Base: `318ae3b99f7af1e7fdc83d33dc7acd9da2917efb` (master at measurement).
- Tested source identity:
  `e66dfb990044fd595d0969df728c0ecbfb0f8a0114570aa48796d8c041aefc95`.
- Same actual ELF SHA256 on both devices:
  `8cd1bf438532d52c763cc95e817845e7566d6f911fea565f5189e1dc719ea1dd`.
- Actual finite allocations: A100 LSF **225852**, `gpu_chem/gpu22`, 18 min;
  RTX5090 Slurm **16344**, `main/node1`, 18 min. Scheduler device visibility
  is retained and never overridden. CPU compilation uses verified
  **sccache 0.16.0**, preserving the shared cache.
- All **16 native samples / 32 independent repeat pairs** pass the original
  static reference gates: energy `1e-5 Eh/atom`, force max `1e-4 Eh/Bohr`,
  force RMS `3e-5 Eh/Bohr`. Worst errors are `7.37e-12 Eh/atom`,
  `4.81e-7 Eh/Bohr`, and `2.01e-7 Eh/Bohr`. The pinned independent A100
  reference is the existing
  `direct-range-warm-anchor-20261105/publication.json.xz`, SHA256
  `6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f`.
  Its old timing is not compared with this new cohort.
- Host checks: 11 pass / 22 explicitly skipped after repairing the temporary
  extracted-source test stub. The initial one-failure report remains retained.
- Initial small-water checks: **2 failed / 1 passed**. Both failures were the
  new `1e-7` cross-trajectory force-parity assertion, not cold/warm lifetime;
  max force differences were `2.012e-7` and `1.879e-7`. The finite-difference
  force check passed. These failures are not relabeled as passes.
- Subsequent actual Slurm16344 checks: **3 pass** with the *same* `1e-7`
  parity gate and *tighter* SCF controls (`1e-12` energy / `1e-10` density),
  plus **2 pass** from the existing independent default-profile GPU4PySCF
  tests at original/moved water geometries. Default scientific gates were
  never weakened to make trajectory parity green. The first small test's
  scheduler ID was not retained; accounting is disabled, so none is invented.

Expired SSH authentication, stale namespace detection, channel timeouts, an
unsuccessful read-only node inspection, and a pipefail/SIGPIPE verification
wrapper are retained separately from GPU results. The complete A100 archive
was transferred despite a channel-close timeout; separate remote SHA256
proof, LSF DONE and driver exit zero authenticate it. No GPU job was repeated
for transport failures. The owned transfer server is stopped with a separate
remote-zero completion receipt.

The reused mutable n1 workspace is restored to its pre-experiment qualified
source `ff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa`
and ELF `4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a`,
with allocation-free identity and policy-v2 receipts. This is restoration of
that existing owner's image, not a claim that this older image is new master.

## Offline replay

```bash
python benchmarks/results/wb97mv-scf-controls-20261106/replay.py
```

Only Python's standard library is needed. `evidence.json.xz` retains every
endpoint sample, full force array, resolved controls, precision/work history,
scheduler/image receipts, initial failures, executed tests, build/cache recipe,
prototype patch and cleanup/restoration receipts. `summary.json` is derived,
not another measurement. Verbose build/authentication logs remain ignored
local artifacts identified by SHA256 and byte count; authentication contents
are deliberately not published.

The scripts embedded in the evidence describe the exact finite allocations
and experiment; restoring their stale host paths or reusing old job IDs is
not a runnable allocation request. A fresh reproduction must rebuild the
retained patch on its stated base, authenticate its image, and obtain its own
finite scheduler allocation. No new ranking or default promotion follows
from offline replay.

Rationale and revisit conditions:
[rejected decision](../../../.agents/notes/rejected/2026-11-06-wb97mv-minao-orthogonal-diis.md).
External memory: issue **#2976**; preceding diagnosis: PR **#3018**.
