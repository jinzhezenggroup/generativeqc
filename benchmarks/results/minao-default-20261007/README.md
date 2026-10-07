# Guarded MINAO default: frozen qualification and follow-up smoke

The historical CUDA qualification uses frozen parent
`2b0feff1d5577e58f3a44fbecbc1974be92df4e3` plus
[`qualified-source.patch`](qualified-source.patch), **before default promotion**.
It does not inherit the later master cross-chunk exact-K default or assign
historical performance to subsequently modified source. Both native arms use
one library with SHA256
`f0f066d23ed8508df2bd9bbb9346a30e78dabf6b2c3754fa69570d365c9361d9`.

## Protocol and results

RTX 5090 on n1, Slurm job 6404, preserved `CUDA_VISIBLE_DEVICES=3`, Release
sm_120, CUDA 12.9.1, strict FP64 exact PBE0/def2-SVP, spherical AOs, no DF,
unpruned `GridSpec(48, 16, 32)`. Energy/density/screening tolerances are
`1e-12 / 1e-10 / 1e-12`, maximum target iterations 100. Three fresh-owner,
fresh-process samples per arm interleave as
`core0, minao0, minao1, core1, core2, minao2`; no trajectories are discarded.

The complete cold E+force timer includes preparation, seed construction,
transfer/admission, SCF, physical closure, analytic forces and synchronization.
It excludes imports, context initialization and Calculator construction, per
the README endpoint protocol. Existing compiler/JIT caches are reused.

| System | Hcore median / s | MINAO median / s | Reduction |
| --- | ---: | ---: | ---: |
| 48-atom water, 384 AOs | 99.047145 | 86.841994 | 12.32% |
| 96-atom water, 768 AOs | 211.832632 | 181.389462 | 14.37% |
| 16-atom peroxide holdout, 152 AOs | 40.162046 | 35.969546 | 10.44% |

[`summary.json`](summary.json) retains all timings, semantic work counts,
paired native density comparison results and the warm replay. [`samples/`](samples/)
retains all 18 native endpoints and three independent GPU4PySCF reference
records with exact geometries, basis identities, settings, final energies/forces,
work counters, acceptance gates and library identities. All endpoints pass the
unchanged `1e-8 Hartree / 1e-7 Hartree/Bohr` reference gates. Maximum paired
native density difference is `3.10e-11`; all MINAO cold attempts build zero
preliminary Focks and use one target attempt. The retained warm replay skips
preparation. There is no CPU speedup or whole-process peak-memory claim.

Native density arrays and shared libraries are not committed. Their paths and
hashes remain in the records; the full 37 MiB artifact set is retained on n1 at
`/data/jzzeng/qc-2068-minao-profit-20261007/results` and mirrored at
`/home/jzzeng/codes/qc-branch-audit-20260922/evidence-2068/results`.

## Reproduction

Reconstruct the frozen qualification from the pinned parent and patch, rather
than assuming current master represents the measured source. Build a CUDA
Release/sm_120 library with verified compiler-cache launchers (e.g. CMake CXX
and CUDA launchers `ccache`); keep the existing cache. Install the repository
Python package plus its reference-test/GPU4PySCF requirements and expose that
checkout on `PYTHONPATH`, with `GENERATIVEQC_LIBRARY` pointing to the rebuilt
library. Export an accurate `QUEUE_SOURCE_LABEL` for the reconstructed source.

[`cold.py`](cold.py) supports the historical explicit `--guess core` (None) and
`--guess minao` arms, plus `--guess default` that omits the keyword for new-source
smoke checks. Every real-GPU run requires a finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python benchmarks/results/minao-default-20261007/cold.py \
  native --atoms 48 --guess minao --reference reference-48.json --output minao0-48.json
```

Generate independent references with the same harness's `reference --atoms 48`
and `reference --atoms 96` commands under Slurm. `--holdout` selects the peroxide
case (use `--atoms 48 --holdout`); `--warm-probe` adds an untimed retained-density
replay. Repeat both native arms in the stated interleaving for each case.
[`summarize.py`](summarize.py) recomputes the report from all raw JSON records and
newly generated `.density.npy` arrays, failing closed on missing/failed samples,
reused cold states or mixed library identities. Do not override device visibility.

Follow-up default-path smoke is separate evidence, not another three-repeat
profitability campaign. See the linked Agent Note for promotion boundaries and
the explicit Hcore rollback.

## Default-path validation

[`default-smoke/`](default-smoke/) records source parent
`7c07fa309cf7f9123cde697e460169e481bfca01` plus this repair/default switch,
including that parent's cross-chunk exact-K default. The new library SHA256 is
`c4e2cb2d509a045adf928a46ac0765e4a51cf821d2d349ace55126b11b2f4f6c`; historical
qualification library `f0f066...` is preserved. Slurm job 6427 on node1 retains
its assigned device 1 and uses a finite 15-minute limit.

- 236 focused Python tests pass, including CPU/GPU omission/default selection,
  explicit Hcore rollback, domain/resource guards, checkpoint/warm precedence,
  independent CPU RHF E/force and native ensemble-oracle contracts.
- 122 public Calculator/checkpoint/Hessian tests pass, with two skips; native
  CPU preliminary and real-GPU MINAO probes pass.
- Omitted-keyword CUDA cold E+force samples take 85.024941 seconds for water-48
  and 35.922225 seconds for peroxide, with independent E/force gates passing,
  zero preliminary Focks and one target attempt. Water-48 warm replay skips
  preparation and passes the same gates.

These are single-sample default smoke timings, not a new statistical speedup
comparison or qualification of subsequent upstream source. CPU inclusion does
not imply uniform benefit: the linear H3+ / STO-3G compatibility example takes
nine MINAO iterations versus six for Hcore, while final-state gates agree.

## Rationale

- [GPU solver ownership](../../../.agents/notes/implemented/performance/2026-10-07-cuda-minao-eigen-owner.md)
- [Guarded CPU/CUDA promotion](../../../.agents/notes/implemented/performance/2026-10-07-minao-default.md)
