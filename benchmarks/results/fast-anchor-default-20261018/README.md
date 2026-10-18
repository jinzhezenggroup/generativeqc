# Production Fast fixed-anchor delta qualification

This qualifies the guarded production default, not the earlier environment-only
experiment. The frozen base is `5d1dbbcaf2f04fb83980c8f988795df28b41e392` plus
the three production input changes retained in `qualification.json`. No unrelated
changes from the original dirty working checkout are part of this source.

## Admission and retained fallback

The default applies only to relaxed application-budget Fast WB97M-V restricted
exact-Direct cold VV10 outer solves already admitted by the existing arithmetic
and shape policy. FP64, strict caller controls, other models, DF, UKS, public
resource ledgers, device chunks and same-owner warm solves retain their paths.
`GENERATIVEQC_KS_FAST_ANCHOR_DELTA=0` rolls back; unset or `1` cannot force
unsupported domains. Optional device OOM retains ordinary Fast; other errors
propagate. Five extra FP64 AO matrices and one scalar are charged to state
capacity, adding 23,592,968 bytes at 768 AO.

## Complete cold endpoint protocol and results

Same water geometries, spherical def2-SVP, WB97M-V/RKS, `48 x 16 x 32` grid,
native automatic Modified Huckel guess, no DF, history eight, maximum 100 target
iterations and unchanged independent policy gates as the
[frozen experiment](../wb97mv-anchor-delta-20261018/README.md).
Cold time includes construction, preparation, first synchronized energy plus
analytic forces, host/density export, teardown and cache release. Imports, basis
parsing, provenance and independent-oracle comparison are outside the timer.
References are read only after the endpoint; they never seed production.

All six observations run in exclusive finite Slurm job **1142**, node4 RTX 5090,
with scheduler-assigned visibility preserved. Order: Fast-12 default/off,
FP64-12 default/off, Fast-96 default/off. Existing compiler/artifact caches are
retained; no first-use time is subtracted. The optional resident-source allowance
is 2 GiB in every arm.

| Atoms | Policy | Selection | Cold E+F (s) | Target Focks | VV10 pair sweeps | Max force error (Eh/Bohr) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 12 | Fast | default | 36.052 | 17 | 2 | 2.31e-6 |
| 12 | Fast | off | 12.058 | 17 | 2 | 4.36e-5 |
| 12 | FP64 | default | 24.184 | 30 | 4 | 1.19e-11 |
| 12 | FP64 | off | 24.263 | 30 | 4 | 1.19e-11 |
| 96 | Fast | default | 460.816 | 16 | 2 | 3.19e-6 |
| 96 | Fast | off | 579.282 | 18 | 3 | 6.46e-6 |

Every endpoint passes its unchanged independent energy/maximum-force/RMS-force
gates. At 96 atoms the production default reduces observed complete cold time
by **20.45% (1.257x)** with two fewer target Focks and one fewer VV10 pair sweep.
The operator inventory retains four strict J applications in both arms, versus
twelve/fourteen mixed J applications for default/off. Preliminary Huckel work
is included in endpoint time and recorded separately from target Focks.

This is one observation per arm, not repeated statistics. Water-12 has unchanged
Fock/sweep counts and a slower first default observation; no small-size speed
claim or first-use adjustment is made. FP64 default/off differ by only 1.14e-13
Eh, 8.97e-14 Eh/Bohr maximum force and 2.05e-13 maximum density; no FP64 speedup
is claimed. Fast remains a different convergence contract from FP64.

## Build and validation

The source-authenticated production library identity is
`8be4a7fcfc9c653688cac2cb9f269dfca4acb6a78213c6fbe7e92db679305318`;
library SHA256 is
`0ad9d62d76fd2f2b0635e8fcf17901186952ec475e51f7d54ebe9e84d0d2629f`.
Builds use verified sccache 0.16.0 on n1. Original builds remain unchanged;
unchanged objects are reused only after full source-inventory comparison, and
every changed-source/header dependency plus the identity owner is recompiled.
Compiler commands and before/after shared-cache statistics are retained.

- Matching-source Python regression suite: **269 passed, 32 skipped**. The skips
  are opt-in device cases, not accepted device evidence. An initial local run
  selected an unrelated stale library; the reported result uses this authenticated
  library and matching Python sources on n1.
- `generativeqc_ks_cuda_tests --fast-anchor-delta`: passes under finite Slurm on
  n1. The real 24-AO GPU owner is compared to an independent CPU WB97M-V
  composition for default/on/off, FP64 and strict-controls exclusion, warm
  replay, injected runtime failure/recovery, resource-ledger fallback and
  exact optional state-capacity accounting. Internal counters positively verify
  anchor/delta execution rather than assuming the environment selected it.
- A retained one-off UBSan host probe passes four algebra/admission/allocation/
  failure cases, including optional device OOM and propagation of other errors.
  It extracts controller bodies for diagnosis and is not a new required-CI
  source-shape contract.
- The broad native executable does **not** pass: it stops at the existing
  strict two-AO WB97M-V chunk-history assertion,
  `bounded WB97M-V nonlocal SolverRegion changed the converged trajectory`.
  Explicit Fast-anchor rollback reproduces the same failure, before this new
  Fast test runs and outside Fast's 24-AO admission floor. Both logs are retained;
  this PR does not claim a passing complete native suite or fix unrelated chunks.

## Evidence and reproduction

`qualification.json` retains all six accepted reductions, raw-receipt hashes,
unchanged-library checks, work/precision inventories, independent gates, paired
state differences, build authentication and compiler-cache snapshots.
Full arrays, raw receipts, test logs, executable source archive and build harness
remain in ignored `.artifacts/fast-anchor-default-20261018/` in the PR worktree
and `/data/jzzeng/fast-anchor-default-20261018/` on n1/n4 as applicable.

The retained `endpoint.py` and `run-endpoint.sh` reproduce each arm. Source the
CUDA/Python environment, set `EXPERIMENT_ROOT` to an authenticated directory with
`source/`, `library/`, `def2-svp.json`, `reference12.json`, `reference96.json` and
the scripts, and preserve the same 2 GiB source allowance. The reference receipts
are the same independently reconverged oracles described by the frozen experiment;
their hashes are retained with each acceptance result.

```bash
srun --exclusive --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash "$EXPERIMENT_ROOT/run-endpoint.sh" \
  fast 96 auto "$EXPERIMENT_ROOT/new-fast96-auto"
python benchmarks/results/fast-anchor-default-20261018/analyze.py \
  .artifacts/fast-anchor-default-20261018/suite \
  --build-receipts .artifacts/fast-anchor-default-20261018/build-receipts \
  --output /tmp/fast-anchor-qualification.json
```

Run the corresponding `0` arm in a fresh output directory. Use an allowed node
(`n1`, `n2`, `n4` or `n5`), a compatible GPU request and finite scheduler time.
The toolchain on n4 uses the retained matching private glibc loader through its
Python wrapper; do not replace the Slurm device environment to work around host
toolchain differences.
