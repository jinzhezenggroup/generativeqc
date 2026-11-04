# Occupied XC: exact original-D restriction

Only bitwise-original-D-identical finite features, including signed zero, may
reuse the occupied candidate. Otherwise publish the entire original vector.
The error bound is only a prefilter. KS defaults to matrix products; explicit
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` requests the verified experimental route.

## Source-bound evidence

- `receipt.json.xz`, source `0b9adb1a6dfcb7bae88f922235d247d52bf09bb9`:
  **162 focused + 168 consumer host cases**; 510 optional CUDA skips are not
  qualified. Two bitwise regressions fail against the prior template.
  Slurm **15811**: **138 passed per normal/memcheck/synccheck/racecheck cohort**,
  zero errors/hazards. Includes real GPU Gram and ragged compact batch kernels.
- Supplemental **15819**, source `60faa4bac6afd8aee4d17f6eee473add8bd77e4b`:
  again **138 per cohort**, zero errors/hazards; the actual GPU Gram producer
  yields stored D00=0 and dense/verified rho=0; ordinary rho=.125 is unchanged.
- `native-receipt.json.xz`, source **`def31af3f6c9b8476e8ff9d892f7f74589d168c7`**:
  **212 host tests**, **four native tests** (Slurm **15825**) and **four native
  memcheck tests, zero errors** (**15827**). Default/explicit/rollback/invalid
  controls exercise cold/warm/moved lifetimes. Independent reconverged PySCF
  2.14.0/Libxc 7.0.0 default-grid-3 energy/grid-response force oracles cover RKS
  and charge-one-doublet UKS, original/displaced three-atom spherical def2-SVP
  water (24 AO). CPU scientific derivative fallbacks are forbidden.
- **160 complete E+F observations**, ten alternating pairs per spin/phase,
  pass unchanged **2e-9 Eh / 1e-7 Eh/Bohr** independent/paired gates. Maximum
  errors: **6.964e-13 Eh / 3.155e-10 Eh/Bohr**. Paired iterations, Fock builds
  and complete semilocal J/K work match. All vectors/costs are retained.

Final native library SHA-256:
`e0a208a73b832d314631280ad771b46bbd89ab25a92c6bbcd010d894614ff5a5`.
Release/C++20/O3/sm_120/`--fmad=false`, verified sccache 0.16.0, caches retained.
Actual-library cuobjdump admits all eight changed/fallback density launches;
feature registers are 80/96, stacks 0/80 bytes. Declared local bytes are zero;
spill counts, achieved occupancy, allocator peak and whole-force-region
profitability are not measured. These facts are not standalone-probe proxies.

Complete tracing-off costs include admission: verified cold/moved is about
22% slower for RKS and 51–53% slower for UKS; warm UKS about 16% slower. Cold
includes construction/preparation/E+F in a shared process/context with existing
caches, not process startup/cache-miss compilation. No speedup/default promotion
is claimed. All timing samples, vectors and resource identities are in the receipt.

## Reproduction

Decode receipts with standard-library `lzma.open`/`json.load`. Select the exact
library with `GENERATIVEQC_LIBRARY`, set `GENERATIVEQC_PROFILE=off` and
`PYTHONPATH=.:python:tests/python`. Run through finite Slurm:

```bash
srun --input=none --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:15:00 env GENERATIVEQC_TEST_OCCUPIED_XC=1 \
  python -m pytest -q -x tests/python/test_xc_occupied_default_cuda.py
```

Jobs 15825/15827 preserve assigned visibility 0/1. Inside the same scheduler
contract, prefix pytest with `compute-sanitizer --tool memcheck
--target-processes all --error-exitcode 99`. `readers/native_endpoint.py --help`
describes cost reproduction; `independent_oracles` supplies its exact input
vectors, never a production fallback. `native_resources.py` assesses demangled
final-library cuobjdump. Offline:

```bash
python benchmarks/results/occupied-xc-exact-identity-20261104/readers/replay_native.py
```

This recomputes every retained vector/work gate and historical input hash.
Raw compiler/cache/trace/sanitizer histories remain ignored, with byte/hash
identities; initial setup, build-budget and reader mistakes are not hidden.
CI/review and merged delivery remain required before closing #2751. No
all-functional/planner qualification or cause for #2915 is inferred.
See the [decision note](../../../.agents/notes/implemented/numerics/2026-11-04-occupied-xc-exact-identity.md).
