# Initial qualification: default bounded bisection for large four-jet semilocal XC

Status: implemented
Date: 2026-11-10

This records the **initial four-work-jet, pure-semilocal qualification**. The final
admission and additional functional measurements are in
[`2026-11-10-default-xc-bisection-functional-extension.md`](2026-11-10-default-xc-bisection-functional-extension.md).
Its broader gate supersedes this initial gate; the source identities and numbers
below remain the original populations rather than being relabelled final runs.

## Decision and invariants

An unset `GENERATIVEQC_CUDA_XC_SPATIAL_ORDER` now selects tile-aligned
bisection for strict-FP64 RKS pure-semilocal owners with at least four AO and
work jets, 512 AOs and 524,288 points, on compute-capability 8+ devices reporting
FP32/FP64 throughput ratio greater than two. Selection uses work/capability
properties, never a molecule, functional or product-name match.

Small, one-work-jet GGA, hybrid/range-separated, unrestricted, mixed-precision,
nonlocal, full-rate-FP64/unknown-device and explicit reference/public resource
budget paths retain their incumbent schedules. Explicit `1` is the center-group
rollback, `2` remains the bounded bisection request, and `0` stays canonical.
The native policy is in `src/dft/ks_tile_policy.hpp`; the constructor invokes it
inside the existing optional spatial-storage admission.

No quadrature point/weight, AO cutoff, SCF stopping gate, initial density or
precision changes. The canonical resident grid still owns moving-grid forces.
Existing checked host/device allowances, transactional publication, allocation
refusal and graph/lifetime contracts are unchanged. No CPU/reference imports
enter native endpoints, and no new persistent cache or provider is introduced.

## Why this default

The preceding [proposal](../../proposed/2026-11-10-df-r2scan3c-spatial-bisection-cold.md)
showed a 1.76-second opportunity on n4, already larger than the requested
0.5 seconds. The acceptance below executes the **new binary with the selector
genuinely unset**, not the old opt-in relabelled as an automatic run.

The diagnostic default has 262,959,168 point-active-AO visits and
66,625,602,624 point-active-AO-square work per Fock, versus the preceding
center diagnostic's 300,677,440 and 89,166,908,992: 12.54% and 25.28% less
work respectively. It retains ten cold XC/Fock submissions and 32-tile batches.
The spatial copy has the same allocation allowance; work reduction rather than
a larger memory promise explains the gain. Full-rate FP64 providers are not
promoted from these consumer-GPU observations.

## Current-source complete endpoint qualification

Master base: `d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`.
Executed source identity:
`ecd7f50df26f79df1a450ba65d9d4d6bfff6dee3a091c9a21a7e8ee7c0d014e0`.
Native library SHA256:
`3b94e04d6174b7251e5a32e893fae6f7039f8f0294899b694717153489375937`.

The qualification build is an isolated source-bound incremental CMake-recipe
replay/relink, **not a clean rebuild**. It verifies that only `cuda_ks.cpp` and
`ks_tile_policy.hpp` differ in the native source/include trees from the frozen
805de425a build, and rebuilds the complete changed-header C++ closure
(`cuda_ks.cpp`, `dft_method.cpp`, `cuda_rhf.cpp`) plus the new build-identity TU.
Unchanged CUDA/generated objects are borrowed with individual SHA256 receipts.
Verified sccache 0.16.0 wraps every actual compilation; cache statistics and
commands are retained. The runtime source identity matches the current checkout.

n1/Slurm node1 RTX5090, CUDA12.9.1, one numerical thread, fresh process and owner
per population. Cold includes constructor, preparation, SCF, analytic forces,
D4/gCP where applicable and host export, after context initialization; process
startup and cache-miss compilation are excluded. Each native process then runs
three same-owner calls from its own frozen cold density.

| Endpoint | Pairs | Center cold median / s | Unset default / s | Saving / s |
| --- | ---: | ---: | ---: | ---: |
| DF r2SCAN-3c96 | 3 | 23.826760 | 22.075337 | 1.751423 |
| DF r2SCAN-3c48 | 2 | 8.457433 | 7.778170 | 0.679264 |
| DF r2SCAN96 | 1 | 13.050831 | 12.308207 | 0.742624 |
| Direct r2SCAN-3c96 | 1 | 29.277254 | 27.226835 | 2.050419 |
| DF r2SCAN-3c48, fresh moved owner | 1 | 8.243961 | 7.728297 | 0.515664 |

The main three-pair saving is 7.351%. All six main cold calls report ten
iterations/Fock builds; every warm call reports one. The main warm ratio is
0.96832. One-pair supplemental results are scoped guards, not independently
established universal speedups. PBE DF48/96, PBE0 DF96 and r2SCAN-3c DF12
remain outside automatic bisection; observed cold variations are within 1.9%
and all warm ratios within 0.3% of one. No gain is claimed for these guards.

Slurm17064/17065/17066 retain 24 native populations / **96 complete native E/F
calls**, plus nine fresh independent GPU4PySCF populations / 18 calls. Every
native export passes unchanged paired gates (energy1e-8 Eh, force max1e-7 and
RMS3e-8 Eh/Bohr) and the independent stock-grid common gates (energy/atom1e-5,
force max1e-4 and RMS3e-5). Observed maxima are 1.82e-12 Eh / 9.21e-12
Eh/Bohr paired and about1.35e-6 Eh/Bohr independently. Every native worker
rejects PySCF/GPU4PySCF imports. These are default-grid comparisons, not new
dense-grid convergence certificates or GPU4PySCF timing-ranking claims.

## Resource, geometry and negative observations

- Slurm17068 adds ten cold/warm/moved/moved-warm/fresh-moved native calls on
  DF48. Automatic versus center at each corresponding phase passes the original
  tight paired gates; all independent default-grid comparisons pass. The existing
  moved-owner versus fresh-owner force difference is about2.76e-6 Eh/Bohr in
  **both** schedules. It is retained, not called a 1e-7 fresh-owner parity pass
  or attributed to bisection. No tighter grid/topology equivalence is claimed.
- The configured suite passes **444 tests plus four subtests**, including new
  executable policy boundaries and existing ledger, AO, point-batch, fleet,
  precision and spatial resource/lifetime checks. Initial local native-dependent
  runs lacked a configured library; the final source-bound rerun passes.
- The existing native `--spatial-grid` contract passes independent CPU E/V,
  PBE/r2SCAN RKS/UKS, exact permutations, tails, changed-density graph replay,
  canonical force ownership and exact one-byte resource refusals. Compute
  Sanitizer memcheck reports **zero errors**. Test-only hook objects are separate
  from the measured production library. Preliminary test-hook link/dependency
  directory setup failures are retained, not counted as scientific passes.

Complete full-precision exports, allocations, diagnostics, the recomputing
`summarize.py`, source patch and build/test receipts remain in
`.artifacts/df-r2scan3c-default-20261110/` and
`n1:/data/jzzeng/df-r2scan3c-default-20261110/`. These are local qualification
artifacts, not a new externally published archive. The tracked benchmark budget
has only6,472 bytes free, so this task neither raises that cap nor removes
unrelated scientific evidence to add another archive. The preceding proposal's
raw n4 observations remain separately retained.

## Revisit when

Requalify complete endpoints, independent observables and capacity/lifetime
contracts before broadening to full-rate FP64, fewer work jets, mixed precision,
hybrids, UKS, nonlocal or public resource-ledger consumers. Keep center/canonical
rollback and optional allocation refusal even when a broader schedule wins.
