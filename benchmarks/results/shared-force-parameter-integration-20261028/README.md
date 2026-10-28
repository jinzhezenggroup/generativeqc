# Shared force and parameter integration qualification

## Integrated R1 and structural parameter package

The `final/` records qualify source commit
`150c9d5bdfd1c04cf1a3601ac0c3bebf6c319ebc`. This includes the R1 native force
owner, structural parameter discovery and the standalone COSX link repair.
All 5,951 tracked build, scientific and test input blobs on node2 matched that
revision. The subsequent ancestry-only merge of the landed storage repair
changed none of those inputs. `final-manifest.json` binds the library, CLI,
drivers, original receipts and compressed storage. This is correctness and
integration qualification; it makes no new performance promotion.

All 18 automatic cases passed: KTBM 0/1, VWN 1/4, APBE, LTA, PBE_SOL, BR78 and
WIGNER in both spin layouts. Their cold/warm/moved phases cover 54 complete force
endpoints and 186 independently reconverged finite-difference directions. KTBM
uses full vectors; the remaining cases use the existing directional gate. The
test checks independent Libxc/PySCF energy, density, Fock and force agreement,
native ownership, work diagnostics and resource lifetime at unchanged tolerances.

All four PBE-D3/WB97M-V CLI compositions and four native CTests passed. CPU
derivative AOT is enabled in this build: the R1 CPU suite passed 44 tests, with
12 existing CUDA-only split-hybrid skips and 56 CUDA cases deselected. CLI
force comparisons still use CPU/GPU energy finite differences at two steps;
the separate CPU suite exercises native analytic forces with independent oracles.
The complete Slurm job (`3701`) returned zero for all four suites.

The first R1 build exposed missing CPU derivative shards in the standalone COSX
test. Its original failed build/configuration/cache logs are retained alongside
the successful retry; no scientific failure or skipped case was relabelled.
The retry used verified sccache 0.16.0, retained existing build outputs, and
reported 47 compilation requests, 47 misses, zero hits and zero compile failures.

Recheck exact bytes, artifact ownership and recorded force/CLI gates without a GPU:

```sh
PYTHONPATH=python:. python benchmarks/results/shared-force-parameter-integration-20261028/verify.py
```

To reproduce on the matching node2 environment, restore the measured Git source,
copy each `final-*` driver to its `execution_path` in the manifest, and decode
`final/build-inputs.json.xz` to `.artifacts/2835-final-inputs.json`. Adapt checkout
and environment paths consistently, then run both phases through finite Slurm
allocations, preserving the assigned device visibility:

```sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --time=00:45:00 bash .artifacts/build.sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:50:00 bash .artifacts/2835-final-test-n2.sh
```

## Historical integration before R1

The `pre-r1/` records qualify source commit
`84c30caafaa2e8393cbb5cb7a6406300f63a16f2`, before integration of the subsequently
merged R1 force owner. They are historical integration evidence, not a claim
about later source revisions. There is no performance promotion.

The build used GCC, CUDA 12.9.86, SM120 and verified sccache 0.16.0 on node2.
Every real-device command ran in a finite Slurm allocation with its assigned
device visibility. All 3,184 tracked scientific/build input blobs matched the
recorded Git revision. `pre-r1-manifest.json` records that check and original
and stored identities for every compressed receipt. XZ decoding restores the
exact original bytes; no timing sample or failure outcome is relabelled.

The automatic force matrix covers KTBM 0/1, VWN 1/4, APBE and LTA in both spin
layouts, with cold, repeated and moved-geometry endpoints. KTBM uses full-vector
finite differences; the other registrations use the established directional
gate. The existing test independently reconverges PySCF/Libxc and checks energy,
density, full Fock, forces, artifact ownership and work diagnostics. All twelve
cases passed. Production calls forbid Python scientific fallbacks.

Four native CLI compositions cover PBE-D3 and WB97M-V in both spin layouts.
Their complete forces agree with CPU and GPU energy finite differences at two
step sizes; WB97M-V also retains its independent PySCF/Libxc center-energy gate.
The CPU derivative AOT option was disabled in this build, so the CLI comparison
uses CPU **energy finite differences**, not an unavailable CPU force endpoint.
The original attempt with grid options on a base PBE route was rejected before
execution and corrected to the repository's existing named-composition fixture.

The four accompanying drivers reproduce the source-matched build and finite
Slurm test commands after adapting only checkout/environment paths. They do not
authorize rerunning GPU work outside Slurm or admitting these old receipts to a
different compiler subject. Complete build logs and binaries remain ignored
local artifacts; the manifest identifies the measured library through the raw
endpoint receipts.

Replay this older phase explicitly with `verify.py --phase pre-r1`. Both replay
modes preserve their measured source scope; neither grants admission to a
different compiler revision.
