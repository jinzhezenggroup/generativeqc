# Native cross-functional force AO acceptance

This campaign addresses #1853's residual **public native-controller** adoption,
not the already-qualified explicit/compatibility map mathematics in #2881 or
the fitted explicit-consumer reuse measurements in #2905. The directory records
the campaign's start date; real-device acquisition is on November 18, 2026
(Asia/Shanghai).

## Decision and scope

Current master already admits a shared native exact-jet force map across its
supported ordinary/composite and Direct/DF bindings. The obsolete description
that all public native forces are dense is not a reason to add another selector
or functional-name exception. This change preserves that production policy and
adds a decline-only, same-binary dense control:
`GENERATIVEQC_CUDA_FORCE_ACTIVE_AO=0`. It also corrects `indexed_grid` tracing
when the separate FP32 grid route bypasses a retained FP64 map.

Map admission retains four-/ten-jet capability, `2**35` dense point×AO² work,
`1e-16` cutoff, at most 16 MiB optional peak storage, and mean active fraction
at most 0.8. No test lowers these guards or injects a Python force profile.
Mandatory dense capacity, independent nonlocal quadrature and all scientific
bindings remain unchanged. This is qualification of an existing guarded native
default, **not a new universal performance promotion**.

## Scientific inputs and arms

- RKS: the first 48 atoms of the retained 4s4 water cluster, 384 spherical
  def2-SVP AOs. Ordinary fitted and PBE0/Direct UKS cases add a distant hydrogen
  radical, giving 49 atoms, 389 AOs, charge zero and multiplicity two. This
  avoids the competing hole-localization SCF roots of a large water cation
  without sharing a native density guess. WB97M-V UKS in both provider domains
  uses 61
  separated H atoms, 305 AOs, charge zero and multiplicity 62: a fully polarized
  open-shell fixture including an empty beta channel. Its 390400 grid points
  still exceed the unchanged `2**35` work guard. Low-spin fitted water/H2
  fixtures and Direct water-plus-H/OH candidates failed strict independent SCF
  convergence; they are retained as rejected diagnostics, not passing evidence.
  The other UKS
  cases retain both occupied spin channels and the water oxygen d shells.
- Retained orbital and auxiliary basis files, their SHA256 identities, every
  Cartesian coordinate, charge, spin and grid setting are in each input receipt.
  Primary quadrature is `GridSpec(32, 10, 20)`; WB97M-V has a distinct
  `GridSpec(12, 4, 8)` nonlocal quadrature. The moved geometry displaces the
  second atom by 0.002 bohr in x.
- Ordinary fitted cases use retained cc-pVDZ-JKFIT. WB97M-V/DF instead uses the
  retained orbital def2-SVP snapshot as its auxiliary fixture. Independent
  libcint/NumPy spectra at both geometries must establish full rank under the
  unchanged native relative metric cutoff `1e-10`, for Coulomb and LR metrics.
  This matches GPU4PySCF's independently factorized analytic-force Hamiltonian;
  it is not a claim about the quality or all rank-truncated LR auxiliary fits.
- Both spins cover LDA/DF, PBE/DF, PBE0/DF, r2SCAN/DF, WB97M-V/DF, PBE0/Direct
  and WB97M-V/Direct. WB97M-V/DF explicitly requests complete fitted LR exchange
  with `GENERATIVEQC_DF_RANGE_EXCHANGE=df`; its oracle uses the same full RI-J/K
  Hamiltonian. This control is not evidence for every automatic LR-admission or
  exact-LR fallback branch. Direct WB97M-V uses no such override.
- Independent GPU4PySCF reconverges **both** geometries, with moving-grid and
  fitted auxiliary response. Full-Fock adapters are installed after density
  fitting. Native public calls forbid Python force orchestration and CPU
  scientific fallback; the actually loaded native library is checked.
- Every case executes `dense`, production `auto`, and true `minimum` capacity,
  each with cold/warm/moved/moved-warm E+F. The minimum arm first permits the
  automatic force tile to retire, re-queries its bounded envelope, rejects
  minimum-minus-one and then executes at the actual minimum. Merely subtracting
  one from the original automatic-tile envelope is not an OOM test.

Acceptance requires energy error ≤`1e-8` Eh, every force component ≤`3e-7`
Eh/bohr, physical residual in `[0, 1e-9]`, and paired dense/selected E/F
differences ≤`1e-9`. Actual point-AO, AO-jet and projection work must decrease;
warm replay must not rediscover maps. Minimum and dense arms must retain dense
work. `acceptance.json.xz` retains full-precision oracle/native arrays, capacity
receipts, SCF/Fock counts and work counters, not only a passed flag.

The final receipt contains **28 successful native cases / 336 complete E+F
calls**: all seven method/provider domains for each spin on each of RTX5090 and
PRO6000. Each device has 14 cases / 168 E+F calls. There are 112 calls per arm
(dense, auto, minimum), with all four lifecycle stages retained. The independent
reference is reconverged twice per case; it is not counted as a native endpoint.

| Scientific check | Unchanged acceptance gate | Maximum observed |
| --- | ---: | ---: |
| Independent energy error, Eh | `1e-8` | `1.3583303370978683e-9` |
| Independent force-component error, Eh/bohr | `3e-7` | `3.5500491435414006e-10` |
| Native physical residual RMS | `[0, 1e-9]` | `3.0410681495460085e-12` |
| Paired dense/auto/minimum energy difference, Eh | `1e-9` | `2.2737367544323206e-12` |
| Paired force-component difference, Eh/bohr | `1e-9` | `6.906741845114084e-11` |

Every auto arm executes indexed work, both geometries rebuild the correct map,
and both warm stages reuse it without discovery. Dense and true minimum arms
execute full dense work, including exact minimum-minus-one rejection after tile
retirement. The additional source-matched resident selection smoke test passes
eight identity/noncontiguous/empty/dense feature and density-jet cases.

Representative executed work at the original geometry (dense / auto). These
are complete force-grid counters, including the separately traversed dense
nonlocal quadrature where present, not an eligibility estimate:

| Domain | Tile points | Point-AO evaluations | AO-jet evaluation values | Projection FMA pairs | Additional discovery jets on cold/moved |
| --- | ---: | ---: | ---: | ---: | ---: |
| r2SCAN/DF RKS | 1024 | 117964800 / 60082176 | 1179648000 / 600821760 | 181193932800 / 53915697152 | 1179648000 |
| WB97M-V/DF or Direct RKS | 256 | 125042688 / 53441280 | 1250426880 / 534412800 | 192065568768 / 43998286848 | 1179648000 |
| WB97M-V/DF polarized H61 UKS | 256 | 126216320 / 22078336 | 1262163200 / 220783360 | 307967820800 / 27642551296 | 1190720000 |

Discovery inspects the entire primary force-order jet domain once per geometry;
it is **additional** work, not part of the smaller evaluation column. Thus
cold/moved total AO-jet values can exceed dense even though projection work
decreases. Warm/moved-warm discovery is zero, and all discovery cost remains
inside the complete endpoint timer. A separate nonlocal grid stays dense
(`shared_nonlocal_grid=0`); the primary map never authorizes its selections.

## Source and build identity

- Measured base master: `2791504cc238fe83d328fc1718080642b288222a`.
- The PR is forward-ported onto master
  `46942b9f05de4ec39cdef4102e828a5b05fccf86`. The retained GPU receipts
  identify the measured source above, not the subsequent VV10/Direct changes
  in #3111/#3112; those changes are not relabelled as requalified here.
- Loaded core SHA256:
  `a9a73394d27a1198c9d2c37743a04f9f26652b37270d7ac19753e2b1ed2f4b0b`.
- ELF build ID: `3fe9be63e26c7fdf92064dffb3409ef39228d5c9`.
- Release, GNU C++ 11.4, CUDA 12.9.86, real SM120 images;
  verified sccache 0.16.0 launchers for both C++ and CUDA. Cache statistics are
  retained before/after compilation without clearing the shared cache.
- Built native stationary profiles: LDA, PBE, PBE0, r2SCAN and WB97M-V, each RKS
  and UKS. The build disables the separate stationary-force/CPU-force libraries,
  automatic Libxc AOT, CLI/tests, and **integral AOT shells**. That last setting
  limits the performance claim to this identified build; these data do not
  certify another release packaging configuration or change stock AOT policy.
- The compressed receipt embeds build flags, source-file hashes, package and
  BLAS identities, device/driver UUIDs, Slurm allocations and acquisition hashes.
  Source hashes are opt-in historical provenance, not required-CI function-body
  fingerprints.
- Host policy/resource/lifecycle/ABI/evidence/link checks: **557 passed, six
  subtests passed**. One existing build-profile-specific test is excluded:
  `test_unbuilt_descriptor_capacity_preserves_output_and_validates_abi` assumes
  PBE is unbuilt, whereas this core deliberately builds PBE and correctly
  returns success. It is not counted as passing and no production test is
  weakened. The default-promotion inventory has 114 entries / 181 registered
  controls. Ruff checks/format, native clang-format and `git diff --check` pass;
  the full Sphinx HTML build passes with warnings treated as errors.

Timing allocations (all use finite `srun`, preserve assigned visibility and use
only the allowed nodes):

| Domains | Host / Slurm job | Device UUID | Driver |
| --- | --- | --- | --- |
| RTX5090 WB97M-V/DF | node1 / 17731 | `GPU-d79031c2-173e-535f-3563-5520bab0293b` | 580.95.05 |
| RTX5090 r2SCAN/DF, WB97M-V/Direct | node1 / 17734 | `GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6` | 580.95.05 |
| PRO6000 WB97M-V/DF | node2 / 5373 | `GPU-cacd0aaf-c80f-41eb-d7a2-4c3a5970f282` | 595.99.02 |
| PRO6000 r2SCAN/DF, WB97M-V/Direct | node2 / 5374 | `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a` | 595.99.02 |

Each pair stays on its allocation's same device and scientific SCF/Fock branch.
Matrix acquisitions use additional allocated UUIDs retained in the provenance.
Node-local Slurm accounting storage is disabled; retain that limitation rather
than inventing historical accounting records. Actual acquisition headers and
the final completed jobs' `scontrol` receipts are included.

## Replay and reproduction

Host-only replay recomputes scientific, work, lifecycle and paired timing gates
from the retained arrays:

```bash
python benchmarks/results/native-force-ao-20261117/replay.py
```

Acquire the native matrix with the matching core and reference environment:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:50:00 env \
  GENERATIVEQC_DFT_CUDA_TEST=1 GENERATIVEQC_LIBRARY=/absolute/path/libgenerativeqc.so.0 \
  GENERATIVEQC_PROFILE=off OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=python:.:tests/python python -m pytest -xq \
  tests/python/test_native_cross_functional_force_ao_cuda.py \
  --junitxml=native-5090.xml -o junit_family=legacy
```

For a PRO6000 allocation use its compatible `gpu:pro6000:1` request. Preserve
Slurm-assigned visibility; never run these commands on the maintenance node.
`collect.py` accepts successful trusted local JUnit receipts and an explicit
provenance JSON, checks the arrays, and writes the compressed evidence.

Fresh native timing runs are separate from reference acquisition:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:10:00 env \
  GENERATIVEQC_LIBRARY=/absolute/path/libgenerativeqc.so.0 PYTHONPATH=python:. \
  python -m benchmarks.native_force_ao_endpoint \
  --receipt benchmarks/results/native-force-ao-20261117/acceptance.json.xz \
  --method r2scan-rks --mode auto --pair 0 --device-label 5090 --output auto.json
```

Compare `auto`/`dense` in alternating order over five independent process pairs
on the same allocation, for r2SCAN/DF and WB97M-V/DF/Direct (`--direct`). Each
process includes calculator construction, native preparation and its first E+F
in cold time, then three frozen-warm-start replays, moved reconvergence and
three moved-warm replays. Freezing warm-start updates does not skip the public
endpoint's convergence checks or fabricate zero SCF/Fock work. Interpreter/import, basis parsing, receipt validation
and reference acquisition are outside timing. Timed processes reject importing
CuPy, PySCF or GPU4PySCF. `endpoints.json.xz` retains all E/F and semantic work;
replay rejects unequal SCF/Fock branches or non-fresh/unmatched process pairs.
Fresh-process cold means a new calculator/native owner, not an erased compiler
or persistent JIT artifact cache. Existing caches are deliberately reused.

## Complete endpoint observations

The separate timing campaign contains **60 fresh native processes / 480 complete
E+F calls**: five interleaved auto/dense process pairs for each of three RKS
domains on each device model. All pairs pass independent energy/force/residual,
actual sparse-work and identical SCF/Fock-branch gates. Each process measures
cold, three warm, moved and three moved-warm calls; all samples are retained.

Median complete endpoint seconds, **auto / dense** (each warm entry first takes
the within-process median, then the five-process median):

| Device | Domain | Cold | Warm | Moved | Moved-warm |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX5090 | r2SCAN/DF | 5.828450 / 5.761888 | 0.744433 / 0.993138 | 3.595913 / 3.773516 | 0.744051 / 0.994236 |
| RTX5090 | WB97M-V/DF | 14.672876 / 14.698076 | 5.933546 / 6.254525 | 10.805583 / 11.041471 | 5.937234 / 6.257582 |
| RTX5090 | WB97M-V/Direct | 21.350262 / 21.611313 | 3.664983 / 3.960790 | 15.840453 / 16.070800 | 3.689035 / 4.031371 |
| PRO6000 | r2SCAN/DF | 5.467492 / 5.611388 | 0.708008 / 0.957390 | 3.457621 / 3.642886 | 0.707710 / 0.958351 |
| PRO6000 | WB97M-V/DF | 13.079775 / 13.327901 | 5.245576 / 5.558452 | 9.789245 / 10.038008 | 5.244961 / 5.559148 |
| PRO6000 | WB97M-V/Direct | 19.399222 / 19.667636 | 3.428810 / 3.738623 | 14.481414 / 14.714222 | 3.422631 / 3.733493 |

The replay's decision statistic is the **median of paired auto/dense ratios**,
not the ratio of the independently aggregated medians in the preceding table:

| Device | Domain | Cold | Warm | Moved | Moved-warm |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX5090 | r2SCAN/DF | 0.980948 | 0.749739 | 0.953308 | 0.748386 |
| RTX5090 | WB97M-V/DF | 0.996826 | 0.948864 | 0.979384 | 0.949334 |
| RTX5090 | WB97M-V/Direct | 0.990961 | 0.922099 | 0.984651 | 0.919402 |
| PRO6000 | r2SCAN/DF | 0.973194 | 0.738486 | 0.948998 | 0.738818 |
| PRO6000 | WB97M-V/DF | 0.981623 | 0.943554 | 0.974375 | 0.943483 |
| PRO6000 | WB97M-V/Direct | 0.986557 | 0.915421 | 0.983550 | 0.916750 |

These observations support retaining the existing guarded native selection on
the tested workloads, particularly warm replay. They do **not** establish stable
cold benefit: RTX5090 cold ratios exceed one for r2SCAN/DF pair 2 (1.024175),
WB97M-V/Direct pair 0 (1.002214), and WB97M-V/DF pairs 3 and 4 (1.001338 and
1.011521). No sample is discarded. Do not infer a universal cold/device win,
change the production crossover, or promote another build/profile from these
results. Unadmitted work remains dense, and the decline-only control provides
an explicit dense choice without a functional-name scheduling exception.

## Retained boundaries

Small work, unsupported jets, ECPs, optional capacity/allocation refusal and
excessive occupancy remain dense. Distinct nonlocal quadrature is independently
dense even when the primary XC grid is indexed. Empty/noncontiguous selections,
stale-state rejection and map lifetime retain the existing independent tests;
they are not replaced by this matrix. The Python explicit/compatibility profile
registry remains separate and its unqualified composite defaults are unchanged.

## Rejected comparator and retained diagnosis

The first large WB97M-V/DF attempt used the same cc-pVDZ-JKFIT auxiliary as
ordinary cases. Its **dense baseline**, before map selection, differed from
GPU4PySCF by `2.695855982892681e-8` Eh and failed the unchanged `1e-8` gate.
The normalized independent 1856-dimensional LR metric retains 869 modes under
the native relative cutoff (largest eigenvalue `664.6047216377914`, absolute
cutoff about `6.646e-8`); GPU4PySCF's fallback uses an absolute `1e-7` cutoff
on its own representation. These are not the same fitted Hamiltonian. Tightening
the native cutoff is not a safe comparator fix: the `1e-12` diagnostic failed
native numerical qualification. Neither attempt enters passing evidence.

The def2-SVP auxiliary fixture has all 384 modes retained for the original
RKS geometry: LR minimum `5.242710622446932e-7`, maximum
`1566.0380068870336`, well above its `1.566e-7` relative cutoff. Every final
WB97M-V/DF reference independently checks its own spectra, including the UKS
and moved geometries. The original failure and spectra are retained separately
in `diagnosis.json.xz`; no tolerance or scientific production policy changes to
make an unmatched oracle pass. Existing independent fixed-cutoff LR response
tests remain the authority for rank-truncated fitting, not this AO-map fixture.

The low-spin water-plus-H independent solve also failed after 200 iterations
with a 20-vector DIIS diagnostic: gradient norm `0.33876924473559916`, RMS
`0.0015200771814510002`. A separated 30-H2-plus-H doublet likewise failed the
unchanged SCF gate. Direct water-plus-H and water-plus-OH references also failed
the strict gate on both devices, before native force execution. Increasing
iterations/DIIS space or weakening convergence is not a qualification strategy.
The fully polarized H61 fixture must converge independently at both geometries
before any native E/F comparison. WB97M-V UKS evidence is limited to that
explicit spin-polarized domain, not the rejected low-spin clusters or all
practical auxiliary bases. Existing independent explicit-map UKS coverage is
retained; a force map cannot repair an unconverged independent SCF.

Each six-case UKS attempt had five intact passing native tests and one rejected
Direct water-plus-H reference. `diagnosis.json.xz` retains both **entire original
JUnit XMLs**, including their failing statuses, plus the exact five-case receipt
split and its hashes. Acceptance reuses only those five successful cases and
separately qualified WB97M-V fixtures. No failing/skipped testcase is accepted by
`collect.py`, no failed endpoint is turned into a passing reference, and the
mixed campaigns are never reported as six passes.
