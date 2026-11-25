# H100 energy and force timing along a water96 displacement sequence

Per-step comparison for [#3169](https://github.com/jinzhezenggroup/generativeqc/issues/3169).
The unit of observation is one trajectory per method, integral mode and engine
on one device. Step 0 starts from the engine's default cold guess. Steps 1–10
use the engine's own preceding converged state. This measures a prescribed
sequence of geometry changes within a single process and owner lifetime.

## Results on `22f72d069`

All 176 timed endpoints converged. The 160 moved calls used their own previous
converged state, with zero native seed fallbacks. All 24 sampled paired gates
passed at steps 1, 5 and 10. Each row reports one trajectory per engine, with
ten moved calls and one cold endpoint. The moved sum excludes the cold call.

| Water96 | Mode | Cold E+F GQC / GPU4PySCF (s) | Moved median GQC / GPU4PySCF (s) | Ten moved steps GQC / GPU4PySCF (s) | Sampled gates |
| --- | --- | ---: | ---: | ---: | ---: |
| ωB97M-V | Direct | 33.63 / 77.17 | 36.54 / 28.76 | 361.28 / 288.60 | 3/3 |
| ωB97M-V | DF | 20.31 / 39.90 | 16.05 / 25.99 | 160.55 / 260.12 | 3/3 |
| PBE | Direct | 4.21 / 8.34 | 4.32 / 3.48 | 42.20 / 34.75 | 3/3 |
| PBE | DF | 3.88 / 7.69 | 2.93 / 2.84 | 29.48 / 28.33 | 3/3 |
| PBE0 | Direct | 6.75 / 12.15 | 8.96 / 6.44 | 88.06 / 64.37 | 3/3 |
| PBE0 | DF | 4.71 / 7.65 | 3.91 / 3.26 | 39.16 / 32.72 | 3/3 |
| r²SCAN | Direct | 4.90 / 9.92 | 4.75 / 4.71 | 46.67 / 47.08 | 3/3 |
| r²SCAN | DF | 4.64 / 9.01 | 3.37 / 4.13 | 33.79 / 41.41 | 3/3 |

For ωB97M-V, GenerativeQC took 25.2% longer across the ten Direct moved calls
and 38.3% less time across the ten DF moved calls. The cold-plus-moved totals
were 394.91 s versus 365.77 s for Direct and 180.86 s versus 300.03 s for DF.
The cumulative columns in [steps.csv](steps.csv) include the cold call.
[summary.json](summary.json) retains the moved-only and cold-plus-moved sums.

## Work behind the moved timings

The GenerativeQC cold target solve used five full-target cycles and eight
provisional frozen-VV10 cycles. Every moved solve used five full-target and
five provisional cycles. The cold Hückel preparation included seven
preliminary Fock builds, recorded separately. The moved solves skipped
that preliminary work and retained one analytic-force endpoint per step.
The native `coulomb_recurrence` counter recorded 1,873,221,716 completed
resident-source values for cold execution and 1,872,218,159 to 1,873,108,665
per moved step. Native source preparation charges completed values even if a
later admission gate declines the cache. The count measures completed source
values and carries no standalone wall-time attribution.
[Source and counter](https://github.com/jinzhezenggroup/generativeqc/blob/22f72d0692a760c2892cd7791022584d9ae1c3e5/src/dft/cuda_ks.cpp#L802-L823)

The automatic same-geometry warm proof requires retained state without a
supplied initial-density buffer. A moved solve transports the prior density
to a new geometry and enters a new full-target convergence check. This
explains how early SCF work fell while full-target cycles and resident
source construction persisted. GPU4PySCF measured seven SCF cycles and
eight `get_veff` calls at each moved step. Those are distinct counters
from native Fock builds. The recorded complete-endpoint timer includes
geometry updates, SCF, forces and device synchronization. These timings
support the joint cost of that work. The individual stages have no separate
timer in this campaign.
[Warm admission and source policy](https://github.com/jinzhezenggroup/generativeqc/blob/22f72d0692a760c2892cd7791022584d9ae1c3e5/src/dft/cuda_ks.cpp#L2674-L2719)

PBE Direct illustrates the automatic cold-only admission rule. Its native cold
counter records three full anchors, six delta builds and one post-SCF full
build. Every moved PBE solve used eight Fock builds with automatic incremental
Direct work inactive. The prepared capability requires native cold-guess
provenance for an automatic supplied-density solve. PBE's moved median was
4.32 s and its cold endpoint took 4.21 s.
[Automatic incremental selection](https://github.com/jinzhezenggroup/generativeqc/blob/22f72d0692a760c2892cd7791022584d9ae1c3e5/src/dft/ks_incremental_policy.hpp#L33-L41)

The same admission change appears in Direct r²SCAN and PBE0. Both recorded
six cold delta builds and zero moved delta builds. PBE0's median moved
endpoint took 8.96 s, with eight Fock builds, and its cold endpoint took
6.75 s, with nine Fock builds.

For ωB97M-V DF the native cold call used 15 Fock builds, including four
full-target cycles and three VV10 pair sweeps. Each moved call used nine
Fock builds. The first moved call recorded three full-target cycles and
two pair sweeps. Its 16.05 s moved median is 4.26 s below its 20.31 s
cold endpoint.

## Source, allocation and costs

The measured source is `22f72d0692a760c2892cd7791022584d9ae1c3e5`. The
fresh Release/sm90 build used sccache 0.16.0, with 3,254.31 s of Ninja
compilation and all four stationary functional profiles passing the native
cubin audit. All timed and reference processes used one physical H100 80 GB
HBM3. [provenance.json](provenance.json) carries the platform receipt, hashes,
build commands, cache statistics, AOT inventory and the source archive identity.

Both qz Jobs were in CQ-NexChem / 分布式训练空间 with a single H100 at
1 point per allocation hour. The launcher requested priority 4 and usage
reported 4. Job status reported effective `NORMAL / 20`.

| qz Job | Outcome | Estimated H100 hours | Estimated points |
| --- | --- | ---: | ---: |
| `gqc-3169-h100-trajectory-22f-1125` | Succeeded and deleted | 1.594 | 1.594 |
| `gqc-3169-h100-trajectory-1125-a` | Superseded, stopped and deleted | 0.186–0.219 | 0.186–0.219 |

The current estimate spans the primary-container start and scheduler-assignment
clocks through the platform finish time. The superseded estimate uses the stop
request and deletion acknowledgment as its end bounds. The rate comes from the
live quota row. Actual per-job billing was unavailable. Both Jobs initially
reported a scheduling resource warning and later obtained an allocation.

The superseded job used `d154546d2` and completed 11 native and five GPU4PySCF
ωB97M-V Direct endpoints. Its native moved sum was 363.41 s on a separate
H100. The original interrupted records and acquisition drivers are retained
with their data provenance in [legacy-archive.json.xz](legacy-archive.json.xz).

## Protocol

- Water96 is the nested 32-water fixture used in the [paper's H100
  cohort](../paper-hfdft-cold3-20261120/README.md). The same ordered coordinates
  and atom identities, charge 0, singlet state and spherical def2-SVP basis are
  used for every endpoint. A fixed PCG64 direction (seed 3169) is scaled from
  0.005 to 0.050 Bohr per-atom vector RMS in increments of 0.005 Bohr. The
  recorded Bohr vectors and their hashes are the replay inputs.
- Both Direct and DF run for ωB97M-V, PBE, r²SCAN and PBE0. The native DF
  calculation uses the paper's frozen GPU4PySCF default auxiliary-basis
  export. The reference DF calculation checks its actual auxiliary basis at
  every step. Each engine applies its own DF metric, screening, exchange
  approximation, defaults for XC and VV10 and SCF stopping rules.
- GenerativeQC Standard uses one public prepared owner with warm updates
  enabled. Each moved call supplies the new coordinates to that owner.
  GPU4PySCF 1.9.0 keeps one RKS or DF-RKS instance. Its public molecule update
  and RKS reset invalidate geometry-dependent state. It passes its own
  previous density explicitly as `dm0`. Reference gradients enable moving-grid
  response and preserve the DF auxiliary response default.
- Cold time includes constructor, preparation, SCF, analytic forces and host
  output. Moved time includes geometry update/reset, SCF, force calculation
  and host output. The reference's carry-forward density preparation is
  included where another step follows. Imports, initial CUDA context setup,
  receipt serialization and separate oracle work are outside the timers.
  A CUDA synchronization immediately before and after the endpoint brackets
  both measurements. Native Fock builds and GPU4PySCF `get_veff` invocations
  have distinct semantics and are reported with their own labels.
- Per-case first-arm order is fixed as Standard first for Direct and
  GPU4PySCF first for DF. Each method/mode/engine uses a fresh process. Descriptive
  moved medians and sums cover steps 1–10, with step 0 reported separately.
  The population covers one fixed displacement path on one H100.

## Sampled numerical gates

Each of steps 1, 5 and 10 uses separate fresh Direct reference owners with
full-density Focks, 1e-11 Eh energy, 1e-7 orbital-gradient and 1e-14 integral
screening tolerances. The references use GPU4PySCF 1.9.0 with XC levels 4 and 5,
VV10 levels 2 and 3, and analytic forces with complete grid response. Their
converged densities are never passed to the timed runs. Both integral modes
are checked against the common level-5 Direct target. Across both timed
engines and all sampled cases, the largest errors were 7.36e-6 Eh/atom,
4.97e-5 Eh/Bohr for a Cartesian force component and 1.57e-5 Eh/Bohr for
force RMS.
The thresholds are 1e-5 Eh/atom in absolute energy, 1e-4 Eh/Bohr in maximum
Cartesian force component, and 3e-5 Eh/Bohr in Cartesian force RMS. Grid-pair
stability uses one quarter of each threshold. Record timings even when an
observable gate fails. The tighter references use the independent GPU4PySCF
implementation for GenerativeQC and independent solves of the same library
for the timed GPU4PySCF arm.

## Reproduction

```bash
python3.11 benchmarks/results/h100-moved-trajectory-20261125/replay.py
python3.11 benchmarks/results/h100-moved-trajectory-20261125/replay.py --steps-csv
```

For new device acquisition, stage the exact `22f72d069` Git archive as
`source/` with its `source.tar.gz` and `source.sha256`. Place
[`trajectory.py`](trajectory.py), [`build_qz.py`](build_qz.py) and
[`run_qz.sh`](run_qz.sh) beside that archive. The launcher takes a new attempt
directory name and requires an allocated H100. It uses the established read-only
software environment and auxiliary inputs, then builds Release/sm90 and audits
the native stationary AOT modules before acquisition. The launch config keeps
scheduler-provided device visibility and one numerical thread in each arm.

To regenerate the frozen auxiliary tables from the pinned scientific inputs,
use PySCF 2.14.0 and the source archive's own input writer. Match the
per-mode SHA-256 values in `records.json.xz` before comparing device runs.

```bash
PYTHONPATH="$source:$source/python" python3.11 -m benchmarks.readme_defaults \
  inputs --inputs "$inputs"
```

Full-precision energies and Cartesian forces, effective settings and actual
semantic counters are in `records.json.xz`. `steps.csv` retains all planned
rows, including empty timing fields for unavailable endpoints. `summary.json`
retains the grid-pair and production-to-reference comparisons. The raw qz
receipts are hash-pinned in `provenance.json`; the published configuration
strings replace only the personal shared-storage prefix. The canceled
`d154546d2` attempt is retained in `legacy-archive.json.xz` as a historical
comparison and is excluded from replay and the current-revision summaries.
