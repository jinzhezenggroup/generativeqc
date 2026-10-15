# Decision: share r2SCAN branches only in the FP64 stationary consumer

Status: implemented; 12/48-atom direct and DF endpoints qualified
Date: 2026-10-15

## Problem

The r2SCAN stationary force point independently lowers the energy and all seven
polarized feature partials. They repeatedly traverse identical piecewise
predicates, even though the existing scalar emitter can share lazy regions
across simultaneous outputs. This work is shared by ordinary r2SCAN and the
electronic component of canonical r2SCAN-3c.

The historical semilocal SCF coalescing experiment in
`2026-10-14-xc-shared-branch-regions.md` also rescheduled the SCF consumer
and observed an extra-build moved-endpoint regression. A faster force consumer
does not justify repeating that SCF experiment or changing convergence gates.

## Decision

Expose the existing scalar emitter's branch schedule at the shared r2SCAN
program boundary. Opt in only from the FP64 stationary Cartesian consumer,
for both general polarized and proven equal-spin inputs. Keep the default
program schedule, all CPU/SCF entry points and FP32 force arithmetic unchanged.

The scalar emitter still owns branch sharing and lazy evaluation. No expression
is hoisted from an unknown inactive arm. Independent polarized AD precedes
equal-spin binding; component-specific Libxc work floors, original-density
energy rescaling, the tail switch and the force tau/2 convention are unchanged.
Scheduling changes emitted-source/artifact identity, not mathematical identity.
No new runtime flag, functional whitelist, allocation or fallback is needed.

## Invariants

- Do not use force-point timings as complete E+F speedups.
- Preserve SCF arithmetic, grid, screening, tolerances and actual Fock histories.
- Keep canonical def2-mTZVPP, D4(BJ)-EEQ-ATM and gCP for r2SCAN-3c.
- Retain both spin partials, inactive singular-arm safety and strict publication.
- Reject an endpoint regression rather than relaxing acceptance to rescue it.

## Scalar and interface qualification

Based on master `5f80ba629`. Eight focused host tests pass, including strict
bitwise parity of all eight raw output lanes, the independent 113-bit Libxc
empty-spin/tail fixture with spin exchange, actual Cartesian finite differences
and tau/2. Compiler structure checks 540 modules with zero dependency errors.

On n5 RTX 5090, finite Slurm 1583, the actual production Cartesian point body
is compiled with the native stationary consumer's `--fmad=false`. RKS and UKS
outputs are bitwise identical to master at 257, 8192 and 65536 points, including
a partial final block. Independent Libxc errors are below 1.9e-13. Six paired
CUDA-event samples per population show 1.74x RKS and 1.81–1.84x UKS point
speedups. Generated Cartesian source shrinks from 213082 to 120270 bytes.
These are diagnostic point measurements, not molecular endpoint claims.

The final candidate host suite passes 71 tests (13 resource/optional tests
skipped) with PySCF 2.14.0 / Libxc 7 and the newly built native library. Six
Slurm public native CUDA force tests pass, including RKS/UKS, empty-spin and
changed-geometry consumers that forbid Python force/compiler fallback. The tail
suite passes its CPU case and both actual CUDA SCF/force consumers against the
independent 113-bit fixture. Slurm 1585 memcheck reports zero errors for both
spin schedules, zero/empty-spin inputs and partial blocks of 1/129/257 points.

An earlier probe with NVCC's default FMA setting is retained separately: it
passes independent numerical gates but is not bitwise equal and does not match
the production native stationary compiler flags. It is not the qualification
receipt.

Complete endpoint, build, sanitizer and source-preservation evidence is retained
under ignored `.artifacts/r2scan-force-branches-20261015/` and
`/data/jzzeng/qc-r2scan-force-branches-20261015` on n5 during qualification.
Complete molecular measurements run on n1 in finite Slurm 8536, preserving its
assigned physical device 5. The native core has both RKS/UKS stationary profiles;
large domains additionally require the separately built RKS SPD force DSO and
its generated manifest. Initial Slurm 8531 omitted that DSO, so both arms failed
with missing-artifact errors rather than numerical errors. Those incomplete
setup records are retained separately, not relabeled as accepted measurements.
The corrected run preflights both artifacts and retains their hashes alongside
the main native image. Both arms use the same Release/sm_120, strict FP64,
OpenBLAS and compiler-cache configuration; no cache is cleared.

## Complete endpoint results

The unmodified `benchmarks/run_readme_dft_benchmarks.sh` driver requests cold,
five fixed-density warm calls, moved geometry and five moved-warm calls. Both
arms complete all 12 calls for each method/size/Hamiltonian: 192 native calls,
all independently rechecked against the retained GPU4PySCF/Libxc 7 references.
The 3c oracle additionally uses independent CPU dftd4 4.3.0 and gCP. Maximum
energy/force errors are 7.9354e-11 Eh and 4.9928e-11 Eh/Bohr, below the unchanged
1e-8/1e-7 gates. Local and n1 reducers produce the same summary SHA-256.

The moving grid is 48 x 16 x 32 per atom, with spherical def2-SVP for r2SCAN
and the defining def2-mTZVPP for r2SCAN-3c. DF has its own explicit auxiliary
basis and independent oracle; it is not a faster direct Hamiltonian. Table
entries are complete seconds in **cold/warm/moved/moved-warm** order; warm
entries are medians of all five samples, with preparation/export inside the
endpoint. No time is normalized by SCF work.

| Method | DF | Atoms | Master seconds | Candidate seconds | Cold/moved Fock builds, both arms |
| --- | --- | ---: | --- | --- | --- |
| r2scan | none | 12 | 5.852/0.872/3.646/0.877 | 5.711/0.772/3.540/0.769 | 17/12 |
| r2scan-3c | none | 12 | 9.478/1.777/6.758/1.765 | 9.325/1.692/6.707/1.678 | 19/13 |
| r2scan | none | 48 | 37.874/5.425/27.540/5.399 | 37.358/5.165/27.500/5.181 | 18/12 |
| r2scan-3c | none | 48 | 84.574/22.551/62.993/22.540 | 84.998/22.275/62.815/22.241 | 19/13 |
| r2scan | cuda | 12 | 5.671/0.626/3.040/0.624 | 5.571/0.531/2.957/0.531 | 17/12 |
| r2scan-3c | cuda | 12 | 6.892/0.741/3.987/0.740 | 6.934/0.645/3.971/0.644 | 18/13 |
| r2scan | cuda | 48 | 32.901/4.807/24.146/4.803 | 32.542/4.303/23.718/4.309 | 17/12 |
| r2scan-3c | cuda | 48 | 51.170/6.233/37.190/6.241 | 50.603/5.726/36.700/5.723 | 19/13 |

Every warm and moved-warm call uses one Fock build. Actual iteration and Fock
histories match across arms in every phase, not just their medians. Warm E+F
time falls 1.22-15.15%, depending on the cohort; moved-warm falls 1.32-14.97%.
Do not hide the 0.5% direct-3c/48 cold or 0.6% DF-3c/12 cold regressions, or
extrapolate the point-kernel gain to the full endpoint. In particular, direct
3c/48 has only a 1.2% warm improvement: its expensive integral work is unchanged.
No 96-atom population is measured here, no SCF-tail problem is declared fixed,
and no broader 48/96 convergence/promotion gate is closed by this campaign.

## Reproduction and receipts

Source reconstruction is master `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`
plus the retained `scientific.patch`; tests/notes do not alter that patch.
`run_endpoints.sh`, `reduce.py`, source-preservation/probe scripts, all timing
samples, force arrays, SCF histories, setup failures and native/AOT hashes remain
under the ignored campaign directories. Local endpoint records are in
`.artifacts/r2scan-force-branches-20261015/remote-records/`; the same campaign
directory on n1 retains the runnable source trees, binaries and dependencies.
The float receipt binds the float point body; an existing mixed consumer's
strict FP64 fallback naturally receives the optimized double point.

Core builds use Release, architecture 120, a verified ccache 4.5.1 launcher,
OpenBLAS, native `r2scan_rks;r2scan_uks` profiles and no CPU force AOT. Build
`generativeqc_stationary_r2scan_rks_spd_manifest` with stationary force AOT
enabled for each arm, and stage that DSO/JSON alongside the corresponding core.
The finite reproduction command on n1 is:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=01:30:00 \
  bash /data/jzzeng/qc-r2scan-force-branches-20261015/run_endpoints.sh
```

The recorded compiler cache counters are shared with other tasks; they prove
launcher use, not that every reported hit/miss belongs to this campaign. No
benchmark-results checkout budget is increased and no unrelated evidence is
removed. Compact receipt hashes:

```text
scientific.patch         d8e40a5c38bf978f9e9814b8bbcf8dd5ed5f6ebe2e3c0c942a374ace814f1d98
source-preservation.json 317f72262d180b8116f8b7f00dea2c3851570bb27fd5d928c8fc6730fa75d802
point-results.json       c34db541a95abe8c1b9826c79bcc5309f53e350f9976fff88bd7eaf937297e00
endpoint-summary.json    58837bdb135c9482e273cd19cb75506839cc7cb303fe345f227c285896811f6d
```

## Revisit when

A new graph or target changes the register/spill or endpoint tradeoff, or a
separately qualified SCF scheduling change establishes all-phase improvements.
Do not infer either condition from this force-only scalar result.
