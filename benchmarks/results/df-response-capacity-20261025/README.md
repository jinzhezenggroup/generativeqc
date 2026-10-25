# Pure Coulomb response across scratch-capacity boundaries

The full-rank raw-source pure-J force now selects its charge-first algebra before
optional tensor allocation. Previously, an allowance between one and two complete
AO-pair/auxiliary tensors selected a fitted tensor before checking the pure-J
route. That added `O(Nao² Naux²)` inverse-factor work, so intermediate systems
could take longer than larger ones with less available storage per tensor.

## Complete endpoint observations

All 36 native populations / 144 complete energy-and-force calls are retained:
Standard/Fast, r2SCAN-3c at 12/24/36/42/48 atoms, and PBE/B3LYP at 36/42 atoms.
Cold includes calculator construction, preparation, SCF, analytic forces and
host export. Warm is the median of three calls from the frozen cold density.
[Every observation and warm range](timings.md) remains visible, including
unchanged and slower cases. These are individual cold observations and repeated
warm calls, not a universal or statistically interleaved performance ranking.

| Arm | r2SCAN-3c atoms | Control cold (s) | Candidate cold (s) | Control warm (s) | Candidate warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 36 | 11.064 | 7.113 | 6.155 | 2.219 |
| Standard | 42 | 16.118 | 8.766 | 10.206 | 2.884 |
| Fast | 36 | 10.326 | 6.362 | 5.663 | 1.714 |
| Fast | 42 | 15.165 | 7.797 | 9.588 | 2.237 |

All 72 paired samples have identical energies, iterations and reported Fock
counts. Maximum force change is **2.853e-12 Eh/Bohr**. The independently qualified
12/24/48-atom routes and PBE/B3LYP controls have no material speedup in this cohort.

Two additional same-binary Standard controls use the existing explicit
`GENERATIVEQC_DF_COULOMB_RESPONSE=panels` comparison. At 36 atoms, cold/warm
changes from 11.026/6.138 to 7.100/2.217 s; at 42 atoms, from 16.136/10.204 to
8.835/2.891 s. Their 16 complete calls remain separate from the main cohort.
The override recreates the old tensor algebra in this capacity range; it is
not a replacement default or a pooled timing sample.

## Scientific work and invariants

At 36 atoms (408 orbital / 2556 auxiliary functions), the old response performs
4,350,127,804,416 inverse-factor FLOPs and unpacks 425,481,984 raw values. At
42 atoms (476 / 2982), those counts are 8,059,148,810,496 and 675,649,632.
Both now perform zero such tensor transforms/unpacks. Their metric-vector work
is 26,132,544 / 35,569,296 FLOPs. These are semantic counts, not hardware FLOP
measurements. Existing authenticated packed raw values feed the charge directly.

The metric cutoff, raw source identity, FP64 arithmetic, density, full derivative
and moving-grid response remain unchanged. Exchange, retained fitted/occupied
factors, truncated metrics, explicit panel/dot diagnostics and allocation retry
retain their established paths. No benchmark size, molecule or GPU model enters
selection, and no memory allowance is enlarged.

Diagnostic traces independently attribute the old warm force response to
4.487/8.088 s at 36/42 atoms, versus 0.550/0.756 s after the repair. Profiling
times do not enter the endpoint tables. `profiles.json.xz` retains response work,
leading component regions and hashes of the complete local traces.

## Qualification and retained failures

159 final related test cases pass across the broad run and targeted reruns.
Coverage includes independent extended-precision signed-density adjoints,
ill-conditioned metrics, rank/exchange refusals, PBE/r2SCAN-3c energy and full
forces, geometry rebuilding, response capacity/retries and borrowed-state replay.
The new three-capacity test checks independent PBE endpoints and bounds semantic
work. Its intermediate-capacity case fails on the unmodified control and passes
on the candidate. Low and high capacity controls pass on both.

The broad run passed 148 tests and exposed 11 preexisting error-category
assertions. The unmodified control reproduces all 11: invalid force controls now
preserve `INVALID_ARGUMENT`, whereas the tests expected `NUMERICAL_FAILURE`.
Updated expectations preserve rejection and subsequent independent replay checks;
all 11 reruns pass. The first capacity fixture used a response override that this
DFT owner does not consume; the next used a public allowance below mandatory
preparation storage. Both attempts remain in `regressions.json.xz`, alongside
the final meaningful boundary test. They are not production failures or timing
samples, and no scientific tolerance was relaxed.

All **12 moved-geometry/property checks** pass against independent GPU4PySCF
at 36/42 atoms, in both profiles, for E+F → E → E+F replay. Common gates remain
1e-5 Eh/atom, maximum force 1e-4 and RMS force 3e-5 Eh/Bohr. Separate small
matched-model tests retain their tighter 2e-7 Eh / 2e-6 maximum-force gates.
This qualification does not turn the old README's default-grid accuracy failures
into passes. The complete README cohort remains at its original measured source.

## Provenance and reproduction

Control: `2ed9b394a2889b3a373dc91ebbaaed03931056eb`.
Candidate: `0c26bbf11af084dabceb7b825bbfdd0f624ec03a`.
The main candidate receipts originally record the unambiguous short SHA
`0c26bbf11`; replay resolves it against the full archive/build binding without
rewriting receipts. Subsequent test/documentation and C++ formatting changes preserve the measured
production behavior. The candidate also includes master's full-range hybrid host
provider; the same-binary controls isolate the pure-J response change.

All 12,761 archived candidate files match the measurement checkout and actual
incremental-build inputs. CUDA 12.9.1, GCC 13.3, Release sm120, verified sccache
0.16.0, 557 cache-wrapped compiler rules, cache statistics and core/AOT
hashes are retained in `provenance.json.xz`. Every real-GPU job uses finite Slurm
allocations on node1 RTX 5090 with its assigned visibility, four CPU cores, and
OMP/OpenBLAS/MKL thread counts of one. Scientific settings and all executed
commands/allocations remain in receipts and `jobs.json.xz`.

```bash
PYTHONPATH=python:. python benchmarks/results/df-response-capacity-20261025/reproduce.py \
  --output .artifacts/df-response-capacity-replay
```

This rechecks every complete paired and moved-property numerical gate and
regenerates the timing table. GPU reexecution uses the retained scripts after
adapting historical installation paths; it requires compatible finite `srun`
allocations. The fixed auxiliary tables are retained in the preceding
[README campaign](../df-dft48-work-20261025/README.md). Source archives, binaries
and full profiler traces remain in ignored local/node1 artifacts under
`hf96-df-response-20261025`; no external release assets are required.

## Separate Direct HF experiments

The preceding Direct HF repair is already in the
[README figure cohort](../df-dft48-work-20261025/changes.md). New exploratory
48/96-atom Standard Direct HF measurements retain all six populations separately
in `hf-trials.json.xz`. Explicit ordered DIIS gives 96-atom cold 19.341 → 18.456 s,
with unchanged 11 Focks and force differences below 4.7e-13 Eh/Bohr. It has not
received broader default qualification. Forcing a 16-GiB resident ERI allowance
instead regresses 96 atoms to 24.357 s and 48 atoms from 6.116 to 7.727 s.
These experiments change no HF default and do not replace README data.
