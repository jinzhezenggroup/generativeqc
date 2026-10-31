# DF schedule executable sharing

Baseline: `2152ae76722a338a67d2b399ec765caeed79b777` (the full measured revision
is authoritative in `receipts.json.xz`). Candidate: that revision plus the
retained production patch. Measurements used NVCC 12.9.1, SM120, Release,
sccache 0.16.0 and a Slurm PRO6000 allocation on node2.

| Measurement | Before | After |
| --- | ---: | ---: |
| Ordered angular classes / CUDA objects | 80 | 80 |
| Distinct full schedule contracts | 240 selectors | 173 executables |
| CUDA kernel entries | 1,260 | 932 |
| DF object bytes | 356,273,672 | 343,371,360 |
| Complete library bytes, same configuration | 703,195,096 | 691,818,680 |
| Timed DF compile phase, seconds | 272.187450 | 188.911811 |

The reductions are **26.0% kernel entries, 3.62% DF object bytes, 1.62% complete
library bytes and 30.6% observed DF compile time**. The timing is one build per
arm with the existing compiler cache retained: candidate compilation had 14
CUBIN-stage hits. It is not a cold-cache timing distribution. An unrelated
Direct J/K compile later restarted sccache and fell back to local compilation;
that phase is excluded from the compile-time claim. Other AOT families were
disabled identically in both arms; this is not a full-distribution size claim.

All 1,260 original entry points map to **identical machine instruction bytes
and REG/STACK/SHARED/LOCAL resources** in the candidate. The reduction keeps
large, distinct schedules and removes smaller duplicate executions; entry
count alone would overstate the binary-size saving.

Three alternating fresh-process pairs cover water RHF with compact/packed/warp
and packet/on/off buckets, OH UHF, water batch size four, water8 at 64 MiB
(streamed response), and water16 at 1 GiB. Each process includes cold preparation,
SCF and forces, three warm recomputations from a frozen seed, and displaced
geometry (+0.015 Bohr). A separate untimed call records semantic work. Across
210 timed calls and 42 traced calls, energies match exactly, maximum force
change is 8.18e-14 Eh/Bohr, and SCF iterations, Fock builds, tiles and all semantic
counters agree. Only diagnostic packet-preparation nanoseconds differ.
The largest candidate/baseline median time ratio is 1.0031; every cold/warm/moved
cohort meets the 1.05 non-regression gate. No runtime speedup is claimed.

Independent validation: 151 host dispatch/ledger checks, 470 mathematical tests
(12 unavailable cases skipped), and 72 GPU complete-force matrix cases against
PySCF, covering scalar/BLAS, Cartesian/spherical, RHF/UHF and all three schedules.
The matrix explicitly selects panel response because its BLAS counter assertions
do not describe automatic occupied response. The original test had the same six
counter failures on both revisions; automatic selection remains exercised by
the endpoint campaign.

Exploratory water16 at 64 MiB failed before producing a sample. Water16 at
256 MiB and water8 at 48 MiB were cancelled after several minutes without a
complete sample. These are recorded limitations, not successful qualification
at those budgets. The successful 64 MiB water8 campaign already covers streamed
response. Raw transient logs and binaries remain in ignored experiment storage.

`receipts.json.xz` retains full precision endpoint outputs/timings/work records,
object/resource/instruction hashes, binary hashes, source reconstruction and
compiler-cache statistics. Recompute the historical gates without a GPU:

```bash
python benchmarks/results/df-schedule-sharing-20261027/reduce.py
```

`endpoint.py` reproduces a fresh-process endpoint (set `PYTHONPATH` and
`GENERATIVEQC_LIBRARY` to the matching source/library and invoke through finite
`srun`). `census.py` inspects the 80 objects offline with `cuobjdump`; it never
initializes a GPU. Machine-code hashes are one-off migration evidence, not a
permanent required-CI source fingerprint.
