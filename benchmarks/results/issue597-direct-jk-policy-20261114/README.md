# Direct J/K runtime-policy qualification

Baseline: master `5e34e2769915ead77ffd1a93f9b9bf82c1c7aba8`.
Candidate: the source-bound working-tree change identified in `provenance.json`,
`source-files.sha256` and `source.patch.xz`. No new measured tuning profile is
shipped or promoted by this qualification.

## Gates and results

All 48 endpoint groups pass. Each binary contributes six complete observations
per group: two fresh-process populations, three fresh prepared-owner repetitions
each, alternating baseline/candidate then candidate/baseline. No sample is dropped.
The replay checks each declared process index (0/1) and owner repetition (0/1/2)
exactly once per group, and matches iteration/Fock work for each process/owner
pair. Process indices come from the `<variant>-<index>.jsonl` filenames. Equal
aggregate counts alone do not qualify a population.

Groups cover 2/18 spherical AOs, RHF/UHF, energy/energy-and-force, batches 1/2 and
cold/warm/+0.03-Bohr moved-warm calls.

All four timed-process stderr streams were empty. Their byte counts are retained
in `provenance.json` instead of publishing standalone transient `.err` files.

| Acceptance | Observed |
| --- | --- |
| Maximum median endpoint regression <= 5% | 1.416342% |
| Independent energy error <= 1e-8 Hartree | 4.618528e-14 |
| Independent force error <= 1e-7 Hartree/Bohr | 1.052275e-10 |
| Default routes and iteration/Fock work preserved | All 48 groups |
| Tracked numeric peak does not grow | All 48 groups |
| Queue and allocator-fallback memcheck | Zero errors in both |
| Focused host tests | 708 passed |

The numeric allocation ledger is **not total process GPU memory**. The 50-ms
NVML trace observes 8026 MiB baseline versus 8038 MiB candidate maximum process
memory in both process pairs: a retained 12-MiB non-ledger overhead. Its specific
driver/code/opaque-library cause is not isolated here. Do not report the unchanged
numeric peaks as an unchanged total-VRAM claim. See `process-memory-summary.json`
and the complete compressed trace.

The energy-only 32-AO s/p/d/f capacity fixture has direct/resident numeric peaks
606640/8651336 bytes. A 4628988-byte ledger limit rejects residency once, completes
the bounded direct CUDA route against independent PySCF, reuses that fallback and
releases all tracked storage. Both queue guards/optional-FP32 reset and this
allocation/cleanup path pass Compute Sanitizer memcheck.

## Scope and reproduction

Hardware: RTX 5090 on authorized `n5` / Slurm `node5`, driver 570.86.16. Job 2040
uses `main`, `gpu:5090:1`, an exclusive finite 20-minute allocation and unchanged
Slurm visibility `CUDA_VISIBLE_DEVICES=0`. CPU numerical libraries use one thread.
Both binaries use Release, GCC 11.4.0, CUDA 12.9.86, sm_120 and explicit
`GENERATIVEQC_CUDA_FAST_COMPILE=ON`. This is not a qualification of other targets,
the non-fast-compile build or a new schedule winner.

Timers surround `run_*_cuda_bucket_cached`, including host packing, target
admission, allocation, SCF and finalization. Fixture construction and process
launch are outside the timer. Every cold call has a new prepared owner; warm and
moved calls consume the immediately preceding physical density. Process-first
provider/context effects remain in the recorded populations. The unchanged
controls are FP64, screening 1e-12, energy/density tolerances 1e-12/1e-10 and at most
100 SCF iterations. These are complete CUDA bucket endpoints, not isolated GEMMs.

Build `generativeqc_direct_jk_policy_benchmark` against both source trees and run
both through one compatible finite Slurm allocation. `qualify-gpu.sh` retains the
exact historical worker, including interleaving, memory sampling and sanitizer
commands; adjust its historical root for a new run, never bypass Slurm.

From this directory, reproduce the independent oracle and comparison with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python generate-oracle.py > oracle-replayed.json
python compare.py --baseline 'baseline-[01].jsonl' --candidate 'candidate-[01].jsonl' \
  --reference oracle-replayed.json --output report-replayed.json
```

PySCF is used only by this independent qualification generator, never production
execution. Synthetic profile-ordering inputs in the native test establish real
consumer activation, cache invalidation and science gates, not measured rates or
a profitable production profile. Benchmark evidence does not grant permission to
replace the conservative defaults without separate complete-endpoint tuning.

Compiler-cache statistics are shared-cache observations. Final CXX/CUDA builds
invoke verified ccache 4.5.1 with the existing cache and data-disk build/temporary
storage; they are not uncached builds or exclusive task-attributed hit counts.
