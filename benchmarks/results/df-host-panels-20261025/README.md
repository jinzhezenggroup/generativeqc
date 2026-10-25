# Bounded DF host-panel qualification

Measured source `0fab1aa77`; CUDA library SHA-256
`a4c6ad3298791c805bdd69cc48a38f21fd98f2f6784c83d956172ec4cfe35574`.
NVIDIA RTX PRO 6000 Blackwell, n2, finite Slurm jobs 3575 (tests/timing) and
3576 (separate traced work/retained-capacity receipts). CUDA 12.9, FP64,
ccache 4.5.1 for native compilation and sccache 0.16.0 for runtime JIT.
The standalone probe and Python loader resolved the same final library.

`records.json` retains every sample, independent error gates, SCF/Fock counts,
trace counters, graph/eager execution labels and allocation diagnostics.
Raw arrays/logs remain in ignored `.artifacts/df-host/` and the node staging
bundle. Three alternating repeats follow unreported initialization in every
policy. Cold includes calculator/prepared-owner setup plus the complete first
E+F call. Warm reuses the owner; moved changes one H coordinate by 0.025 Bohr.
Tracing is disabled for timings and enabled only for separate work receipts.

## Complete E+F medians

Seconds; spherical def2-SVP orbital and auxiliary basis, native DF-RHF.
The budget is the public complete DF value/response allowance, not GPU size.

| Workload / policy | Host disabled: cold / warm / moved | Automatic RAM: cold / warm / moved |
|---|---:|---:|
| 96 AO tetramer, 20 MiB, forced dense streamed | 1.3014 / 0.4331 / 1.2315 | 0.8596 / 0.3811 / 0.8175 |
| 96 AO tetramer, 20 MiB, natural full GPU | 0.1901 / 0.1092 / 0.1784 | 0.1910 / 0.1089 / 0.1777 |
| 192 AO octamer, 64 MiB, forced dense streamed | 8.0248 / 1.9317 / 7.2458 | 5.6336 / 1.6609 / 5.1232 |
| 192 AO octamer, 64 MiB, natural streamed | 8.0221 / 1.9311 / 7.2461 | 5.6391 / 1.6605 / 5.1289 |

Natural 192-AO cold improves **29.7%**, warm 14.0%, moved 29.2%. Cold ranges
are 8.0209–8.0262 s disabled and 5.6342–5.6391 s automatic. The 96-AO natural
full-GPU path remains faster than the dense fallback and is never displaced.
These two workloads establish bounded local qualification, not universal gains.

All arms retain identical SCF/Fock counts: tetramer cold 16/17, warm 2/3,
moved 15/16; octamer cold 18/19, warm 2/3, moved 16/17. Independent PySCF
energy/force gates are 1e-9 Ha and 1e-8 Ha/Bohr. Maximum observed errors are
5.23e-12 Ha and 4.09e-12 Ha/Bohr. References are computed outside native clocks.

## Capacity and data movement

The matched diagnostic host-resident difference is the cache allocation:
7,077,888 bytes (all 9,216 pair rows) for the dense tetramer; 45,596,160 bytes
(29,685 of 36,864 pair rows, **80.53%**) for the octamer. Octamer total host
peak is 52,104,680 bytes and device peak remains 51,765,809 bytes in both arms.
There are no new pinned/device allocations. The final natural octamer cold
trace records 532 host hits, 1,673,206,272 H2D bytes, and 26,188,992 generated
uncached raw values. Dense tetramer records 102 hits and 120,324,096 H2D bytes.

The disabled source trace includes graph construction (four capture records);
its raw counters do **not** count later graph replays. Do not compare its sum
with the eager RAM sum as total executed work. Complete SCF/Fock counts above
are the semantic endpoint counts; the RAM movement counters are executed eager
calls. Derivative integrals and metric response remain independent of this cache.

## Validation and reproduction

38 GPU tests pass: 14 host-panel cases (full/partial capacity, GPU priority,
malformed policy and open-shell cold/warm E+F) plus 24 existing streamed endpoint
cases. Five compute-sanitizer memcheck cases pass with zero errors, including
partial boundary crossings and UHF (`--target-processes all` includes the native
probe). Repository pre-commit checks pass.

Within a finite Slurm allocation, using the qualified source/library and a
verified compiler cache:

```bash
python -m benchmarks.df_host_panels --budget-mib 20 --repeats 3 \
  --source-sha 0fab1aa77 --output .artifacts/tetramer20.json
python -m benchmarks.df_host_panels --case water-octamer-s4-def2-svp-spherical \
  --budget-mib 64 --repeats 3 --source-sha 0fab1aa77 \
  --output .artifacts/octamer64.json
```

`--trace-only` retains the capacity/work receipts without timing repeats.
The evidence-only driver extension does not change native code or binary.
