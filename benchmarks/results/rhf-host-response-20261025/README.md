# Exact RHF frame RAM tier qualification

Source: `0a989656c`, stacked on full-range provider commit
`3cb888fa3f2991a1c663f1144c647801883f0c7d` (PR #2762).
[records.json.xz](records.json.xz) retains all complete endpoint outputs, source/binary/
input identities, controls and observable differences. RTX 5090 on node4,
finite Slurm jobs 1344/1345; one fresh process per endpoint. These are individual
samples, not a statistical hardware-wide crossover model.

| Complete 230-AO DF-CCSD(T) E+F configuration | seconds | response seconds |
| --- | ---: | ---: |
| Response Direct, common GPU ceiling 256 MiB | 323.397 | 106.565 |
| Incumbent unmasked GPU capacity refusal, RAM disabled | 323.277 | 106.531 |
| Automatic GPU fit + RAM, common GPU ceiling 256 MiB | 291.670 | 74.985 |
| Explicit GPU partial mask, RAM disabled | 409.851 | 100.493 |
| Same explicit GPU partial mask + RAM | 384.632 | 74.930 |
| Full GPU residency, no ceiling override | 276.979 | 70.535 |

Automatic RAM is 9.8% faster than Direct at the constrained resource setting.
The explicit matched-mask pair improves 6.2%. That mask also affects the separate
RHF-SCF owner: reference time rises from 27.5 to 120.4 seconds in **both** masked
arms. Do not call the masked RAM sample a repeat of the unmasked automatic
configuration, or compare their complete times as a host-cache regression.
Full GPU residency remains the preferred path and uses no host staging.

All arms execute 17 response J/K actions and 12 Z iterations. RAM retains
4,340,811,640 bytes in addition to 264,303,680 GPU bytes; 16 replays upload
69,452,986,240 bytes. Host values plus pinned/device staging occupy
5,414,553,808 bytes. Unmasked host setup costs 15.391 seconds, charged to the
complete endpoint. Action recurrence evaluations fall from 9,785,870,055
(Direct) or 9,257,262,695 (partial) to 575,639,415: the final independent audit
still recomputes. Preparation separately evaluates the cached source once.
Nested phase times must not be added to complete endpoint times.

Maximum energy/force differences across these controls are 8.53e-14 Hartree /
2.91e-10 Hartree/Bohr; the independent Z residual is below 1.4e-13. Independent
small-system validation, rather than agreement among native variants alone,
qualifies the scientific result: 164 response tests, two executable host/GPU
failure-injection suites, two complete CCSD/CCSD(T) independent energy and
two-step force finite-difference cases, and six memcheck cases (zero errors).
The response suite includes a 70-AO automatic-capacity case, full GPU priority,
Cartesian/spherical d/f bases, moved geometry, and refusal/error publication.

## Reproduction

Build `libgenerativeqc` for sm_120 with the verified ccache/sccache launcher,
then compile `benchmarks/df_ccsdt_force_endpoint.cpp` against that library.
Use `benchmarks/results/rhf-phase-values-auto-20261010/ethane230.input` and
endpoint arguments after INPUT/OUTPUT:

```text
1 1 1 1 8 8 8 0 1 2 1 30 0 0 1 auto
```

Use final `0` for response Direct. Set
`GENERATIVEQC_RHF_RESIDENT_MAX_BYTES=268435456` for constrained cases,
`GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES=0` to disable RAM, and
`GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK=258998275` for **both** explicit-mask
arms. Leave the host ceiling unset for automatic RAM; leave all GPU controls
unset for full residency. J/K profiling is identically enabled in every arm.
Execute through `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
--time=00:15:00` with the same CUDA visibility and runtime libraries.

Compile `tests/native/rhf_frame_response_probe.cpp` into a shared probe and run
`test_rhf_host_response_cuda.py` plus `test_rhf_frame_response_cuda.py` under
finite Slurm with `GENERATIVEQC_RHF_FRAME_CUDA_TEST=1` and
`GENERATIVEQC_RHF_FRAME_PROBE`. The complete-force probe uses
`GENERATIVEQC_TEST_RHF_REQUIRE_HOST=1`, GPU mask 1, positive 64 MiB response
allowance, and 512-byte host tiles on the two water independent-direction cases.
Full raw logs, compiler-cache receipts and binary artifacts remain under ignored
`.artifacts/rhf-host/` and the isolated node2/node4
`/data/jzzeng/rhf-host-response-20261025/` bundles.
