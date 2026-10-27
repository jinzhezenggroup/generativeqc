# Default cold Direct HF96: screened FP64 proposals and bounded pages

Three alternating native/GPU4PySCF rounds use fresh processes on one RTX 5090
under Slurm job 14429. The unchanged #2786 endpoint includes construction,
preparation, first energy and analytic forces, and host export, followed by
three calls from each engine's frozen cold density. Ordinary convergence
settings are retained. The only native environment controls are the campaign's
existing spatial order `2` and SCF validation GEMM `1`; no research screening or
page-size override is enabled.

| Endpoint | Native cold median [range], s | GPU4PySCF cold median [range], s | Native / GPU4 warm median, s |
| --- | ---: | ---: | ---: |
| Direct HF96 | 11.3787 [11.2870, 11.3961] | 11.9044 [11.7138, 11.9298] | 2.7127 / 3.3566 |

The median cold advantage is 4.4%, and the measured ranges do not overlap.
All six populations and 24 complete calls are retained in `receipts.json.xz`.
Native cold performs eight screened FP64 and three target FP64 builds, with no
mixed-precision or omitted preliminary work; frozen warm performs one target
build. GPU4PySCF exports iterations but not complete Fock work. This comparison
does not establish a speedup for every molecule, GPU or convergence target.

## Accuracy and retained failures

All main cold/warm populations pass the unchanged common independent gates:
`1e-5 Eh/atom`, `1e-4 Eh/Bohr` maximum force error, and `3e-5 Eh/Bohr` RMS force
error. References and their reuse contracts come from
[`readme-final-20261022`](../readme-final-20261022/README.md).

Against a separate tighter GPU4PySCF reference, the default native cold energy
error is `1.73e-9 Eh` and maximum force error is `3.02e-7 Eh/Bohr`. The previous
production path has `5.10e-8 Eh/Bohr` maximum force error; the new trajectory
differs from it by `2.55e-7 Eh/Bohr`. Thus an additional `1e-7` force diagnostic
fails at default settings. This is **not a reproduced baseline failure**, and
the performance table does not claim the stronger diagnostic passed.

Explicitly tighter native requests (`energy=1e-12`, `density=1e-10`,
`screening=1e-14`) pass unchanged `1e-8 Eh` / `1e-7 Eh/Bohr` independent gates
for cold, frozen warm, displaced/restored geometry, and checkpoint import.
The independent GPU4 targets use energy `1e-10` and orbital residual `1e-8`;
an earlier reference failed convergence at energy `1e-12` and is retained.
These tighter native runs qualify controllable target accuracy and do not
replace the default benchmark receipts or loosen an observable gate.

The four opt-in CUDA tests also cover energy-only iteration limits and a real
4-GiB ledger rejection of the optional 6-GiB task page. The bounded fallback
passes independent observables, respects the ledger peak and releases all live
storage. Native precision-policy tests and two existing direct/DF complete-work
CUDA checks pass. Full logs, including both unsuccessful qualification attempts,
are retained in the compact summary. Another 329 host control/inventory checks
and 97 resource-contract checks pass. The initial local resource run had no
native library; the latter checks were rerun against the frozen native image.

## Mechanism and provenance

Large unseeded single-system restricted bounded quartet owners may use at most
eight `8e-8` screened FP64 proposal steps. A fresh target DIIS/energy epoch then
converges the original requested operator. The stages share the user iteration
allowance. A local proposal graph prevents leakage into warm/imported owners.
The larger reusable task page reduces repeated compaction; failed allocation
retries the original 8M-task ceiling before any graph capture or upload.
Admission and fallback contracts are in
[`hf_force_finalization.md`](../../../docs/developer/hf_force_finalization.md).

The timing/qualification library is frozen `hf-default-v1`, based on
`7205e9a32b3c94f84d94eef96f7b44ee0eef2035`, source archive SHA-256
`bbc6f864a3d6dbd4b1eaae4e63c8f88343299b469c9fe103cc48cbcde8c29f65`.
The summary records its patch identity, binary hashes, verified sccache 0.16.0
statistics, build allocation, compiler commands hash and exact benchmark
commands. CUDA 12.9.1/GCC 13.3, Release/sm120 and one BLAS/OpenMP thread were
used. Scheduler-assigned GPU visibility was preserved; caches were not cleared.
Publication adds documentation, the inventory, the independently run Python
qualification module, comments and formatting; frozen versus publication source
hashes are retained. The source is not claimed byte-identical to the final PR.

The first production timing campaign is retained separately: GPU4 cold times
were unusually slow near independent oracle initialization. It is not mixed
into the main table. The research 16M/32M page comparison is also retained as
supporting evidence, with its explicit controls and original work receipts;
the production image fixes the preliminary-work accounting.

Full archives, patches, logs and build outputs remain at
`n1:/data/jzzeng/cold-opt-20261025/`. Reproduce with `hf-default-paired-v2.py`
and a new output directory under a finite Slurm allocation; do not overwrite
historical receipts. No GPU is needed to audit the tracked original bytes and
rerun their independent qualification:

```bash
PYTHONPATH=python:. python benchmarks/results/cold-hf-defaults-20261026/recheck.py
```

The [decision note](../../../.agents/notes/implemented/performance/2026-10-26-cold-hf-screening-pages.md)
records scope, rejected alternatives and conditions for revisiting the policy.
