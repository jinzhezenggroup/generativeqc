# Decision: reserve optional XC panels before prepared admission cutover

Status: rejected (the unreserved cutover)
Date: 2026-10-26

This is a diagnosis of closed, unmerged PR #2586, not acceptance of a new
implementation. Historical candidate d474d0b265e59e0ed36c819839f9ad2a5293b75f
and baseline a97112c4e703f90d21ba48f032d49d2d2e9c19fe were reused without
changing their binaries or installed headers. SHA-256 identities were checked
against their original compile receipts before and after the experiment.

The retained public SDK probe includes context/system/composition construction,
preparation, cold/warm execution, a moved-geometry re-preparation, and teardown.
The diagnosis used PBE RKS H2/STO-3G, Direct E/F and DF energy, with the original
2 GiB numeric cap for the candidate. Each of five configurations was run for
21 samples after three warmups, then in reverse configuration order. These
420 complete endpoints retain iteration counts and energies/forces within
1e-12 of the historical default baseline. Runs used n5 Slurm main/gpu:5090:1
with a finite 15-minute allocation; assigned visibility was preserved.

| Configuration | Direct ratio (forward / reverse) | DF ratio (forward / reverse) |
| --- | --- | --- |
| Baseline, default | 1 / 1 | 1 / 1 |
| Baseline, point batching disabled | 1.23404 / 1.23329 | 1.38009 / 1.38342 |
| Baseline, AO maps disabled | 1.00615 / 1.00397 | 1.00528 / 1.00684 |
| Baseline, both disabled | 1.26097 / 1.25776 | 1.42307 / 1.42396 |
| Historical native-admission candidate | 1.26113 / 1.25920 | 1.42406 / 1.42447 |

Separate Nsight Systems 2025.1.3 runs of the Direct endpoint show:

| Count | Default baseline | Both disabled | Candidate |
| --- | ---: | ---: | ---: |
| CUDA kernel launches | 6149 | 7293 | 7293 |
| XC point-evaluation calls | 25 | 720 | 720 |
| XC total-accumulation calls | 25 | 720 | 720 |

The source gates in `CudaKsPlan` disable optional AO maps and point panels
whenever a public device ledger is bound. This is deliberate: selected resource
inventories reserve the dense one-tile owner and later fleet/force lifetimes,
not optional panels. The measurements isolate that schedule change as the
principal regression for these two endpoint cases; they do not establish that
all 100 historical case/order comparisons share a single cause.

A recovery must explicitly size and reserve optional XC panels across retained
fleet and force lifetimes, retain the one-tile fallback, and requalify the full
original matrix. Removing the active-ledger condition or spending currently free
ledger capacity would violate future-owner reservations. CPU observer overhead
or serialization is not supported as the primary cause by these measurements.
The 1.05 endpoint gate and original model/property coverage remain unchanged.

A recovery on newer source must also inventory optional owners introduced after
these measured revisions, including the dependent spatial-grid execution copy.
The historical ablations do not qualify that newer storage or its scheduling.

Raw clean measurements, runner, product receipts, profiler summaries and the
three separate traces are retained under
`n5:/home/jzzeng/generativeqc-qualification/prepared-device-diagnosis-20261026/`.
The original failed campaign remains unchanged under
`prepared-device-d474d0b265e59e0ed36c819839f9ad2a5293b75f-v1/ks-cli-endpoints-v5/`.

## Retained evidence and references

The [compact diagnostic publication](../../../benchmarks/results/prepared-admission-xc-diagnosis-20261026/README.md) retains all 420 clean endpoint samples, the exact runner and SDK probe, binary identities, scheduler receipt and separate Nsight summaries. It records rejection of the historical candidate, not acceptance of a replacement. Full optional-panel lifetime/admission work remains open under [#2227](https://github.com/jinzhezenggroup/generativeqc/issues/2227); [#2586](https://github.com/jinzhezenggroup/generativeqc/pull/2586) remains closed and unmerged.
