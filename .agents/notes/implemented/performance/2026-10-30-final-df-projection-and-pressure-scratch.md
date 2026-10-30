# Decision: consume final fitted projections before reusing their storage

Status: implemented
Date: 2026-10-30

## Problem

The native stationary controller did not pass the final KS occupied projection
`U=B*Cocc` into the prepared DF derivative provider. Rebuilding it repeatedly
was particularly expensive on devices with low FP64 throughput. Merely passing
U was insufficient: a later nonlocal force owner could reduce available memory
below the occupied-response workspace threshold on warm calls. The bounded
fallback remained memory-safe but repeated the fitted projection for many
response panels.

On the 96-atom A100 WB97M-V DF case the warm response had about 1.456 GB free.
The repeated-fit route performed 1,156 fitted-panel GEMMs, about 2.626e14
projection FLOPs and 290 GB of staging. The final U allocation was still live
and already charged to the value owner, although its contents became dead once
the response contracted `Cocc^T*U`.

## Decision

Authenticate the projection against the final density owner, solve epoch,
generation, device, stream and dimensions. The provider additionally checks
the actual pointer, rank, fitted representation and allocation extent. Exchange
consumes this one-shot lease before Coulomb can invalidate/reuse provider scratch.
An unsuccessful allocation retains the existing bounded ordinary response;
retry cannot relend a consumed or overwritten U.

For an authenticated fitted projection, prefer 512 auxiliary response columns;
unqualified/raw/UHF factors retain 64. The public budget, current device memory,
resource ledger and bounded allocation retries still limit the actual panel.
No device name selects the route.

When the preferred independent scratch does not fit, finish `Cocc^T*U` into
disjoint owned factors and recycle U's exclusive interval for subsequent
same-stream work. The factors must fit separately and the borrowed interval
must fit irreducible scratch. Borrowed storage stays charged to its original
owner. Future SCF work rebuilds U before exporting another lease. Independent
scratch remains preferred when ample memory makes it faster.

## Evidence

The complete portable-profile README matrices contain 64 cases per device,
cold plus three frozen-engine-density warm energy/force calls, on A100,
RTX 5090 and RTX PRO 6000. They retain one math thread, original scheduler IDs,
source/binary identities and all failed independent accuracy gates. The follow-up
evidence bundle is `benchmarks/results/ecnu-df-projection-20261030/`.

At 96 atoms, PBE0 DF warm medians changed from 10.98 to 7.17 seconds on A100,
28.29 to 7.45 seconds on PRO 6000, and 70.59 to 8.04 seconds on RTX 5090.
A100 WB97M-V DF changed from 62.23 to 34.42 seconds, including the nonlocal AO
mapping change recorded in the accompanying note. Its final warm trace shows
one occupied-projection BLAS call, 1.422e11 projection FLOPs, 740,556,800 bytes
of owned occupied-response storage and 2,415,919,104 bytes of recycled scratch.
Instrumented traces are attribution, not clean endpoint measurements.

The 24-MiB public-budget, 12-atom test demonstrates completed recycled-scratch
hits on cold, warm and changed geometry; 64 MiB keeps independent scratch.
Both agree with the no-projection path within 1e-9 Eh and 3e-8 Eh/Bohr.
All three device tests pass on A100 and RTX 5090 and under PRO 6000 CUDA memcheck
with zero errors. Host tests cover foreign/stale leases, insufficient extents,
J/K ownership, OOM after overwriting U, and planner capacity boundaries.

## Rejected alternatives and limits

- Always recycling U was slightly slower with ample memory; use it only under
  pressure. Increasing a panel cannot override the public resource allowance.
- Asking `prepare_force_memory` to reserve the entire DF response allowance
  evicted optional SCF caches and reduced A100 WB96 warm time to about 47.6 s.
  It was inferior to explicit dead-storage reuse and was removed.
- Projection reuse with the old 64-column panel improved the largest cases but
  regressed some 48-atom cases. The separately measured 512-column preference
  removes that crossover without extending the unqualified factor domain.
- Old Direct HF 48/96 timings omit force contributions and are invalid speed
  baselines. Their correctness repair is recorded separately.
- These are portable-profile measurements, not proof that auto-selected tuned
  SM120 builds are slower or faster. Nor do they establish universal GPU4PySCF
  superiority: A100 and several Direct cases still lose.
- HF DF fails the common dense-direct energy gate at all four sizes; independent
  GPU4PySCF DF exhibits the same fitting error. M06-2X at three atoms fails the
  default-grid RMS force gate in both modes. Neither gate is relaxed here.

Revisit the panel preference when another measured workload/device improves
complete endpoints with an equally bounded plan; preserve the fallback and
observable correctness instead of adding a GPU-name whitelist.
