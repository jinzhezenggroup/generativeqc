# Complete cold-start DFT96 qualification

Follow-up: [1024-point CUDA defaults](../default-cuda-xc-1024-20261103/README.md)
now apply the measured tile to ordinary semilocal-XC calls without a profile.
This campaign and its source-matched archives remain unchanged historical
evidence; the application paragraph below describes the campaign itself.

## Result

On the exact 96-atom water cluster in
`tools/dft_mp_v1/inputs/water32.json`, PBE0-RKS/def2-SVP spherical, Treutler
level 3, FP64, XC tile **1024** versus the native default **256**:

| Timed boundary | Baseline median | Candidate median | Speedup | One-sided lower 95% |
| --- | --- | --- | --- | --- |
| Complete fresh worker process | 113.122897 s | 105.615331 s | 1.071084x | 1.067975x |
| Calculator construction + prepare + first E/F + prepared close | 111.816374 s | 104.245246 s | 1.072628x | 1.069830x |

The complete-process result is **6.64% less elapsed time**; the inner native
application boundary is **6.77% less elapsed time**. Do not call a 1.071x
speedup a 7.1% reduction in elapsed time. No density, orbitals, prepared owner
or converged state is shared across the ten samples.

Slurm job **1467** uses n4 / RTX 5090 / SM120, five serial alternating-order
pairs, finite 40-minute GPU allocation, and unchanged Slurm device visibility.
The worker-process timer includes Python startup/imports, source/binary
validation, native context setup and complete process teardown. The inner
timer starts before Calculator construction and ends after prepared close;
post-execution native identity probing and remaining process shutdown belong
only to the complete-process timer. Neither boundary includes the independent
CPU oracle solve or a tuning campaign.

## Scientific and work gates

Every cold sample converges in **17 iterations**, with **17 actual Fock builds,
17 Coulomb and 17 exchange applications**, and **1,078,336 grid points**.
Energy tolerance `1e-12`, density tolerance `1e-10`, screening `1e-14`, DIIS
history 8, maximum iterations 100, geometry, functional and J/K policy stay
fixed. Both native resource owners, independent E/F, schedule parity, source
identity and equal semantic-work gates pass. The maximum baseline/candidate
energy difference is `1.3642420526593924e-12` Hartree and force difference is
`1.8118839761882555e-13` Hartree/Bohr.

The independent reference and its strict recomputed full-Fock orbital-gradient
gate are the same as in `../dft96-local-autotune-20261103/README.md`. This is a
new cold campaign, not a reinterpretation of the earlier resident workers'
single initial-execute measurements, whose timers excluded preparation.

Owned numeric device peak is **623,529,284 -> 861,759,980 bytes**. This is a
latency tradeoff, not a memory reduction or a total driver/context/RSS claim.

## Identity, retention and application

Canonical checkout and native source identity both are
`413127726534f605780873d80f46ed23e717969d07423c3e067a6293cbd32496`.
Native library SHA-256 is
`54ae88e0820c08d869fd47ed7ccff744a6bd9e67e8982d62662769d77cd58328`.
The fully executed cold controller, its launch script and the unchanged
resident-controller comparison/gate implementation are retained in the archive
with individual hashes. All ten complete E/F vectors, iterations, work/resource
records, timings and worker logs are retained; no native binary is published.

`receipts.json.xz`: **29,308 bytes**, SHA-256
`0b859ca50329e19a84a6600b372545f9b16a29062c9487161daf7382e814430c`.
Its `report.evidence` checks the inner boundary; `process_endpoint` revalidates
the same paired observations using their complete-process clocks. These two
views are not twenty independent samples and must not be pooled.

This benchmark **does not automatically install or activate the 96-atom
parameters**. Ordinary Calculator defaults and the profile cache are unchanged.
For this exact workload, explicitly pass
`KsOptions(grid=GridSpec.treutler(3), tile_points=1024)`. Existing qualified local
XC profiles can be applied by the normal exact-workload profile-selection path,
but a benchmark receipt by itself is not an installed profile. Explicit caller
tile choices always take precedence. No cold speedup is extrapolated to another
molecule, basis, grid, GPU, source build, UHF/UKS or density-fitted Hamiltonian.
