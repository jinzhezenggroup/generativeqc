# Native CUDA basis-capacity consumer qualification

The candidate is `9b29bd65dc7a6515b1e17d9d872ab08c60883095`; the baseline is
`228a2a68c08d8b27ffec1798e9c3addfb200153f`. Native source, headers and CMake inputs
are byte-identical. Both frontend arms use the same frozen native library,
SHA-256 `72efe46682fed58ecbb9f776a7a1dcb13977344f510afdaa5383cb368419dd69`.
The candidate poisons portable basis-shape arithmetic; native counts must serve
actual budgeted HF/KS preparation and replay. This is a frontend ownership
cutover, not a new native scientific capability.

Slurm job **14851**, node1/main, one assigned RTX 5090, finite one-hour limit,
passed 103 real-backend checks and 1,792 complete timed calls in eight alternating
ABBA/BAAB groups. RHF/UHF and PBE RKS/UKS cover scalar and ragged H2/water fleets,
energy/forces, cold/warm/repeated and displaced-coordinate replay, including
construction and teardown. Both arms use identical 2 GiB host/device budgets,
STO-3G spherical basis, default SCF controls and the 24/12/24 KS grid.

Maximum median time ratio: **1.0175495775083907**, below the unchanged 1.05 gate.
Raw populations and every per-case ratio are in `cuda-samples.json.xz`.
Scientific/work tolerances and exact semantic comparisons are in `qualify.py`.
Maximum E/F differences are 5.69e-14 Eh / 8.46e-15 Eh/Bohr.
Scalar HF does not publish a Fock-build count: it remains explicitly unavailable;
batch counts, KS diagnostics and actual GPU activities are retained.

The complete untimed Nsight comparison is identical: 71,783 kernels, 100 graph
calls, 1,926 explicit dynamic allocations, 27,673,726-byte live peak and zero
final live bytes. All copy counts/bytes match. Static, driver and opaque vendor
storage is excluded from that peak. Raw profiler databases, XML and bulk logs
remain ignored; their identities and source/binary provenance are retained here.

The initial job **14850** passed 103 tests but failed before timed collection
because its recorder assumed scalar `Result.fock_builds`. Its log/input hashes
are preserved in `identities.json`; the corrected campaign uses a new immutable
source root. Slurm accounting storage is disabled and completed controller job
records expired before collection; no accounting receipt is invented.

A diagnostic D4 profiler occupied a separately allocated GPU on the same node
during part of this campaign; these are alternating matched samples, not an
exclusive-node measurement.

Run the retained `run-gpu.sh` only through a finite `srun`, adapting immutable
source/artifact paths. The job preserves scheduler-assigned device visibility.
Host-only dimensions were independently checked against PySCF (177 checks,
one explicitly GPU-only skip). The small evidence addition is offset by
losslessly encoding the historical issue471 autotuning JSON as XZ, with its
original Git blob and decoded/encoded checksums retained separately.
