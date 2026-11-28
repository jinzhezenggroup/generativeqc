# Conventional CCSD(T) C5–C7 forces

Follow-up to [the full-DF and first conventional results in #3174](https://github.com/jinzhezenggroup/generativeqc/tree/86016450ecc5af32f8e4528112badad7dec2d279/benchmarks/results/ccsdt-external-forces-20261125).
The measured GenerativeQC revision is `22f72d0692a760c2892cd7791022584d9ae1c3e5`.

## Protocol

The driver calls the hash-verified #3174 protocol for all-electron conventional
RHF-CCSD(T). It retains the same C1 symmetry, atom order and geometry, explicit
spherical cc-pVDZ basis, no frozen orbitals and engine-specific convergence
controls. The independent programs are Psi4 1.10.2 and PySCF 2.11.0. Forces are
the negatives of the analytic gradients. Endpoint time includes preparation
inside the scientific call and ends after the energy and full Cartesian force
return. Each sample runs in a fresh Python process. The timing table compares
the recorded CPU and H100 configurations. The reused C5 Psi4 trials
used eight threads on a Xeon CPU Max 9462 with a 92 GiB program memory
limit. The C5 PySCF reference used 32 threads on a Xeon Gold 6530.
The wrapper passes `--memory-mib` numerically to both CPU engines.
Psi4 interprets it in MiB. PySCF interprets its advisory `max_memory`
value in decimal MB, so its configured value of 172,032 MB is 160.22 GiB.

For each molecule, two H100 endpoints and a complete analytic CPU reference
are required. All available H100 versus CPU pairs are checked. Limits are
`1e-7 Eh` for total energy, `3e-6 Eh/Bohr` for the largest absolute force
component, and `1e-6 Eh/Bohr` for the Cartesian force RMS.

The C5 baseline in `records-C5.json` preserves one H100 endpoint and two
independent Psi4 endpoints from #3174. The replay reports a median only
for configurations with at least two complete samples. Resource records
separate the run's configured memory admission budget, Linux worker RSS
high-water mark, any exposed cgroup peak, and sampled GPU device memory.

## Acquisition status

| Molecule | H100 endpoint (s) | CPU reference | CPU endpoint (s) | Worst energy error (Eh) | Largest force error (Eh/Bohr) | Worst force RMS (Eh/Bohr) |
|---|---:|---|---:|---:|---:|---:|
| C5 | 3,830.93 median (2) | Psi4 | 2,495.80 median (2) | 3.0241e-11 | 1.7405e-06 | 5.8795e-07 |
| C5 | 3,830.93 median (2) | PySCF | 6,760.91 (one) | 4.6327e-12 | 1.1114e-08 | 5.0135e-09 |
| C6 | 11,642.08 median (2) | Psi4 | 8,607.15 (one) | 3.2401e-12 | 1.587e-06 | 3.4604e-07 |
| C6 | 11,642.08 median (2) | PySCF | 17,533.39 (one) | 1.2449e-11 | 2.1165e-08 | 7.6774e-09 |
| C7 | 24,256.72 median (2), H200 | Psi4 | 18,777.03 (one) | 3.7062e-11 | 3.6041e-06 | 8.6067e-07 |
| C7 | 24,256.72 median (2), H200 | PySCF | 44,309.32 (one) | 4.4338e-12 | 2.1027e-09 | 8.1318e-10 |

A single sample is shown with its own wall time. The replay uses every
available H100-to-reference pair and reports incomplete cases explicitly.
C5 and C6 pass the energy and force limits with both CPU programs.

Both C7 CPU programs returned the full 69 force components. The CPU-to-CPU
energy difference was `4.1496e-11 Eh`. The force RMS difference was
`8.6069e-7 Eh/Bohr`, and the largest component difference was
`3.6041e-6 Eh/Bohr` on the fourth atom (carbon), along `z`. This exceeds
the `3e-6 Eh/Bohr` component threshold used for the H100 comparisons.
The replay reports this pair under `cpu_reference_pairs`.

The pinned C7 path needs a 106.46 GiB single-card allocation, so its two
GenerativeQC endpoints ran on H200. They agree with PySCF to `4.4e-12 Eh`
and `2.1e-9 Eh/Bohr`. A Richardson-extrapolated central difference of the
fourth carbon's `z` force agrees with GenerativeQC to `1.6e-10 Eh/Bohr` and
with PySCF to `1.4e-10 Eh/Bohr`. The Psi4 component differs from it by
`3.6e-6 Eh/Bohr`, so the only C7 limit excess belongs to the Psi4 reference
vector. See the [C7 addendum](C7-addendum.md).

## Resource sizing

### C7 CPU placement

The stock Psi4 1.10.2 C1 triples-gradient loop allocates an `E` DPD
heap block of `nocc^3` doubles for each of the `nvir^3` ordered virtual
triples. The closing frees omit that block and the next iteration overwrites
its pointer. The accumulated double arrays alone account for 249.91 GiB
at C6 (`nocc=25, nvir=129`) and 601.09 GiB at C7
(`nocc=29, nvir=149`). The measured C6 process peaked at 266.98 GiB.
[Psi4 allocation loop](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/cc/cctriples/T3_grad_RHF.cc#L812-L847),
[Psi4 loop cleanup](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/cc/cctriples/T3_grad_RHF.cc#L1320-L1345).

The DPD allocator evicts file4 cache entries against the configured
`memory` and uses the heap for these blocks. A smaller `memory`, a larger
scratch volume and fewer threads still leave the same accumulated DPD
blocks. The notebook's 250 GiB total working limit and a 20-CPU HPC
allocation with at most 100 GiB are too small for this stock C7 path.
The 1,200 GiB HPC request uses 32 solver threads and a 1,024 GiB Psi4
allowance. The original job queued behind the personal 800-CPU ceiling
and then waited for a node with enough free CPU and memory. It started at
08:09 UTC after about 6 h 40 min in queue. The Xeon Gold 6530 worker
finished the gradient and peaked at 634.56 GiB RSS. The shared notebook
was not used for this calculation.
[DPD allocator](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/libdpd/block_matrix.cc#L71-L106).

PySCF 2.11.0 sizes triples Lambda and gradient-density blocks from its
advisory `max_memory`, and its gradient writes the two-particle density
intermediates through HDF5 scratch. For C7, decreasing `max_memory` from
172,032 to 70,000 MB would reduce the nominal virtual block edge from 50
to 37 before deducting other live memory. The two corresponding triples
arrays would drop from 45.43 to 18.41 GiB, plus other allocations. The
original 200 GiB job finished with 32 solver threads on Xeon Gold 6530
and 118.45 GiB peak RSS. The source allows smaller blocks. A 100 GiB
configuration was not tested while that job was running.
[PySCF Lambda sizing](https://github.com/pyscf/pyscf/blob/v2.11.0/pyscf/cc/ccsd_t_lambda.py#L44-L62),
[PySCF density sizing](https://github.com/pyscf/pyscf/blob/v2.11.0/pyscf/cc/ccsd_t_rdm.py#L204-L220),
[PySCF gradient scratch](https://github.com/pyscf/pyscf/blob/v2.11.0/pyscf/grad/ccsd_t.py#L21-L36).

### GPU sizing

The C5 native complete-force plan reported a 61.46 GiB bound with a
72 GiB application allowance. Scaling the bound by the fourth power of
the orbital-basis dimension suggests planning allowances above roughly
121 GiB at C6 and 216 GiB at C7. The projection guides allocation choice
and the driver separately records the observed process and device peaks.

The pinned CUDA Hamiltonian response owner reserves one device allocation.
Replaying its aligned layout from the build's generated arena functions gives
30.30 GiB for C5, 59.66 GiB for C6 and 106.46 GiB for C7. Each tested H100
reports 81,559 MiB of total device memory, or 79.65 GiB. The C7 allocation
exceeds the whole device capacity before accounting for any other GPU
allocation. This layout uses one device even if the job reserves more GPUs.
See `device-capacity.json` and the local layout calculation in `raw/`.

Both 155 GiB C7 one-card runs failed with an out-of-memory error after
5,271.05 s and 5,269.80 s. Worker RSS reached 19.74 and 19.78 GiB,
respectively, with zero host cgroup OOM events. The two one-second device
traces covered 5,295 and 5,265 samples and each reached 53.72 GiB. The
106.46 GiB layout calculation describes the requested allocation. The
sampled peak describes device memory already in use. All three C7 GPU
attempts have been archived and the qz jobs were deleted.

## Local raw captures

The complete endpoint rows and their SHA-256 values are retained in
`records-C5.json`, `records-C6.json` and `records-C7.json`.
`provenance.json` records each remote output path and the hashes of
the archives under `raw/`. The local archives
contain the original worker, engine and device telemetry files. They stay
untracked because they include verbose logs and time series. The original
C5 H100 and two Psi4 record bytes were also copied from their #3174 remote
output and matched against all three baseline SHA-256 values.

The second C5 H100 run reached 8.76 GiB worker RSS and 44.74 GiB sampled
device memory at the matching 72 GiB application allowance. The first C7
H100 attempt failed during MO provider allocation at a 320 GiB allowance.
Its only GPU sample preceded that allocation. The subsequent 155 GiB C7
attempts record one-second device samples. The first C5 PySCF launcher
rejected its trial label before computation and was retried with C5.1.
Both C6 H100 runs used a 155 GiB application allowance inside a 200 GiB
host allocation. The first reported a 145.68 GiB planning bound. Both runs peaked at
16.04 GiB worker RSS and 74.25 GiB sampled device memory to two decimal
places. They used two fresh processes in the same job. All five
follow-up GPU jobs were deleted after archiving their output.

## Resource accounting

The allocations used qz CQ-NexChem. GPU receipts returned NORMAL/20
for requested priority 4. CPU Slurm receipts returned HIGH/35 for
requested priority 10. Card-hours use scheduling through terminal
status. Allocated CPU core-hours use the compute worker's scheduled
and release events and exclude Slurm control-plane resources.
The quota rates were one point per H100 card-hour and zero points
for CPU HPC. Points are estimated from these rates and wall times.
The first C5 PySCF launcher used 0.789 core-hours before its rejected
trial label was corrected. The scheduler reported its Slurm step as
succeeded. The scientific record contains no endpoint for that step.

| qz task | Scheduler status | Card-hours | CPU core-hours | Estimated points |
|---|---|---:|---:|---:|
| `gqc-3167-conv-h100-c5-1126-a` | SUCCEEDED | 1.146 | — | 1.146 |
| `gqc-3167-conv-h100-c6-1126-a` | SUCCEEDED | 6.498 | — | 6.498 |
| `gqc-3167-conv-h100-c7-1126-a` | FAILED | 0.072 | — | 0.072 |
| `gqc-3167-conv-h100-c7-retry-1126-r2` | FAILED | 1.497 | — | 1.497 |
| `gqc-3167-conv-h100-c7-repeat-1126-a` | FAILED | 1.468 | — | 1.468 |
| `gqc-3167-conv-psi4-c6-1126-a` | SUCCEEDED | — | 132.978 | 0.000 |
| `gqc-3167-conv-psi4-c7-1126-a` | SUCCEEDED | — | 524.250 | 0.000 |
| `gqc-3167-conv-pyscf-c5-1126-a` | SUCCEEDED | — | 0.789 | 0.000 |
| `gqc-3167-conv-pyscf-c5-1126-r2` | SUCCEEDED | — | 76.133 | 0.000 |
| `gqc-3167-conv-pyscf-c6-1126-a` | SUCCEEDED | — | 195.356 | 0.000 |
| `gqc-3167-conv-pyscf-c7-1126-a` | SUCCEEDED | — | 493.044 | 0.000 |
| Follow-up total | | 10.681 | 1422.550 | 10.681 |

#3174 separately recorded 3.140 card-hours and 114.569 CPU
core-hours. Those past costs are outside the follow-up total. They
include 0.207 card-hours for
`gqc-3167-ccsdt-h100-1125` at the superseded `d154546d2`.
The prior job was cancelled and deleted. Its results are excluded
from the follow-up comparisons.

## Reproduction

Place the #3174 `run.py`, `fixtures.json` and orbital basis file in the
protocol directory. Use the environment locks from that result directory.
The pinned native build is checked against the measured library SHA-256.
Use `qz-run.sh` below an allocated Slurm worker or in an H100 job. Set
`GQC_PROTOCOL_DIR`, `GQC_PYTHON`, and the requested arguments. For the
native arm also supply `GQC_NATIVE_SOURCE`, `GQC_NATIVE_ARCHIVE`,
`GQC_NATIVE_BUILD`, `GQC_CUDA_ROOT` and `GQC_BLAS_LIB`. Use separate
output paths for independently launched samples.

```bash
python3 benchmarks/results/ccsdt-external-forces-conv-20261126/replay.py
```

## C7 H200 and directional finite-difference addendum

The H200 runs use the same pinned SM90 production library, spherical cc-pVDZ
basis, FP64 method, reference convergence settings and 155 GiB host
correlation allowance as the one-card C7 H100 trials. Each run owns an H200
and its own fresh Python process. The wrapper records the installed driver,
compute capability and SM90 ELF image count, verifies the native build identity,
and samples single-device memory at 1 Hz.

For the independent directional check, Psi4 conventional RHF-CCSD(T) energy
jobs displace only the fourth atom's z coordinate in the fixed C1 geometry.
The basis and electron count match the analytic measurements. The three
central-difference steps are 0.004, 0.002 and 0.001 Bohr. The main energies
use SCF and CCSD energy-change thresholds of 1e-13 Eh and a SCF density
and CCSD amplitude RMS threshold of 1e-11. Both signs at 0.001 Bohr also
run at 5e-14 Eh and 1e-12 for the convergence check. Every job starts from
a fresh geometry and solves the energy without a gradient. The replay uses
the force convention `(E(-h) - E(+h)) / (2*h)` and two Richardson levels.

See [C7 measurement results](C7-addendum.md) for the H200 endpoints, the
finite-difference table and the allocation ledger.
