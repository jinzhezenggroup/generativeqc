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

The C5 baseline in `records.json` preserves one H100 endpoint and two
independent Psi4 endpoints from #3174. The replay reports a median only
for configurations with at least two complete samples. Resource records
separate the run's configured memory admission budget, Linux worker RSS
high-water mark, any exposed cgroup peak, and sampled GPU device memory.

## Acquisition status

| Molecule | H100 endpoint (s) | CPU reference | CPU endpoint (s) | Worst energy error (Eh) | Largest force error (Eh/Bohr) | Worst force RMS (Eh/Bohr) |
|---|---:|---|---:|---:|---:|---:|
| C5 | 3,830.93 median (2) | Psi4 | 2,495.80 median (2) | 3.0241e-11 | 1.7405e-06 | 5.8795e-07 |
| C5 | 3,830.93 median (2) | PySCF | 6,760.91 (one) | 4.6327e-12 | 1.1114e-08 | 5.0135e-09 |
| C6 | 11,946.41 (one) | Psi4 | 8,607.15 (one) | 3.2401e-12 | 1.587e-06 | 3.4604e-07 |
| C7 | — | — | — | — | — | — |

A single sample is shown with its own wall time. The replay uses every
available H100-to-reference pair and reports incomplete cases explicitly.

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
allowance. The job remains queued after scheduler reports of the personal
800-CPU ceiling. No Psi4 worker was launched on the shared notebook.
[DPD allocator](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/libdpd/block_matrix.cc#L71-L106).

PySCF 2.11.0 sizes triples Lambda and gradient-density blocks from its
advisory `max_memory`, and its gradient writes the two-particle density
intermediates through HDF5 scratch. For C7, decreasing `max_memory` from
172,032 to 70,000 MB would reduce the nominal virtual block edge from 50
to 37 before deducting other live memory. The two corresponding triples
arrays would drop from 45.43 to 18.41 GiB, plus other allocations. The
current 200 GiB job has already sampled above 118 GiB of worker RSS. The
source allows a smaller block configuration, but a complete endpoint under
a hard 100 GiB ceiling remains unverified. The original job continues and
no duplicate PySCF worker was launched.
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
`records.json`. `provenance.json` records the remote output path for each
job and the hashes of any completed archives under `raw/`. The local archives
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
The first C6 H100 run completed with a 155 GiB application allowance
inside a 200 GiB host allocation. Its reported plan was 145.68 GiB.
Worker RSS peaked at 16.04 GiB and sampled device memory at 74.25 GiB.
The first C6 run's raw output was archived while its second fresh
process started in the same allocation. The C7 CPU reference remains
in progress in this local snapshot.

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
