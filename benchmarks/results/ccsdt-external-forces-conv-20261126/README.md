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
the recorded CPU and H100 configurations.

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
| C6 | — | — | — | — | — | — |
| C7 | — | — | — | — | — | — |

A single sample is shown with its own wall time. The replay uses every
available H100-to-reference pair and reports incomplete cases explicitly.

## Resource sizing

The earlier stock Psi4 run hit the DPD allocator's memory limit. In its C1
gradient, each virtual triple allocates an occupied-triple-sized `E` block.
Its missing release implies about 250 GiB of cumulative allocation at C6
and 601 GiB at C7 for the block alone. These source-based estimates exclude
other allocations and guide node selection. [Psi4 1.10.2 source](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/cc/cctriples/T3_grad_RHF.cc#L950-L955).

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
C6 acquisition and the C7 CPU reference remain in progress in this local
snapshot.

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
