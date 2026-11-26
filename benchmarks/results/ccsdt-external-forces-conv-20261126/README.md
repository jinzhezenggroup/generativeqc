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

| Molecule | H100 complete endpoint (s) | Psi4 complete endpoints (s) | Maximum energy error (Eh) | Maximum force error (Eh/Bohr) | Worst force RMS (Eh/Bohr) |
|---|---:|---:|---:|---:|---:|
| C5 | 3,633.460 (one baseline sample) | 2,427.486 and 2,564.104 | 3.0241e-11 | 1.7405e-6 | 5.8795e-7 |
| C6 | Pending | Pending | — | — | — |
| C7 | Pending | Pending | — | — | — |

The retained C5 errors compare the single H100 baseline against both Psi4
samples. The second H100 acquisition and the larger-molecule acquisitions are
pending qz session renewal. They have no fresh timing, error or memory-peak
measurements yet.

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
