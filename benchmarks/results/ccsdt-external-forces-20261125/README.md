# Independent RHF-CCSD(T) Cartesian forces

This campaign addresses [#3167](https://github.com/jinzhezenggroup/generativeqc/issues/3167).
It compares all-electron, restricted closed-shell CCSD(T) total energies and
complete Cartesian forces at the cc-pVDZ correction basis. It uses the exact
QM7 C1–C7 geometries and three neutral polar controls (water, ammonia and
formaldehyde) from the
[previous CBS campaign](../cbs-h100-5090-7109b3d2-20261106).
The benchmark measures single-basis CCSD(T) endpoints. The CBS extrapolation
and MP2 components are outside this experiment.

## Acquisition status

The initial snapshot retains the standalone CPU reference and finite-difference
energy records. Three polar control groups pass the two-trial conventional
Psi4/PySCF comparison. The `22f72d069` H100 build is still active at this
snapshot, and the primary native comparisons stay incomplete. The superseded
CUDA records are retained separately below.

## Scientific inputs and gates

- The conventional panel uses GenerativeQC CUDA CCSD(T), Psi4 1.10.2 `CCENERGY` with
  conventional PK RHF, and PySCF 2.11.0 conventional RHF/CCSD with the
  CCSD(T)-specific Lambda and analytic gradient. See the [PySCF 2.11.0 gradient source](https://github.com/pyscf/pyscf/blob/v2.11.0/pyscf/grad/ccsd_t.py).
- The full-DF panel uses GenerativeQC CUDA `df-ccsd(t)` with `full_cuda` and Psi4 1.10.2
  `QC_MODULE=OCC` (DFOCC) with `DISK_DF` RHF. The reference and correlation
  share the same cc-pVDZ-JKFIT auxiliary basis within each engine. See the [Psi4 1.10 DFOCC routing source](https://psicode.org/psi4manual/1.10.x/_modules/psi4/driver/procrouting/proc.html).
- The source coordinates are in Bohr, in the recorded atom order, without
  translation, rotation or symmetry reduction. Basis snapshots are byte
  matched to the previous campaign's member SHA-256 values. The three
  implementations consume the same orbital basis decimals. The two fitted
  implementations consume the same auxiliary basis decimals. All are
  spherical, neutral singlets with zero frozen orbitals. Forces are the
  negative gradients in the original Cartesian frame.
- SCF and coupled-cluster convergence controls are pinned in the measured
  drivers. PySCF explicitly checks the RHF, CCSD and CCSD(T) Lambda convergence
  flags. Psi4 exceptions and native failures are retained. Engine residual
  definitions and initial guesses differ. The check compares physical
  energies and force components directly.
- For each comparison and molecule, all four combinations of the two
  independent trials must meet `1e-7 Eh` in energy, `3e-6 Eh/Bohr`
  in maximum component error and `1e-6 Eh/Bohr` in component RMS error.
  These thresholds follow the earlier CBS campaign.
- C1 finite differences reconstruct the full 15-component force using each
  sign of `0.001` and `0.0005 Bohr` in every Cartesian component. The
  conventional energies come from PySCF and the full-DF energies from Psi4
  DFOCC. The conventional check is the primary independent tie-breaker. The
  full-DF check is supplementary. Each displaced endpoint reconverges from
  a fresh reference. The
  maximum step-to-step component difference must be at most
  `1e-6 Eh/Bohr`. At the finer step, the maximum analytic versus difference
  error must be at most `3e-6 Eh/Bohr`.

## Timing and provenance

The complete timed endpoint starts before molecule or calculator construction
and ends after total energy and the complete analytic force have returned to
the host. Fixed basis-table decoding, Python process startup, package imports,
postflight validation and serialization are excluded. Lazy backend startup
inside the endpoint is included. Every trial launches a fresh process and
solver owner without a density, amplitude, integral or gradient seed.
The two observations provide descriptive medians and ranges. CPU and GPU
wall times are cross-hardware reference points and include each program's
chosen implementation and hardware configuration.

The primary GenerativeQC source is master
`22f72d0692a760c2892cd7791022584d9ae1c3e5` (tree
`5b587f68163deeb5ae767591fa75512998040935`), the revision shared by
#3166–#3171. The H100 job builds SM90 from an exact Git archive with the
verified sccache launcher and checks the loaded binary's native source
identity. The independent CPU reference drivers import only Psi4 or PySCF;
their geometry and basis snapshots are shared with the current native run.

`superseded-native-records.json` preserves the 15 complete full-DF samples
from the initial `d154546d2` run. The initial job was cancelled when the
required revision changed. Main results and gate calculations use
`22f72d069` exclusively for GenerativeQC.

## Resource and software limits

The first 20-CPU, 100-GiB Slurm requests could not be placed and were
withdrawn before any solver calls. The standard CPU endpoints use eight
solver threads with a 24-GiB application budget inside an 8-CPU, 32-GiB
qz allocation. The two energy-difference batches use a single solver
thread and an 8-GiB application budget in separate four-CPU allocations.
The per-attempt CPU model and resource receipts are in `execution.json`.

At C4, Psi4's conventional path printed a CCSD(T) total energy before
raising `dpd_block_matrix: No memory left` while forming the triples-gradient
intermediates. The v1.10.2 RHF implementation allocates an `E[Gij]` DPD
buffer inside its virtual-triple loop. The closing frees for that loop omit
the buffer. The failed 24-GiB attempt and resource-capped larger attempts are
retained. The separate 100-GiB allocation request tests a larger application
budget on C4 and C5. This resource constraint affects the pinned Psi4
implementation, and a completed energy row alone yields no analytic-force
comparison. [Psi4 v1.10.2 RHF triples-gradient source](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/cc/cctriples/T3_grad_RHF.cc#L950-L955).

Initial software probes exposed a Psi4 ABI mismatch with Einsums 1.1.6, a
LibXC 7.1.2 alias mismatch, and a PySCF 2.11.0 incompatibility with NumPy
2.4.6's einsum planner. Probe outcomes and the compatible package pins are
retained in `provenance.json`.

## Replay

Use Python 3.11 or newer to recompute the comparison.

```bash
python benchmarks/results/ccsdt-external-forces-20261125/replay.py --require-passed
```

`records.json` retains measured total energies, full host force vectors, wall
times, convergence/work metadata and displaced energies. `execution.json`
binds the samples to the hardware, allocations, acquired rows and failure
records. `provenance.json` contains source, package and platform provenance.

`run.py` is the readable shared driver. Apply `gpu-measured-driver.patch`
or `cpu-measured-driver.patch` to a copy in the same directory as
`fixtures.json` and `basis/` to recover the exact measured bytes.
The superseded driver can be recovered from the measured CPU copy by
reversing `old-gpu-driver.patch`. The qz shell launchers record the build and
worker commands. Install each CPU environment from its explicit Conda lock
file, then run the recovered CPU driver on the appropriate allocation.
