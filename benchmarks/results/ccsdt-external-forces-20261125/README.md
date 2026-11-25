# Independent RHF-CCSD(T) Cartesian forces

This campaign addresses [#3167](https://github.com/jinzhezenggroup/generativeqc/issues/3167).
It compares all-electron, restricted closed-shell CCSD(T) total energies and
complete Cartesian forces at the cc-pVDZ correction basis. It uses the exact
QM7 C1–C7 geometries and three neutral polar controls (water, ammonia and
formaldehyde) from the
[previous CBS campaign](../cbs-h100-5090-7109b3d2-20261106).
The benchmark measures single-basis CCSD(T) endpoints. The CBS extrapolation
and MP2 components are outside this experiment.

## Observed results

The figures below come from the retained raw records. An observed case has at
least one complete cross-program comparison. The two-trial gate requires both
complete trials in every arm and checks all four pairings.

| Comparison | Cases observed | Two-trial passes | Worst energy (Eh) | Worst max force (Eh/Bohr) | Worst force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GenerativeQC conv. / Psi4 conv. | 8/10 | 6/10 | 3.02e-11 | 1.74e-06 | 5.88e-07 |
| GenerativeQC conv. / PySCF conv. | 7/10 | 6/10 | 5.23e-12 | 4.64e-08 | 2.67e-08 |
| Psi4 conv. / PySCF conv. | 7/10 | 6/10 | 1.37e-11 | 5.59e-07 | 2.21e-07 |
| GenerativeQC full DF / Psi4 full DF | 10/10 | 10/10 | 1.45e-10 | 3.11e-10 | 8.35e-11 |

### Complete endpoint wall times

Values are in seconds. Two samples are summarized by their median and range.
A single available sample is marked with its count. A dash indicates a missing
complete endpoint. The C4/C5 Psi4 conventional memory-control rows use the
larger allocation when available. CPU and GPU times describe the recorded
cross-hardware configurations.

| Case | GenerativeQC conv. | Psi4 conv. | PySCF conv. | GenerativeQC full DF | Psi4 full DF |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 4.478 [4.279–4.678] | 6.093 [2.627–9.560] | 6.181 [4.698–7.664] | 1.523 [1.412–1.635] | 11.066 [5.835–16.297] |
| C2 | 30.280 [29.962–30.598] | 22.476 [22.192–22.760] | 65.083 [64.316–65.851] | 2.600 [2.573–2.628] | 14.850 [13.228–16.471] |
| C3 | 184.990 [184.808–185.172] | 146.434 [138.486–154.381] | 469.428 [464.511–474.345] | 6.873 [6.864–6.883] | 56.271 [53.012–59.530] |
| C4 | 972.055 (1) | 648.870 [642.701–655.038] | 2360.793 (1) | 19.542 [19.475–19.609] | 190.397 [189.975–190.818] |
| C5 | 3633.460 (1) | 2495.795 [2427.486–2564.104] | — | 50.444 [50.352–50.536] | 643.288 [616.384–670.193] |
| C6 | — | — | — | 121.690 [121.646–121.734] | 1637.015 [1429.223–1844.806] |
| C7 | — | — | — | 273.638 [273.585–273.691] | 3656.906 [3618.855–3694.958] |
| water | 2.965 [2.772–3.157] | 2.975 [1.777–4.173] | 3.669 [1.920–5.417] | 1.316 [1.309–1.324] | 5.633 [5.481–5.784] |
| ammonia | 3.268 [3.232–3.304] | 2.197 [2.159–2.234] | 3.049 [3.029–3.069] | 1.353 [1.349–1.356] | 5.848 [5.671–6.024] |
| formaldehyde | 7.920 [7.891–7.949] | 5.351 [5.215–5.487] | 15.142 [14.956–15.329] | 1.797 [1.793–1.801] | 7.286 [7.231–7.341] |

### Methane finite differences

- PySCF conv. gives a maximum step-to-step change of 1.91e-07 Eh/Bohr. Across the available analytic endpoints, the worst finer-step difference is 5.08e-07 Eh/Bohr.
- Psi4 full DF gives a maximum step-to-step change of 1.86e-07 Eh/Bohr. Across the available analytic endpoints, the worst finer-step difference is 6.55e-08 Eh/Bohr.

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
  negative gradients in the original Cartesian frame. A separate PySCF/libcint
  check of the diagonal-normalized auxiliary Coulomb metrics gives a smallest
  relative eigenvalue of about `4.57e-8` at C7, above the fit cutoff setting
  of `1e-10`. Solver arithmetic and stopping-rule definitions still differ.
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
The two primary Psi4 full-DF C6 observations came from separate eight-thread
hosts with the same Xeon CPU Max 9462 model, memory allocation and solver
settings. Each producer numbered its own C6 trial zero. The replay uses
the origin labels and original per-record hashes to verify both observations.
A separate one-thread PySCF C3 diagnostic returned in 588.198 s. Its raw
record and the cancelled single-thread Psi4 C4 probe are kept outside the
eight-thread timing cohort.

At C4, Psi4's conventional path printed a CCSD(T) total energy before
raising `dpd_block_matrix: No memory left` while forming the triples-gradient
intermediates. The v1.10.2 RHF implementation allocates an `E[Gij]` DPD
buffer inside its virtual-triple loop. The closing frees for that loop omit
the buffer. The failed 24-GiB attempt and cancelled larger attempts are
retained. C4 and C5 subsequently returned complete Psi4 conventional endpoints
with a 92-GiB application budget in a 100-GiB allocation. This resource
constraint affects the pinned Psi4 implementation. A completed energy row
alone yields no analytic-force comparison. [Psi4 v1.10.2 RHF triples-gradient source](https://github.com/psi4/psi4/blob/v1.10.2/psi4/src/psi4/cc/cctriples/T3_grad_RHF.cc#L950-L955).

PySCF conventional returned C4 in one of the planned trials. C5 reached the
2700 s cap and C6/C7 reached the 5400 s cap without returning full forces.
The C6 log reports converged CCSD and the perturbative triples energy correction.
After both 5400 s timeouts, the unfinished larger-case repeats were stopped.

Initial software probes exposed a Psi4 ABI mismatch with Einsums 1.1.6, a
LibXC 7.1.2 alias mismatch, and a PySCF 2.11.0 incompatibility with NumPy
2.4.6's einsum planner. Probe outcomes and the compatible package pins are
retained in `provenance.json`.

The fresh native build acquired both full-DF trials, then conventional C1–C4.
A separate H100 allocation reused the verified binary for the two conventional
C1–C3 and polar trials. The original job was stopped after C4 to give the
conventional C5 endpoint a dedicated, longer time window. C5 first failed
the simultaneous host-memory admission check with a 32-GiB application
correlation budget. With a 72-GiB budget, the same binary returned one complete
C5 endpoint in 3633.460 s. Its host-memory plan reported 65,988,191,372 bytes.
The maximum force difference across the two independent Psi4 observations
was `1.74e-6 Eh/Bohr`. Conventional C6 and C7 were left outside that H100
follow-up after the larger independent reference runs became resource-limited.
The allocation receipts include the stop and cleanup outcomes.

## Replay

Use Python 3.11 or newer to recompute the comparison.

```bash
python benchmarks/results/ccsdt-external-forces-20261125/replay.py
```

The optional `--require-passed` exit check requires a complete passing cohort.
Per-comparison verdicts remain available when any planned comparison is incomplete.

`records.json` retains measured total energies, full host force vectors, wall
times, convergence/work metadata and displaced energies. `execution.json`
binds the samples to the hardware, allocations, acquired rows and failure
records. `provenance.json` contains source, package and platform provenance.

`run.py` is the shared driver for further acquisitions. It checks the revision
of a clean Git checkout or reads the commit ID from the accompanying Git tar
archive. It also verifies the staged native inputs against the pinned build
identity and checks that identity against the loaded library. An archive
staged beside `source/` defaults to `source-22f72.tar.gz`; pass
`--source-archive` for a different location.

To recover the exact measured driver bytes, apply the appropriate patch to a
copy of `run.py` alongside `fixtures.json` and `basis/`.
`gpu-measured-driver.patch` recovers the original master H100 driver,
`gpu-recovery-measured-driver.patch` recovers the 72-GiB C5 driver, and
`cpu-measured-driver.patch` recovers the independent CPU driver. Reverse
`old-gpu-driver.patch` on the measured CPU copy to recover the superseded
driver. Install each CPU environment from its explicit Conda lock file.
The qz shell launchers record the build and worker commands. To reproduce the
conventional C5 recovery, run from the staged source workspace containing the
library, inputs and recovery driver.

```bash
GQC_BENCHMARK_DRIVER=run.py bash qz-gpu-conventional-controls.sh run-c5-replay \
  --cases C5 --repeats 1 --timeout 7200 --correlation-budget-gib 72
```
