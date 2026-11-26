# CRBH20 barrier heights with MP2-assisted CCSD(T)/CBS

This benchmark covers 20 cycloreversion barriers built from 40 fixed,
neutral singlet geometries. Measurements used GenerativeQC
`ae1086d511060662b7d728eb80d3e53981443ec1` with a production Release SM90
AOT build on H100. See the [source archive receipt](source.json),
[build identity](build-identity.json), and [build qualification](build-qualification.json).

## Results

| Recipe | CUDA endpoints | Barriers | MAE (kcal/mol) | Max abs. error (kcal/mol) | Within 1 kcal/mol | Median / max endpoint (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DT_D | 40/40 | 20/20 | 1.763 | 3.095 | 6/20 | 29.54 / 117.57 |
| TQ_D | 40/40 | 20/20 | 0.269 | 0.676 | 20/20 | 46.27 / 157.55 |

For each geometry, both recipes ran on the same physical card. TQ_D meets the
complete-set gate of at most 1 kcal/mol absolute error per barrier. The
comparison measures the change from D/T to T/Q extrapolation at fixed DZ
correction. [Barrier values and signed errors](barriers.csv),
[endpoint times and energies](endpoints.csv), [offline summary](summary.json).

## Sources and protocol

The reference values are from Table 2 of Yu et al., using W1-F12 for entries
1–10 and W1w for entries 11–20. The driver divides the table values in kJ/mol
by 4.184. The 40 geometries are from the pinned Molpro `Karton_CRBH20` mirror.
The acquisition script checks atom order, positions to within `1e-8` Bohr,
charge, and multiplicity against pinned MGCDB84/ACCDB XYZ files. The
[inputs](inputs.json) retain the reference values, coordinates, signed
stoichiometry, and file digests. [Yu et al.](https://doi.org/10.1016/j.chemphys.2015.07.005),
[Molpro snapshot](https://github.com/molpro/pymolpro/blob/1d45741076c19a0b1f87464a53a256254a662035/pymolpro/share/database/Karton_CRBH20.json),
[ACCDB snapshot](https://github.com/peverati/ACCDB/tree/bd1f68947daab2370edccd4339cc0390dc562c29),
[geometry license](CRBH20-geometry-license.txt).

| Recipe | HF and MP2 extrapolation | CCSD(T) minus MP2 correction |
| --- | --- | --- |
| DT_D | cc-pVDZ / cc-pVTZ | cc-pVDZ |
| TQ_D | cc-pVTZ / cc-pVQZ | cc-pVDZ |

Both recipes use all-electron spherical orbital bases and cc-pVXZ-JKFIT
auxiliaries for RHF and the correlation operators on the full-DF CUDA path.
HF extrapolates exponentially with exponent 1.63, and MP2 correlation uses
inverse cubic extrapolation. The strict FP64 policy also uses FP64 CCSD
preconvergence. SCF energy and density tolerances are `1e-12` and `1e-11`.
CCSD energy and residual tolerances are `1e-12` and `1e-10`.

The energy-only jobs use a fresh subprocess per endpoint with a 7,200 s
process limit, a 64 GiB correlation numeric-buffer allowance, and a finite
scalar-work limit of 50,000,000,000,000 per phase. Endpoint timings span CBS
owner construction through energy return. The endpoint table also gives
whole-process timings. Barrier differences exclude zero-point, thermal,
scalar-relativistic, and counterpoise corrections.

The [build driver](build_qz.sh) uses CUDA 12.9, GCC 12, OpenBLAS, and sccache
0.16.0. Fast CUDA compilation is disabled and shell AOT uses
`portable_cuda`. Six s/p and s/p/d stationary CUDA force profiles were
packaged and audited. The measurements request energy only.

## Independent conventional anchors

Entries 1 and 11 were selected for conventional all-electron PySCF RHF,
MP2, and CCSD(T) at DZ, TZ, and QZ. The CPU driver checks the Intel Xeon CPU
Max 9462 model before opening an endpoint journal. RHF uses `1e-12` Eh
energy and `1e-9` orbital-gradient tolerances. CCSD uses `1e-10` Eh energy
and `1e-8` amplitude-update tolerances.

| Recipe | Anchor basis | Complete anchors | CUDA composite MAE / max (kcal/mol) | Conventional composite MAE / max (kcal/mol) |
| --- | --- | ---: | ---: | ---: |
| DT_D | TZ | 2/2 | 1.502 / 1.694 | 1.465 / 1.644 |
| TQ_D | QZ | 0/2 | — | — |

Nine of the 12 canonical endpoints completed. QZ endpoints `01ts`,
`11react`, and `11ts` reached the 7,200 s cap.
[Canonical endpoint outcomes](canonical-endpoints.csv).

## qz allocation

All jobs used CQ-NexChem. The two GPU measurement jobs used an estimated
0.987 H100 card-hours. The shared production build used 0.904 card-hours.
Including the entire shared build, the estimate is 1.892 H100 card-hours and
1.892 qz points. The CPU job used 91.094 allocated core-hours at zero points.
Estimates span scheduling to terminal status. [Per-job accounting](allocation.json).

## Offline replay

Run with Python 3.11 or newer. Replay validates the input and raw bundle
digests, hardware and build provenance, precision and convergence metadata.
It reconstructs native composite energies within `1e-8` Eh, then rebuilds
the barriers and canonical comparisons.

```bash
python benchmarks/results/crbh20-cbs-20261125/replay.py .artifacts/crbh20-cbs-replay
```

The [acquisition ledger](acquisition.json) preserves the input and driver
hashes recorded by each job and binds them to the retained raw bundles and
CRBH20 input projection. The CRBH20 coordinates, references and canonical
anchor selections were verified against the original acquisition. The
retained input subset and scoped driver copies have their own file identities.
The two GPU bundles are byte-identical to the acquisition copies. The CPU
bundle substitutes an interpreter installation prefix in stack traces.
Its scientific journal and original acquired-file digests are preserved.

The drivers describe the acquisition protocol for the pinned source. Use
[prepare_inputs.py](prepare_inputs.py) to regenerate the CRBH20 input plan
and [plan_gpu.py](plan_gpu.py) to generate the two paired-device shards.
The qz entrypoints are [run_qz_gpu.sh](run_qz_gpu.sh) and
[run_qz_cpu.sh](run_qz_cpu.sh). The replay command needs only the files
retained here and the Python standard library.

The reproduction runner reaps interrupted endpoint processes and cleans
each CPU endpoint's scratch after the child exits.
