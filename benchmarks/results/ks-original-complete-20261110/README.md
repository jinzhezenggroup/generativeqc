# Original KS correctness gates, 2026-11-10

Numerical acceptance is scoped to the identified RTX 5090 runs, not production,
performance, actual CUDA Graph execution, or unmeasured A100/PRO6000 devices.
Original #2287 failures and the prior inconclusive 2026-11-09 publication keep
their original statuses and source identities.

Integrated source: `d3f8a62df6aa371ad75449406053d7ab3333c97a`.
Native input: `02dd0d238d7572be343496b4f3267a48b48ee2b32b9f8917035996a7ae2365cb`.
Library SHA-256: `41efe7a154cfd8e428edbc9ff4c25c9a99f42c380839866d893dbc9002c8c34e`.
Release/sm120/fast-compile, GNU 11.4, CUDA 12.9, verified sccache 0.16.0;
PySCF 2.14.0 / Libxc 7.0.0 supply independent oracles, never production work.

## Water96: original grid, controls and sequence

PBE-RKS, spherical def2-SVP orbital/auxiliary bases, DF 512 MiB, FP64,
Treutler level 3, force tile 1024. Default E/D: 1e-9/1e-5; tighter-SCF-only:
1e-12/1e-10. Both retain two-item cold solve, rejected invalid occupation with
atomic rollback, frozen replay, atom 1 z+0.025 Bohr and moved/fresh owners.
Energy/force gates remain 1e-8/1e-7. Each arm retains six completed observations.

| Binary/controls | Max moved/fresh force error | Failing components |
| --- | ---: | ---: |
| Parent/default | 4.794283938736044e-7 | 45/288 |
| Parent/tight | 1.4027001782324078e-11 | 0/288 |
| Reviewed repair/default | 3.398517023356362e-10 | 0/288 |
| Reviewed repair/tight | 1.4004797321831575e-11 | 0/288 |
| Integrated repair/default | 3.398525905140559e-10 | 0/288 |
| Integrated repair/tight | 1.4007461857090675e-11 | 0/288 |

Parent and reviewed repair are paired on the same physical GPU in each arm
(Slurm 17028/17029). Parent source/library:
`a305bde4a845e554e5f04ca4ab593b26166b7e31902635dcd5a900194532aea3` /
`d6383c769ed44e089a61b731f2e3614032ee7e2cbc0b9ac321955fa1a58d5604`.
Reviewed repair source/library:
`8f9e0093709c9923bf7a8df4a3e04558741ae4d1bb84ed3dd77f4af5ef8e9b39` /
`f3ec752a5a4ca2aad49853cfca2fd476174662756ede11093336dee63d8b3877`.
Integrated endpoints use separate jobs 17048/17049, not relabelled paired data.
The baseline reproduces the reported 45-component discrepancy; tightening only
SCF on the original grid removes it. Its default moved/fresh residual maxima
are 2.16e-8/1.12e-8, versus repaired 1.10e-11/5.66e-12. This establishes
stopping sensitivity in the matched present-day base, not whether #2910
introduced the historical behavior. The repair binds force stationarity
without changing caller controls, grid, budgets or numerical assertions.

## UKS and original range-only RSH

All 8 original large-spd UKS E/E+F × masks 0/127 Cartesian/spherical controls
pass: 40 states and 20 independent analytic-force observations. Max canonical
error 9.267731582163208e-9 remains below 1e-8; max force error is
1.0525362970081886e-8 Eh/Bohr. Cartesian closure no longer destroys an already
proved physical state; the independent spherical reconstruction gate stays.

The original PBE range-only H2/H3 fixture passes both spin/representation
layouts through cold/warm/moved/warm-moved E/F/state: 4 controls, 16 states.
Its exact primitive decimals, geometry, grid and SCF controls are pinned by
`test_ks_original_rsh_admission_cuda.py`. DF full-range J/K plus exact Direct
LR-K is matched by PySCF; independent five-point derivatives reconverge the
moving-grid energy including auxiliary-center response. Max E/D/Fock errors
are 1.87e-14/8.27e-11/3.20e-11, force error 6.686293893228869e-12, canonical
error 5.61e-14. No WB97M-V or fitted-LR substitution is used.

RSH/UKS execute in an overlapping step of job 17028; no performance claim is
made. The earlier missing-SR/LR-sum exploratory failure (0.1292169721472736
Eh/Bohr) is retained separately, not relabelled as a current failure or pass.
All full force/reference vectors, final diagnostics, raw-receipt identities,
artifact witnesses and exact history hashes are in `samples.json.xz`.
Complete iteration streams remain in ignored `.artifacts/full-original-gates/`.
Recover JSON with `lzma.open(path, 'rt')`; no lossy rounding is used.
