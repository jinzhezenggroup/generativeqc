# Evidence: complete README timing after master integration

Status: implemented bounded HF/DFT continuation; full README goal remains open
Date: 2026-10-14

## Protocol and identity

Slurm n1/RTX5090 job 8444 ran the actual `benchmarks.readme_dft` public CLI,
with independently prepared reference/ordinary-FP64/Fast owners. The source is
master `c2c670e2` plus the retained application-budget continuation; main library
SHA256 is `3d31a43ddd62630393c6bcd352e6b3333cf9c8004b68258532681e4e488f7ac5`.
This binary predates the private typed DF shell/Rys work. It is not HF-DF FP32
or correlated/CBS evidence.

Every population includes cold preparation plus E/F, five frozen-seed replays,
one moved-geometry E/F and five moved-frozen replays. Warm table entries use
medians. The grid is 48/16/32; WB97M-V and M06-2X use def2-SVP, 3c its defining
def2-mTZVPP; fitted models use the pinned cc-pVDZ-JKFIT input. Basis inputs,
grids, model composition, initialization and convergence controls match. The
reference includes auxiliary/metric, moving-grid, VV10 and 3c response as
applicable; fitted WB97M-V retains exact long-range exchange in both engines.

All 24 engine/population runs completed: 288 independently gated calls. Every
one of the 96 Fast calls reports one solve, positive FP32 semilocal-geometry
work and a mixed returned gradient, with no whole-gradient retry. Every matched
native strict/Fast pair has identical SCF iterations and Fock-build counts.
Warm calls execute one SCF iteration. Geometry points are 73,728 for three
atoms, 147,456 for six and 294,912 for twelve. These count attempted geometry
points, not FP32 instructions or narrowed SCF/VV10 work.

## Results and limits

Ratios are ordinary FP64 seconds / Fast seconds, so values below one lose.
Maximum force error includes every phase, against the independent oracle.

| Model | Atoms | Cold | Frozen | Moved | Moved frozen | Max error Eh/bohr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WB97M-V direct | 3 | 1.013 | 1.162 | 1.014 | 1.160 | 4.76e-8 |
| WB97M-V direct | 12 | 1.006 | 1.097 | 1.009 | 1.101 | 7.57e-8 |
| WB97M-V DF | 3 | 1.018 | 1.183 | 1.019 | 1.177 | 4.55e-8 |
| WB97M-V DF | 12 | 1.004 | 1.118 | 1.010 | 1.116 | 7.83e-8 |
| r2SCAN-3c DF | 3 | 1.000 | 1.173 | 0.976 | 1.178 | 1.54e-7 |
| r2SCAN-3c DF | 6 | 1.015 | 1.340 | 1.055 | 1.337 | 1.69e-7 |
| M06-2X direct | 3 | 1.016 | 1.627 | 1.097 | 1.624 | 2.35e-7 |
| M06-2X direct | 12 | 1.076 | 1.565 | 1.075 | 1.559 | 7.11e-7 |

Do not discard the small-3c moved loss or generalize frozen speedups to cold
singlepoints. These small domains do not establish all-size README coverage,
all-FP32 arithmetic, long-time dynamics, or broad 2–7x complete acceleration.
Further SCF/integral/VV10 and full-DF CCSD(T)/CBS work remains necessary.

## Retention

Raw 24 JSON/log pairs, summary, source receipts, command script and job log
remain under ignored `.artifacts/readme-budget-fast-20261013/`, with the source
and original records under the matching n1 directory. No external evidence
host or release was created. Failed earlier missing-AOT-package qualification
is retained separately; complete package job 8443 passed 126 endpoint tests.

This supersedes earlier timing comparisons for these integrated populations,
not the earlier evidence itself. See `../../proposed/2026-10-14-readme-budget-fast.md`.
