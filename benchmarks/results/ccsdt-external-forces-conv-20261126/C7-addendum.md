# C7 H200 and directional force check

Measured GenerativeQC revision `22f72d0692a760c2892cd7791022584d9ae1c3e5`.
See the [conventional reference measurements](README.md) for the original
C7 analytic CPU vectors and the failed H100 allocation.

## Directional finite difference

Psi4 energy-only conventional RHF-CCSD(T) on Xeon Gold 6530 HPC
workers produced the displaced energies. Each job requested 20 CPU
and 100 GiB. Psi4 used 20 threads and a 76 GiB memory allowance.
The main SCF and CCSD energy-change threshold was 1e-13 Eh.
The SCF density and CCSD amplitude RMS thresholds were 1e-11.
The finer convergence repeat used 5e-14 Eh and 1e-12, respectively.
Both used 1e-14 integral screening and the identical all-electron
spherical cc-pVDZ basis, C1 geometry, and cold start.

| Step (Bohr) | Convergence | E(−h) (Eh) | E(+h) (Eh) | Force (Eh/Bohr) |
|---:|---|---:|---:|---:|
| 0.004 | tight | -275.590785646627467 | -275.590782892980712 | -3.4420584427153e-04 |
| 0.002 | tight | -275.590788719356851 | -275.590787343001011 | -3.4408895999150e-04 |
| 0.001 | tight | -275.590789315582015 | -275.590788627462928 | -3.4405954352223e-04 |
| 0.001 | check | -275.590789315579912 | -275.590788627461734 | -3.4405908877488e-04 |

First-level Richardson estimates are -3.4404999856482e-04 and
-3.4404973803248e-04 Eh/Bohr. Their difference is
2.605e-10 Eh/Bohr. The tighter fine-pair
convergence check changes the direct estimate by 4.547e-10
Eh/Bohr. The two-level extrapolated force is
**-3.4404972066366e-04 Eh/Bohr**.

| Analytic result | Force minus extrapolated force (Eh/Bohr) |
|---|---:|
| Psi4 | +3.603916e-06 |
| PySCF | -1.426851e-10 |
| H200 trial 0 | -1.594123e-10 |
| H200 trial 1 | -1.594154e-10 |

The extrapolation supports the PySCF analytic force for the fourth
carbon's z component. The stability check measures changes across
step pairs and convergence settings.

## H200

Each H200 reported 143,771 MiB device memory, compute capability 9.0 and
driver 570.124.06. The Release SM90 build exposed 181 SM90 ELF images
and matched the #3174 library and native source identity. The original
production binary ran with the same FP64 physics and 155 GiB host
correlation budget as the single-card H100 attempts.

| H200 trial | Complete endpoint (s) | Peak device memory (MiB) | Energy (Eh) |
|---|---:|---:|---:|
| 0 | 24222.29 | 124,134 | -275.59078928496791 |
| 1 | 24291.14 | 124,134 | -275.59078928496791 |

Two-sample endpoint median is 24256.72 s.

| H200 trial | Analytic reference | ΔE (Eh) | Max absolute force difference (Eh/Bohr) | RMS force difference (Eh/Bohr) |
|---|---|---:|---:|---:|
| 0 | Psi4 | 3.70619e-11 | 3.60408e-06 | 8.60668e-07 |
| 1 | Psi4 | 3.70619e-11 | 3.60408e-06 | 8.60668e-07 |
| 0 | PySCF | 4.43379e-12 | 2.10274e-09 | 8.13178e-10 |
| 1 | PySCF | 4.43379e-12 | 2.10274e-09 | 8.13178e-10 |

Both H200 trials differ from the extrapolated finite-difference force by
-1.594e-10 Eh/Bohr in the fourth carbon's z component. Against PySCF they
pass every gate with margins of more than three orders of magnitude. The
Psi4 pairs exceed the 3e-6 Eh/Bohr maximum-component gate only in that
component, where the Psi4 analytic force itself differs from the
extrapolated force by +3.604e-06 Eh/Bohr. The disagreement therefore
belongs to the Psi4 reference vector, not to GenerativeQC.

## qz allocation and cleanup

| Task | Project | Status | H200 card-hours | CPU core-hours | Estimated points |
|---|---|---|---:|---:|---:|
| `gqc-3167-c7-fd-h1m-check-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.611 | 0.000 |
| `gqc-3167-c7-fd-h1m-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 5.544 | 0.000 |
| `gqc-3167-c7-fd-h1p-check-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.700 | 0.000 |
| `gqc-3167-c7-fd-h1p-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.211 | 0.000 |
| `gqc-3167-c7-fd-h2m-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.383 | 0.000 |
| `gqc-3167-c7-fd-h2p-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 5.867 | 0.000 |
| `gqc-3167-c7-fd-h4m-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.544 | 0.000 |
| `gqc-3167-c7-fd-h4p-tight-1126-a` | cq-nex-chem | SUCCEEDED | — | 6.422 | 0.000 |
| `gqc-3167-c7-h200-0-1126-a` | CQ-NexChem | SUCCEEDED | 6.769 | — | 6.769 |
| `gqc-3167-c7-h200-1-1126-a` | CQ-NexChem | SUCCEEDED | 6.787 | — | 6.787 |
| Total for archived jobs | | | 13.556 | 50.283 | 13.556 |

H200 jobs requested priority 4 and the platform applied NORMAL/20.
CPU HPC jobs requested 10 and the platform applied HIGH/35.
The ledger uses the compute worker's scheduled-to-release interval,
the selected quota's 1 point per H200 card-hour, and 0 points for CPU HPC.

Local raw outputs are under `raw/c7-addendum/`. The source archive,
production library, driver, scheduler receipts and per-job raw files
are identified in `c7-addendum-provenance.json`. Replay the compact
records from this directory with `python3 replay_c7.py`.
