# Decision: qualify a larger SCF XC tile without promoting a global default

Status: implemented (benchmark/evidence and public-control documentation only)
Date: 2026-10-06
Related: #1893, #1894, #1876; follows the mapped-density crossover evidence

## Problem

The earlier mapped-density experiment suggested about 23% lower fixed-density
XC wall time at 96 atoms for generated tile 512 versus 256. Its density was a
positive diagnostic, not a converged PBE0 state, and its result could not qualify
complete energy/analytic-force endpoints or an indexed GEMM promotion.

## Decision

Compare the existing public SCF tile controls in one native binary on the
retained frozen source. Keep generated/local-AO execution, cutoff 1e-16 and
ordinary force tile 256 unchanged. Each arm uses an independently converged
native warm snapshot, frozen publicly; do not call the density bytes identical.
Include actual one-iteration/one-Fock-build SCF work in every timed E+F replay.

n2 Slurm job 2506, RTX PRO 6000 Blackwell, CUDA 12.9, ccache and five interleaved
pairs per size/geometry establish these complete median improvements:

| Atoms | Warm | Moved-warm |
| ---: | ---: | ---: |
| 96 | 2.088% | 2.174% |
| 48 | 3.074% | 3.000% |

All four rows exceed the shared 2%/relative-MAD descriptor. Independent retained
GPU4PySCF 1.8.1 references bound maximum energy/force errors at 1.015e-10 hartree
and 3.134e-11 hartree/bohr under unchanged 1e-8/1e-7 gates. These are configuration
results, not a new compiler layout, kernel, library provider or generic selector.

## Work and resources

At 96 atoms, SCF map tiles halve from 9,216 to 4,608, while per-traversal local
AO-square summands increase from 75,674,112,000 to 88,940,603,392. Point/AO visits
increase from 373,868,544 to 398,149,632. Fewer tiles therefore do not mean less
arithmetic. Original-geometry AO-map device reservations are 69,340,448 versus
48,915,744 bytes; those are not concurrent endpoint device peaks. The discovered
map statistics persist as geometry receipts, not warm rediscovery counters.

Force work is identical across arms: 2,359,296 points, 9,216 phased batches,
10,758,389,760 primal and reverse visits each, and 64,512 Becke launches per
96-atom replay. No force/Becke tile enlargement is attributed to this result.
Cold setup uses different actual histories (26/29 iterations at the original
96-atom geometry); retain them without normalizing or claiming cold speedup.

## Provenance and limits

Measured source is base 83109befe5b97a5542b30e169da74c81d04719e6 plus retained
patch, source identity b0e32abf342aa53877f2a350801a1dfdc1bc74d341df0cebe82a738d74429d2f.
This includes the ordered normalization experiment shared by both tile arms;
the new PR is independent and does not include that production patch. Core SHA
20100959aa8c2643b1f341f9c140e4778d618e496dda8f216548c52676137931 identifies this
fresh ccache-enabled rebuild. Do not relabel these data as current-master
performance or add their percentage to a separate normalization comparison.

Two prepared arms coexist during replay, and whole endpoint peak memory is not
measured. The observed core rebuild is 70.84 seconds with a shared ccache;
uncached whole-core/first-wrapper compilation cost is not qualified. No new
sanitizer campaign, physical UKS96 gate or other-device claim is made.

## Rejected alternatives and revisit conditions

Do not promote indexed GEMM, enlarge force tiles, or change the global default
from this narrow configuration comparison. Automatic 512 selection requires
compiler-visible occupancy/resource accounting, broader numerical/domain
qualification and an explicit bounded 256 fallback. Default/explicit tile
semantics remain intact. This does not close #1893 or #1894.

See `benchmarks/results/pbe0-xc-tiles-20261006/README.md` and the independent
consumer `tools/generativeqc_validation/pbe0_xc_tiles.py`. The prior
[mapped-density crossover decision](2026-10-05-mapped-xc-density-crossover.md)
remains valid negative provider evidence; this qualification resolves only its
larger-tile endpoint question.
