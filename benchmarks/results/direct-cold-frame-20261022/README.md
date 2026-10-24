# Direct-HF cold determinant reuse

HF/def2-SVP, spherical, 96 atoms, complete energy+force endpoint on n1 RTX 5090:

| Seconds | Baseline | Candidate | GPU4PySCF |
| --- | ---: | ---: | ---: |
| Cold | 22.480 | 20.106 | 11.958 |
| Warm | 2.917 | 2.913 | 3.414 |

Medians use three interleaved fresh processes per variant, each with three frozen
warm calls, in one finite Slurm allocation with one math thread. Cold includes
constructor/preparation, SCF, forces and export. Smaller 3/12/48-atom populations
have one cold sample each. All 72 calls pass independent gates; maximum candidate
force error is 6.38e-7 Eh/Bohr. Target-basis cold Focks fall 13 → 11; iterations stay 11.
See the [decision note](../../../.agents/notes/implemented/performance/2026-10-22-direct-huckel-force-frame.md).

The compressed evidence retains timings, work, errors/gates, maximum residuals,
source patch, binary identities and rejected ablations. Baseline is master
`afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`; candidate adds the archived patch.
Full arrays/logs remain in ignored local/n1 storage, with a retained checksum.

```bash
PYTHONPATH=python:. python benchmarks/results/direct-cold-frame-20261022/reproduce.py
# Optional original-array requalification:
PYTHONPATH=python:. python benchmarks/results/direct-cold-frame-20261022/reproduce.py .artifacts/direct-cold-frame-full.json.xz
```
