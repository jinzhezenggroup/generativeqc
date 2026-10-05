# Direct J/K work census and rejected density reuse

This publication retains two separate observations for #1892/#1895. Neither
promotes a production default or completes independent J/K plan optimization.

- Slurm 5779: an intrusive, energy-only native solve counted actual streaming
  shell-task admissions separately for J and K. It is not clean endpoint timing
  and supplies no force measurements.
- Slurm 5785: clean public exact-direct PBE0 energy + analytic-force endpoints
  compared the optimized composition with one-transform J/K density reuse.
  Each 48/96-atom arm retained cold, five warm, moved, and five moved-warm calls.
  The candidate did not establish an endpoint improvement and is rejected.

| Atoms | Phase | Control median s | Candidate median s |
| ---: | --- | ---: | ---: |
| 48 | warm | 11.22592 | 11.30720 |
| 48 | moved-warm | 11.16198 | 11.24895 |
| 96 | warm | 32.53611 | 33.44083 |
| 96 | moved-warm | 32.55717 | 33.06780 |

Processes ran control then candidate at 48 and candidate then control at 96,
with separate initial stationary artifact caches. These ordered observations
are not interleaved causal timing estimates. The apparently faster 96-atom cold
candidate also changed the SCF trajectory from 28 to 26 iterations. All 288
native/reference and 144 control/candidate same-geometry E/F comparisons pass
1e-8 Eh / 1e-7 Eh/Bohr. Historical public Fock counts stay null.

The separate census observed 47 and 50 actual Fock builds at 48 and 96 atoms.
Original warm per-build admissions were J=32,816,365 / K=22,776,236 at 48 and
J=142,757,104 / K=72,116,584 at 96. These are shell-task admissions, not primitive
recurrences. The probe checks 55-entry arrays, 21 present shell classes, actual
owner counts, strict physical residuals and 144 independent energy pairings.
Its results must not fill missing fields in the clean E/F campaign.

## Verify retained observations

From the repository root, without a GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-jk-negative-work-20261005/verify.py
```

The verifier authenticates the publication, recomputes all numerical gates and
phase medians, checks executed sparse/phased/native force routes, and audits
exact census geometry/basis/probe-source inputs. The independent census oracle
is the permanent sibling `pbe0-composed-baseline-20261005` publication; the basis
is `pbe0-def2-svp-20261003/def2-svp-ho.json`. Executable binaries are not tracked:
original binary hashes and successful remote checks are receipts. They do not
constitute a fresh executable audit.

## Reconstruct measured sources

Create an isolated checkout at `1de139fa5`. Apply the permanent baseline
publication's `composed-source.patch`, then this publication's
`control-source.patch`. Stage source paths before computing canonical source
identity with `generativeqc.autotune.source_identity`. It must equal
`cb81f4c481e5cadd85999a7ff791f93a95a61f9fcf7bb5917ba3c3bfffa849b8`.
Applying and staging `candidate-source.patch` must then produce
`35a26a50a7e794b2b21ac9b5210fdc6af4d7b93bbf5dc37d28831b68951bf316`.
Both reconstruction identities were independently checked before publication.
The patch chain preserves scientific source ownership without retaining a
private branch as the only recovery path.

Exact original endpoint/census drivers and environment setup are retained as
`.txt` reproduction inputs. Paths identify the original frozen checkouts and
must be adapted when rerunning. Compile with ccache and the recorded CUDA 12.9
sm_120 configuration. Build `census/ks-work-probe.cpp.txt` as C++ against the
control library and its native headers. Real-device execution must use a finite
Slurm `main` allocation with `--gres=gpu:5090:1` and preserve its device visibility.
Keep census and timing processes separate. Raw JSON compression is lossless;
original scientific hash fields have not been rewritten.

The rejected implementation and wrapper-only predecessor are explained in
`.agents/notes/rejected/2026-10-05-shared-direct-jk-density-transform.md` and
`.agents/notes/rejected/2026-10-05-unused-j-preparation-wrapper.md`.
