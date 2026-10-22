# Cold DFT preparation: interim evidence

This bundle retains the complete endpoint observations supporting the
[cold-factor and Hcore decision](../../../.agents/notes/implemented/performance/2026-10-23-dft-cold-density-factors.md).
It does not claim that all 96-atom Direct/DF methods beat GPU4PySCF.

`cooperative-clean.json.xz` and `cooperative-moved.json.xz` retain independent
five-process alternating-order comparisons of explicit component-lane versus
cooperative Hcore schedules using the same untouched base library. The moved
cohort additionally displaces atom one by 0.001 Bohr along z, reconverges each
owner and freezes its own moved density. Every sample includes energy,
analytic forces and host export. These cohorts are separate; no samples are
pooled or selected by latency. The 48-atom cold gain fails the unchanged 2%/MAD
floor and remains visible. Warm and moved-warm gates pass at both sizes.

Source: `bbd41a42764e8d22d15c1348f70a633961d4dcf5`. Base core SHA-256:
`620c11f502513aef6faef9dc4a1fdfbeed6691f0619ad59de69d0bd701690fd9`.
Release sm120, CUDA 12.9.1, GCC 13.3, sccache 0.16.0, n1 RTX 5090, finite
Slurm allocations and one host numerical thread are recorded in every receipt.
The base build and experiment sources remain in ignored local/remote artifacts.
No reference density enters a native endpoint. Work counts, residuals, grids,
software versions and individual forces are retained in the compressed JSON.

Stationary qualification reuses the exact original dense-grid and CPU/GPU
reference receipts in `../readme-final-20261022/`, with the audited reuse
manifest and unchanged observable gates. Moved qualification separately checks
the same geometry against independent default GPU4PySCF; it is a schedule
equivalence check, not a dense-grid convergence claim for a moved production
ranking. The historical failed Hcore-only cohort remains in its original note.

```bash
PYTHONPATH=python:. python benchmarks/results/dft-cold-stage1-20261023/reproduce.py \
  > .artifacts/dft-cold-stage1-reproduced.json
```

Native cold-factor, warm-factor and Fock-composition tests pass, as do 40 GPU
Python tests for one-electron derivatives, DF occupied factors and KS seed
admission. The baseline and candidate both fail the provider suite's existing
default-MD census expectation and both pass with MD disabled. The tentative
triangular-Schwarz optimization is excluded. Build/test logs, initial failed
attempts and profiler streams remain in ignored artifacts rather than being
used as clean timing samples. Final combined-source timing and validation
receipts are recorded separately from these mechanism cohorts.

## Final combined source

Measured implementation: `3f3d67d1ec1bde3f5eab5d703cd57a800123b64d`.
Core SHA-256: `3e63ca3347135feeeca13f3d1196b852e21b036192a4805721a2f0c43ea0bb8f`.
The 1,832 compiler/source inputs were checked before the clean build. All
changed native owners were rebuilt through verified sccache, including the
reverted experimental direct kernel and DF layout. Separate immutable library
directories keep later experimental builds outside these timing populations.

Five alternating processes per library and mode at PBE0/water96 each execute
one cold and three frozen warm calls. Warm statistics first reduce the three
calls within each process, then take the median across five processes. All 80
complete calls pass the existing independent observable gates. Every displayed
gain exceeds the unchanged 2%/relative-MAD floor.

| Mode | Phase | Base median (s) | Candidate median (s) | Gain |
| --- | --- | ---: | ---: | ---: |
| Direct | cold | 29.649493 | 28.650683 | 3.369% |
| Direct | warm | 7.984821 | 7.038636 | 11.850% |
| DF | cold | 37.371538 | 33.023845 | 11.634% |
| DF | warm | 11.128437 | 10.282697 | 7.600% |

`stage1-pairs-direct.json.xz` and `stage1-pairs-df.json.xz` retain every sample.
`stage1-matrix.json.xz` adds one cold/one warm observation for PBE, PBE0, B3LYP,
r2SCAN, r2SCAN-3c and M06-2X in Direct and DF modes: all 24 complete calls pass.
This matrix is compatibility evidence, not repeated per-functional speedup
qualification. The final image passes native cold-factor, warm-factor and
composition tests and 40 GPU Python tests. The full provider suite passes with
MD disabled; the focused ERI-tile matrix failure also reproduces on the
untouched base. Both baseline failures are retained in local run logs.

The final source also passes 302 focused host tests, 19 documentation tests
with 61 subtests, all pre-commit hooks, Sphinx warnings-as-errors, and the
materialization review (nine baseline/candidate sites; no added or changed
sites). CUDA ownership changes by -3 scientific-classified lines, from selector
simplification, with no new generated mathematics or kernel retirement claim.

The twelve Fast populations in `stage1-fast.json.xz` also pass all 24 complete
energy/force checks. `df-moved.json.xz` retains another five alternating pairs
at each of 48/96 atoms, with cold, warm, moved and moved-warm calls. All 80 calls
pass the stationary dense or moved independent-default reference gates. The
48-atom cold/warm/moved/moved-warm gains are 2.278% / 2.420% / 1.273% / 2.464%;
the 96-atom gains are 11.886% / 8.218% / 3.230% / 8.423%. The 48-atom moved
gain does not exceed the 2% floor and is not claimed as a significant speedup.
No measured phase regresses. These trajectories stay separate from the static
three-warm-call cohorts above.

CI identified one additional historical test that pinned the old selector
expression. That assertion is replaced by a reference to the existing compiled
default/override selector probe; the ownership audit and independent GPU
derivative checks remain. No numerical or lifecycle gate is removed.
