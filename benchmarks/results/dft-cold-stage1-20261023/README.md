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
receipts will be recorded separately from these mechanism cohorts.
