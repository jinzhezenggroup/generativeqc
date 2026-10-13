# Decision: complete cold/warm DFT comparisons in the README

Status: implemented
Date: 2026-10-13

The HF/DF scope and mixed-work plotting rule are superseded by
[complete endpoint statistics](2026-10-13-readme-hf-df-endpoint-statistics.md).

## Decision

Reuse the independent moving-grid energy-plus-force protocol for PBE, PBE0,
B3LYP, r2SCAN, r2SCAN-3c, M06-2X and WB97M-V. Publish cold as calculator
construction, topology preparation and first synchronized execution. Collect
binary hashes and device provenance outside that interval. Retain existing
compiler/artifact caches; this is an API cold start, not an empty-cache build or
process-launch benchmark. Native calculator construction is now included to
match reference engine construction.

r2SCAN-3c retains its defining def2-mTZVPP basis and both D4/gCP corrections.
Its reference combines GPU4PySCF electronic energy/analytic forces with CPU
`dftd4` and the independent gCP qualification implementation. Every correction
evaluation is inside the reference timer, including geometry-dependent setup;
the figure explicitly labels these CPU components. Native execution uses its
public composite endpoint and never calls those reference implementations.

## Invariants

- Keep identical geometries, spherical basis data, moving quadrature and the
  existing absolute `1e-8 Eh` / `1e-7 Eh/Bohr` gates in both engines.
- Gate cold, all fixed-density repeats, changed geometry and all moved repeats.
- Return forces to host before stopping either timer.
- Use one engine-local post-cold/post-move density for all corresponding replays;
  never supply a reference density to the native solver.
- Keep stock reference DIIS and count full-density potential rebuilds explicitly.
  Equal iterations do not assert equal work or stopping rules across engines.
- If replay iteration/Fock counts differ, show the observations/range rather
  than a pooled median. Incomplete or failed native records have no plotted time.
- Retain source and loaded-binary identities, all scalar measurements, work and
  residuals, independent cold/moved reference arrays and raw-file hashes.

## Rejected alternatives

Plain r2SCAN timings cannot represent r2SCAN-3c. Omitting correction time from
the reference or timing only a warmed electronic kernel would change the
endpoint. Purging shared compiler caches would destroy unrelated reusable
artifacts without establishing a fair API cold comparison. Historical candidate
timings are not relabeled as measurements of the current master.

## Evidence and reproduction

`benchmarks/run_readme_dft_benchmarks.sh` runs finite per-point subprocesses
inside Slurm. `tools/render_readme_dft.py` reuses the complete endpoint gate
before reduction and preserves unsuccessful outcomes. The
[dated result report](../../../../benchmarks/results/readme-dft-20261013/README.md)
records the measured revision and hardware. Host tests cover composite identity,
correction signs and timing inclusion, alongside the existing all-endpoint
publication gates.

Build the full production preset for reproduction. Building only the
`generativeqc` target produces the core library but omits packaged stationary
force DSOs and their contracts. Initial trials with that incomplete installation
converged SCF, then failed the force endpoint; their logs remain separate from
the final campaign. Existing packages from another build were rejected because
their contracts differed. The final measurements use the freshly built core
and matching force packages.

Revisit the reference labeling if GPU4PySCF gains a qualified GPU implementation
of both canonical 3c corrections. A changed comparator requires new measurements.

## Force-memory follow-up

The initial master measurements exposed post-SCF OOMs at 96 atoms. The
[force-phase cache handoff](2026-10-13-ks-force-cache-handoff.md) fixes this
without reducing SCF cache admission. The report now explicitly identifies
eight remeasured points from that source snapshot, while retaining the original
master provenance for other points. Unmodified master raw records and report
remain under the ignored artifact directories.
