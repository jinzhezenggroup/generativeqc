# Decision: report stock endpoint latency separately from qualified rankings

Status: implemented
Date: 2026-10-20

Superseded for the active README refresh by
[default alignment and observable qualification](2026-10-20-readme-default-alignment-and-ranking-reversal.md).
The measurements below remain historical evidence; the reference Fast profile
and unequal force-response settings are not part of the replacement campaign.

## Problem

The requested README refresh fixes GPU4PySCF's default XC and VV10 grids and
asks native Standard/Fast to follow them. The previous README policy instead
described choosing the fastest independently qualified configuration. Those are
different experiments. The runner proposed in #2623 also covered only cold
Direct DFT, whereas the requested refresh includes HF, all seven historical
DFT series, DF, and warm reuse.

## Decision

Keep production science at master `e5fa4d2f0cd13775327261a0b6ced1eab6006c62`.
Measure one fresh-owner cold and three warm complete E+F calls per population.
Use the same frozen, engine-local cold density for every warm call. Run public
`PrecisionCalculator(policy="fast")`, including real audits/fallbacks in its
timing and solve census. Do not replace that facade with `precision="auto"`.

GPU4PySCF Default retains its stock controls. GPU4PySCF Fast copies the actual
native Fast candidate's energy/density tolerance numbers to `conv_tol` and
`conv_tol_grad`; the latter checks an orbital gradient, not density RMS. Retain
that distinction and the method/DF-dependent strict fallback controls. No
monkeypatch of GPU4PySCF's convergence algorithm is part of this experiment.

DF uses GPU4PySCF's unmodified automatic auxiliary selection. A separate host
input-preparation step snapshots the selected raw basis tables for native DF.
Native endpoints consume the immutable snapshot without invoking PySCF.
Reference endpoints independently choose their defaults and verify the executed
auxiliary table against that snapshot after timing. This avoids imposing the
historical cc-pVDZ-JKFIT choice on the reference's default configuration.

Default grids do not imply identical force semantics: GPU4PySCF leaves
`grid_response=False`, while native includes moving-grid response. Publish
absolute latencies, with no shared-accuracy or equal-force-work speedup claim.
Dense-oracle/displaced-geometry production qualification is separate; this
campaign deliberately contains only cold and warm. CBS is unchanged.

## Invariants

- No production grid, numerical, precision, or resource overrides are introduced.
- Missing/failed endpoints and incomplete warm populations remain visible.
- Cold includes construction, preparation, SCF, analytic forces and host export;
  imports/CUDA context initialization precede it. Any runtime compilation inside
  the endpoint remains timed. Compiler caches are retained, never cleared.
- Preserve exact driver and native binary identities, settings, SCF/precision
  work, and Slurm outcomes. Default timing does not promote numerical accuracy.
- README contains plots and a brief protocol link; detailed evidence belongs in
  `benchmarks/results/readme-defaults-20261020/`.

## References

The finished campaign retained 765 successful calls: all 64 native Direct
populations, all 64 GPU4PySCF Default populations, 63 complete GPU4PySCF Fast
populations and the remaining reference cold endpoint. All 64 native DF
populations failed preparation because the default auxiliary contains `l=4`
and current CUDA `df_metric` admission ends at `l=3`. The reference HF/96 Fast
warm call did not converge within its stock 50-cycle limit. Neither control
was changed to replace these negative observations. All native Fast solves,
including eight two-solve audits, remain in the complete endpoint timings.

- #2623: default comparison protocol and real Fast provenance.
- #2626 and #2633: default Treutler/Lebedev/NWChem grids and phased partitions.
- `benchmarks/readme_defaults.py` and `benchmarks/readme_defaults_report.py`.
