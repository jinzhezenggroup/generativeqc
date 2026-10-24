# Decision: rank qualified production configurations in the README

Status: implemented (publication policy only; production remeasurement pending)
Date: 2026-10-18

## Problem

The homepage HF/DFT plots used a fixed, unpruned 48/16/32 quadrature for both
engines. This is useful for comparing implementations on identical quadrature,
but it does not establish the fastest way for either engine to meet a common
scientific accuracy target. The user explicitly chose best production
configuration, rather than matched quadrature, as the homepage comparison.

## Decision

The homepage HF/DFT ranking requires independently selected production settings
that pass shared energy and analytic-force gates and have measured complete
endpoints. Each engine may use a different qualified quadrature, pruning,
screening or precision. The authoritative publication requirements are in
`docs/maintainer/performance_engineering.md`.

Remove the historical HF/DFT plot embeds from the headline comparison, retain
links to their original evidence, and mark production comparison pending. This
does not implement pruning, change calculation defaults, rerun either engine,
alter historical evidence or qualify an existing driver/renderer for production
publication. The separate DF-CCSD(T)/CBS section is unchanged.

## Rejected alternatives

- Keep identical grids as a homepage requirement: would conflate identical work
  with equal scientific accuracy and could suppress useful production paths.
- Enable pruning only in the reference and reuse old measurements: new settings
  need independent force/energy qualification and actual endpoint timings.
- Relabel existing plots as best production results: their measurements do not
  demonstrate that claim, and some retained observations lack accuracy checks.

## Invariants

- Shared scientific task, independent accuracy gates and complete endpoint cost.
- No production dependency on reference densities or hidden oracle work.
- Historical timings, provenance, failures and unchecked observations remain
  unchanged in their retained reports.
- No speedup claim based only on point-count or nominal pair-count reduction.

## Evidence

`python/generativeqc_compiler/dft/grid.py` currently accepts only
`pruning="none"`. `benchmarks/readme_wb97mv.py:reference_engine` reconstructs
fixed angular rules and overrides `gen_atomic_grids` for both semilocal and
nonlocal grids; changing its `prune` attribute alone cannot add pruning to that
custom constructor. A production-grid comparison needs configuration and
qualification work, not a plot-label change.

## Revisit when

Both engines have source-matched, independently qualified production settings
and complete endpoint populations ready to replace the pending homepage entry.
