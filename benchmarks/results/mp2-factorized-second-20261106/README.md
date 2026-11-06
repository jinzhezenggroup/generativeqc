# Factorized MP2 second-stage CPU qualification

`qualification.json` retains scoped CPU measurements, not an endpoint speedup
or issue-closeout claim. Its baseline is `1affb86a7`; the exact native candidate
patch, pullback/library SHA-256 identities, compiler, build configuration,
thread/affinity settings and each raw complete endpoint sample are included.

The later [six-site qualification](../mp2-six-site-20261106/README.md) supplies
final measured retention dispositions and matched helper controls; the scoped
claims and measurements in this earlier snapshot remain historical.

- The actual common-pullback probe compares the prior compact-first-only source,
  the new compact second buffer and the full staged fallback. The first-only
  source is uninstrumented, so its accumulation counters are explicitly unknown.
- Complete H2/LiH/water cold and warm prepared E+F calls use one CPU-pinned ABBA
  process ordering. Preparation and execute timings remain separate. The small
  sample population and non-default disabled packaged-derivative configuration
  do not qualify a speedup or representative method-wide profitability.
- Independent PySCF 2.14.0 references use the exact native STO-3G primitive
  inventory, not its differently rounded built-in basis. Every sample passes
  the declared energy, full-force and net-force gates. Baseline/candidate
  observables match exactly. Whole-endpoint measured allocation peak is unknown.
- This initial CPU qualification claims no GPU runs, scientific payload lineage
  or complete wait coverage. The separate scheduled n4 follow-up is retained in
  `benchmarks/results/residency-n4-20261106/`.
  Both #1626's full per-site dispositions and #1629's actual-path residency
  qualification remain incomplete.

Executable component/reference coverage lives in
`tests/python/test_mp2_derivative_pullback_probe.py`,
`tests/python/test_mp2_cpu_reference_source.py`, and
`tests/python/test_mp2_gradient_fast.py`. Durable rationale and scoped remaining
work are in
`.agents/notes/implemented/performance/2026-11-06-compact-factorized-mp2-second-stage.md`.
