# Remdesivir-77 default endpoints and capacity boundaries

Retained evidence for a 77-atom drug-like molecule (remdesivir, PubChem CID
121304016, C27H35N6O8P) measured with the same
[`#2786`-style default framework](../readme-refresh-20261026/README.md) and the
same benchmark driver conventions as the water README cohort: one cold and three
frozen-density warm complete energy-and-force calls per fresh process,
GenerativeQC Standard against stock GPU4PySCF 1.8.1, Direct and DF, without the
public Fast arm.

The point of this cohort is a geometry the README water clusters do not cover:
a compact, chemically heterogeneous molecule with a third-row element (P) and
`spd` shells. It is a capacity and scaling observation, not a new accuracy
claim.

## Measured source

| Item | Value |
| --- | --- |
| Checkout revision of the driver and library | `0e33e02` |
| Measured revision of the published README cohort | `46b916ac` (identical source identity) |
| Native source identity | `2443a52d8f2c97de9d866c8856993641c5b926b7f29823cf8837c0998b787ac8` |
| Native library SHA-256 | `fa65de6814a6f45f907218a20af0dc86a3e2cc2950a9693f9152a946f5f22319` |
| Accelerator | NVIDIA GeForce RTX 5090, 33,668,988,928 bytes reported |
| CuPy / PySCF / GPU4PySCF | 13.6.0 / 2.14.0 / 1.8.1 |

A four-point rebuild check of the published water-96 cohort with this library is
retained in `build-check.json.xz`: warm medians agree within 2.6 % and HF cold
within 3.4 %, while PBE cold is 17.6–18.7 % slower than the published table.
Cold-start absolute values therefore should not be compared cell-by-cell with
that cohort; warm values are commensurate.

## def2-SVP, all eight methods

Speedup is GPU4PySCF latency divided by GenerativeQC latency (above one favours
GenerativeQC). Absolute values and per-repeat ranges are in [timings.md](timings.md).

| Mode | Method | Cold speedup | Warm speedup |
| --- | --- | ---: | ---: |
| DF | HF | 1.80× | 3.24× |
| DF | PBE | 1.12× | 1.75× |
| DF | PBE0 | 1.27× | 2.40× |
| DF | B3LYP | 1.17× | 2.00× |
| DF | r²SCAN | 1.15× | 2.60× |
| DF | M06-2X | 1.34× | 2.26× |
| DF | ωB97M-V | 0.36× | 0.63× |
| Direct | HF | 0.58× | 0.84× |
| Direct | PBE | 0.64× | 0.46× |
| Direct | PBE0 | 0.53× | 0.54× |
| Direct | B3LYP | 0.48× | 0.38× |
| Direct | r²SCAN | 0.71× | 0.65× |
| Direct | M06-2X | 0.62× | 0.48× |
| Direct | r²SCAN-3c | 0.89× | 0.88× |
| Direct | ωB97M-V | no endpoint | no endpoint |

- **DF favours GenerativeQC in six of seven measured methods (1.12–3.24×).**
  ωB97M-V is the exception: 2.8× slower cold and 1.6× slower warm.
- **Every measured Direct endpoint favours GPU4PySCF (0.38–0.89×).** This
  reverses the water-96 Direct ordering and is the main geometry effect here.
- r²SCAN-3c DF and ωB97M-V Direct have no GenerativeQC endpoint; see the next
  section.

## Capacity boundaries

`capacity-evidence.json` retains, per failure, the retained status, the verbatim
error, the campaign outcome record and the tail of the driver log. Failures were
not retried with loosened tolerances, removed forces, changed grids or a
different model.

| Endpoint | Observation |
| --- | --- |
| def2-SVP ωB97M-V Direct | `GENERATIVEQC error 7: out of memory` during generated forces; GPU4PySCF completes in 436.1 s cold |
| def2-SVP r²SCAN-3c DF | `GENERATIVEQC error 7: density-fitted stationary response exceeded the prepared resource budget` |
| def2-TZVP (1,518 AOs) | no GenerativeQC endpoint in any of 14 cells: immediate OOM, `std::bad_alloc`, DF metric budget refusal, 5,400 s point timeouts, or SCF non-convergence |
| def2-TZVP reference | GPU4PySCF completes 9 of 14 cells; its own DF path also raises `cudaErrorMemoryAllocation` for HF, PBE0, B3LYP, M06-2X and ωB97M-V |

A probe with `density_fitting_memory_budget_bytes` raised to 22 GiB also fails
for def2-TZVP HF DF (`CUDA runtime error`), so the automatic budget is not the
binding constraint at that size. The relevant device is a 32 GB part. This molecule has 767 AOs, essentially
the published 768-AO water count, yet its Direct ωB97M-V cell is refused while
the water cell admits: the trigger is system-dependent rather than a pure AO
threshold, so no AO cutoff should be inferred from this record.

Related retained capacity evidence elsewhere in this tree: the 768-AO
generated-force OOM in
[`pbe0-fast-scaling-20261013`](../pbe0-fast-scaling-20261013/README.md), the
ωB97M-V source inventory refusal in
[`vv10-range-followups-20261014`](../vv10-range-followups-20261014/README.md),
and the direct J/K arena policy in
[`#597`](../../../.agents/notes/implemented/performance/2026-09-21-direct-jk-target-resource-policy.md).

## Independent-accuracy status

No independent accuracy gate is introduced here: comparisons are cross-engine
differences at the same geometry and basis, exactly as the README cohort
qualifies its own cross-engine deltas. Relative to GPU4PySCF 1.8.1,

- energy differences are 7.7e-13 to 1.2e-11 Eh/atom for every pair except
  ωB97M-V DF, which is 1.18e-9 Eh/atom (9.1e-8 Eh total);
- maximum force-component differences are 1.0e-06 to 3.7e-06 Eh/bohr;
- force RMS differences are 2.5e-07 to 9.2e-07 Eh/bohr.

The ωB97M-V DF offset is consistent with the VV10 weight/density-fitting
conventions documented in [conditions.md](conditions.md); its magnitude is far
below chemical accuracy but two orders larger than the other pairs, so it is
reported rather than averaged away.

## Reproduce

```bash
python benchmarks/results/remdesivir-77-20261027/verify.py
```

`verify.py` reads only the six retained bundles and recomputes
[timings.md](timings.md), `timings.csv` and `comparisons.csv` in memory,
failing if any committed table disagrees. `--write` regenerates them.

The endpoints themselves were measured with
[`remdesivir_defaults.py`](remdesivir_defaults.py), a geometry and basis adapter
over `benchmarks/readme_defaults.py`, on the unmodified geometry in
[`remdesivir.xyz`](remdesivir.xyz). Commands and environment are recorded in
`provenance.json`.

## Retained files

| File | Role |
| --- | --- |
| `standard.json.xz`, `gpu4pyscf.json.xz` | def2-SVP records, one entry per method/mode |
| `tzvp-standard.json.xz`, `tzvp-gpu4pyscf.json.xz` | def2-TZVP records including failures |
| `repeat.json.xz` | four Direct cells re-measured in a second independent process |
| `build-check.json.xz` | publication-parity check against the water-96 cohort |
| `provenance.json` | source identity, library hash, environment, commands |
| `capacity-evidence.json` | per-failure status, error, outcome and log tail |
| `timings.md`, `timings.csv`, `comparisons.csv` | rendered tables |
| `verify.py`, `remdesivir_defaults.py`, `remdesivir.xyz` | verification and drivers |

These are descriptive observations from one fresh process per population, as in
the water cohort; they are not confidence intervals and do not establish a
universal ordering across geometries.
