# Scoped automatic cold-HF qualification and rejected DFT experiment

This capsule separates the retained HF policy from the rejected force-channel
reassociation. It does not claim that every HF/DFT method beats GPU4PySCF or that
one fixed-order cold pair establishes a statistical ranking.

## Reproduce the offline gates

```bash
PYTHONPATH=python:. python benchmarks/results/hf-cold-defaults-20261107/replay.py
PYTHONPATH=python:. python -O benchmarks/results/hf-cold-defaults-20261107/replay.py
PYTHONPATH=python:. pytest -q tests/python/test_hf_cold_default_evidence.py
```

`cohort.json` binds three distinct normal Release source/ELF images, actual
scheduler allocations, compilation/cache/restoration receipts, and checksums of
separate compressed per-card populations. `summary.json.xz` is a derived reduction,
not a replacement for raw measurements. Production-source fingerprints are not
required-CI spelling checks; these identities describe immutable historical
executable measurements.

## Accepted HF boundary

The final image includes master `f27bdb3b0` and no DFT reassociation. Automatic
admission covers large singleton through-d restricted FP64 bounded Direct HF
owners. A screened proposal has at most eight iterations, then a fresh target
DIIS/energy epoch and the strict physical AO-maximum residual gate. Both epochs
share the caller's iteration allowance. The exact reusable task page has finite
memory/capacity bounds and retries the incumbent page before upload/capture if
the larger allocation is refused. Generic Direct/KS policy is unchanged.

Each card supplies six independent HF/default/lifetime/allowance/page-refusal
cases, sixteen independent Libcint signed J/K cases, a raw-AD/numerical-fallback
probe, and four-method complete cold-plus-three-warm negative controls. 5090
also supplies memcheck and racecheck. HF observable gates are absolute energy
`1e-8` Eh and maximum force `1e-7` Eh/Bohr, including default requests; public
controls remain energy `1e-9`, density `1e-5`, screening `1e-12`. The external
GPU4PySCF oracle never seeds native SCF or enters its timed production process.

All timers include energy, analytic forces and host export. The first cold call
is retained; warm calls freeze the owner's own cold density. Full Fock counts
include proposal and target work, not just reported target iterations. Paired
DFT controls retain energy/force, J/K and VV10 semantic work; the reader rejects
a protected cold or warm ratio above 1.03. These one-owner screens do not prove
universal non-regression for every system size or GPU.

HF has valid full Fock-stage counts, but its pre-existing fine-grained precision
operator inventory reports `complete=false` and `operator_inventory_complete=false`
in both arms. Preserve these flags: this is not delivery of a complete primitive
precision audit. DFT negative controls retain their complete operator inventories.

## Rejected and adverse history

`rejected-*.json.xz` contains the combined HF-plus-DFT image, actual LSF 227637
and Slurm 16671. The DFT mapping change is mathematically qualified but yields
essentially no r2SCAN-3c endpoint benefit on either card. It is removed, not
retained behind a new opt-in, and its image cannot be relabeled as the final
HF-only executable. Its exact compiler patch and original A100 archive checksum
remain in `cohort.json`.

`history-*.json.xz` preserves the earlier strict-tail/incumbent-page experiment
(A100 227478, 5090 16661), including the slight 5090 cold slowdown, and the
isolated existing explicit-page experiments (227514/16669). These are not final
automatic-default measurements. The A100 history also retains the repaired
independent r2SCAN-3c reference; the earlier missing-`libmvec` reference failure
was never a valid performance observation. The older failed default-force
diagnostic remains in
[`cold-hf-defaults-20261026`](../cold-hf-defaults-20261026/README.md).

The host-only intermediate images are not GPU-qualified populations. Real GPU
work runs only under finite LSF/Slurm allocations with assigned device visibility
preserved. Private runtime repair and transfer services do not alter shared
system libraries or production chemistry.

See issue #2976, the current
[HF force contract](../../../docs/developer/hf_force_finalization.md), and the
[rejected DFT decision](../../../.agents/notes/rejected/2026-11-07-density-force-relative-map.md).
