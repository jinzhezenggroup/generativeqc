# Complete SCF XC tile qualification

`benchmarks.pbe0_xc_tile_pairs` compares existing public 256/512-point SCF
controls with the same tuned native binary, scientific grid/basis, FP64 policy
and fixed 256-point ordinary force policy. It does not enable a new provider,
change defaults, or implement automatic tile selection.

Each arm independently converges its native state and freezes updates through
the public warm-start API. Density bytes are not asserted identical. Setup and
priming are retained outside replay timing; every timed call must execute one
physical SCF iteration and Fock build without warm fallback. Solver histories,
whole forces, actual selected tiles, local-AO work and force telemetry are saved.

Run real GPU work on a finite n2 Slurm allocation, preserving assigned device
visibility. Use ccache for core and dynamic wrapper compilation. Provide an
independent GPU4PySCF reference generated for the exact protocol:

```bash
python -m benchmarks.pbe0_xc_tile_pairs --atoms 96 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
  --reference .artifacts/reference-96.json --repeats 5 \
  --output .artifacts/xc-tiles-96.json
```

The program compares original and moved warm geometries with five interleaved
A/B pairs each. Independent gates remain `1e-8` hartree for energy and `1e-7`
hartree/bohr for every force entry. The offline consumer independently checks
the shared greater-than-2%/relative-MAD descriptor, actual solver work, selected
SCF tiles and unchanged force phase visits/launches. This is not a confidence
interval or a cold/moved-start speed claim.

AO telemetry combines solve-local XC evaluations with one-traversal map counts
and retained geometry-discovery receipts. Do not charge the cached discovery
fields as repeated warm work or confuse AO-map reservations with endpoint peak.

```bash
PYTHONPATH=python:. python -m pytest -q tests/python/test_pbe0_xc_tile_pairs.py
```

The [retained publication](../../benchmarks/results/pbe0-xc-tiles-20261006/README.md)
binds the frozen source patch, source manifest, raw vectors/histories and recipe.
Only its exact RKS water/grid/basis/device cohort is qualified. UKS, other
functionals, high-occupancy/tiny domains and automatic resource-aware selection
need their own gates. Complete endpoint memory/uncached compilation promotion
is not inferred from replay medians; production defaults remain unchanged.
