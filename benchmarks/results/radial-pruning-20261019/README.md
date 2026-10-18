# Qualified radial-pruning defaults, 2026-10-19

This is a water-96/STO-3G qualification, not a whole-README ranking or VV10
performance claim. `summary.json` retains synchronized samples, work counts,
independent-reference errors, source identities and integrity information.

| Engine / method | Old/unpruned (s) | New/pruned (s) | Before/after ratio |
| --- | ---: | ---: | ---: |
| GenerativeQC LDA | 23.158432 | 21.044082 | 1.1005x |
| GenerativeQC PBE | 24.355397 | 14.995148 | 1.6242x |
| GPU4PySCF LDA | 8.611896 | 6.591725 | 1.3065x |
| GPU4PySCF PBE | 10.201068 | 7.813512 | 1.3056x |

These are medians of six alternating repeats after one complete warmup per
candidate, with fresh owners and no converged-state reuse. Timers include
construction, preparation, SCF, analytic forces, cleanup/GC and synchronization.
Slurm jobs 10323/10324 each used one RTX 5090 on node1/main with a finite
20-minute allocation and scheduler-assigned visibility; host/BLAS threads were one.

The new native maxima are 64/20/40 (LDA) and 64/16/32 (PBE), followed by
`radial-v1` angular pruning. The previous explicit unpruned policy remains
54/16/32. GPU4PySCF uses its own level-3 radial/Lebedev grids, with actual NWChem
pruning, not the native quadrature. Both engines' pruned candidates pass common
energy and maximum-force gates against independently stable level-8/9 grids.

The error budget remains 1e-6 Eh per water and 5e-5 Eh/Bohr for LDA, 1.5e-6 Eh
per water and 3e-5 Eh/Bohr for PBE. Final qualification also reserves margin for
observed dense-reference variation; this is not a rigorous error bound. The
old native baseline fails that conservative acceptance rule, so its ratio is a
before/after-default timing, not a comparison of two equally qualified grids.
Both GPU4PySCF candidates qualify. GPU4PySCF remains faster on these endpoints.

The unchanged raw v1 reports classified candidates using raw errors. The
retained summary applies the stronger v2 margin rule separately, without
rewriting original measurements. Raw reports, source capsules, scripts and the
verified 357,965-byte evidence archive remain in the ignored local
`.artifacts/pruning-gpu-20261019/` directory; their hashes are in the summary.
This is not a public hosted archive. No Release or external upload was created.

Reproduce with the source-matched Release CUDA library, optional GPU4PySCF
dependencies, a verified compiler cache and a finite allocation:

```bash
export GENERATIVEQC_LIBRARY=/path/to/libgenerativeqc.so
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 python -m benchmarks.grid_pruning_gpu \
  --atoms 96 --repeats 6 --output .artifacts/pruning96.json
```

The [decision note](../../../.agents/notes/implemented/numerics/2026-10-19-qualified-pruning-defaults.md)
retains rejected lower-resolution candidates, the resolved GPU ledger test
issue, build-dependent CPU-AOT test limitations and conditions for expansion.
