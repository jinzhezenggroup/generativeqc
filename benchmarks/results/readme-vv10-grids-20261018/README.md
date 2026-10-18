# WB97M-V grid evidence

Supplementary 3/12-atom water/def2-SVP study on n1 RTX 5090 under finite Slurm:
shared-grid native/reference E+F, 2026-10-18, master `2a62a17e8`;
reference-only split-grid feasibility, 2026-10-19. No defaults or other results change.

[Protocol, gates and reproduction](../../readme_grids.md), [timings/work](grid-timings.md),
[shared-grid convergence](grid-convergence.md).

`manifest.json` binds `grid-timings.json.xz`: all 192 shared-grid endpoint samples,
force arrays and semantic counters; `split_grid_feasibility` holds 20 reference
population summaries and raw hashes, not native speedups. Full journals remain at
`n1:/data/jzzeng/readme-{vv10-grids-20261018,split-vv10-20261019}` and ignored
`.artifacts/vv10-grid`.
