# Independent spherical Ni second-order references

These are permanent, offline scientific inputs for the opt-in installed endpoint
qualifiers in `tools/qualify_global_hybrid_ni.py` and
`tools/qualify_global_hybrid_ni_grid.py`, not local native restart checkpoints.
`manifest.json` records shapes, dtypes, units, exact input hashes, independent
versions, frozen gates and original producers. No array requires pickle.

The basis is pinned by `../manifest.json`. The explicit grid has 5760 points;
the native default Treutler/Lebedev grid has 59872. `reference-matrix.npz`
contains all 15 independent analytic-gradient-difference columns at three
steps, without symmetry repair. `reference-actions.npz` contains three
independent moving-gradient HVPs. `reference-components.npz` partitions six
independent gradient groups. `reference-center.npz` separately saves an
independently converged central state. `default-reference.npz` is a separate
three-step moving-gradient oracle for the default quadrature.

Grid primals are quadrature inputs, not electronic-state seeds. Some historical
files include native density/energy **comparison receipts**: the `density`
member in `reference-actions.npz`, and `native_*` members in `default-grid.npz`.
Those are not independently generated oracle densities and must never seed an
independent SCF. The actual independent central densities are retained in
`reference-center.npz` and `default-reference.npz`.

The compressed single-file producers preserve their exact historical source
bytes, including failed independent guesses and bounded independent Newton
retries. They use the original `/tmp/gqc-2843-2849` transport paths. Audit or
regenerate them in a fresh sandbox rather than overwriting retained evidence.
Their finite differences and PySCF calls are validation only; neither enters
the installed production Hessian.

Native acceptance receipts and negative qualification history belong to
`benchmarks/results/global-hybrid-public-20261105`, not to these oracle inputs.
The complete production-scale 25–40-atom Ni gate remains unresolved under #2843.
