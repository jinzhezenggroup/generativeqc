# Decision: compose range-DF with current source and grid ownership

Status: implemented
Date: 2026-10-23

## Integration boundary

The benchmark selector repair in `ca5c605a6d5ad72147cc9edd81fa9d8a01bffd62`
could not receive current CI because PR #2526 conflicted with master
`bbd41a427`. The merge retains both the opt-in LR operator and master's newer
independent-grid, source-only Coulomb, auxiliary-g and derivative-only resource
contracts. Neither the default LR selector nor numerical acceptance gates change.

## Invariants preserved by the resolution

- Generated shell derivative consumers remain full-range only. LR derivatives
  pass omega through the generic consumer, including six-term auxiliary g.
- Explicit packed responses require both full metric rank and the existing
  non-g occupied-owner domain. Rank-truncated response still differentiates
  whitened factors and reserves its additional complete whitened tensor.
- Source-only raw Coulomb response retains its full-rank guard. Its new callbacks
  cannot select the rank-truncated whitened response accidentally.
- Exact LR derivative owners use master's derivative-only capacity purpose;
  fitted LR owners do not acquire an unnecessary exact LR allowance.
- Native LR approximation and independent-grid proofs both contribute to
  snapshot identity. Detached host fixtures must bind decoding methods to the
  cloned owner, rather than retaining methods bound to the original fixture.

Master also changes the declared VV10 molecular density cutoff to `1e-10`.
The tiny native API test's fitted LR energy references were independently
reconverged with PySCF 2.14.0 / Libxc 7.0.0 at that explicit cutoff using
`tools/verify_mixed_df_rsh_cpu.py --vv10-density-cutoff 1e-10`. The fitted WB97M-V
RKS/closed-shell UKS value is `-1.284926849692006` Hartree and the doublet value is
`-0.8133366861625834`. The original `2e-10` native API gate remains unchanged.
The reference CLI now parses both the cutoff and LR selector together.

## Evidence retention

Combining the existing PR receipts with current master exceeded the unchanged
64 MiB tracked-evidence budget. The two PR-owned xz bundles now restore from
existing Git revision `37a7ca142064d65e2fca3411588b6eeb85c6ca8f`, using the
repository's offline verified restoration tool and a pinned size/SHA-256
manifest. Encoded and decoded digests and all 149 original members were verified.
Historical negative observations and scientific records are unchanged; no new
external archive or Release was published. Both campaign READMEs document the
restoration commands and the stable-response verifier accepts the restored path.

## Validation boundary

The merged CPU library passes its native DFT API contract, including the
independently regenerated energies. The combined benchmark/Hessian/host response
run passes 99 cases with 34 explicit dependency/device skips. The host capacity,
packed-response and snapshot checks pass 440 cases; the six cloned-owner fixture
cases also pass after rebinding. Real-GPU and independent PySCF endpoint results
are recorded separately in the PR as they complete; CPU checks alone do not
qualify the merged CUDA implementation or any new performance claim.
