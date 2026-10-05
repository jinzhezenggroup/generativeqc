# Decision: do not promote primitive-local root reuse from this campaign

Status: rejected for promotion on the retained evidence
Date: 2026-10-05

The [complete follow-up records](../../../benchmarks/results/pbe0-force-followups-20261005/README.md)
preserve a source-matched compiler experiment that caches reached Cartesian
Coulomb roots within each order-4/5/6 primitive gradient. Root arithmetic,
subset accumulation order, scientific screening and precision remain unchanged.
Independent host/full-LR native gates and device sanitizers pass.

Warm complete E+F improves 3.04% at 48 atoms and 3.80% at 96 atoms. However,
48-atom moved regresses 5.80% with builds 12 to 13, and 96-atom moved regresses
30.04% with builds 14 to 20. The latter dominates the isolated warm benefit.
The measured implementation is retained as a reconstruction patch, not shipped.
This decision rejects promotion under current evidence; it does not establish
that caching force roots caused the SCF trajectory changes.

A separate force-map cold diagnostic initially reused generated artifacts
between processes. Its first arm paid about 26 s of owner setup while later
arms paid about 3 s, making cold medians misleading. The repeated campaign
uses a fresh generated-artifact cache per process and retains compiler ccache.
Its positive observations do not erase the earlier regressions or prove the
profile's interpolation assumptions. Both cache protocols and all rows survive.

Revisit root reuse only with a changed mechanism or an adequately instrumented
complete endpoint campaign that resolves cold/moved behavior. Capture actual
SCF histories and final residuals; never normalize time by iterations or
rewrite missing historical measurements. The source-level root count reduction
is not an observed molecular primitive count or an endpoint speed guarantee.

The separate Wick-subtree pruning experiment starts from the unmodified
composition so it can be judged independently of this root table. Neither
experiment revives the rejected order-7/8 scalar-center alternative or the
thirteen-pass angular schedule as a production default.
