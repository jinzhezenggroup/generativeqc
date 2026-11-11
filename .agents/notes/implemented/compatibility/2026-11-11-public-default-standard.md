# Decision: public precision defaults select the measured Standard endpoint

Status: implemented
Date: 2026-11-11

## Problem

The README benchmark explicitly selected Standard, and ordinary Calculator
already used the shared native Standard controls. However, the public
PrecisionCalculator facade and a default-constructed PrecisionPolicy still
selected Balanced. Balanced requests an independent observable audit, adding
two or three complete endpoints. A user omitting the policy therefore did not
execute the measured endpoint, even after HF/KS convergence defaults aligned.

## Decision

Both public precision constructors default to Standard. The README Standard
arm now omits the policy argument instead of supplying a benchmark-only choice.
Method, device, orbital representation/basis and explicit Direct/DF auxiliary
inputs still identify the compared physical model. Convergence, initial guess,
grid, scheduling, precision profile and optional environment selectors are not
overridden by the benchmark's Standard constructor.

Balanced, Strict, audit_auto and explicit targets remain opt-in audits with
their complete comparisons and acceptance gates unchanged. Consumers that
intentionally test these audits now request Balanced explicitly. No numerical
gate, GPU4PySCF timed control, final-state proof or resource fallback changes.

## Evidence and invariants

The permanent public-boundary regression exercises all eight README methods
in Direct/DF mode with both an omitted policy and PrecisionPolicy(). It checks
the same resolved controls as the benchmark constructor and one FP64 endpoint
without inventing observable-budget evidence. Existing explicit-audit fallback,
invalid-reference and independent native-oracle tests retain their full work
and acceptance requirements.

The policy implementation participates in the complete source identity even
though no C++/CUDA numerical implementation changes. A new clean source/image
and complete benchmark acquisition are therefore required for the replacement
publication; the earlier source-bound campaign remains retained and is never
relabeled. This default correction does not establish a causal timing gain.

## Source-matched qualification

The new frozen source is `a50c80f627455a801e1e2196b26259e64e5e9de2`, complete
source identity
`f502f9e5471d7de71bc41098273f059a6ec340852dbbc8cff1cb1a47a33edacf`, ELF SHA256
`4e4315e0d8317566185f45806b81feebb7ba1ab90ef0bbcc3f19145124197509`.
The verified sccache 0.16.0 build retains Release/SM120, fast compile OFF and
device link ON without clearing caches or overriding source identity.

On this exact image, 724 host/resource/default/precision cases pass with the
same 17 explicitly optional skips, all 47 DF final-state/eigen/projection cases
pass, and all 30 independent complete E/F cases pass in one coherent run.
Actual ordinary/captured Direct/DF CUDA reductions and block/refinement controls
also pass. The pre-publication offline suite passes all 512 cases. The
temporary old-image host probes and their setup/fixture failures remain
separate; they are not substituted for these current-image qualifications.

Every measured process uses an unset precision-profile environment selector,
in addition to unset RHF/KS convergence and XC-order selectors. Worker receipts
retain only the selected library deployment path and the declared common
single-thread math environment, not hidden performance or numerical switches.
Complete acquisition, comparison and source/build/cache receipts are retained
under ignored `.artifacts/readme-public-default-20261111/` and the matching
`n1:/data/jzzeng/readme-public-default-20261111/` tree.

## Complete replacement publication

The predeclared replacement acquisition completes 128 processes / 512 complete
E+force calls with no failures or retries. Direct has 32/32 cold and warm wins;
DF has 31/32 cold and 32/32 warm wins. All 58 independently accuracy-qualified
pairs remain faster in both phases. The same six independent fitting/grid
failures remain visible; all 1024 full-vector cross-engine comparisons pass
without replacing those separate gates. Warm Direct WB97M-V96 speedup is
9.385239135857269.

The single cold loss is DF HF96: native 17.860284987837076 s at 11 iterations /
12 Focks, versus GPU4PySCF 17.327834360301495 s at 10 iterations / 11 effective
potentials. Native warm samples are 3.72396944090724, 3.710203818976879 and
3.716286364942789 s, versus 6.850647389888763, 6.979983024299145 and
6.970473736524582 s. Native target work remains unchanged from the preceding
source cohort, whose single cold observation happened to win. Do not select
that earlier 64/64 cohort in preference to these public-default measurements
or infer a causal policy regression from cross-cohort cold timing noise.

The prior publication remains recoverable at
`2a7f52f7e475297246be8ab80ce3596518cc3409` and its original
`.artifacts/df-hf-aligned-final-20261111/` / matching node1 artifact root.
The older 63/64 and 56/64 cold publications remain at
`e25b70c24d8464c32444bec25bccd33d2f494795` and
`28ac57e819405bd9bae06e8fae60ac25b98d14a1`. The initial `0b74eb57f` source
cohort remains in `.artifacts/df-hf-aligned-20261111/` and matching node1 storage.
These historical recovery pointers live here rather than growing the bounded
current benchmark summary; no raw observations or unrelated evidence are removed.
The first publication bookkeeping commit exceeded the fixed aggregate evidence
budget by 799 bytes. Compacting only this campaign's summary restores the cap;
the numerical capsule, plots and complete endpoint population are unchanged.
The earlier same-image strict/default HF ablation is unchanged and remains
distinct from either complete README cohort.

After replacing the publication, all 512 offline tests pass again. Both normal
and optimized Python replay the complete vector audit and compact scalar
publication, reproducing each tracked SVG byte-for-byte; both charts are
visually inspected. The new-image HF48/96 comparison against retained tight
same-fitted-model oracles also passes, with maxima E/atom 4.4858931384320994e-12
Eh, Fmax 5.2886742096802664e-8 and Frms 2.5530112380328044e-8 Eh/Bohr across
eight complete endpoints. Default-policy inventory, native CUDA ownership,
source-identity and fixed-cap evidence checks pass; CBS bytes are unchanged.

## References

- `docs/user/application_precision.md`
- `python/generativeqc/precision_policy.py`
- `benchmarks/readme_defaults.py`
- `.agents/notes/implemented/numerics/2026-11-11-df-hf-orbital-stopping.md`
