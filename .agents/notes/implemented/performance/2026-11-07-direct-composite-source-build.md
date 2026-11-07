# Decision: construct coefficient-bound exchange sources directly

Status: implemented
Date: 2026-11-07

## Problem

PR #3030 combines a complete GPU LR source with a separately built Full source.
It saves repeated K applications but still constructs two complete canonical
inventories and requires a second approximately 15-GB temporary source in Direct96.
The retained RTX5090 inventory is complete; its fallback is not evidence that
disabling LR caching would help. Removing that transient allocation may admit a
composite on smaller resource envelopes without changing retained storage.

## Decision

Bind exact signed Full/LR exchange terms before constructing an unpublished strict
restricted GPU source. The canonical rows, screening, omega and persistent FP64
allocation remain unchanged. The compiler-owned order-3--9 whole-shell consumer
adds Full and LR radial roots, then shares one Coulomb recurrence and component
contraction. The existing native radial owner computes both seed families; no
second scientific recurrence algebra is introduced. Low/high-order component
fallback borrows the existing paired evaluator: it is not claimed to have only
one radial recurrence.

Publish only after complete finite auditing. Optional capacity refusal preserves
the raw source/fallback contract; numerical or driver failures propagate. Partial,
mixed, DF, incremental, graph-chunk and application-ledger owners are not extended.
Immutable coefficient/operator/spin identity continues to guard all borrowers.
The conservative source allowance includes bounded temporary inverse metadata;
retained device bytes remain the one source plus incumbent plan/scratch.

`GENERATIVEQC_KS_COMBINED_RANGE_BUILD=0` selects the incumbent two-producer route.
Its separately admitted transient allocation, pre-mutation raw fallback and fatal
post-mutation failures remain intact. `GENERATIVEQC_KS_COMBINED_RANGE_SOURCE=0`
still restores raw LR replay. Neither comparison disables effective LR caching.

## Scientific and work invariants

- Strict FP64, identical basis/order, grids, screening and convergence semantics.
- No CPU/PySCF work enters production. Independent CPU oracles are validation only.
- Complete cold includes construction, preparation, SCF, analytic forces and export.
- Warm is the first same-owner call from its own frozen cold density.
- Count constructed resident outputs separately from both represented radial terms;
  whole-shell and paired component fallback do not have identical primitive work.
- Count actual K contractions. Analytic-force and raw profiling owners are unchanged.
- No GPU-name dispatch or unbounded allocation is introduced.

## Rejected alternatives

Do not disable complete LR residency to manufacture a streamed fusion opportunity.
Do not retain the second complete source merely to simplify the optimized producer.
Do not substitute half precision, altered grids/stopping criteria or a CPU guess
for construction work reduction. A new composite must not be exposed to raw LR/J.

## Evidence

### Original pre-master cohort

Native identity: `56034cd22620e5ded0fd231556609642ee7595cebfe34f98961ff9b3822bc478`.
SM80/SM120 Release ELF: `1c193f6ae50516613de16c2da1c5662d771b4edfd644e1256309f4e6cb1d37ff`.
Same-image rollback (`BUILD=0`) versus genuinely unset default, WB97M-V/full Direct96:

| GPU / allocation | Incumbent cold | Default cold | Incumbent warm | Default warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226550 | 89.088205 s | 83.401132 s | 13.842545 s | 13.903494 s |
| RTX5090 / Slurm16639 | 178.695741 s | 167.787624 s | 13.652372 s | 12.697615 s |

A100 saves 5.687073 s / 6.38% cold. Its first warm is essentially unchanged
(+0.061 s / +0.44%), as expected for construction-only work with the same K replay.
5090 saves 10.908117 s / 6.10% cold and 6.99% first warm. Both retain 16 cold
iterations/Focks, 11 provisional + 5 target Focks, four cold VV10 sweeps, one warm
Fock and zero new warm sweeps. A100 resident outputs halve 3,746,443,432 ->
1,873,221,716; 5090 retains 1,873,221,716 outputs but K actions halve 32 -> 16 cold
and 2 -> 1 warm because it no longer needs the second-source transient allocation.
The approximately 14.99-GB retained source and its 16-GiB ceiling do not grow.

Maximum paired energy discrepancy is 3.79e-14 Ha/atom and force discrepancy
1.01e-12 Ha/Bohr. All complete endpoints pass the retained independent E/F oracle;
no fresh gpu4pyscf latency ranking is claimed. The offline reader additionally
checks the prior qualified default, so a slower rollback kernel cannot manufacture
a same-image performance win. Those historical observations are not relabeled.

Independent host radial/reachability/derivative quadrature passes 184 cases, including
signed/cancelling roots through ffff and omega zero/finite/large. Slurm16639 passes
six independent signed/displaced/f-shell fixtures with zero skips and memcheck
reports zero errors. A100 independently passes the 96-AO composite and usable
raw-fallback gates. Incomplete, wrong-spin, nonfinite-coefficient and wrong-omega
requests all retain independently checked usable raw LR values.

Source 3f90e9d6419f was superseded before GPU qualification after a host-only finite
lookup and formatting correction. Its owned build was interrupted with exit130;
the borrowed original runtime was restored/hash-verified. It has no timing claim.
The final candidate is 56034cd22620e5ded0fd231556609642ee7595cebfe34f98961ff9b3822bc478.
Retain failed harness attempts separately rather than relabeling images or skips.
Slurm16638 passed the new signed composite matrix gates but failed three added
raw-fallback tests: the fixture accidentally requested derivative order one.
Changing that probe request to order zero fixes the gate without changing native
identity or ELF. No endpoint ran in the failed allocation. The verified A100
download's owned SCP EOF wait was retired only after authenticated remote/local
SHA256 agreement; the finite transfer service was stopped after receipt recovery.

### Final master synchronization

Master `dafa29f83` (#3016, MP2/DF residency) arrived during finalization. It changes
the native source inventory even though the Direct scientific owners do not
overlap. Merge it through integration commit `2e85a854c`, rebuild normal Release
SM80+SM120 with verified sccache0.16.0 (99.68% cache hits), and qualify a separate
final image rather than relabeling the original measurements. The frozen source is
`ea83781aafef1d8e4f8159033ce12f2aed096cb7aaeb07de397d71da32ee9a13`, ELF SHA256
`e27f500d7bba457002cec893fb7762ee6a411a84846843525702082f4c97428f`.
The borrowed original source/runtime is restored and hash-verified after the build.

| GPU / finite allocation | Incumbent cold | Default cold | Incumbent warm | Default warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226634 | 88.980278 s | 83.334962 s | 13.913736 s | 13.918371 s |
| RTX5090 / Slurm16640 | 178.994173 s | 168.198248 s | 13.645767 s | 12.760090 s |

Final A100 complete cold saves 5.645316 s / 6.34%; warm changes +0.004634 s /
+0.03%, not a warm speedup. Final5090 saves 10.795925 s / 6.03% cold and 6.49%
first warm. All retained iteration/Fock/stage/VV10/source-output/K-work gates stay
unchanged. Largest paired E/F discrepancies are 3.79e-14 Ha/atom and 1.48e-12
Ha/Bohr; the independent full E/F oracle, six zero-skip signed/displaced/f-shell
GPU fixtures, A100 raw/composite refusal gates, and zero-error memcheck pass.

The reader replays both separately identified cohorts and checks their integrity
under Python `-O`. Retain the original `pre-master-evidence.json.xz` and its
failures. A local host-test selection initially included seven CPU diagnostics
without a local native library; retain that unavailable-runtime log, do not call
it scientific qualification, and use the actual scheduled endpoint J/K work gates.
The final A100 download also required retiring its owned SCP EOF wait only after
authenticated complete remote/local SHA256 agreement. The LSF job is DONE, Slurm
is COMPLETED/0:0, and both owned finite transfer services are stopped.

### Reviewed dependency integration after paired measurement

The active PR #3030 ref had advanced to `ac66f7fab`; refresh active dependency refs
as well as master before freezing future cohorts. Do not infer a dependency SHA
from a branch name or cached tracking ref. A non-fast-forward push was rejected
safely; preserve that independently updated branch rather than force-pushing it.

Merge `768f4a514` retains its four-line compact-mask guard in the legacy optional
combination wrapper and the independent compact raw-LR oracle alongside the new
composite oracle. This changes the inventory to
`0644bc072403a966f646672cabdd8210cff351bacf1be1eb812781db9b14e60a`, ELF SHA256
`2dd1887b8e29f21bdec8d2b2764fc6835df29193c9e6b17b1035c2e710c2215f`.
The normal cached rebuild records 99.89% hits and restores the borrowed runtime.

The guard is outside the qualified complete-composite domain. Preserve the EA
image's paired performance evidence byte-for-byte; use separate default-only
complete A100/RTX5090 endpoints, independent compact/matrix/force gates and a
historical default non-regression guard for the integrated source. Do not invent
another rollback pair or relabel old timings as the later image. The dedicated
`replay_dependency.py` reader keeps this boundary explicit under Python `-O`.

The integrated default records A100 LSF226710 cold82.588231 s / warm13.920897 s
and RTX5090 Slurm16641 cold168.205025 s / warm12.722292 s. Independent full E/F,
signed/displaced/f-shell and compact raw-fallback matrices, zero-error memcheck,
default-unset controls and unchanged 16/1 Focks, VV10 and source/K work pass.
Do not recompute the paired improvement using these later default-only timings.
Both default-only allocations complete successfully. Their finite transfer service
is stopped after the complete A100 download is authenticated and its owned SCP
EOF wait retired; no scheduler job or shared VPN/compiler service is terminated.

All 125 reviewed-dependency host regressions pass after updating three stale
host-double signatures/default declarations to the new optional argument. Exact
HF lifetime/retry doubles still explicitly reject nonzero composite scales. Retain
both fixture failure logs separately; these are test-only fixes, not native changes.

## Revisit when

Broaden only with independent numerical/resource and complete endpoint evidence.
Separate angular/class tuning or further radial fusion must preserve bounded
fallbacks, construction accounting and full cold performance on both GPUs.

## References

- Tracking issue #2976, comments 6701064222 and 6701277684.
- PR #3030 and `2026-11-07-coefficient-bound-range-source.md`.
- `tests/python/test_reachable_coulomb.py`.
- `tests/python/test_range_resident_values_cuda.py`.
- `benchmarks/results/combined-range-build-20261107/replay.py`.
