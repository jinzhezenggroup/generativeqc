# Direct relative-response contraction: PR #2558

Baseline #2556: `523499b09750d2922548fc4168b671147d5d731d`, including #2541/#2549.
Candidate: `6526c392711b55bf1a105a8cd5d75746a2e753b0`. Native identities:

- Baseline `730c1f1c4f1d334af6749d5b8b966fb3c7b48aa689542b7e8e9449d89fc800c1`.
- Candidate `30657c71068e5eb14ce73b33256571ea9442512e4ee2ace2e8d2ac2d40beb201`.

Four native inputs change;1754 total. Every call verifies its native identity.
Stacked-source comparison, **not current-master timing**; #2553 excluded.
Exact Direct FP64 J/K, screening, convergence, canonical
def2-mTZVPP (544/1088 spherical AOs), moving 48×16×32 Becke grid per atom,
original references and 1800-second population limits remain unchanged.
PBE0/HF retain def2-SVP. Slurm job1645: one assigned node5 RTX5090,90 minutes.
No visibility/clock/tolerance/cache override, trimming or prewarming.

## Complete E+F wall time

Eight complete populations,96 calls: cold + five warm + moved + five moved-warm.
Five-call warm medians; cold/moved include preparation; positive means regression.

| r²SCAN-3c/96 | Baseline s | Candidate s | Change |
|---|---:|---:|---:|
| Cold |383.888035|373.634581|-2.671%|
| Warm |62.350698|51.800627|-16.921%|
| Moved |232.498299|221.635959|-4.672%|
| Moved-warm |62.499865|51.678825|-17.314%|

| Control | Cold s | Warm s | Moved s | Moved-warm s |
|---|---|---|---|---|
| r²SCAN-3c/48 |68.904710→66.426167|14.402611→11.680185|50.309942→47.564831|14.428178→11.685657|
| PBE0/48 |22.781601→22.521541|4.274582→4.054773|22.443889→22.335013|4.272208→4.070798|
| HF/48 |11.183760→11.179372|0.749126→0.749382|7.265011→7.271700|0.749186→0.749792|

Matched cold/moved iterations/builds:96 24/24,13/13;48 19/19,13/13;
PBE0 17/18,12/12; HF21/22,16/17; all warm1/1, no normalization.
PBE0 cold -1.142%: single observation; historical +3.06% retained.

## Source-matched diagnostic, separate device times

| Warm device group | Baseline s | Candidate s |
|---|---:|---:|
| SCF Direct J |8.305609|8.277665|
| XC/AO/grid |4.546549|4.546148|
| Order6 |13.804019|13.865095|
| Order7 |7.363464|2.018463|
| Order8 |5.810217|0.455951|
| Other stationary force/Becke |9.404342|9.405148|

Order6 arithmetic unchanged (+0.442% variation); order7/8 decrease72.588%/92.153%.
Enumeration/admission/publication unchanged. Actual executed work:

| Order | Bra preparations | Ket preparations | Contractions |
|---|---|---|---|
|7|2606140→1027752|3147016→1250544|1937135584→754811008|
|8|231424→95976|231424→95976|270222624→107642016|

Primitive products remain2400504912/1250544/95976 for orders6/7/8; Coulomb
preparations remain1250544/95976 for7/8. Counters measure executed preparation
and contraction invocations, **not FLOPs**: one shared nine-response basis
replaces per-atom contractions. Hardware FLOP counters: `ERR_NVGPUCTRPERM`.

Mutually exclusive warm wall owners (not device times):

| Owner | Baseline s | Candidate s |
|---|---:|---:|
| SCF/publication |12.946022|12.917465|
| Stationary force/publication |43.782045|33.158927|
| Final snapshot |4.233840|4.229824|
| Final validation |0.588381|0.589387|
| Other exclusive owners |0.840837|0.835910|
| Complete diagnostic |62.391125|51.731514|

Cold/warm wall/device ledgers reconcile separately. Asynchronous order markers
share host preparation/synchronization owners; do not add kernel times. Cold J/XC
device times are199.918086/109.108998→199.662481/109.110204s; preparation
8.823115→8.797689s. Cold remains SCF-dominated, not planner-dominated.

## Correctness, resources and limits

96 primary +4 diagnostic calls pass independent unchanged gates
1e-8Eh/1e-7Eh/Bohr. Maximum errors:3.865352482534945e-11Eh and
6.014522213604323e-11Eh/Bohr. All populations finish below1800s.
47 host tests,542-module compiler audit and6 linked Libcint GPU tests pass.
Jobs1024/1646: signed RKS/UKS J'/K' fixtures, memcheck/synccheck zero errors.
29160 host responses: AD error1.11022e-15; displaced-value error1.014e-12.
Tests check bounded work, not exact internal calls; scientific assertions stay.

Order7 registers214→173, dynamic shared42984→27176bytes; order8
224→150 and26888→11080bytes. Nsight reports local bytes/thread0 for both.
32/44-KiB bounds, fallbacks and Gaussian algebra remain. No method/molecule/AO/GPU
hardcoding. Default on in this unmerged branch, not master; no selector promotion.

Rejected order6 polynomial/DAG trials are not retained; no scheduling-only rerun.
Optional #2556 PBE0 force-spy tests fail after passing numerical assertions;
not claimed fixed/passing. The implementation Agent Note records full details.

## Retention and next work

Lossless float64 arrays/references, every timing/gate/count, source/protocol,
convergence trajectories and matched diagnostic ledgers/resources/census remain.
Raw-only: component-energy decompositions and unprofiled force-wall detail
(hash-bound). `verify-arrays.py`: all100 gates; `--restore-references`: oracles.
`build.sh`: verified cache/all AOT. Reproduction: immutable drivers, Fock wrapper,
reducer/analyzer, toolchain/cache/sanitizer receipts and raw hashes.

Only this task's packet is superseded. Previous packet at this path in
`11ea887b1606c67062b0542a2ac84434abf07c37` links older #2541/#2549 evidence.
No cap increase, foreign deletion, Release, tag or merge.

51.800627s remains1.95× historical GPU4PySCF26.59s (not a new competitor run).
Next cold: budgeted exact density-first J reuse with original AO screening and
fallback. Existing layout~188MiB exceeds128MiB cap; no ledger/cap bypass.
