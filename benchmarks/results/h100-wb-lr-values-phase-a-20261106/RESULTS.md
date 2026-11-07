# WB-LR phase-A standalone result

**PASS for the bounded standalone fixture.** This does not qualify native integration, molecular source replay, endpoint speed or the all30 goal.

Source commit `1fa3ba2f9d579e868103bfd5877c83cc5d225f5d`, tree `4cb25a54dd8cad13bb03947103e30a7bb84a7144`. The five committed file hashes match the approved source review. Bundle manifest `576802cd3906682b47808ae8d4b693ccdf4e02a743312fbc0e44dba07fc16357`; executable `47e60836806b2225ff2ec6a48c6e48bd144de5f7689a236abb663f238e427706`; object `20e0ab79c66f0ca516a899d4f654b2c56693cd2da8aef2a010a6e098f820b58e`. Actual H100 UUID `GPU-24ec451f-eb55-820b-9107-7314fac7c0ba` (SM90, MIG disabled).

All 3,120 ordinary cases and 6,240 graph replays passed (9,360 reports). The CPU reduction independently rechecked the 3,120 retained ordinary raw arrays against the existing frozen reference bytes, every visit/guard/hole and all work counters. Each sanitizer covered its approved fixed 264-case slice, with 264 ordinary executions and 528 graph replays (792 reports/tool). Memory and synchronization summaries report zero errors; racecheck reports zero hazards, errors and warnings.

No Libcint/reference regeneration, GPU execution or rebuild was performed during reduction.

| Recomputed check | Entries | Maximum absolute error | Maximum gate fraction |
|---|---:|---:|---:|
| Candidate values, emitted gate | 126,792 | 1.010302952e-14 | 1.752706807e-03 |
| Candidate values, complete gate | 126,792 | 1.010302952e-14 | 2.890420979e-04 |
| Incumbent scalar values | 126,792 | 1.010302952e-14 | 1.752803216e-03 |
| Candidate vs scalar | 126,792 | 4.440892099e-16 | 1.187830249e-05 |
| Candidate full raw K | 558,960 | 2.060851489e-15 | 6.489313498e-05 |
| Scalar full raw K | 558,960 | 2.026157020e-15 | 6.380065796e-05 |
| Ordinary Fock full raw K | 558,960 | 2.053912596e-15 | 6.467463957e-05 |

Both original component gates are unchanged: `2e-12 + 2e-10*abs(reference)` and `3e-11 + 3e-11*abs(reference)`. Raw-K and candidate/scalar comparisons use the original complete-component bound. Independent orbit recomputation reproduces the stored reference raw K exactly; retained candidate/scalar raw-K arrays also match the CPU recomputation exactly.

Per one ordinary execution of every case, actual work totals are 10,128 bra preparations, 46,704 ket/radial preparations, 1,654,656 component contractions and 126,792 publications. Every ordinary, graph and sanitizer report matches the independently computed per-task counts. All 624 zero-live cases perform zero source work; all 624 nonempty LR(0) cases publish exact zeros with one-write coverage. These are work counts, not a speedup.

| Order | Symbol | Registers | Attribute local B | ptxas stack B | Attribute shared B | Linked shared B | Spill stores/loads B |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | candidate | 148 | 512 | 512 | 6584 | 7608 | 0/0 |
| 3 | scalar | 139 | 1664 | 1664 | 0 | 0 | 0/0 |
| 3 | fock | 164 | 512 | 512 | 6584 | 7608 | 0/0 |
| 4 | candidate | 202 | 512 | 512 | 6864 | 7888 | 0/0 |
| 4 | scalar | 143 | 2752 | 2752 | 0 | 0 | 0/0 |
| 4 | fock | 170 | 640 | 640 | 6864 | 7888 | 0/0 |
| 5 | candidate | 248 | 512 | 512 | 7312 | 8336 | 0/0 |
| 5 | scalar | 146 | 3920 | 3920 | 0 | 0 | 0/0 |
| 5 | fock | 214 | 768 | 768 | 7312 | 8336 | 0/0 |

Candidate shared storage passes the 16-KiB bound in both resource views. Linked cuobjdump shared bytes exceed the attribute/ptxas view by 1,024 B for candidate/Fock symbols; both values are retained without inferring a cause. Linked `LOCAL=0` does not erase the reported stacks or establish favorable occupancy/dynamic local traffic.

The linker’s generic future deprecation warning for pre-sm75 offline targets is retained unchanged in `cuda-completed-v1/link.stderr`; the actual compile argv explicitly targets compute90/sm90. All raw streams and failures from earlier stages remain untouched.

Machine reduction: `reduction-v1.json`, SHA256 `c6a0f59ce45815b21a53a3c7ac9513c99c778f758a011d01c0b9570a799b493f`. CPU reducer: `reduce-wb-lr-results-v1.py`, SHA256 `b6ef8627a26082da096e8b8d68c27090e6e7ffbb24116152a598197c900ec815`. Additional committed-tree binding: `reduction-source-binding-v1.json`. Raw file hashes, every gate maximum, all resource rows and strict sanitizer summaries are retained in the reduction JSON.
