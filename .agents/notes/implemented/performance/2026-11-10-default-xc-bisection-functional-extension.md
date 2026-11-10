# Decision: admit pure GGA and global meta-GGA hybrid spatial bisection

Status: implemented
Date: 2026-11-10

The work-jet/exchange distinction below is superseded by the
[uniform global-hybrid policy](2026-11-10-uniform-global-hybrid-xc-bisection.md).
The measurements and source identities here describe the earlier policy only.

## Decision and rationale

The initial [four-work-jet pure-semilocal qualification](2026-11-10-default-large-semilocal-xc-bisection.md)
established the DF r2SCAN-3c opportunity. The final automatic policy also admits
pure GGA with one work jet and global hybrid components with at least four work
jets. It is a work/capability policy, not a functional or benchmark-name allowlist.

The common prerequisites are strict-FP64 RKS, at least four AO jets, 512 AOs and
524,288 points, compute capability 8+ and reported FP32/FP64 throughput ratio
greater than two. Nonlocal correlation, range separation, mixed precision, UKS,
explicit reference-memory budgets, public resource ledgers and unqualified
devices retain their incumbent execution. Pure LDA and smaller owners also keep
center grouping. Existing resident-grid, local-AO and optional spatial-storage
admission remains authoritative.

The execution permutation reduces the union of active AOs in a tile without
dropping quadrature points or weights. AO cutoffs, precision, initial guess,
grids, stopping gates, exact exchange and DF approximation do not change. The
canonical grid remains the authenticated moving-force owner. Host/device bounds,
transactional publication, allocation refusal and graph/lifetime contracts stay
unchanged. Explicit selector `1` rolls back to center grouping, `2` requests the
existing bounded bisection path and `0` retains canonical execution.

## Why not all hybrids

On the current source's explicit-selector pilot, DF96 PBE0 saves 0.408271 s
(1.99%) and B3LYP 0.467273 s (2.26%); DF96 M06-2X saves 0.915065 s (3.78%).
Each has one pair and three warm calls per process, with paired and independent
default-grid gates passing. These are **pilot observations, not repeat-qualified
speedup claims**. They use the initial qualification's source identity
`ecd7f50df26f79df1a450ba65d9d4d6bfff6dee3a091c9a21a7e8ee7c0d014e0`
and library `3b94e04d6174b7251e5a32e893fae6f7039f8f0294899b694717153489375937`,
not the final automatic-policy binary. The fewer-work-jet hybrids are close to the 2% measurement
floor, while their exchange endpoint is unchanged. They retain opt-in bisection
rather than being promoted from marginal single-pair evidence. This is not a
claim that bisection cannot help them. Nonlocal/range-separated methods such as
WB97M-V have additional consumer contracts and are not broadened here.

PBE's one-work-jet density product still consumes four AO jets; excluding it by
work-jet count alone unnecessarily excludes a useful large-grid optimization.
M06-2X has the higher-width meta-GGA component and is separately repeat-qualified
with the final selector-unset binary. Pure r2SCAN/r2SCAN-3c remain eligible.

## Final-source qualification

Master base: `d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`.
Final source identity:
`98c023c4ea3923b9d5cd23c9d5716a048394cfd17492ed7efa579cf5518c303a`.
Final production library SHA256:
`c513f859e087522ece5a6e2c032bddf880e0d56aa6aa96d7708723c0668b47bc`.

As in the initial qualification, this is an audited changed-C++-closure replay
of the frozen CMake recipes and relink, **not a clean full rebuild**. Exactly
`cuda_ks.cpp` and `ks_tile_policy.hpp` differ in native source/include inputs.
The four rebuilt translation units are `cuda_ks.cpp`, `dft_method.cpp`,
`cuda_rhf.cpp` and the build-identity TU. Unchanged CUDA/generated objects have
individual SHA256 receipts. Verified sccache 0.16.0 wraps actual compilation;
before/after cache statistics and exact commands are retained. The final source
inventory independently matches the checkout.

Final endpoint populations execute on n4's Slurm-assigned RTX5090 with CUDA
12.9.1 and one numerical host thread. Compare same-binary explicit `1` against
**genuinely unset** selection, alternating order across repetitions. Each fresh
process owns one cold plus three frozen-density warm calls. Cold includes
constructor, preparation, SCF, analytic forces, applicable D4/gCP and host export;
process/import/context startup and cache-miss compilation are excluded. Native
workers reject PySCF/GPU4PySCF imports; independent references run separately.

| Complete endpoint | Pairs | Center cold / s | Unset cold / s | Saving / s |
| --- | ---: | ---: | ---: | ---: |
| DF r2SCAN-3c96 | 3 | 23.622041 | 21.872686 | 1.749355 |
| DF PBE96 | 3 | 10.327776 | 9.853488 | 0.474288 |
| Direct PBE96 | 2 | 11.311532 | 10.805892 | 0.505641 |
| DF M06-2X96 | 3 | 24.192837 | 23.286171 | 0.906666 |
| DF r2SCAN96 | 1 | 12.612575 | 11.818759 | 0.793817 |
| DF r2SCAN-3c48 | 1 | 7.981343 | 7.442850 | 0.538493 |

The repeated promoted cohorts exceed the declared cold gain floor
`max(2%, 2*(center relative MAD + candidate relative MAD))`; their gains are
7.406%, 4.592%, 4.470% and 3.748%, respectively. One-pair rows remain scoped
supplemental observations, not independently established universal speedups.
PBE48/12 and PBE0/B3LYP are unchanged-domain guards. Retain all three warm calls
per native process; a sub-2% warm change is not claimed as a warm speedup.

Slurm1557 retains 34 clean native populations / **136 complete native E/F
calls** and ten independent populations / 20 calls. All pass the original
paired and stock-grid independent gates below. Maximum force discrepancies are
about `2.18e-12` Eh/Bohr paired and `1.35e-6` Eh/Bohr independently. Cold
iterations/Fock builds agree across paired arms, and every frozen warm call has
one iteration/Fock build. Separate lifecycle and intrusive diagnostics do not
enter these clean timing medians.

The readable compact timing samples and final acceptance summary are retained
in `benchmarks/results/xc-spatial-default-20261110/summary.json`. Full-precision
E/F exports, all repetitions and failures, recomputing summarizers, source patch,
allocation/build/cache/test receipts and diagnostic traces remain under
`.artifacts/xc-spatial-functional-survey-20261110/`,
`.artifacts/xc-spatial-gga-default-20261110/` and corresponding n4/n1 working
directories under `/data/jzzeng/`. These local raw artifacts are not an externally
published archive. The compact summary does not replace raw arrays for a future
independent re-audit. No storage cap is raised or unrelated evidence removed.

## Acceptance and limitations

Every native cold/warm export must pass paired energy `1e-8` Eh, maximum force
`1e-7` and RMS force `3e-8` Eh/Bohr. Separate stock-default-grid GPU4PySCF gates
are energy/atom `1e-5` Eh, maximum force `1e-4` and RMS force `3e-5` Eh/Bohr.
This checks preservation against independent observables; it is not a newly
established dense-grid convergence certificate or a GPU4PySCF timing ranking.

Final-source native/resource regressions pass **453 tests plus four subtests**.
The compiled spatial contract passes independent CPU E/V, PBE/r2SCAN RKS/UKS,
exact permutations, tails, changed-density graph replay, canonical force
ownership and exact one-byte host/device refusal. Compute Sanitizer memcheck
reports zero errors. Test-only hook objects remain separate from the measured
production library. The first qualification attempt lacked pytest in the
benchmark venv; the passing rerun uses the existing configured test interpreter.

Slurm1556's preliminary narrower-policy cohort was deliberately stopped after
the first PBE pair to broaden and rebuild before final qualification. Its
complete and interrupted populations are retained but excluded from final
timing medians. It is not a silently dropped slow population. The initial
r2SCAN-3c moved/fresh-owner discrepancy remains documented in the linked initial
qualification; neither schedule is claimed to fix it.

The final-source PBE96 geometry lifecycle adds ten cold/warm/moved/moved-warm/
fresh-moved native calls. Automatic-versus-center energy and force parity passes
the tight paired gates at every corresponding phase; the maximum force
difference is about `2.3e-13` Eh/Bohr. All moved phases pass the independent
stock-grid gates, with maximum force error about `3.3e-7` Eh/Bohr. Both schedules
retain a moved-owner versus fresh-owner force difference of about `3.92e-7`
Eh/Bohr: this is **not** a `1e-7` fresh-owner parity pass or a bisection fix.

Separate final-source PBE96 diagnostics record semantic work per Fock:
point-active-AO visits change `206131264 -> 178102976`, and point-AO-square work
changes `41225797696 -> 30228019776`. The bounded point-batch planner admits
8 versus 16 tiles with retained arena bytes `87338648 -> 133113992`; those are
owner arena sizes, not whole-endpoint peak-memory measurements. Faster execution
does not mean lower retained capacity. Existing optional admission/refusal
remains necessary. Intrusive diagnostics are not used as clean timing samples.

## Reproduction and revisit

Use `benchmarks.readme_defaults` with a verified source-bound release library,
the same exported auxiliary basis inputs and one numerical thread. Generate
those inputs outside timed populations with its `inputs --inputs DIRECTORY`
arm. Under finite `srun --partition=main --gres=gpu:5090:1 --time=00:30:00`, run
fresh `standard --method METHOD --mode df --atoms 96 --warm-repeats 3` workers,
alternating `GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=1` and an unset selector for three
pairs. Supply `--inputs`, `--source-revision` and a unique ignored `--output` for
every worker. Run the independent `gpu4pyscf` arm separately. The retained local
campaign scripts additionally enforce the native import firewall and recompute
every exported observable gate.

Broaden fewer-work-jet hybrids, nonlocal/range-separated consumers, UKS, precision
or resource promises only after separate complete-endpoint, numerical,
canonical-force and capacity/lifetime qualification. Do not replace the existing
bounded fallback or center/canonical rollback with an unqualified global switch.
