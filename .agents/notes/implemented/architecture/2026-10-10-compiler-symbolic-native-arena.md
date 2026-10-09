# Decision: compiler ownership of symbolic retained native arenas

Status: implemented
Date: 2026-10-10

## Problem and audit boundary

The audit starts from master
`1db02ce3657ff92ce713f4db461c6b0c8541dc7a`, fetched on 2026-10-09.
GitHub issue bodies, PR state/head/files, and the active optimization diffs were
inspected before editing. The original checkout contains unrelated staged and
unstaged DF force work; this change is isolated in a clean master-based worktree.

The optimizer foundations #832 and #833 are closed. #682, #1580, #597,
#1886, #1889, #1890, and #1763 remain open. #2090 is closed. #2160 is merged.
#2153, #2159, #2161, #2163, #2164, #2165, and #2166 are open, not master
capabilities. These are audit-time facts, not a permanently maintained status list.

Master subsequently advanced to
`740262e2ecf4a0449403c25d2c960f66f6ec16e3` through #2149's native API/CLI
change. Its changed paths do not overlap this slice or the inspected TensorIR
planners. #2167 is also open and owns native DF-guess work admission; this slice
does not modify its resource/default policy.

## Ownership and gap matrix

| Optimization | Current owner | Production consumer on audited master | Compiler support | Duplication / gap | Integration action |
| --- | --- | --- | --- | --- | --- |
| Source traversal reuse / invariant hoisting | Compiler dependency/batch analyses; native epoch owners | MP2/provider ordered source scans, opt-in conventional CPU RCCSD reuse, CUDA DF Lambda core retention | `common/source_reuse.py`, `tensor/iteration_reuse.py` | Symbolic coloring resides in `tools/generate_rccsd_native.py`, also reached by Lambda emission; cold KS promotion is open #2164 | Move that sole coloring implementation into TensorIR without changing native state or DF policies |
| Specialization / fission | IntegralIR and MethodIR generators; native launch admission | Generated shell classes, merged #2160 force-queue split | `common/gpu_profitability.py`, `common/schedule.py`, IntegralIR route contracts | General profitability is not automatic qualification; stationary point/AO split is open #2166 | Reuse #832/#833 and preserve backend-specific admission; do not copy #2166 |
| Independent J/K lowering and queues | Native direct-Fock selector plus integral leaf generation | Incumbent direct J/K routes | IntegralIR recurrence/queue emission, `integral/direct_fock_schedule.py` and `operator_route_schedule.py` | Open #2153 owns the active independent lowering/scheduling selector integration | Depend on its frozen typed selector and leaf identities, not a competing selector |
| Provider-neutral tensors | Compiler semantic requests/providers; native resource owners | Tensor lowering/native contraction bindings | `tensor/lowering.py`, `common/lowering_selection.py` | Native vendor calls remain, e.g. `src/scf/cuda/df_occupied_exchange.cpp`; representation is not migration | Continue #1886/#1889/#1890 with one actual native consumer per slice |
| Budgeted Lambda batching/replay | Native CC owner and existing generated actions | `src/cc/df_lambda_cuda.cu` | Shared invariant proof, existing symbolic arena/ordered accumulation and matrix contraction lowering | Budget/default changes belong to open #2163; arena ownership is still a tool-level dependency | Share storage lowering here; leave #2163's admission/default changes untouched |
| Multi-output triples fusion | TensorIR scientific producers and generated CC tiles | Existing fused triples/response paths from closed #2090 | Shared producer and runtime-domain/fusion mechanisms | General resource-profitability/consumer adoption remains #1763 | Preserve equations, ordered accumulation, stationarity and independent audit |
| CPU/CUDA recurrence preparation | Shared IntegralIR equations, backend-specific lowerers | Current separately scheduled integral kernels | Existing recurrence IR/codegen | CPU high-angular-momentum preparation reuse is open #2165, not CUDA qualification | Reuse mathematical preparation while keeping schedules and numerical acceptance independent |
| Compensated Fock / incremental DIIS | Integral codegen/numerical policy; native solver history | Existing baseline accumulation and DIIS controllers | Compiler precision contracts and shared runtime controller boundaries | Open #2161 and #2159 change numerical/performance policy, not generic purity | Do not infer generic admission or change their defaults in this ownership slice |

For the J/K overlap, the inspected #2153 branch is
`codex/unified-k-selector-2133-2135` at
`9db85cf78451134d037513f73b583751e1c84ce7`. Its prepared typed K contract keeps
lowering and Fill/Work scheduling independent. A follow-up compiler integration
must consume that contract and its qualified leaf producers; native queues,
screening, stream lifetime and allocation remain outside the scientific IR.
No selector code from that unmerged branch is implemented here.

## Decision

Move the established materialized FP64 symbolic arena planner, including
exclusive retained-slot reservation, into `tensor.native_arena`. The RCCSD
generator keeps only the domain-to-runtime-extent binding and existing selected
execution order. DF Lambda and triples/response already reach this same planner
through the shared native emission helpers. There is no second scientific
implementation, pass manager, scheduler, provider selector, or cache owner.

This is the smallest independent production-connected slice: it removes method
tool ownership of a reusable dependency/lifetime optimization without waiting
for #2163/#2164 or changing their policies. Generic tests use non-CC density and
geometry inputs; the native tests exercise an actual conventional CPU solve.

## Invariants and retained boundaries

- Reserve retained storage before any dynamic slot, including invariants whose
  original position follows a same-shaped dynamic producer.
- Share only equal symbolic capacity products, never equal representative sizes.
- Release after the last reader, never during the producing operation; keep
  borrowed outputs live through return.
- Keep the existing operation order, arithmetic, generated checked capacities,
CPU/CUDA layouts and native complete-budget/failure/publication behavior.
- Treat the storage-only identity as a layout identity, not numeric validity.
  Purity comes from `analyze_iteration_reuse`; epoch validity comes from the owner.
- Materialized dense FP64 is the lowering contract. Do not infer aliases or admit
  mixed precision. The existing common alias analysis and ScheduleIR remain
  authoritative for their own contracts.

## Rejected alternatives

- A new general caching runtime would duplicate allocation, invalidation and
  solver ownership and overlap #2164.
- Replacing symbolic products with concrete `common.storage` byte sizes would
  admit unsafe reuse when representative dimensions coincide but runtime extents
  differ. Shared storage lifetime analysis is an independent test oracle here.
- Changing coloring/fusion/defaults alongside ownership would invalidate exact
  generated-source equivalence and require a new complete-endpoint promotion.
- Moving CC index names/representative orbital partitions into generic code
  would move method semantics across the ownership boundary.
- Revalidating a supplied order through repeated `Program.live_nodes` queries
  unnecessarily repeats full liveness analysis. An initial diagnostic increased
  iteration planning from 5.46 to 14.93 ms and Lambda core planning from 1.18 to
  21.29 ms. The retained implementation validates the existing output-reachable
  dependency inventory once and binds each immutable index once. A regression
  test rejects another full liveness query when an order is supplied.

## Evidence and qualification boundary

Qualification uses independently emitted baseline and candidate CPU/CUDA RCCSD,
triples, triples-Fock and DF Lambda artifacts from the same master source.
Compare all eight complete files, not excerpts or old binaries. The existing CPU
reuse tests compare against the independent determinant oracle and exercise
changing references/amplitudes, short arenas, exact budget misses, allocation
failure, full-evaluator fallback and unchanged physical replay.

The main host run passes 106 tests with eight skips (seven real-GPU Lambda cases
and one unavailable `clang++` expression-depth compile check). The separate
Hamiltonian/force-seed, Lambda and triples-response codegen run passes nine
tests. Native compilation reuses the existing ccache launcher where supported;
no cache is cleared. Ruff, focused `ty`, compiler structure (505 modules, zero
errors), ledger, public-docstring, provider-boundary and codegen-ownership checks
pass. Sphinx builds with `-W --keep-going` without warnings. Full compiler `ty`
retains 145 errors and ten warnings from the unchanged baseline environment;
excluding only the new module produces byte-identical diagnostics. These
unrelated diagnostics are not fixed or presented as a passing full type gate.

All eight baseline/candidate emitted files compare byte-identically:

| Artifact | SHA-256 |
| --- | --- |
| RCCSD CPU header | `b60eb861a75fef8efd30d7aaf2022ef01623fac2d8b1dd03421eac20b7bc97b2` |
| RCCSD CUDA source | `20b921c153593a95c1cc89d5f22700e71dcde3b79055f238c815f07aa4021640` |
| Triples CUDA source | `6d0ac876f3fece300ab687b62e5e6e4dd97cab45ff9310109a8d37df93186490` |
| Triples-Fock CPU header | `a0d151d2e1c2688424965c2517d3c1dc02815638bb1ed81e8b96fb431e1bfcd2` |
| Triples-Fock CUDA source | `a6b823a5a5022d71a49bb0bb4264925cf254cec88469e3194693a4e0a57c8a60` |
| DF Lambda CPU header | `d3b1e9dda4f0838628bd8feef98535bdffcab5572d88852e3eef034cdd92b27c` |
| DF Lambda CUDA header | `dc776542d58093ca3d4786a6116b6c0437e5ee710f6f6fec8aa87c3da9db12fa` |
| DF Lambda CUDA source | `b777b326cb9fe7d77bac13f6866744c52d0d75861144a16ccf23d0711a01859c` |

The compiler inventory naturally changes because ownership/source files change;
the equivalence claim is for scientific/generated execution source, not a newly
built whole-library binary or an unchanged inventory/cache key.
The measured compiler planner SHA-256 is
`7c0055baa520815ef03cafb1f13e27bc64f1d5c34592561d8f76b740be1d63a1`;
the native generator adapter SHA-256 is
`08214b6824eb32ad0dc0512a12319d4bdd2a4a5d595492eab26532ada59963c5`.

Three interleaved/reversed same-process pairs compare complete **warm source
emission**, replacing only the planner with the old master implementation while
keeping the same compiler graphs and warmed caches. All emitted hashes match:

| Emission endpoint | Baseline seconds | Candidate seconds |
| --- | --- | --- |
| RCCSD CPU header | 14.903581, 14.930886, 14.926921 | 14.960784, 14.969658, 14.952374 |
| DF Lambda CUDA source | 6.614258, 6.597451, 6.606771 | 6.601029, 6.604595, 6.612619 |

Median changes are +0.23% and -0.03%, respectively. This is a preparation-cost
diagnostic, not a scientific endpoint promotion: unrelated host validation can
overlap and the sample count is small. Individual planner medians increase from
5.428 to 5.912 ms and 1.179 to 2.226 ms with the additional fail-closed validation.
There is no native runtime work/schedule change or claimed speedup. Raw vectors
and the matched old-planner harness are retained locally with the artifacts.

GPU resources were checked with `sinfo -N -o '%N %P %t %G'`. This host exposes
only drained `node3 main gpu:5090:1`. No GPU command is run and node3 is not
scheduled. New cold/warm/moved-warm energy/force endpoints, registers, spills,
occupancy, launch counts and measured device traffic are **not qualified here**.
No speedup or performance-default promotion is claimed. The acceptance rationale
is compiler ownership and unchanged complete generated execution artifacts.

Reproduction commands and current ownership contract are in
`docs/developer/iteration_reuse.md`. Raw local baseline/candidate artifacts are
under `/tmp/gqc-compiler-audit-20261009/`; this path is not a durable publication.

## Remaining independent slices, in priority order

1. P0-A: integrate source preparation selection with explicit immutable epochs
   and complete-budget admission after #2164, reusing `common.source_reuse` and
   the existing invariant proof. Do not infer reuse from shape/pointer identity.
2. P0-C: compiler-owned joint J/K plans consuming #2153's typed selector boundary;
   use #597 for target/resource tuning rather than adding another native switch.
3. P0-B: resource-qualified producer/consumer specialization after #2166, using
   #832/#833 and #2160's established force-queue classes, not device names in IR.
4. P1-A: #1886/#1889/#1890, one semantic native operation and qualified provider
   migration at a time, with precision/materialization/preparation costs intact.
5. P1-B: #1763 shared producers and profitable bounded fission/fusion, with #2163
   owning Lambda budget/default changes; preserve independent FP64 audit gates.
6. P1-C: represent #2165 preparation in shared recurrence/codegen without copying
   CPU scheduling onto CUDA. Continue symbolic contraction reasoning under #1580.

## Revisit when

A qualified view/donation lowering or another native TensorIR consumer needs
symbolic storage, or measured endpoint evidence justifies changing the existing
coloring. Extend common storage/ScheduleIR rather than inventing method-local
planners. GPU qualification must use finite Slurm jobs and assigned visibility.

## References

- `docs/developer/compiler_architecture.md`
- `docs/developer/iteration_reuse.md`
- `.agents/notes/implemented/architecture/2026-10-06-shared-iteration-invariants.md`
- `.agents/notes/implemented/performance/2026-10-09-df-lambda-core-invariant-reuse.md`
- #682, #831, #832, #833, #933, #1886, #1889, #1890
