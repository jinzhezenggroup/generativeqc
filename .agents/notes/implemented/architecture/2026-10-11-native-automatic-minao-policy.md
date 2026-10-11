# Decision: query automatic MINAO admission from the native owner

Status: implemented
Date: 2026-10-11

The conservative CPU batch-selection decision is superseded by
[native per-item automatic cold starts](2026-10-11-native-per-item-cold-start-policy.md).
The context-free discovery, compatibility and qualification boundaries remain.

## Problem

The #2227 source baseline at `140160231f23d61cea6ce103ec5a46a0da915b31`
already contains the merged native MINAO capacity query, native CPU DF-PBE/PBE0
batch-force owner and its Python routing, and native CLI resource/profile
commands. Those changes do not complete the runtime migration. The Python
Calculator still selects automatic MINAO by repeating native method/backend/
precision/DF/spin restrictions; system-level selection repeats H-Ar/ECP rules.
The projection, strict density admission, seed budget and bounded retry are
already native. Moving their equations again would introduce a second owner.

The call-path inventory distinguishes this selection debt from existing native
seed execution and from the unrelated progressive/force-aware controllers.
It also records that a module or function named "native" can still orchestrate
scientific response/HVP work in Python. Wrapper parity and source-level
coverage are not numerical or installed-SDK parity.

## Decision

Add a context-free, shape-only `generativeqc_initial_guess_auto_kind_v1` query.
The request names a registered carrier and explicit scientific spin, backend,
precision and DF facts. The optional element list and ECP flag extend
model-only admission to the element domain. A result of zero selects Hcore;
MINAO remains a candidate subject to native system, basis, seed, density and
resource checks. The query allocates nothing and runs no scientific kernels.

Share its domain and element predicates with existing prepared native MINAO
admission. Python marshals facts and uses the native result. Query failures
are errors, not compatibility fallbacks. Older libraries lacking the additive
symbol retain the previous positively discovered MINAO policy, including
Hcore on missing capability/version queries. Explicit `InitialGuessSpec` and
`None` choices do not invoke automatic selection.

Retain the existing conservative batch descriptor: one unsupported element or
ECP disables automatic MINAO for the whole batch, including its resource
inventory. This change does not claim per-item native progression, complete
resource authority, native CUDA force publication, or native molecular HVP.
Those remain in the full R0-R6 tracker, not redefined as success here.

## Rejected alternatives

- Treat a positive global MINAO provider bit as model/system admission. It
  describes provider availability, not the qualified domain.
- Rewrite projection, occupation or retry equations in another C++ path.
  The existing native seed and target owners already implement them.
- Catch any query failure and reuse Python policy. This would silently retain
  a second production authority after positive native discovery.
- Enable partial MINAO in ragged batches without migrating the shared
  descriptor and planner inventory. That would change execution/resource
  policy rather than mechanically moving admission ownership.

## Invariants

- No change to functional, grid, basis, ECP, DF, convergence, force sign or
  derivative semantics.
- Warm/imported/explicit density takes precedence, and preparation/target
  failure remains bounded by the existing native state machine.
- Failure preserves the query output; invalid descriptors, Boolean flags,
  atomic numbers, pointer/count pairs and enum values are rejected.
- The source inventory is baseline evidence, not a promotion or a complete
  capability matrix. Exact-head execution and independent scientific gates
  remain mandatory for every actual cutover.

## Evidence

- `tests/native/test_preliminary_initial_guess.cpp`: native policy domains,
  atomic output on errors, and existing cold/warm/retry/budget density checks.
- `tests/python/test_native_initial_guess_policy.py`: effect tests demonstrating
  native selection, explicit-choice bypass, ECP fact transport, missing-symbol
  compatibility and visible errors.
- `tests/python/test_minao_default.py` and
  `tests/python/test_preliminary_initial_guess.py`: unchanged public defaults,
  resource planning, endpoint energy/density and prepared lifecycle behavior.
- `tests/python/test_native_runtime_ownership.py`: source symbols and complete
  tracker scope, with negative tests for wrapper-as-native parity claims.

These are reproducible test anchors, not claims of CUDA execution evidence.

## Revisit when

The existing native prepared batch can retain per-item automatic policy and
the corresponding native resource owner admits and accounts for each seed.
Then migrate the conservative descriptor and resource contract together,
including independently qualified ragged, failed-neighbor and shortage gates.

## References

#2227, #933, #934; `docs/user/initial_guesses.md`;
`manifests/maintenance/python_native_runtime_ownership.json`.
