# Decision: share phase-local Direct ERI values between RHF and Global-Hybrid KS

Status: implemented, opt-in KS qualification required
Date: 2026-10-10

## Problem

PR #2206's optional RHF values bundled a method-neutral Direct source with RHF
assembly and iteration graphs. Copying that owner into KS would duplicate RHF
equations and couple KS XC/final-state lifetimes to an unrelated solver.

## Decision

Extract `DirectJkSourceLease` for source admission, immutable canonical FP64
values, compensation, raw restricted J/K, diagnostics and final finite/census
audit. Keep RHF assembly and graphs in its existing adapter. KS borrows the
same public Direct capabilities and retains its own scientific equations.

KS selection is explicitly opt-in through `GENERATIVEQC_KS_RESIDENT_VALUES`.
Default promotion requires separate complete endpoint evidence, not RHF's
existing qualification. Automatic admission keeps the cold generic f-shell
heuristic; explicit force cannot bypass capacity or capability requirements.
The canonical replay consumer retains the target Schwarz predicate while
using immutable unscreened source values. No new integral recurrence is added.

## Rejected alternatives

- Renaming the old RHF owner without separating assembly/graphs would leave
  hidden RHF normalization and graph-lifetime assumptions in KS.
- Mutating the geometry-cached prepared provider would let optional quartic
  values survive into subsequent solves and derivative consumers.
- Allocating opportunistically under a public resource ledger would bypass
  its complete plan because the current planner reserves no quartic source.
- Promoting KS automatically from RHF timings would ignore grid/XC work,
  different iteration trajectories and whole-endpoint force cost.

## Invariants

- Source reuse never changes precision, exchange coefficients or XC selection.
- Source construction is finite-audited before publication; completed replay
  reports the full canonical visit census and zero ERI recurrence evaluations.
- Complete mandatory consumer and source setup inventory precedes optional
  values; the 8-GiB value cap and 256-MiB free-device reserve remain.
- Public ledgers, UKS, DF, RSH/nonlocal, mixed and incremental schedules keep
  bounded incumbent paths. Driver/numerical errors are not capacity refusals.
- Consumers retire captured borrowers before sources and keep borrowed streams
  alive until destruction. A source never crosses a terminal/restart boundary.

## Evidence

The existing stubbed RHF ownership test now compiles both the adapter and common
owner and checks raw Global-Hybrid coefficients/threshold propagation without
RHF assembly. The host policy and maintenance checks retain RHF's qualified
default envelope. The Slurm-only Hybrid probe asserts independent PySCF signed
raw J/K, replay screening, same-grid energy, cold/warm/seed behavior, public
ledger refusal and low-budget exact fallback. Its optional public energy mode
includes provider preparation, source/Fock work counts and complete energy parity.
The base revision's `DftMethod::execute` rejects all KS nuclear-gradient requests
with `NOT_IMPLEMENTED` before SCF. Retain a separate strict E+F probe mode rather
than inventing force evidence or substituting an energy-only endpoint. E+F
qualification/default promotion must wait for that independent capability.

Runtime qualification results and source-matched evidence must be retained
separately; implementation alone is not a performance claim or default promotion.

### Initial qualification on 2026-10-10

Base: `1419595e35d40291532369149bf3081de6d3ec72`, plus the retained complete
working-tree patch (including new files). CUDA 12.9.86, sm120 Release, local
GCC 11.4, ccache 4.5.1; Slurm node1 RTX 5090, driver 580.95.05; independent
PySCF 2.14.0 / NumPy 2.3.5.

- Host regression selection: 495 passed, 56 device-gated tests skipped;
  dependency/promotion checks, Ruff and whitespace checks passed.
- Slurm job 7167: four Hybrid cases passed (water/methane aug-cc-pVTZ,
  chunk widths 1/2, both PBE0 and B3LYP). Signed raw J/K passed the independent
  `2e-9` maximum-entry gate; same-grid returned-density energy error was at most
  `2.71e-13` Eh across the eight functional/case/chunk comparisons.
- The same job's four complete cold **energy** endpoint pairs included public
  preparation. Baseline source construction was zero; resident water built
  15,487,395 FP64 values and methane 73,090,095, with 13--14 submitted Fock
  assemblies. Maximum paired energy difference was `2.28e-13` Eh.
- Slurm job 7170: all three existing RHF geometry/lifetime/independent-energy
  cases passed, retaining the lower-angular automatic refusal.
- The original full native KS suite did not complete: with the new selector
  unset, its existing density-provider admission assertion failed at route 5
  after routes 0--4 passed. This run is not evidence of full-suite qualification;
  no unrelated density-provider implementation or test was changed.
- Slurm job 7171 separately passed native restricted/unrestricted incremental
  KS physical-state and fresh-solve gates. Its complete water Hybrid probe also
  passed CUDA Compute Sanitizer memcheck with zero errors, including source
  retirement, warm/seeded phases, screening and bounded fallback.
- Slurm job 7166 confirmed the base public E+F refusal before SCF. No Hybrid
  force agreement, complete E+F speedup or automatic-default promotion is claimed.

Local ignored evidence is under `.artifacts/hybrid-phase-values-20261010/`, with
commands, complete patch, binary hashes, compiler-cache snapshots, raw logs,
independent oracle snapshots and endpoint/work-count JSON. Single ordered energy
pairs (with other node workloads present) are correctness evidence only, not a
controlled performance comparison; repeat/ABBA complete E+F qualification remains
required before performance/default-promotion claims.

## Superseding capacity decision and force clarification

The screened source layout and explicit KS capacity decision are superseded by
`2026-10-10-screened-hybrid-phase-source.md`. The initial force statement above
applies only to the legacy direct C-API `DftMethod::execute` owner/probe. It does
not apply to the high-level Python compiled stationary PBE0 E+F endpoint,
which already exists in this base. The initial evidence remains energy-only;
new 96-atom E+F evidence belongs to the superseding decision, not those runs.

## Revisit when

The complete public resource planner reserves optional ERI storage, and paired
cold/warm E+F measurements justify wider domains or a KS automatic default.
UKS compensation and SR/LR source identity need independent work before reuse.

## References

- PR #2206, base `1419595e35d40291532369149bf3081de6d3ec72`.
- `docs/developer/scf_module_boundaries.md`.
- `tests/python/test_ks_phase_resident_values_cuda.py`.
