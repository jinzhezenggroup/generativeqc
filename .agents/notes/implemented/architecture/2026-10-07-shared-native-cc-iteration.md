# Decision: one native CC host iteration controller

Status: implemented
Date: 2026-10-07

## Problem

CPU and CUDA already consume the same RCCSD equations, but their native solve
functions independently owned the energy/residual convergence test, mandatory
physical replay, iteration budget, previous-energy state, and accepted-trial
reuse lifecycle. Keeping those scientific decisions in two backend owners made
future changes susceptible to drift. This is separate from the TensorIR
iteration-invariant retention work in PR #2041.

## Decision

`src/cc/iteration_driver.hpp` owns the common host control flow. It consumes the
existing method-neutral bounded iteration primitive in `src/solver`; scientific
CC policy does not move into the generic solver infrastructure.

The CPU/CUDA callbacks retain evaluation, scalar observation, diagnostic
publication, independent replay, Jacobi/DIIS lowering and failure recovery.
Optional carried outputs are shallow backend-owned views, admitted only when
DIIS did not modify the trial amplitudes. They do not allocate numerical storage
or count as fresh graph evaluations. Replay remains fresh and independent.

CUDA allocation, kernels, DIIS storage, status copies, event recording/draining,
error checks and final amplitude transfers retain their existing owners and
order. Resource-dependent packed/full/Jacobi fallbacks remain backend choices.
No generator, generated equation, device kernel, launch geometry, or memory
budget is changed. No performance improvement is claimed.

The shared controller runs at most `max_iterations` updates, then one final
observation. The final observation is separate from the unsigned bounded count,
so `max_iterations + 1` cannot overflow. At the unsigned maximum this repairs the
old CPU path's unobserved final update and saturates the diagnostic observation
count rather than overflowing the old CUDA diagnostic. Ordinary budgets retain
identical trajectories and semantic work counts. The old CUDA loop already had
an explicit terminating budget check; it was not an infinite-loop defect.

## Rejected alternatives

- A shared convergence predicate alone leaves two orchestration state machines
  and does not give one owner to independent replay and output carry
- Unifying backend allocation, packed DIIS, or execution schedules is unnecessary
  for this policy extraction and would require separate device qualification
- Moving CC convergence into the generic bounded-loop helper violates scientific
  method ownership
- Rejecting the unsigned-limit option would unnecessarily narrow accepted input

## Evidence

Base: canonical master `c3bb6df4468ad75d34a2e2effb6e2450f2838a5a`.
Validation used a cloud CPU workspace, C++20, verified ccache 4.14.1, and the
existing compiler cache. No NVCC or real GPU execution was available.

- `test_cc_shared_iteration_driver.py`: actual native CPU solves for DIIS 0/2/6,
  damping 0/0.2 and one-update/converged budgets, checked against an independent
  determinant-space oracle; 12 cases. Standalone shared-policy tests cover carry,
  changed-DIIS invalidation, replay refusal, final observation, numerical failure
  conversion, exception propagation, and early convergence with UINT_MAX budget
- The same 12-case native probe compiled against the frozen original CPU owner
  and candidate produced identical 6,592-byte result/amplitude/work-count output
- `test_rccsd_cuda_jacobi_refusal.py`: the live CUDA solve function compiles and
  executes with delayed mock operations; 20 full/packed/replacement/Jacobi,
  rejected-replay and numerical/transport-failure scenarios pass
- Instrumented original/candidate CUDA mock probes produced identical 4,000-byte
  operation/status/result traces across those scenarios. This is host control
  evidence, not execution of CUDA kernels
- The CUDA Owner and all preceding kernels/DIIS helpers, plus the final
  diagnostics/amplitude-copy tail, remain byte-identical outside the new include
- `test_df_cc_native_solver.py -k cpu`: 22 passed, 10 skipped (CUDA-only packed
  storage, conventional binding, matrix-provider and Q-tile cases), 34 deselected;
  includes dense/DF and determinant oracles, denominator trajectory, exact
  capacity, bounded/hoisted fallback, sticky arithmetic failure and pinned
  molecular endpoint checks
- Final focused suite: 14 passed (shared driver, native CPU capacity, live CUDA
  control flow, event timing, preflight, cleanup and shared-owner guards)
- Compiler structure: 482 modules, zero dependency errors; electronic structure
  boundaries: 80 shared modules, zero architecture errors
- Relevant Ruff, clang-format and whitespace checks pass

Fresh generation from a pristine base archive and candidate yielded identical
byte lengths and SHA-256 values for every CC CUDA generation artifact:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| rccsd source | 2862153 | f9b320eab39fb98aad166c77190e0d5335a407160300a70e005c4f64121847e9 |
| df_ccsd header | 900 | 6de28ef667104d53fbe1d80d2efda2b55467bf50fef0e7223dbf46c4659ef83d |
| df_ccsd source | 342187 | 11790d662b277a84422bb5ae42a8bd7603bc2c3759bebb9ceef1637d44df8ce1 |
| df_ccsd_core header | 506 | 39ae12d24d272bb9f2e1612f4c264c30aa601bd51779dc9679242133aa6dff76 |
| df_ccsd_core source | 280954 | ffd1e5dab180bdd5ca6626af22028ad4983d9906089758a559f8da902595c412 |
| df_ccsd_hoisted header | 1692 | bc5bec4a2e2fbf37d82cb7bb3ad880c13f0da185ec8ac05fea987f304296ba8e |
| df_ccsd_hoisted source | 445743 | e7dc9e95d04f13c2855ae3abdca0cd40fb005b3bfa0be94485e187b2cfa3ff63 |

## Consequences and next boundary

Generated-device-code equivalence plus preserved/mock-qualified orchestration
makes a new GPU performance campaign unnecessary for this narrow extraction.
It does not prove CUDA compilation or hardware behavior; normal CI still applies.
Callbacks leave CPU and CUDA data ownership visible instead of concealing device
synchronization behind a generic numerical interface.

The next audit should examine Lambda/response convergence and residual/replay
policy for similar duplication, without forcing its distinct acceptance rules
into the RCCSD driver. PR #2041 can integrate this extraction while retaining its
preparation and `run_iteration`/`run_replay` closures; its TensorIR proof and
retained storage remain unchanged.

## Integration with the invariant-reuse candidate

A clean cherry-pick onto PR #2041 head
`94166bd0befe267df15744b3cd9feda5498e161e` succeeded without conflicts. Its
preparation/DF/evaluation/replay closures remain byte-identical. On that combined
tree, 16 native/control tests passed, including the invariant-reuse oracle,
allocation-failure and budget fallbacks, canonical denominators, DIIS Gram gates
and this controller's tests. Forty shared invariant, canonicalization and default
property tests also passed, as did the electronic-structure architecture guard.
The follow-on remains an independent master-based change; this integration check
does not activate #2041's default-off candidate or change its retained evidence.
