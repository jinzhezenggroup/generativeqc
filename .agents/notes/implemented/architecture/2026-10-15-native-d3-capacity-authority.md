# Decision: query the production D3 owner's capacity layout

Status: implemented
Date: 2026-10-15

## Problem

The D3(BJ) resource adapter duplicated the production ragged owner's storage
formula and reconstructed its native table ABI with ctypes. Although the
numbers agreed, production Python planning could drift from the compiled
owner's table layout or CPU/CUDA workspace lifetime. This is a bounded R2
ownership gap in #2227, separate from correction scientific equations.

## Decision

Expose `d3_resource_usage` from the existing C++ D3 runtime and use it both
in `D3Plan::prepare()` and the private additive
`generativeqc_resource_d3_layout_v1` bridge. The query returns plan host,
execution host, device, table, workspace and complete retained bytes without
constructing a context, plan, coordinates or backend allocations. Shape
validation requires a realizable nonempty uint32 ragged batch with the existing
4096-atom per-system bound; these bounds also keep capacities below int64.

The Python calculator passes its already loaded library. Discovered native
errors propagate, rather than silently recalculating another layout. An absent
symbol or library retains explicit metadata-only portable sizing. Schedule
identity distinguishes these authorities. Local/global budget presentation and
composition remain Python-owned; native preparation retains its existing local
capacity gate. No scientific admission or lease state machine is moved.

## Rejected alternatives

- A second C++ capacity polynomial would only move the duplication. Share the
  production function and its actual compiled `sizeof` table layout instead.
- Loading a backend for dry-run planning would break provider-free planning.
  Pass the caller's existing library and keep the no-library fallback.
- Treating CUDA sizing as CUDA scientific qualification would hide missing
  execution evidence. The selector is scalar metadata and works in CPU builds.

## Invariants

- CPU workspace scales with the maximum item; CUDA workspace scales with the
  aggregate ragged atom count, not the per-system cap.
- Invalid shapes, selectors or output schemas publish no partial output.
- The prepared owner's exact/below-cap behavior, numeric capacities and all
  D3 scientific equations, damping, gradients and replay semantics are unchanged.
- Capacities exclude allocator/process overhead and are not full endpoint RSS
  or measured simultaneously-live allocation receipts.

## Evidence

The native resource contract compares independent layout arithmetic, large
aggregate batches, uint32 boundaries, malformed shapes and transactional
publication. Python contracts compare native and legacy CPU/CUDA selectors,
provider-free planning, authority identities, rejection injection, exact/below
local/global caps and prepared-owner diagnostics, moved replay, isolated failure
and recovery. Existing production D3 independent simple-dftd3 goldens and
calculator composition tests remain the scientific gates. Real GPU execution
is separate and must run under Slurm; scalar CUDA-selector tests are not GPU
evidence.

The focused Python suite passes 21 cases. The broader D3/resource/composition
suite passes 87 cases, including 17 production CPU cases with independent
simple-dftd3 goldens; one real-CUDA production test is explicitly deselected.
The existing HF/KS resource suite passes 267 cases with seven Slurm-only GPU
cases skipped. Five CTest contracts pass, including the new D3 layout contract,
4100-item D3 replay and the Python-free CLI. The compiler mock tests use
registered oracle data restored from the existing pinned xTBloom Git history
through the source registry's digest validation, not regenerated replacement
parameters. These are capacity/scoped execution gates, not complete R2 receipts.

## Revisit when

D3 storage lifetimes or table representations change, or a native global
budget/candidate owner is ready to replace frontend composition. An additional
ABI must preserve explicit old-library behavior without masking native errors.

## References

- #2227 R2; #933 / #934
- `src/dft/dispersion/d3_runtime.cpp`
- `python/generativeqc/resources_d3.py`
- `docs/maintainer/resource_planning.md`
