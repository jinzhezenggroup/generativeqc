# Decision: share the existing native D3 capacity with CLI resource planning

Status: implemented
Date: 2026-10-29

## Problem

Named PBE D3(BJ)/D3(0) programs have native prepared energy owners and a public
shape-only KS D3 capacity query, but the installed CLI refused their resource
inventories. The D4 CLI consumer already composes the same electronic owner.
Repeating Python capacity equations or sizing one standalone ragged correction
would introduce another authority or miss replicated per-item storage.

## Decision

Extend the existing CLI composition to both native D3 damping kinds. Require the
same compiled PBE electronic graph, carrier, spin, domain and backend capability
used by the D4 consumer. Query the existing KS D3 owner and compose its retained
host/device storage with the existing electronic fleet. Keep the equal-share
correction cap independent of the global host/device budgets. A refused native
D3 shape query produces an unsupported result rather than a partial plan.

## Invariants and validation

This is energy-only metadata planning. No CUDA device, SCF, Python runtime,
reference engine or independent scientific equations are introduced. Hybrid,
range-separated, nonlocal, gCP, bound-basis, force and fitted contexts remain
outside the admitted CLI inventory. Native D3 owns its atom-count bound.

Executable CLI tests compare both damping kinds, spins, CPU/CUDA shape metadata
and batch multiplicities with the real public capacity query. Existing CUDA
electronic test leaves are synthetic; they do not establish real-GPU execution.
Exact and one-byte-below local/global limits, native shape refusal, and retained
D4 behavior remain required. Installed command timing includes process startup.

This is an incremental #2227 R5/R2 consumer on top of #2873. It does not qualify
new molecular E/F contexts, repair #2872's CUDA runtime performance, or close a
whole #2227 slice. R3 is outside scope.
