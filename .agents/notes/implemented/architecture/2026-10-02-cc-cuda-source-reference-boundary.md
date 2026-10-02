# Decision: keep native CUDA RHF separate from optional correlation source storage

Status: implemented
Date: 2026-10-02

## Problem

Passing a CUDA `PreparedFockPlan` into `run_prepared_fock_strategy` selects the
host mean-field driver. Exact plans provide no device eigen callback, so that
route also executes reference eigensolves and DIIS on the CPU. It does not
preserve the native CUDA RHF contract merely because its J/K provider is CUDA.
The default prepared-source device allowance is also independent of the
correlation endpoint's budget.

## Decision

Always obtain CUDA physical references through `run_rhf_cuda`. Retire any prior
optional correlation source before that solve. Only after successful reference
publication may the method prepare a CUDA Direct source for its MO transforms.
CPU retains its existing shared reference/source owner.

Use the existing Direct device-byte query and pass its exact generic-provider
allowance. Correlation ERI tiles use that generic source; optional canonical and
generated J/K owners would consume capacity without contributing to these tiles.
A checked, allocation-free preparation planner reserves the detached physical
reference beside the sequential one-electron and Direct preparation phases. It
includes public/Cartesian metadata, shell pairs, coefficient copies, matrix
staging, and the shared packer's PSSS task table using the generated thread count.

If optional admission or allocation cannot fit, use the existing bounded
`RawSource` route. If the extra retained owner prevents MO-provider or solver
admission, release it before retrying that bounded stage. Correlated amplitudes
and independent physical replay are unchanged. The source lifetime still spans
the subsequent force endpoint when retained.

## Evidence and limits

Host probes execute the production reference/source prefix with counted owners:
CUDA native reference precedes optional preparation, failed references publish
no source, prior sources retire before replay, exact generic allowances reach the
constructor, optional failure keeps the fallback, and CPU retains its owner.
The pure planner has exact-cap, one-byte-short, shape, overflow and phase-overlap
tests. Source-lifetime probes continue through MO problem construction.

Native NVIDIA numerical and complete-endpoint performance qualification remain
separate. This change does not claim shared HF preparation, source reuse across
RHF replays, or a speedup. The optional route may be refused conservatively when
its full preparation lifetime cannot be admitted.

Agent: dot
