# Decision: compose prepared density with formal XC admission

Status: implemented
Date: 2026-10-05

## Problem

PR #1907 binds density arithmetic and scalar/tiled launchers once during setup,
replacing the old per-enqueue density-precision enum. PR #1872 adds independent
physical-layout and three-state formal-qualification gates to that old enqueue
API. Keeping only either conflict side would lose a scientific admission gate or
restore runtime selection and a removed API.

## Decision

Keep #1907's preparation and phase-only execution. Require both executable
physical-layout support and `Qualified` point-program mixed-density capability
before preparing non-strict density bindings. The constructor recomputes formal
traits from the functional identity; caller-provided flags cannot admit a path.
The common generated binding still validates the complete arithmetic directive
and preserves canonical request, candidate, and precision identity.

This actual-master integration retains #1907 head
`69b7bc4050fc9d9c6e8ab2e843aaf1a9f84afb09` and real landed master
`a32b6cfd63ad07d66f8494591031822e1483e69f` as its only parents. Master includes
#1872 at author head `3e18aa5723422488932412976f4e39af794500ce`.
The previously reviewed prospective composition is reused only as source
resolution evidence, never as a commit parent. Both author branches carry the
same native precision-census repair, which is retained once; compiler failures
in the host admission fixture continue to report stderr and fail the test.

## Invariants

- Generic local-AO defaults depend on legal physical execution, not a method whitelist
- Mapped density remains strict; independently qualified Coulomb J may remain mixed
- Nonlocal compositions lower J only; strict refinement restores every region
- Full/tail and local launcher tables are setup-only and transactional
- Capture/replay does not select, allocate, widen admission, or change lifetime rules
- Unknown costs retain the incumbent; no equations, thresholds, or defaults change
- Native rejection tests retain no-submit, published-result and generation checks

## Evidence

The bounded host suite combines existing density selection, generic local-AO,
capability/registry, precision census, replay, feature export and compiler tests.
The preparation test executes the real production preparation/binding methods
with generated canonical lowering tables across physical support and all three
formal states. It checks invalid directives, immutable post-evaluation bindings,
exact full/tail launchers, strict local maps including an empty tile, and retained
request/precision identity. The KS test crosses those gates with AUTO/FP64,
nonlocal composition and strict refinement. No new GPU, complete native build,
chemistry or endpoint timing qualification is claimed.

## Revisit when

A physical layout or point program receives new independent numerical
qualification. Such promotion must update its owner-controlled capability census
and qualification evidence; broadening a setup/replay guard is insufficient.
