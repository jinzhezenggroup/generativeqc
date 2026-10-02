# Decision: fixed-density CUDA DFT component profiling

Status: implemented
Date: 2026-10-02

## Problem

Issue #1480 could time complete SCF replays and fixed-final-state forces, but it
still serialized the fixed-density J/K/XC boundary as unavailable. Full-SCF
trace time cannot isolate one physical density because iteration count, DIIS and
convergence control are part of that measurement.

## Decision

Add a private exact-token diagnostic on the resident CUDA KS owner. At one
already-converged density it replays the existing prepared providers separately:

- Coulomb J;
- primary full-range K when present;
- the existing long-range correction provider when present;
- the existing device-fused semilocal XC body.

CUDA events bracket only submitted device work. Event creation, token checks and
scalar error reads are outside the component times. The diagnostic does not call
SCF, alter density/final-state generations, change provider selection or publish
a new XC generation. Nonlocal compositions leave semilocal-XC timing unavailable
because a semilocal-only replay would overwrite the final VV10-composed XC view.

The benchmark runs this only in its separate intrusive profiling pass, repeats
the same fixed-D boundary, reports medians plus raw samples, and preserves null
for components not measured by the selected method.

## Invariants

- No new J/K/XC implementation exists in benchmark code.
- Scientific coefficients, provider approximation, metric/screening policy and
  precision remain those of the prepared KS owner.
- Clean endpoint wall timing and intrusive event profiling remain separate.
- Missing measurements are never inferred as zero.

## References

Issue #1480.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Failure publication audit

The diagnostic borrows the CUDA KS owner's live J/K/range-K buffers and the
semilocal XC owner's potential/totals. After any component submission, an error
drains the same provider stream and revokes the KS final-state, final-frame and
stationary-weight views, including the final generation. It also clears partial
profile results. Preflight rejection preserves the prior token; successful
repeated profiles preserve it as well. The last-good warm density is separate
storage and is retained for explicit recovery.

The executed-host control-flow test covers both spins, exact and fitted provider
routing, nonlocal/host-XC omissions, repeated profiles, and faults during enqueue,
event completion, XC execution and numerical-status checks. These tests validate
publication and buffer routing, not GPU numerical or performance acceptance.

Agent: dot
