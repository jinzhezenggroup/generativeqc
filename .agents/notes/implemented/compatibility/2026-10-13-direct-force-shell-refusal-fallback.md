# Decision: preserve canonical Direct forces after optional shell refusal

Status: implemented
Date: 2026-10-13

## Problem

Some initial pure-J benchmark controls converged SCF but refused forces with
"prepared Direct owner has no retained shell derivative lease". Repeating with
more free GPU memory passed. An isolated sm_120 process leaving only 8 GiB free
reproduced the refusal: the optional generated shell owner's 64-KiB/thread
recurrence-stack reservation could not be admitted. The prepared canonical
Direct owner and its first-derivative scratch remained valid. The Python
stationary bridge could only report a generic budget error (or use its much
slower tiny-system AO fallback).

## Decision

If the optional shell lease is absent, the full-range host-returning derivative
adapter uses the already retained canonical GPU RSH derivative provider with
omega=0: SR is full range and LR has zero coefficient. Reuse the existing
canonical pair domain, screening, density normalization and derivative buffers.
The adapter preserves independent J/K channels or combines their returned
forces, with a finite-result check, as requested by the original ABI.

This fallback allocates no additional GPU buffers and performs no CPU integral
or reference evaluation. Host source transfers follow the existing stationary
bridge contract. The resident-only callback continues to refuse an unavailable
shell lease; the public stationary consumer then uses its host-returning source
adapter. Missing canonical/derivative capability remains an explicit refusal.

## Rejected alternatives

- Removing the established CUDA stack requirement would change unqualified
  kernel resource assumptions.
- Raising caller budgets cannot create free GPU memory and would break caps.
- Retrying SCF or recreating a detached derivative owner wastes work and loses
  the prepared density, geometry and resource identity.

## Evidence

The old frozen integration `aaec57399` refuses a 12-atom r2SCAN-3c force with
8 GiB free. Candidate `0775fc6f0` completes cold, warm, moved and moved-warm
under the same pressure. Independent E/F errors on those four observations
are at most 2.68e-12 Eh and 3.23e-11 Eh/Bohr. Three isolated GPU tests compare
separate and combined derivatives with Libcint for pure J, restricted hybrid,
and unrestricted hybrid two-water systems; all pass. The test starts fresh
processes so an earlier context-stack allocation cannot mask the regression.

These memory-pressure runs are diagnostics, not ordinary benchmark timings.
Keep the original failed controls and ordinary-resource passing populations
separate in retained evidence. The fallback can be slower than shell execution;
it preserves completion under bounded resources, not a new throughput claim.

## References

Benchmark issue #2416 and pure-J scheduling PR #2450.
