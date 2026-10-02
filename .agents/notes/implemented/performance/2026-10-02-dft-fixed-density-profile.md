# Decision: profile CUDA DFT J/K/XC at fixed final density

Status: implemented
Date: 2026-10-02

## Problem

The #1480 force-performance report could repeat a whole SCF under the DF trace,
but that mixes J/K/XC cost with iteration count and left the real fixed-density
SCF component boundary explicitly unavailable.

## Decision

Add a private token-checked diagnostic on the existing CUDA KS owner. It reuses
the exact resident final density and already-prepared Fock/XC owners to replay:

- Coulomb J;
- primary full-range K when present;
- the explicit long-range correction K when present;
- semilocal device XC when the final owner is not a nonlocal composition.

Each submitted component is bracketed by CUDA events on the existing provider
stream. Validation, event creation, status reads and setup are outside the
reported milliseconds. The diagnostic never enters the SCF solver, changes the
density generation, or relabels a warm SCF pass as fixed-density evidence.

The boundary is intentionally intrusive: J/K and XC scratch may be overwritten.
It is therefore benchmarking infrastructure, not a public calculation API.
VV10 remains separately measured because replaying semilocal XC alone would
overwrite buffers belonging to the composed nonlocal final state.

## Invariants

- Method/provider identity comes from the prepared owner; no benchmark-specific
  J/K/XC implementation is introduced.
- Missing components remain null through a present-mask contract.
- Direct and density-fitted full-range providers use their existing production
  device paths.
- RSH correction timing follows the prepared correction owner and is reported
  separately from primary full-range K.
- The exact final-state token is checked before and after profiling.

## Evidence

Device-free source contracts ensure the profiler calls the existing provider/XC
seams, contains no solver entry, and the benchmark consumes it through a live
NativeKsSnapshot rather than calling batch.execute.

Real GPU timings remain evidence to collect; this PR adds the measurement
boundary and does not claim a speedup.

References: #1480, #1553.

Agent: ChatGPT
Model: GPT-5.6 Sol
