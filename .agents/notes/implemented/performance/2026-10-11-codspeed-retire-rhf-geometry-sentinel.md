# Decision: Retire unstable RHF changed-geometry CodSpeed sentinel

Status: implemented
Date: 2026-10-11

## Problem

The `test_cpu_rhf_changed_geometry_pair_walltime` benchmark repeatedly triggered
the repository's 5% CodSpeed regression threshold in unrelated PRs:

- [#2259](https://github.com/jinzhezenggroup/generativeqc/pull/2259)
  compressed historical performance receipts, with no RHF execution changes;
  CodSpeed reported 26.3 -> 27.8 ms (-5.24%) against master `858fe52`
  ([check](https://github.com/jinzhezenggroup/generativeqc/runs/114410399675)).
- [#2260](https://github.com/jinzhezenggroup/generativeqc/pull/2260)
  moved only aggregate CI pass gates to ubuntu-slim, again reporting
  26.3 -> 27.8 ms (-5.24%) against the same master
  ([check](https://github.com/jinzhezenggroup/generativeqc/runs/114411416446)).
- [#2255](https://github.com/jinzhezenggroup/generativeqc/pull/2255)
  reported a similar -5.2% for the same RHF benchmark
  ([check](https://github.com/jinzhezenggroup/generativeqc/runs/114405713190)).

The #2259 and #2260 Actions logs explicitly qualified exact-master-base and
matching recorded CPU/runtime fingerprints. Thus stale-base and detected
cross-CPU comparisons do not explain those reports. This evidence is strongly
suggestive of an unreliable regression signal, **not** a proof of the exact
root cause; compiled layout, runtime state and instrumentation differences
were not isolated.

The benchmark reuses a stateful warm-start batch across successive geometry
updates. It checks convergence and finite energy but does not pin equal
SCF/Fock work or demonstrate a stable same-runner noise envelope.

## Decision

Remove only this changed-geometry CodSpeed benchmark and its now-unused
coordinate-shift helper, plus its URI from the bounded same-runner diagnostic.
Update source-bound policy and diagnostic fixtures so PR/master run four
benchmarks and scheduled/manual full tier runs five.

Retain the existing RHF and PBE warm endpoints, WB97M-V small-grid endpoint,
PBE analytic-force endpoint, and scheduled formaldehyde RHF endpoint. Keep
the existing 5% policy for the remaining eligible performance measurements,
scientific correctness coverage, exact-base/CPU-fingerprint admission, and
artifact-only ABBA diagnostic semantics unchanged.

## Rejected alternatives

- **Raise every benchmark's threshold:** masks real regressions elsewhere.
- **Treat every red check as a required optimization:** fails causal attribution
  for CI- and evidence-only PRs.
- **Silently ignore results while continuing to upload:** perpetuates noisy
  reports and unnecessary instrumented runtime.

## Invariants and consequences

Historical CodSpeed records remain accessible; no measurement is rewritten,
relabelled as a speedup, or published as new evidence. The benchmark definition
digest changes deliberately, so the existing fail-closed baseline qualifier may
skip incomparable PR uploads until a new exact-source master baseline is
published after merge. This is advisory, not a scientific correctness waiver.

Changed-geometry and warm-start behavior remain exercised in other native,
batch and numerical qualification suites; this removes only an unstable
**routine performance sentinel**, not the API or its functionality.

## Revisit when

A replacement identifies and fixes the warm-start state/work dependence,
freezes the actual operator/Fock counts, and demonstrates stable multi-repeat
same-runner baseline/head measurements, including cold/moved/warm definitions
and numerical correctness gates.
