# Decision: select the canonical source for its discrete screening oracle

The later [through-f value policy decision](../performance/2026-10-02-through-f-value-policy.md)
supersedes this note's bounded-default assumption after matched endpoint timing.
The original oracle and unchanged numerical thresholds below are preserved.

Status: implemented
Date: 2026-10-01

## Problem and decision

PR #1666 changes the default full-range through-f value route to the bounded
shell owner. `canonical_screened_values()` still compares with an independent
CPU oracle that screens Cartesian AO pairs before public spherical projection,
and asserts the exact admitted canonical candidate/radial counts. The bounded
shell route also uses shell/density screening, so its coarse-threshold output is
not required to equal that different discrete Cartesian-pair-screened matrix.

The fixture now verifies that both owners were prepared, disables only its local
plan's `bounded_value_capability`, and requires that no full-range shell source
remains selected. The original screening thresholds, independent ERI oracle,
3e-12 matrix tolerance, full/SR/LR operations, both spin channels, moved geometry,
prefix/dense comparison and canonical work assertions are unchanged.

`canonical_value_provider()` continues to exercise the default bounded provider
before its separately scoped fallback probes, with all output masks, Cartesian
and spherical bases, both spin layouts, two geometries and independent CPU ERIs.
Its zero-canonical-work check is route evidence, not total-work evidence. A host
regression protects the separation and the unchanged native acceptance gates.
No production source or default-selection policy is modified by this fix.

## Reported NVIDIA evidence and limitations

The [device receipt](https://github.com/jinzhezenggroup/generativeqc/pull/1666#issuecomment-5942111176)
identifies frozen source `658aed180defb2154a42356c17903c14d8ed4436` and library
SHA-256 `e5f8490a5b7e68806e90ebcece9c72960bbc6f8915791a9fe04c2a68f9dcad5d`.
The original job 12016 remains exit 1 at the screened fixture. It reports passing
unmodified through-f response (including both mixed-census modes), range response,
canonical work and 15 WB97M-V tests, plus one supplemental independent f-shell UKS
gate in job 12020. Jobs 12021/12022 report that selecting the canonical fixture
route passes the same screened assertions and complete canonical-value suite
against that unchanged library. This checkout implements that described semantic
correction independently; the remote raw patch bytes are unavailable here and
byte identity with its recorded patch hash is not claimed. No local NVIDIA run
of this new test source is claimed.

## Default-promotion gate remains open

The receipt records adverse one-shot synthetic full-JK source intervals:
0.420444/0.472187 seconds at 58/116 public AOs versus approximately
0.0185/0.0487 seconds for the frozen #1651 census. These are not matched warmed,
repeated complete endpoints; they do not establish a cause or a speedup. Zero
canonical counters do not count the new bounded-shell work and cannot erase this
negative observation.

Before accepting the new default, the
[performance qualification checklist](../../../../docs/maintainer/performance_engineering.md#performance-qualification-checklist)
still requires matched complete endpoints and scientific settings, actual total
physical-work counters for baseline and candidate, cold/warm/changed-geometry
behavior, a larger case capable of exposing a scheduling cliff, and supported
batch/constrained-memory behavior. Numerical qualification alone does not
complete this performance/default-promotion gate. Use the qualified #1664 parent
as the incremental endpoint baseline: its derivative-scheduler improvement must
not be attributed to #1666's separate value-route promotion.

Agent: dot
