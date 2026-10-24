# Decision: execute benchmark engines independently and retain completed phases

Status: implemented
Date: 2026-10-13

## Problem

The README runner skipped native execution when the GPU4PySCF population was
incomplete. The reducer also discarded all timings when a later phase timed out.
Together these rules hid native completion and reference calls that had already
returned. A reference timeout is evidence about that reference run, not a reason
to suppress the other engine. The user explicitly requested this correction.

This supersedes the whole-population plotting prerequisite in
[the HF/DF publication note](2026-10-13-readme-hf-df-endpoint-statistics.md).
It does not replace the independent numerical gates or the original observations.

## Decision

The README runner executes native calls even when the reference file is absent
or incomplete. The shared endpoint driver offers an explicit
`--allow-missing-reference` mode; other existing consumers retain their strict
reference requirement. Available cold/moved reference baselines validate the
corresponding geometry even when later reference calls fail. A supplied protocol
mismatch remains an error. Missing baselines never use native self-comparison,
the other geometry, or fabricated zero error.

Record finite, converged host energy/forces and execution timing separately from
independent accuracy. Missing accuracy has null errors and a null accuracy gate;
failed completion or a known numerical failure still fails. The reducer checks
raw arrays, reference hashes, geometry/repeat keys and every requested repeat.

Publish a completed cold/moved call or a complete warm repeat population even
when a later phase times out. Preserve all failed/interrupted outcomes beside
these timings. Unchecked timings use open markers and an explicit table symbol.
Whole-population acceptance counts remain separate from phase completion.
Do not convert a whole-point deadline covering setup and multiple calls into a
per-call lower bound or a numerical speedup ratio.

## Evidence

CPU tests exercise the real shell orchestration with missing/partial reference
files, validate partial geometry oracles, reject malformed/nonfinite/unconverged
native endpoints, and preserve complete phases without accepting partial warm
populations. New WB97M-V/48,96 runs use separate finite Slurm allocations with
the original 1800-second point limit and the previously qualified native source
and library. The report retains actual source hashes and outcomes; only the
benchmark and reporting policy changes here.

WB97M-V/48 completes native cold in 781.601 seconds and all five warm calls
at median 89.817 seconds, versus 1216.489/105.410 seconds for the corresponding
completed reference phases. All six native accuracy gates pass. The native moved
call completes in 544.708 seconds with accuracy unchecked; the point limit stops
the first moved-warm replay. Native/96 independently times out in cold/0.
[#2426](https://github.com/jinzhezenggroup/generativeqc/issues/2426) tracks this
native performance finding; #2417 retains the reference investigation. The
completed native advantage and the remaining incomplete sequences are both
visible rather than using one to suppress the other.

The no-reference three-atom Slurm control completes all 12 calls with null
accuracy gates. A separately recorded post-run comparison against an existing
independent reference passes all 12, without altering the original raw statuses
or adding this diagnostic run to the performance population.

## Revisit when

An additional independent oracle becomes available for unchecked geometries.
Any later qualification must record that new comparison explicitly and preserve
the original timing, reference identity and execution outcome.
