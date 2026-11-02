# Frozen WB97M-V fitted-LR production endpoints

This packet qualifies the measured scientific source
`bbf5033f320dec96157df65e5f07e660d4f5de58b2946c34a6128814a361922e`,
not the later review fixes in PR #2932. Fitted long-range exchange is an explicit
candidate; exact/mixed LR and the generic LR derivative remain the defaults.
The matched reference uses GPU4PySCF with the same fitted LR operator, geometry,
basis and checksum-bound auxiliary input. Native timed processes reject imports
of PySCF/GPU4PySCF. Production cores were built with `BUILD_TESTS=OFF`; separate
hook-enabled fixtures supply fault controls, never production timings.

Each device has three fresh native/reference process pairs with alternating
order. Each process measures one complete cold constructor/prepare/SCF/force/export
call and three warm calls from its own frozen cold density. All sixteen cross
energy/force comparisons per pair pass the unchanged common gates. These are
static-geometry paired comparisons, not a new independent dense-grid acceptance
or a moving-geometry/all-functional/all-GPU qualification.

| Device | Native cold seconds | Reference cold seconds | Stable warm reference/native ratio |
| --- | ---: | ---: | ---: |
| A100 | 89.706–90.455 | 86.317–87.580 | 1.165 (three pairs) |
| RTX 5090 | 276.812–279.739 | 373.167–376.373 | 1.891 (two stable pairs) |

A100 cold calls remain about 3–4% slower. Native cold SCF uses 28 iterations;
GPU4PySCF uses 10. Both warm paths use one iteration. The additional 5090
reference-first pair 15482 has reference warm times 278.518, 198.984 and
126.624 seconds, so its timing population is nonstationary. It is retained in
full, including its numerical gates; the reader reports both all-pair and
stable-pair statistics instead of using its large ratio as a stable speed claim.

## Original failures and scope

The capsule retains every original exit code. A100 molecular job 224722 has
83 passes, 48 failures and one skip: 26 compiled host-only probes could not use
the old cluster loader, and 22 GPU tests incorrectly required a Slurm identifier
on the actual LSF allocation. Later fixture corrections and new scientific-source
runs need separate evidence; this run is not relabelled as passing.

5090 job 15481 failed its fixture packaging and was retried in 15567. Job 15518
had one semilocal response failure. Repeats in 15667/15668 and unfiltered memcheck
and initcheck passed, but the original failure is nonreproduced, not established
as fixed. Mixed-owner witness job 15669 also remains a failure. Their complete
logs and recipes remain available alongside the successful endpoints.

## Replay and retention

```bash
PYTHONPATH=python:. python benchmarks/results/wb97mv-production-20261102/report.py \
  --output /tmp/wb97mv-production-replay.json
```

The unchanged exact FP64 v3 codec authenticates each original decoded text.
No numerical value, full force vector or endpoint sample is rounded or omitted.
The packet retains build recipes, CMake configuration, compiler-cache statistics,
source/core/AOT identities and all scientific/diagnostic logs. Verbose successful
compiler transcripts are represented by length and checksum in
`compile_log_inventory`; original build archives remain ignored locally. Binary
files are identified separately rather than duplicated in Git. All existing
storage limits remain unchanged.
