# Decision: include lazy native-library loading in cold timing

Status: implemented
Date: 2026-10-03

## Problem and correction

The comparator verified the loaded library immediately after constructing a
calculator, outside both its constructor and first-calculation timers. For
GenerativeQC the constructor had already loaded the native library. xTBloom's
constructor deferred loading, so the verification call to
`xtbloom.library.load_library()` loaded its native library and CUDA providers
without charging that work to either timer. This understated xTBloom's first
process cold endpoint. It was not evidence that provider loading could be removed
from the production contract.

Verify the actual loaded path after the timed first calculation and before
accepting its sample. A wrong library still rejects the measurement. The timing
contract is now `calculator-cold-v3-lazy-library-load`; mixed old/new reports are
rejected. Cleanup remains separately timed as established by the
[loader-discovery decision](2026-10-03-xtb-loader-discovery.md).

All historical first-process comparisons before this correction exclude some
xTBloom loading cost. This qualifies the first-process claims in the
[device-fact](2026-10-03-cuda-device-facts.md),
[H0 scheduling](2026-10-03-xtb-h0-force-pair-schedule.md),
[bootstrap](2026-10-03-xtb-topology-bootstrap.md), loader-discovery and
[integral-preflight](2026-10-03-xtb-integral-preflight-width.md) notes. Their
warm/changed timings and scientific gates are unaffected. Older constructor
timings also have the separate preceding-cleanup issue described in the loader
note. Do not reconstruct corrected first-process timing by adding an estimated
load cost; remeasure both engines.

## Qualification and evidence

The 26 benchmark tests pass. A fake clock charges 200 seconds for initial native
loading, one for construction, two for calculation and 100 for cleanup. Loading
occurs during GenerativeQC construction or during xTBloom's first calculation;
both cold totals must be 203 seconds. Subsequent constructors exclude preceding
cleanup, and both engines reject an unexpected loaded path. This fails under the
old untimed xTBloom identity-query sequence.

The current native preflight candidate was remeasured on n1 RTX 5090 and n2 RTX
PRO 6000, through finite Slurm jobs. Both 110-sample complete endpoint cohorts
pass all numerical and SCC-count gates and retain ten winning warm and changed
medians. Combined cold totals win 7/10 on n1 and 9/10 on n2 in this cohort. These
counts include a noisy single first-process sample and are not universal claims.

To expose startup variation, six additional fresh Python processes per engine
measure H2 with energy and forces, alternating engine order. Each performs one
cold, one warm and one changed call; all accuracy and SCC gates pass. Constructor
plus first-call medians and ranges, milliseconds:

| GPU | GenerativeQC median | xTBloom median | GenerativeQC range | xTBloom range |
| --- | ---: | ---: | ---: | ---: |
| n1 | 557.674 | 547.780 | 536.652–751.010 | 503.596–641.486 |
| n2 | 393.236 | 381.374 | 387.361–399.549 | 370.574–386.445 |

Thus the remaining measured median startup gap is roughly 10–12 ms, much smaller
than the previous unfair comparison suggested. n1 remains particularly variable;
the larger samples are retained. This endpoint starts after Python package import
and excludes post-sequence cleanup, which is recorded separately. It is not total
OS process startup or a cold-filesystem measurement.

Ignored receipts are `.artifacts/{n1,n2}/cold-library-timing/`, with complete
reports plus `startup-{0..5}-{generativeqc,xtbloom,comparison}.json`. Settings remain
FP64, fresh SCC, 300 K, Broyden 8/0.4, maximum 300 iterations and tolerances
1e-10/1e-8. n1 uses native SHA
`fc23ea615764d458bb3281ceed72749ac0466bd27e32b62bbd0fdbb7fc467145`; n2 uses the
same CUDA objects with source metadata refresh, SHA
`2bb5afe520e8c079785e759f6a621a32b0f2fc987c16cd90c1f7a3c37b12a408`. xTBloom remains
revision `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, binary SHA
`6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Invariant

Metadata verification must not initialize an engine before its timed cold
endpoint. Preserve actual loaded-library verification, every numerical sample
and SCC count, explicit timing contracts, cleanup accounting and startup spread.
