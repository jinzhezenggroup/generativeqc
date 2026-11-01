# Decision: reuse exact VV10 force seeds in the native stationary owner

Status: implemented
Date: 2026-11-01

## Problem and decision

The standalone nonlocal force owner already compared complete VV10 ingredients
before repeating a fixed-geometry pair sweep. The native stationary controller
unconditionally recomputed the same seeds. A complete 5090 WB97M-V endpoint
profile attributed 48.3% of accumulated GPU time to VV10 feature and geometry
kernels. Source-TU and strict-packet specializations failed to improve their
complete consumers; their [rejected alternatives](../../rejected/2026-11-01-range-source-tu-and-vv10-strict-packet.md)
remain documented.

Share the existing device key comparison, pair-skip, seed-pack-skip and key-store
operations. Geometry, grid coordinates/weights and model parameters belong to an
immutable controller lifetime. Seven bitwise-equal finite ingredients cover the
pair closures and row finalizers: omega, kappa, their three local derivatives,
weighted density and density. The current AO/grid geometry pass still consumes
the current gradient direction. This is neither a density tolerance nor a hash.
The previous and current error flags also enter validity.

The existing six seed rows stay resident. The pair workspace's dead energy/index
scratch retains density and validity; the six local-scale rows already remain
intact. The existing unused scalar slot after the force seeds holds the device
comparison predicate. Pair energy instead reduces into dead energy-term scratch,
just as the standalone force owner already permits. After force completion,
only that predicate is downloaded to report the actual cache decision.

No device allocation or capacity receipt changes. This preserves both original
fleet tiers, geometry/AO-map admission and the nonlocal sub-budget. Minimum-
capacity owners can also reuse exact seeds. Every error/early return invalidates
readiness; changed schedules and explicit disable force a new sweep. Geometry
replacement creates a new controller. Public batch owners remain independent.

Unset or empty `GENERATIVEQC_VV10_COMPONENT_REUSE` selects the existing
65,536-point crossover. Explicit `1` admits smaller fixtures; other nonempty
values disable it. Existing pair-domain guards bound the point count and
coefficient. Selection has no GPU-model whitelist. All scientific arithmetic,
density screening and independent acceptance thresholds remain strict FP64.

## Rejected metadata allocation

An earlier prototype added 32 optional bytes for a mismatch flag and counters.
A scheduled real-device probe showed that a real `cudaMalloc` OOM remains in
CUDA's last-error slot after successful device selection, stream creation and
kernel launch. Catching `bad_alloc` alone broke its optional fallback. A corrected
prototype cleared only the expected OOM and passed real allocation-failure gates.

Further fleet admission review rejected even the corrected allocation: raising
the cached quote by 32 bytes can make a tightly bounded two-tier fleet choose its
minimum schedule, evicting much more valuable geometry storage. Borrowing the
existing scalar slot eliminates both the allocation failure and this scheduling
cliff. All prototype images, negative evidence and timings retain their original
identities in ignored local artifacts; they are not final-source measurements.

## Evidence and limits

The [source-matched evidence bundle](../../../../benchmarks/results/stationary-vv10-reuse-20261101/README.md)
records native source
`6527e5625b096f10a8c431ce2829b67d44a98abb656889f486f2b13786dc419c`
and library
`327a417cb66c115b7f5d35ca9fde6a89ec217da632e2d0cb0c589bead84b614f`.
Two complete ABBA cohorts pass ten independent population gates. Warm medians
fall from 58.383 s to 30.079 s on 5090 and 42.947 s to 14.328 s on PRO 6000;
cold medians remain within the observed spread. Cold plus three warm endpoints
improve by 20.70% and 25.48%. The 108-atom cold control is 0.78% slower; its
retained, evicted, moved and moved-warm calls all remain in the bundle.
The 5090 trace records one cold pair sweep followed by two exact-key warm hits
with zero pair sweeps. Both nine-field public capacity receipts are identical.
 Earlier sources `ba54e9ea...`
(initial allocation) and `23850dbc...` (corrected OOM handling) remain separately
frozen; neither is relabeled as the allocation-free implementation.

The native tests compare energy within 2e-8 Eh and every force component within
2e-7 Eh/bohr against independent GPU4PySCF results. README endpoints retain their
existing independent reference/grid gates. Diagnostic trace time is excluded
from clean ABBA timings. Cold plus three warm calls must be reported alongside
individual cold and warm populations. A larger 108-atom six-phase lifecycle
checks retained, evicted, moved and moved-warm behavior. A two-item batch checks
independent state and selective minimum-capacity admission, including a
transactional refusal to shrink an already-live controller.

A100 connectivity remains unavailable for this revision; SM80 compilation and
Blackwell timings cannot stand in for its measurements. No universal device
speed guarantee follows from this bounded exact-reuse change. Revisit the
crossover or owner scope only with complete endpoint, numerical, lifetime and
actual-work evidence on the newly admitted domain.
