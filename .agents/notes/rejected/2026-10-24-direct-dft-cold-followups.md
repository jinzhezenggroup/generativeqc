# Experiments: Direct DFT cold paths not promoted

Status: rejected for default promotion in this campaign
Date: 2026-10-24

## Sampled AO discovery chunks

A bounded native selector grouped up to 128 consecutive sampled-jet map tiles,
borrowing dead projection scratch for flags. The predicate, canonical AO jets,
cutoff, maps, precision and downstream force work were unchanged. Host staging
admitted old retained bytes plus `(12 * tiles + 16) * nao`, with transactional
publication and single-tile/old-library/dense fallbacks. This was a serious
implementation, not an extrapolation from a kernel time.

On frozen master `72e0a7e5a011592428b5625acdfa6f19168645ac`, one same-binary
single/chunked cold-plus-warm pair per method gave these complete Direct
Standard/96 endpoint times (seconds):

| Method | Single cold / warm | Chunked cold / warm |
| --- | ---: | ---: |
| PBE | 17.519362 / 5.225768 | 17.359380 / 5.231781 |
| PBE0 | 21.341514 / 6.738234 | 21.027721 / 6.696261 |
| r2SCAN-3c | 41.553227 / 15.494773 | 41.215277 / 15.469133 |

The cold reductions are only 0.91%, 1.47% and 0.81%, below the prespecified
2% minimum floor even before repeated-population uncertainty. No default is
promoted, and these pilot values are not pooled with another experiment.

PBE discovery transactions drop from 2,107 to 17 and the measured discovery
scope drops from 3.0124 to 0.8340 seconds. This does **not** save 2.18 endpoint
seconds: the old synchronous scope also waited for previously enqueued force
work. All 8,281,620,480 discovery AO jets and selected point-AO work remain.
The complete endpoint is the deciding evidence. The experimental native
`discovery_passes` counter aggregated a whole chunk; it must be moved inside
the tile loop if this candidate is revisited, to distinguish actual AO passes
from host transactions. The host counters already distinguish these quantities.

All 12 endpoint E/F calls pass the independently qualified dense references.
104 host cases and 43 real-GPU cases pass, with nine chunked cases under each
of memcheck/synccheck and zero errors. The retained candidate source, tests,
build/cache logs and samples live at
`node1:/data/jzzeng/direct-dft-map-chunks-20261024/`, with a frozen
`rejected-evidence.tar.xz` (58 files, 113,411,320 bytes) and its per-file manifest.
Archive SHA256: `a121c9c69623ea962310c31a4c8f44bef67e95ae464501cb50b8bb47e2578879`. No implementation from
this losing schedule enters the automatic occupied-XC change.

Revisit only if a producer actually removes AO work or a new complete-endpoint
population demonstrates a robust benefit. Do not infer an endpoint saving by
subtracting overlapping host scopes.

## Old compact-density and native-binding candidate

An earlier local candidate based on `fe709e74a5ed1a1a53285c0336a22374ad406f18`
prepared canonical symmetric density factors once per compact tile in dead
potential scratch and removed a duplicate native-library hash. Its separately
qualified old-base cold gains (3.35–4.13%) are historical. The patch was never
committed or promoted. Master #2717 independently landed the native binding
change before submission, so those combined gains cannot be claimed again.

PR #2703 also introduced an occupied-XC template parameter. **At the current
base it remains explicitly opt-in, not active by default.** The dense factor
optimization therefore was not automatically superseded by that merge; its
remaining matrix-fallback value would need a compatible reimplementation and
new endpoint evidence. The current campaign instead qualifies automatic use
of the already implemented occupied route, which avoids more work on eligible
cold meta-GGA proposals. First/held/fallback density products still exist.

All earlier sources, samples, failures, profiles and scripts remain in the
237-file, 137,546,056-byte archive at
`node1:/data/jzzeng/direct-dft-chase-20261024/qualified-evidence.tar.xz`, also
verified locally in that worktree's ignored artifacts. SHA256:
`c56a8dd277ce5bc338834cddd3945d5d83379fb335dabb5eb18a29685e912505`.
Do not transplant its historical measurements onto current master or discard
its failed controls. No Release, release asset or tag was created.
