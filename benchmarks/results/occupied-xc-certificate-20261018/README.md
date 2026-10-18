# Occupied-XC cancellation-certificate requalification

This packet qualifies the repaired opt-in path, not the former exponent-only
image. The default remains the original stored-density matrix consumer.

- Scientific implementation and proof: [Agent Note](../../../.agents/notes/implemented/numerics/2026-10-18-occupied-xc-error-certificate.md).
- Source-matched control: the same frozen native image with occupied reuse `0`
  versus `1`; force overrides unset, strict FP64, unchanged scientific settings.
- Full returned force arrays (lossless FP64 XOR deltas), independent pinned oracle geometries, every
  repeat's actual SCF/Fock counts and provenance are in `evidence.json.xz`.
- `python benchmarks/results/occupied-xc-certificate-20261018/verify.py` verifies
  checksums and every retained complete population against the independent
  energy/force gates (`1e-8` Eh / `1e-7` Eh/Bohr). Warm summaries are medians
  of all five calls, not iteration-normalized estimates.
- Native probes and endpoints use finite Slurm jobs with assigned visibility;
  no device override, relaxed tolerance or changed grid/convergence is used.

The compact packet retains negative results. Force reconstruction preserves
every original FP64 bit and verifies a separate per-call checksum; it is not
rounded residual storage. Raw profiler databases, full per-file source manifests,
compiled libraries, build logs and frozen source trees remain ignored local
artifacts; their identities and reduced diagnostics are retained rather than
committing transient multi-megabyte profiler/build output. Storage caps and
historical evidence are unchanged.

## Source and execution boundary

Implementation: `46e5f88d39645c0166264957548d1ec3b41d63c3`; native identity
`ef2d34fcc1d1e87faedbd040c2de6f5af338d4815650b00fd65f94051574bd33`;
library SHA256 `72a1fd52a805d63f74923279f29800645ff20a5d427a9e8ca90ddfd3ddc4bb10`.
All 1889 production inputs match the frozen image. Original-image controls
use native identity `f7c524732fe0852a5f4a6ebc18efec6526d349c9e1103be72fb28c5b87382ac5`.
The source receipt preserves the pre-commit freeze base and dirty-patch hash;
the measured image is not relabelled with an unmeasured future commit.
GPU tests use jobs 10120/10133; complete endpoints and profiles share finite
job 10133 and RTX 5090 UUID `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`.

This is the frozen composed repair, **not current-master benchmarking**.
During the campaign #2587 merged as `7915dce5e89064761cfeaabfa3b27138897895b3`,
#2588 advanced to `56dfa3ef85aa8649e4ff68b8e47dda10048a12e0`, and base #2596
remained at `bf99314e4fc4ca7070ea0b048c72b9d07685d0d6`. Rebase/integration
must retain source identity and be requalified before any later default decision.
The repair does not silently incorporate unrelated intervening changes.

## Complete E+F results

Seconds, same-image matrix `0` → certified opt-in `1`. Both force overrides
remain unset. r²SCAN-3c uses canonical spherical def2-mTZVPP, moving Becke
48×16×32 per atom and strict FP64; PBE0/HF retain the original def2-SVP controls.

| Case | Cold | Warm median (5) | Moved | Moved-warm median (5) |
| --- | ---: | ---: | ---: | ---: |
| r²SCAN-3c / 96 | 242.955053→279.241177 | 29.399903→29.456770 | 143.150417→162.011812 | 29.388973→29.523056 |
| r²SCAN-3c / 48 | 59.591552→69.475624 | 7.744333→7.715781 | 41.140342→47.817563 | 7.704989→7.725966 |
| PBE0 / 48 | 23.043093→22.920929 | 3.831007→3.830832 | 21.686443→21.615480 | 3.828974→3.824608 |
| HF / 48 | 10.983082→11.069124 | 0.744169→0.742967 | 7.225771→7.227426 | 0.743173→0.743395 |
| r²SCAN-3c / 3 | 1.601456→1.634616 | 0.176065→0.176571 | 0.465713→0.476463 | 0.176602→0.176843 |
| r²SCAN-3c / 12 | 8.464716→9.142533 | 1.388102→1.380718 | 6.146383→6.651063 | 1.383816→1.383121 |

96 cold/moved regress **14.94%/13.18%**; 48 regress **16.59%/16.23%**.
Primary pairs have matched SCF/Fock trajectories. Reverse96 is
243.595169→287.995689 cold, 29.491720→29.454993 warm,
152.063338→162.034193 moved, 29.496615→29.466373 moved-warm.
Reverse cold counts are 24/24→25/25; moved counts are 14/14→13/13.
All warm calls are 1/1. Do not normalize away these convergence differences.
All six original-image controls and all individual repeats are retained in the packet.
Original-image → repaired-candidate percentages (not isolated certificate gains):

| Case | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| r²SCAN-3c / 96 | +14.566% | +0.170% | +12.960% | +0.223% |
| r²SCAN-3c / 48 | +16.234% | −3.532% | +15.148% | −2.558% |
| PBE0 / 48 | −1.054% | −7.791% | −3.474% | −8.627% |
| HF / 48 | −0.414% | −0.155% | +0.027% | −0.081% |
| r²SCAN-3c / 3 | −1.734% | +0.386% | −2.854% | +0.023% |
| r²SCAN-3c / 12 | +7.715% | −0.406% | +11.022% | +0.514% |

All 20 native populations / 240 calls finish; no incomplete native populations.
Independent maximum native energy/force errors are 2.547e-11 Eh / 5.735e-11
Eh/Bohr, versus unchanged 1e-8/1e-7 gates. This passes the measured scientific
gates but **fails performance/default promotion**. It is not GPU4PySCF parity.

## Source-matched profiling

These are separate counted cold/first-warm calls, not replacements for the
uninstrumented endpoint table. Cold counts differ 24/24→25/25; warm is 1/1.
Top-level wall ledgers are mutually exclusive and sum to each complete call.
Device sums are reported separately and must not be added to those wall times.

| Metric (seconds) | Cold 0→1 | Warm 0→1 |
| --- | ---: | ---: |
| Complete wall | 244.299978→288.774532 | 29.621759→29.619307 |
| SCF + publication exclusive wall | 207.951815→252.454796 | 8.280222→8.276486 |
| Force + publication exclusive wall | 24.816063→24.868757 | 19.440025→19.441231 |
| Final validation exclusive wall | 0.566901→0.562072 | 0.555877→0.547641 |
| Endpoint preparation exclusive wall | 9.169020→9.086957 | 0→0 |
| SCF J kernel device sum | 101.290615→105.435569 | 4.224331→4.222255 |
| XC/AO/grid kernel device sum | 95.051359→135.075474 | 3.961256→3.959826 |
| Generated low J′ device sum | 3.833379→3.832633 | 3.832858→3.832395 |
| Order-6 J′ device sum | 2.610987→2.610573 | 2.611254→2.610556 |
| Order-7 J′ device sum | 0.491188→0.490844 | 0.490961→0.490703 |
| Order-8 J′ device sum | 0.454026→0.453627 | 0.453888→0.453891 |

Full exclusive host/device-window ledgers retain other force/Becke work,
one-electron preparation, publication, synchronization and unattributed gaps.
The candidate cold standalone/compact occupied feature device sums are
25.541778/18.377863 s (6992/25208 launches). Their certificate/refusal work
is not a contraction win; individual refusal counts and hardware FLOPs were
not instrumented and remain unavailable, not inferred from launches.

Force work is identical in both arms and both profiled calls. Order-6/7
primitive-pair product counters are 8,666,248/1,250,544. The density-contracted
subpaths attempt 4,099,576/811,824 shells and accept 4,072,824/809,776;
their density-term counts are 1,882,836,519/453,127,296 and Hermite-product
counts 3,153,588,800/870,721,600. These are distinct counters, not hardware
FLOPs or mutually additive totals. Missing generated-class counters remain
`null` with covered subtotals, not fabricated global totals.

## Fresh same-card GPU4PySCF peer

GPU4PySCF 1.8.1, same Slurm allocation/card and unchanged endpoint protocol:

| Engine / 96 atoms | Cold | Warm median (5) | Moved | Moved-warm median (5) |
| --- | ---: | ---: | ---: | ---: |
| GPU4PySCF | 154.894258 | 26.401694 | 146.203766 | 26.449376 |
| Certified opt-in | 279.241177 | 29.456770 | 162.011812 | 29.523056 |

The repaired opt-in remains **11.57%/11.62% slower** for warm/moved-warm;
the historical approximately 26.6-second target is not substituted for this
fresh observation. Peer SCF-iteration/J-K-build counts are 60/61 cold and
57/58 moved; warm counts are 3/4, 4/5, 1/2, 3/4, 3/4 and moved-warm counts
3/4, 3/4, 5/6, 1/2, 1/2. Native and peer trajectories differ: these are complete
elapsed endpoints, not isolated arithmetic improvements or normalized timings.
All 12 peer calls pass independent gates (maxima 1.819e-12 Eh / 4.848e-12
Eh/Bohr), as do all four diagnostic calls. All **256** returned force arrays
are independently checked and losslessly replayable in the retained packet;
there are zero incomplete populations and the finite controller exits zero.
Slurm accounting storage is disabled; terminal controller/cleanup receipts,
not an unavailable `sacct` history, establish completion.

## Validation and decision

300 host tests plus 55 subtests pass, including 26 certificate cases and an
800-digit independent Decimal oracle. All 104 emitted real-CUDA cases and
16 native-owner cases pass. Each of memcheck/synccheck/racecheck passes 84
emitted cases and all 16 native cases, with zero errors/hazards. The exact
implementation commit has 30 successful CI checks and four skipped checks.
The cache-enabled native build, ownership checks, changed-file hooks and
standard Sphinx warnings-as-errors pass; initial missing-library host setup
and optional nitpicky cross-reference failures are retained as diagnostics,
not hidden by changing tests or suppressing warnings.

Default remains **off**, and the PR remains draft. The smallest next step is
proved early refusal plus bulk reuse of the existing tiled matrix consumer,
avoiding projection/certificate work followed by costly per-point D recomputation.
Do not re-enable on the basis of isolated projection time or old image timings.
