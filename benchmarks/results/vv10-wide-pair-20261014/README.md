# Experimental adaptive VV10 pair tiles

`GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-wide` retains the four-row schedule
below 65,536 active points and uses eight rows with immutable row-input reuse
above that boundary. Both branches use the device count and the canonical FP64
pair roots, admission, output leases and ordered fallback. Defaults are unchanged.

Same-library 3/12-atom off/on populations pass all 48 independent E/F calls at
the original FP64 / 48×16×32 / 100 iterations / 1800-second point protocol.
Both arms enable exact two-electron and geometry-response reuse; the selector
is `parallel` versus `parallel-wide`.

| Atoms | Off cold | On cold | Off warm | On warm | Off moved | On moved | Off moved-warm | On moved-warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 40.51289 | 7.61240 | 0.20375 | 0.20058 | 5.65667 | 5.71443 | 0.20090 | 0.19859 |
| 12 | 31.47373 | 30.60593 | 0.69799 | 0.69285 | 19.91066 | 19.46107 | 0.69781 | 0.70487 |

Seconds cover complete endpoints; warm values are five-sample medians. The
3-atom cold difference is dominated by force preparation (35.068 s off versus
2.230 s on); it does not establish a kernel-caused cold speedup. All setup cost
and the original samples remain included. This
does not qualify a default promotion or completion of #2426.

Qualification on finite n4 RTX 5090 Slurm allocations:

- Job 916: the actual generated templates pass an independent 17-point
  extended-precision oracle and full-output parity through 320k points. Five
  forward/reverse timing rounds retain the 65,535/65,536/65,537 transition.
  At 320k, adaptive feature/geometry ratios are 0.9076 / 0.8884. At 10k and
  65,535 points, the ratio is approximately 1.000. These are kernel intervals.
- Job 917: 330 admission, partial-tile, signed/zero-domain and canary tests pass.
- Jobs 918 / 919: memcheck / synccheck, 16 tests each, zero errors.
- Job 920: 337 tests pass, including six independent molecular E/F and
  reconverged displaced-energy cases and stale/failure recovery.
- Job 921: 29 phase-storage, ownership/stream, component-cache and independent
  nonlocal derivative cases pass. Host qualification: 139 passed.

The small populations use n1 job 8459. A prior invalid comma-separated size
argument exited with status 2 before computation and remains recorded. On the
user's request to wrap up, jobs 8455/8456/8457 (partial-cache 96, 48 off/on,
full-cache 96) were cancelled. The 48 off arm had returned cold and five warm
calls; neither 96 arm had returned cold. These are **user cancellations**, not
timeouts, complete populations or evidence that the 96-atom issue is fixed.
Their input, library identity, partial records and cancellation receipt remain.

Measured/built source: `de90d952d930a0584446dff30425369d45be4eab`.
Library SHA256: `0051d04a4d7dc4d842da3f96a9c485088e8363ae6ad86a09fd76dd142215b38a`.
Native identity: `7f10320b366288679395cd5d638debc67d0da9976ec7abe80df4886e1ac10620`.
The later parent merge contains the portable integer-limit comparison and
evidence maintenance; the frozen measurements retain their original identity.

Run `python benchmarks/results/vv10-wide-pair-20261014/verify.py` with
`defusedxml` installed to verify the archive, all complete populations, gates,
medians, device outcomes and cancellation boundary. The draft remains pending
48/96 complete-endpoint qualification and subsequent default-policy validation.
