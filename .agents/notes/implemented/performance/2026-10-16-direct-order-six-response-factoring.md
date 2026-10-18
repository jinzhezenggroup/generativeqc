# Decision: factor constant nuclear responses outside the order-six contraction

Status: implemented and source-matched endpoint-qualified; PR not merged
Date: 2026-10-16

## Problem and derivation

The generated order-six spatial primitive contracts 64 pairs of Wick terms.
Each term projects one Coulomb-root response onto three nuclear centers using
the same product-center Jacobians. The bra coefficient response also enters
two centers with opposite signs. Repeating those projections inside the sum
executes unnecessary products and additions.

Let A be the summed bra-coefficient response, C the summed ket-coefficient
response, and R the summed Coulomb-root response, separately for each Cartesian
coordinate. The three independent center derivatives before Gaussian decay are
`A + alpha/p R`, `-A + beta/p R`, and `C - gamma/q R`. Accumulate A, C and R
first, then apply these constant Jacobians once. Gaussian decay, normalization,
the fourth-center translational response and shell/force accumulation are
unchanged. This changes FP64 association, not the mathematical derivative.

## Boundaries

Only the existing admitted spatial order-six consumer uses this factoring.
Orders four/five and the nonspatial fallback retain their original response
accumulation. Full/LR kernels share the generated primitive; no functional,
molecule, AO-count or device-specific dispatch is introduced. Nine response
sums replace nine center sums without additional storage or synchronization.

## Preliminary evidence

A real-GPU, finite-Slurm, alternating five-sample primitive probe evaluates
32768 Cartesian components and four primitive products per component. With
second-pair-fastest ordering, median pair-4/2 time changes
1.215756 to 0.990448 ms and pair-3/3 time 1.077028 to 0.830554 ms.
Maximum derivative differences are 2.276e-15 and 1.110e-15, respectively.
Full, partial and singleton active-lane cases pass. These are isolated device
times, not endpoint acceptance or an independent molecular oracle.

The pair-4/2 probe uses 172 to 178 registers; pair-3/3 uses 248 to 242.
Their stack bounds remain 1056/1312 bytes, with zero reported spills and no CTA
barriers. These probe resources are not the production kernel's resources.

Existing independent Hermite-value finite differences cover all pair partitions
of orders four through six, all nuclear centers, full/LR limits and exponent
scales. Added common-center geometries check the zero radial-displacement limit.
All 300 additional correlated-Gaussian quadrature checks pass against independent
raised/lowered analytic integrals, covering all four pair partitions, three
axis rotations, full/LR radial limits, repeated/common centers, diffuse and
separated geometries. Both spatial and retained scalar consumers are checked.
Compute Sanitizer memcheck and synccheck report zero errors for the CUDA probe.

Separate instrumented GPU launches count entered constant-Jacobian products,
not enumerated candidates: 75,497,472 to 1,179,648 for full masks,
68,633,856 to 1,072,404 for partial masks, and 2,359,296 to 36,864 for singleton
masks, in each pair partition. This is a 64-fold reduction in that specific
source operation, not a hardware instruction/FLOP count or a whole-kernel work
reduction. Counted launches are excluded from performance timings.
At this preliminary stage, complete source-matched endpoint and linked-runtime
qualification were still pending. The completed qualification follows below.

## Rejected reuse alternatives

Warp-local Boys/prefactor preparation reduces instrumented full-mask radial
preparations from 131072 to 4096, but regresses pair-4/2 probe time
1.218603 to 1.239000 ms and pair-3/3 1.064220 to 1.067742 ms. Sharing bra Wick
terms with match/shuffle subgroups changes 1.206765 to 1.218532 ms and
1.065542 to 1.063032 ms: no useful consistent gain. Both trials produce exact
zero derivative differences for full, partial and singleton masks. Neither is
enabled. The two probes use different AO layouts and cannot be compared across
trials. Preparation reduction alone does not establish fewer machine
instructions or endpoint benefit; no hardware-stall cause is claimed.

## Completed source-matched qualification

The clean sources are synchronized master `5348b9e66`, recurrence/convolution
incumbent `4bdffe3ce` from PR #2541, and response deployment `00fced59f` from
PR #2549. Later test, documentation and evidence commits preserve the measured
scientific inputs. Master includes the merged #2531/#2532 dependencies; later
master movement is not relabeled as measured.

All GPU work used finite Slurm allocations and assigned visibility, CUDA 12.9.1
and official Release/sm_120 builds with verified ccache 4.5.1. Primary 96-atom
populations share the same n4 GPU; 48-atom populations share one n1 allocation.
Separate diagnostic captures share another n1 GPU and isolated stationary
caches. The canonical def2-mTZVPP basis, moving 48 x 16 x 32 grid per atom,
strict FP64, screening, convergence and 1800-second population timeout remain
unchanged. Each case/source retains cold, five warm, moved and five moved-warm
calls. All 144 unique primary calls complete; none is omitted as incomplete.

Seconds below are complete E+F wall times; warm entries are five-call medians.
No iteration or Fock-build normalization is applied.

| Case | Phase | Master | Incumbent | Response |
| --- | --- | ---: | ---: | ---: |
| r2SCAN48 | cold | 75.817234 | 69.934962 | 69.224673 |
| | warm | 21.438233 | 15.226590 | 14.368431 |
| | moved | 56.877101 | 50.849639 | 49.897404 |
| | moved-warm | 21.439160 | 15.253462 | 14.339476 |
| r2SCAN96 | cold | 411.282020 | 386.116851 | 382.483718 |
| | warm | 90.798641 | 65.477918 | 62.204506 |
| | moved | 260.105874 | 234.961981 | 231.331466 |
| | moved-warm | 90.800574 | 65.707564 | 62.139573 |
| PBE0/def2-SVP48 | cold | 23.735617 | 23.372648 | 24.088161 |
| | warm | 4.559350 | 4.258826 | 4.236261 |
| | moved | 22.465948 | 22.088725 | 22.191909 |
| | moved-warm | 4.575266 | 4.257219 | 4.217512 |
| HF/def2-SVP48 | cold | 11.001145 | 10.937647 | 11.003730 |
| | warm | 0.742660 | 0.741202 | 0.745075 |
| | moved | 7.206905 | 7.204627 | 7.229003 |
| | moved-warm | 0.745415 | 0.742078 | 0.745135 |

The response step additionally reduces 96-atom warm/moved-warm latency by
5.00%/5.43% and 48-atom latency by 5.64%/5.99% relative to the incumbent.
Combined reductions relative to this master are about 31.49%/31.56% at 96
atoms. Primary cold/moved phases also improve. The negative control is explicit:
PBE0 cold regresses 3.06% incrementally (1.49% versus master); other PBE0 phases
change within 1% incrementally, and HF phases within 0.61%. This is retained
scoped endpoint qualification, not a claim of universal or interleaved global
performance promotion.

All calls pass independent original-reference gates of 1e-8 Eh and maximum
force error 1e-7 Eh/Bohr. Across the three sources, maximum errors are
3.865352482534945e-11 Eh and 6.014255760078413e-11 Eh/Bohr. Cold/moved actual
SCF iterations/physical Fock builds are identical across sources:
r2SCAN48 19/19 and 13/13, r2SCAN96 24/24 and 13/13, PBE0 17/18 and 12/12,
HF 21/22 and 16/17. Every warm call is 1/1. HF uses valid returned native
strict/mixed/post-SCF physical-build counters after timing, not estimates.

Matched intrusive warm device times are separate from clean wall times:

| Device group | Master seconds | Incumbent seconds | Response seconds |
| --- | ---: | ---: | ---: |
| Generated low J' | 4.678396169 | 4.678237029 | 4.668890975 |
| Order-six J' | 21.968210321 | 17.549076519 | 13.914501823 |
| Order-seven J' | 28.218228922 | 7.375439397 | 7.356338321 |
| Order-eight J' | 5.809916822 | 5.824592228 | 5.810460079 |
| SCF Direct J | 8.402366718 | 8.353920828 | 8.342424024 |
| SCF shared XC/AO/grid | 4.538392521 | 4.550965613 | 4.538433314 |

The public profile reconciles mutually exclusive host owners and separate device
windows for SCF J/K, XC/AO/grid, order-six/seven/eight J', other force and Becke
response, final-state validation, preparation and synchronization. Kernel times
are never added to parent host waits. Every diagnostic endpoint is also
independently accurate. Entered primitive products remain identical:
2,400,504,912 / 1,250,544 / 95,976 for orders six/seven/eight. Unexposed order-six
preparation counters remain null; only the separately instrumented source
Jacobian products above establish this step's actual removed source work.

Production order-six resources remain 255 registers/thread and 128-thread CTAs;
order-seven uses 214 registers and 256 threads after the incumbent optimization,
unchanged by response factoring. No new barrier, shared-memory allocation or
atomic is introduced. Hardware instruction/FLOP counters were unavailable
(historical ERR_NVGPUCTRPERM); no achieved occupancy or full-process peak-memory
claim is made.

The final response host suite passes 890 tests, including all 300 independent
correlated-Gaussian cases. Six actual-linked independent Libcint GPU cases per
implementation cover both spins, representations and separate/combined J'/K'.
Probe memcheck/synccheck report zero errors; compiler structure covers 541
modules with zero errors. Both generated optimizations are automatic on their
existing admitted paths, with bounded fallbacks retained and no new user flag.
No PR is merged by this qualification.

## Evidence and next step

The original incremental packet is retained in immutable commit
`631f0b003884094f758a349e4a0418f6b54eec6a` at
`benchmarks/results/direct-force-reuse-20261016/`; the preceding master/incumbent
packet is retained at the same path in immutable commit
`4eb9fd859a69a31fa21c5fbbe9b13f7d36a9d90a`. Their 48 shared incumbent calls,
including every E/F bit, timing/count metadata and all eight geometry oracles,
are identical. The union retains all 144 primary calls. `verify-arrays.py`
independently recomputes the published gates. Selected trial generators and
raw timing/count/resource/sanitizer receipts are retained as ordinary lossless
JSON in `primitive-trials.json.xz`, with per-source hashes and replay instructions.
Ignored full build inventories and diagnostic histories remain under
`.artifacts/direct-reuse-next/` and the n1/n4 campaign roots. Only this task's
own same-path publication is superseded; unrelated evidence and storage budgets
are unchanged. The original frozen packet remains at commit
`6369de44fe1f30759b3d07c8f267f4a02d0d70dd`.

Response 96-atom warm latency remains about 2.34 times the historical GPU4PySCF
26.59-second measurement; this is not a fresh competitor comparison. The next
small implementation step is per-component axis-factorized coefficient/response
contraction from canonical compiler Hermite IR, reducing actual Wick/contraction
or primitive-preparation arithmetic without warp-shuffle overhead. Do not
revisit CTA/page-width tuning or the rejected sharing probes without new evidence.

## Follow-up: PBE0 cold observation is not a reproducible source regression

The original 23.372648 to 24.088161-second PBE0/def2-SVP48 cold observation
(+3.06%) remains valid historical data. It was one population per source,
not an established repeatable regression. Its original exclusive force ledger
places 0.067221 to 0.720458 seconds in `python_packing`: the 0.653236-second
increase accounts for about 91.3% of the total 0.715512-second endpoint gap.
This owner includes task preparation and resident AO-map setup; its old receipt
does not isolate a CPU, GC, driver, cache or scheduler cause. Do not invent one,
or equate the spike with slower order-six executed arithmetic.

A finite Slurm allocation on one assigned n1 RTX5090 repeats the unchanged
complete benchmark in ten fresh Python processes, predeclared ABBA BAAB AB
order, five populations per source. Every process retains cold, five warm,
moved and five moved-warm calls, original 1800-second timeout, grid, basis,
FP64, screening, convergence and independent references. No process is
prewarmed, no cache is cleared or disabled, and no sample is trimmed.
The two measured scientific sources and qualified native images are unchanged.

| Complete E+F seconds | Incumbent median | Response median | Change |
| --- | ---: | ---: | ---: |
| Cold | 23.425425 | 23.371673 | -0.229% |
| Warm | 4.274105 | 4.214359 | -1.398% |
| Moved | 22.087280 | 22.112491 | +0.114% |
| Moved-warm | 4.278988 | 4.225750 | -1.244% |

Warm summaries are medians of five per-process five-call medians, with all
25 calls per source/phase retained. Paired cold changes are +0.038%, -0.260%,
-3.039%, -0.211% and -0.150%; even the opposite-sign baseline outlier is kept.
The five response cold packing times are 0.065 to 0.152 seconds, incumbent
0.067 to 0.080 seconds. The cold result is effectively flat, not an additional
statistically established speedup. Existing evidence does not support a
repeatable 3.06% source-induced cold regression.

All 120 primary calls complete and pass the original independent gates;
maximum errors are 7.503331289626658e-12 Eh and 2.4920954189155964e-11 Eh/Bohr.
Each cold remains 17 iterations/18 physical Fock builds, moved 12/12 and every
warm 1/1. Separate matched intrusive captures also pass all four diagnostic
E/F gates. Generated-low/order-six device time changes 0.889814 to 0.837129
seconds cold and 0.896416 to 0.838039 seconds warm; SCF J/K cold device time
is 3.841285 to 3.845416 seconds. Exclusive host/device ledgers reconcile below
1e-7 seconds, without adding kernel sums to host waits. The original warm-only
SQLite reducer rejects the two-call capture; a retained selector removes only
the other public root marker in a temporary copy, preserving all selected
kernel/API events and nested scopes. Both original database hashes are retained.

No production change, PBE0-specific dispatch, GC suppression, clock tuning,
tolerance change, rollback or default change is justified by these results.
If cold tails recur, split the packing owner into bounded subscopes before
choosing another optimization; do not label or fix an assumed cause.
Forty-four focused profiling/benchmark host tests pass. The new scoped packet
retains all 124 primary/diagnostic arrays and replay inputs at
`benchmarks/results/direct-force-response-factor-20261016/`.
The complete preceding implementation qualification remains immutable at
`631f0b003884094f758a349e4a0418f6b54eec6a`, at
`benchmarks/results/direct-force-reuse-20261016/`; the older master/incumbent
bridge remains unchanged. Only our own packet is superseded,
without raising storage limits or deleting unrelated evidence. No PR is merged.

## Lossless integration with the master evidence packet

The master/incumbent campaign remains complete and unchanged at
`benchmarks/results/direct-force-reuse-20261016/`. The distinct PBE0 cold
follow-up is separated into
`benchmarks/results/direct-force-response-factor-20261016/`; all eleven files
retain their exact bytes from commit
`11ea887b1606c67062b0542a2ac84434abf07c37`, including the original publication
manifest, README, verifier, arrays and replay inputs. The README's historical
"same publication path" references describe the original
`direct-force-reuse-20261016/` path at its named immutable commits, not the new
sibling path. No files from these two campaigns are mixed. Run each directory's
`python verify-arrays.py` separately.

This separates publication ownership only: measured source identities, original
negative results, previous recovery commits and scientific bytes are unchanged.
No measurements qualify the integration tree itself. Existing readers and the
64 MiB aggregate limit remain unchanged. No historical evidence is deleted or
recompressed, and no release, tag, external archive or GPU rerun is introduced.
