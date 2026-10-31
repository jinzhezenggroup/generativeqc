# Decision: preserve cooperative and indexed stationary force execution

Status: implemented
Date: 2026-10-30

## Problem

The ECNU A100 complete energy/force endpoints exposed two regressions in the
native stationary consumer. A cooperative AO pullback required the entire AO
gradient panel to fit in dead Becke shared storage. Crossing that capacity
silently moved every AO pullback onto thread zero. The prescribed r²SCAN-3c
basis crosses this boundary at both 48/544 and 96/1088 atoms/AOs. Separately,
the native controller always evaluated dense force tiles even though the
preceding Python consumer already admitted derivative-specific active AO maps.

The original 96-atom r²SCAN-3c trace spent 10.43 seconds in the cooperative
geometry kernel, versus 0.48 seconds after chunking and indexed execution in
the later diagnostic. Kernel measurements are attribution, not endpoint
timings. The corresponding full cohorts, errors, work and identities are
retained in `benchmarks/results/ecnu-stationary-performance-20261030/`.

## Decision

The compiler emits bounded cooperative AO chunks in the existing shared arena.
The XC point value remains live while successive gradient panels reuse the
remaining storage. Each panel is consumed in original AO order before reuse;
barriers protect both subsequent AO writes and the Becke alias. Atom runs may
cross chunks, and indexed maps may be unordered or repeat AO labels. The
reducer resumes from the previously accumulated atom/grid values, preserving
the original order. FP32 retains one point-local accumulator and recovers
translation only after all chunks. Zero available capacity retains the scalar
fallback. No new allocation or recurrence is introduced.

The native controller reuses the existing exact sampled-jet bitmask producer
at cutoff `1e-16`. "Exact" describes the labels selected at this explicit
cutoff, not unscreened mathematics. Discovery uses the force binding's complete
AO derivative order; an SCF map cannot authorize dropping force derivatives.
Ordinary nonlocal-free profiles with 4 or 10 AO jets may discover a map when
predicted dense point×AO² work reaches `2**35`. A 16 MiB cap covers peak device
storage and host/device offsets conservatively. The observed occupancy must
be at most 0.8 to use the indexed consumer. These predicates contain no GPU
model, architecture, molecule or functional-name allowlist. ECP and nonlocal
consumers remain outside this promotion.

The later [nonlocal map decision](2026-10-30-nonlocal-stationary-ao-maps.md)
supersedes the nonlocal exclusion with exact quadrature-identity checks and
additional shared/distinct-grid device validation.

Map storage is optional cached capacity; the dense minimum is unchanged. A
budget which fits the existing geometry cache but not the masks retains that
geometry cache. Optional allocation failure leaves dense execution available;
numerical and driver failures propagate. Host offsets are charged to the
retained host inventory. The controller borrows immutable basis/quadrature
state, and geometry replacement destroys it. The underlying map owner also
checks geometry epoch, point pointer/span, derivative order and identity.

For zero full-range exchange, the Direct adapter uses the existing combined
J-only traversal. The public source request still determines reset/download
extent: a requested J/K view has two channels, and its K channel stays zero.
Nonzero exchange, coefficients, screening, spin conventions and callback
lifetime are unchanged. Traversal and published channel counts are distinct
diagnostic counters.

The existing opt-in component trace now records stationary grid work, including
discovery separately from evaluated AO jets. Counters are per-call differences
of existing host metrics; no new device readback is added. Instrumented timings
are excluded from the clean endpoint cohorts.

## Invariants and qualification

- Production execution does not call an independent reference or CPU oracle.
- Ordered reductions, derivative signs, FP64 default and mixed-path translation
  invariants are retained.
- Cached and minimum capacity queries remain authoritative; neither indexing nor
  cooperative chunks authorize exceeding a user resource allowance.
- Host emulation covers tails, sticky failures, arbitrary maps, chunk boundaries
  and both arithmetic types. Independent ordered sums check the reduction bits.
- Device synthetic r²SCAN tests compare scalar/cooperative/phased execution at
  both capacity-cliff shapes. Memcheck, synccheck and racecheck qualify those
  schedules; molecular energy/force references provide separate scientific gates.
- RKS and UKS PBE finite differences at two step sizes and the minimum-workspace
  fallback qualify the zero-exchange traversal on device.

## Alternatives and limits

Increasing the public XC tile from 256 to 512/1024 improved M06-2X diagnostic
endpoints, but changes SCF and force storage together. These probes do not
promote a new default or remove the small-workspace fallback. A larger automatic
force tile needs a planner that prices both preferred and minimum shapes, plus
complete qualification across device and workload shapes.

Do not repeat the unpromoted normalized/primitive Becke experiments merely to
chase parity. Their retained negative endpoint evidence remains applicable:
`2026-10-06-becke-physical-endpoint-boundary.md`,
`2026-10-09-exact-ao-bitmasks-becke-adjoints.md`, and
`2026-10-09-becke-zero-cotangent-elision.md`.

Matched-grid A100 tests diagnose implementations. They do not establish a
grid-converged production ranking, universal GPU4PySCF parity, or an A100 versus
RTX 5090 hardware ranking. Remaining slower complete cases stay visible.

## Reproduction and retained failures

Raw runs, profiles, build/cache receipts and failed attempts remain under
`.artifacts/ecnu-stationary-performance-20261030/` and the corresponding ECNU
`results/stationary-performance/` directory. ECNU work used genuine finite LSF
job 216475 in `gpu_chem`, with scheduler-assigned GPU visibility unchanged.

The measured master baseline is `e265e0e45` plus disabled grid trace hooks. The
hooks add no CUDA event/fence when disabled. An initial copied library symlink
was dangling; those early diagnostics actually loaded `gap-trial/build` and
must retain that identity. Subsequent cohorts use dereferenced library copies
and explicitly verify the loaded path before measurement.

Other retained setup failures include the system-glibc native launcher,
CuPy-header JIT setup in the synthetic test, an Nsight importer event-order
failure, an unsupported automatic-only work query on a static profile, and a
trace stream-type compile error. Corrected reruns are distinct receipts. None
of these attempts is a completed timing cohort or scientific acceptance.

## Measured work and validation receipts

The v4 component traces (native source `a8d85e7622aef04565484dee300868adb327f127ba1ed726e45912f2376c81c8`)
report the following executed warm force work at 96 atoms. The dense AO domain
is a same-grid comparison count, not a new measured hardware-instruction count.

| Method | Grid points | Active point×AO | Dense point×AO | Logical projection FMA pairs |
| --- | ---: | ---: | ---: | ---: |
| r²SCAN-3c | 2,359,296 | 563,799,040 | 2,566,914,048 | 700,154,220,544 |
| M06-2X | 2,359,296 | 388,141,056 | 1,811,939,328 | 324,228,399,104 |

Both owners discover their exact-jet map once on the cold call and report zero
warm discovery passes. Every grid point remains present. The four synthetic
chunk cases pass each of memcheck, synccheck and racecheck with zero errors or
hazards. The eight compiled native source/controller profiles pass their
lifetime/resource gates, and RKS/UKS PBE pass two-step finite differences and
minimum-workspace replay. Host receipts retain 586 derivative transport tests,
442 reduction/admission tests, and the earlier 72/185 focused tests; these sets
overlap and must not be added as a count of distinct tests.

The final code was rebased onto `6f4d19c61`, which merged the separate
Python force-map binding qualification. Its two runtime Python inputs change
the aggregate build identity to
`bfa2e4c8cb833d53ffebe9ee04db641e2c7f1f791d7e52db7943b02389661443`.
The complete default matrix is retained separately under `head/`; earlier
`final/` and tuning/fallback populations keep their own source/binary identities.
