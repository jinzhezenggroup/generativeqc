# Strict VV10 SCF feature sweeps

This bounded #2976 change specializes **SCF feature** pair sweeps. It does not
remove a pair sweep, change the functional, or solve the historical1.576-second
warm gap. The separate force-output-demand prototype is rejected and removed.

## Decision and mechanism

Enable the strict FP64 SCF packet specialization automatically within the
existing admitted molecular domain. Share the energy's correctly rounded
reciprocal in the derivative prefactor: `-phi*(1/D)` replaces `(2/3)*phi*phi`
for `phi=-1.5/D`. This saves one FP64 multiplication per physical pair, with
separately versioned rounding. Host-known strict arithmetic also removes mixed
admission and unrequested per-pair census. A requested census retains a counted
specialization and the original physical work, including diagonal legs.

TensorIR remains the scientific owner. Geometry, mixed arithmetic, CPU/rVV10,
small automatic domains and refused numerical/output leases retain their
established closures. The packet width, active points/pairs, grids, precision,
solver tolerances, memory capacity and synchronization contract are unchanged.
There is no GPU-name hardcoding, Direct-to-DF substitution or production
CPU/PySCF oracle. Existing `GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet`
provides same-image rollback; unset/empty uses the new automatic SCF closure.
`parallel-packet-reciprocal` explicitly admits small qualification fixtures.
Schedule changes continue to invalidate exact component reuse.

## Complete endpoints

Freshly synchronized master `6987eea52` (#3020), without unmerged #3021 or the
rejected force prototype. Water96, WB97M-V,768 spherical def2-SVP AOs, full
Direct, Standard FP64, native Huckel guess and stock XC/nonlocal grids.
Both arms use physical energy tolerance1e-9, density tolerance1e-5, screening
1e-12, DIIS8 and analytic moving-grid response. Each arm runs in its own process;
artifact caches are warmed, while the timed cold interval includes constructor,
prepare, SCF, analytic forces and host export. Three warm calls restore the same
engine-local cold density rather than a changing sequence of seeds.
Tracing/progress controls and optional force selectors are unset. Production
oracle imports are explicitly blocked in the endpoint runner.

| Actual allocation | Cold rollback -> default/s | Change | Three-warm median rollback -> default/s | Change |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226114 |108.435130 ->107.647631 |-0.73% |15.010082 ->15.092283 |+0.55% |
| RTX5090 / Slurm16599 |188.442832 ->185.653515 |-1.48% |14.320770 ->14.338280 |+0.12% |

Cold saves0.787499s /2.789316s respectively. These are one cold observation
per arm, not a cold-time distribution. Warm changes are noise-scale, not a claim
of literally zero regression or a warm speedup. Every paired warm ratio is below
1.008; the retained guard is1.02 on warm medians and1.05 per call.

All arms retain16 cold Focks (11 provisional /5 physical target),16 J and32 K
applications, four SCF pair sweeps, four pair-cache probes and one pair-cache hit.
Every warm has one Fock/J, two K, zero SCF pair sweeps and one cache hit.
These SCF counters are **not** an endpoint-wide force-pair census; each endpoint
also performs one analytic-force evaluation. Exact scalar physical-pair census
is unavailable in these clean runs, not zero. No MD-J-bypassing trace is mixed
into the latency ranking.

Measured native source:
`f4006c165078b71f1b358734a72a828abc6df7b6963552327256307fe75acfaa`.
Same complete ELF on both cards:
`107fcc113cc5909dd66808c2ba03e9f28d175a0766e56a4a1940d91234fecde7`.
The Release build uses verified sccache0.16.0 and SM80-real/SM120-real. Source,
loaded identity, full ELF hash, compiler/cache receipts and finite scheduler
allocations are retained. The borrowed build tree's original runtime is restored
and its checksum verified. A100 receives an exact authenticated executable
delta, independently reconstructed and fully hashed on both hosts.

## Later master integration

Before opening the PR, master advanced to `f1212d5aa` (#3021/#3022). The branch
is fast-forwarded and rebuilt, including the newly merged shared force panels.
The final integrated native source is
`85c68f6b865ccddd9d22f0cf193b0bb6ec9d26290d6902d4391302f9c97da324`,
ELF `ef29a74ec63fe73f39a64ecac527eb736caa661951deef78cde0d1cd25a9aa7f`.

| Default-only integration | Cold/s | One frozen warm/s | Focks |
| --- | ---: | ---: | ---: |
| A100 / LSF226125 |106.364494 |14.867500 |16 /1 |
| RTX5090 / Slurm16606 |188.678345 |13.930641 |16 /1 |

Both complete endpoints and eight additional independent-reference comparisons
pass. Cross-image native E/F differences also pass the tighter paired gates;
grids/settings and SCF work remain unchanged. Each card passes412 selected SCF
array/admission cases on the integration source; its local host cohort reports
451 passes with the same explicit skips/deselections. These are **default-only
integration observations**, not a new paired performance ranking. In particular,
do not relabel the old-image cold deltas as measurements of this combined image.

The integration's A100 submission SSH timed out without a returned job receipt.
Independent HTTP result and LSF DONE audits identify the already-completed
job226125. It is collected and authenticated without submitting a duplicate.
The original timeout and recovery audit are retained.

## Independent gates

The static independent GPU4PySCF energy/force receipt in
`../direct-range-warm-anchor-20261105/publication.json.xz` is reused only for
accuracy, not timing. Replay authenticates that original bundle and checks the
scientific input contract. All32 comparisons pass the existing Standard gates
of1e-5 Eh/atom,1e-4 max and3e-5 RMS Eh/Bohr; observed worst errors are
8.99e-12 Eh/atom,1.05e-6 max and4.76e-7 RMS Eh/Bohr. Raw endpoint metadata
keeps its original `accuracy_qualification: unverified`; this replay computes
the qualification separately rather than rewriting raw observations.

The tighter paired-native gates remain1e-9 Eh/atom,1e-7 max and3e-8 RMS
Eh/Bohr. Observed worst errors are9.48e-15,9.46e-13 and1.42e-13 respectively.
Independent90-digit scalar derivatives cover both legs and bounded extremes;
extended-precision device arrays cover signed/positive-zero weights, empty
domains, padding/canaries and refused admission. Mixed refusal still equals a
separately executed incumbent strict closure **bit for bit**; it is not compared
bitwise against an intentionally different SCF derivative order.

- Slurm16599:1450 consumer/device tests pass; one unrelated test-only failure
  hook is explicitly deselected rather than claimed.
- LSF226114:412 selected SCF admission/array cases pass.
- Slurm16600: memcheck and synccheck each pass eight focused cases, zero errors.
- Final local host cohort:450 pass,1516 explicit GPU/native skips, two consumer
  cases deselected because the checkout lacks a native library. Both consumer
  cases pass against the built image in Slurm16599.
- Compiler dependency audit:586 modules, zero errors. Default-promotion
  inventory and formatting/source checks pass.

## Component evidence, not endpoint rankings

The motivating320000-point prototype is not the production ELF. A100 LSF226105
measures0.494041 ->0.460531 ->0.448553s for incumbent, strict/no-census, then
shared reciprocal (-9.21% combined). RTX5090 Slurm16597 measures
1.938669 ->1.924075 ->1.856434s (-4.24%). Each card passes96 independent array
cases. Actual SM80 packet registers fall141 ->128 with no spills. Static SM120
SASS has59 ->55 DMULs across four unrolled row bodies, with unchanged DFMA,
DADD and reciprocal-instruction counts. These are static instruction counts,
not an executed-instruction census.

Full-tile/fringe splitting adds only about0.04% beyond strict specialization
on5090 and is discarded. Earlier strict-only Slurm16595 and split16596 records
are retained without inventing an A100 timing for the split schedule.

## Failed attempts and rejected force work

LSF226101 compiled but lacked optional `cuobjdump`, stopping before numerical
tests/timings. Corrected226105 explicitly records unavailable A100 SASS.
Slurm16598 initially compares the two intentionally different FP64 derivative
orders bitwise. The amended test preserves the exact incumbent mixed-refusal
gate and the unchanged independent new-order gates. Its failed results remain
in the bundle. An initial uninstalled host run and a delta-driver PATH failure
are also retained, not relabeled as passes.

The `rejected_force` section preserves the earlier source/ELF, raw dual-card
endpoints and exact unshipped source overlay. Four versus six force accumulation
fields gave a component win but no A100 endpoint benefit: cold
111.895463 ->112.448947s. Its implementation is absent here. Its large static
reference gate was not completed, and missing UKS Python AOT artifacts prevented
the supplemental UKS/sanitizer attempt; neither is claimed as passed.

## Replay

```bash
python benchmarks/results/vv10-scf-features-20261106/replay.py
```

`evidence.json.xz` retains complete vectors, controls, work counts, identities,
test logs/XML, runner sources, prototype bundle, exact source overlays and
bounded failures. `summary.json` is the derived replay result. Large build logs
and binaries remain ignored local artifacts with retained checksums, not Release
assets or external backups. No fresh GPU4PySCF ranking, fleet-wide speedup or
original warm-gap closure is claimed.

Durable rationale:
`.agents/notes/implemented/performance/2026-11-06-vv10-scf-shared-reciprocal.md`.
