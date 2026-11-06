# Decision: specialize strict VV10 SCF feature sweeps

Status: implemented; bounded dual-card endpoint gates pass
Date: 2026-11-06

## Problem

The cold WB97M-V endpoint still executes four expensive SCF feature pair sweeps.
Compaction, bilateral symmetry, packet rotation and bounded FP64 reciprocal
lowering already exist. Earlier FMA spelling and packet-width experiments did
not remove useful work; repeating those sweeps is not an optimization plan.
The separate force-output-demand experiment also failed its A100 endpoint gate.

## Decision

Keep the admitted scientific domain and all physical pairs. For the strict
FP64 SCF packet, share the energy reciprocal in the common derivative factor:

```text
D = g_i*g_j*(g_i+g_j)
reciprocal = correctly-rounded-FP64(1/D)
phi = -1.5*reciprocal
old partial factor = (2/3)*phi*phi
new partial factor = -phi*reciprocal
```

This removes one FP64 multiplication per evaluated pair. It is not approximate
reciprocal arithmetic, mixed precision, extra screening, or a new functional.
The compiler owns the canonical TensorIR change and its separately versioned
rounding policy. The native owner continues to own admission, storage, streams,
selection and error publication.

Host dispatch already knows whether arithmetic is strict and whether an
arithmetic census was requested. Compile those conditions out of the ordinary
strict feature loop. Retain a counted specialization for a requested census,
including diagonal ordered legs and the existing positive-zero response path.
There is no extra allocation, point-count readback or synchronization.

Automatic admitted strict SCF execution selects this specialization. Explicit
`GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet` retains the previous closure;
`parallel-packet-reciprocal` explicitly selects the new route for small tests.
Geometry, mixed arithmetic, CPU/rVV10, refused-domain and small automatic
consumers keep their established closures. Schedule changes invalidate exact
pair reuse as before. No GPU-model whitelist or fixed A100 tuning is added.

## Evidence and interpretation

The motivating 320000-point prototype is distinct from the production image.
Actual A100 LSF226105 measures 0.494041 -> 0.448553 seconds (-9.21%); RTX5090
Slurm16597 measures 1.938669 -> 1.856434 seconds (-4.24%). Each passes 96
independent long-double array cases. SM80 registers fall 141 -> 128. Static
SM120 SASS has 59 -> 55 DMULs across the four unrolled row bodies, with
unchanged DFMA/DADD/reciprocal-instruction counts. These are component and
static-code observations, not endpoint or executed-instruction counts.

Production source is frozen over master `6987eea52` (#3020), preserving its
16-Fock cold provisional handoff and unchanged physical target convergence.
The complete cold/warm, work-count and independent static-reference gates are
retained and replayed under `benchmarks/results/vv10-scf-features-20261106/`.
Both complete endpoint gates pass: cold observations improve0.73% on A100 and
1.48% on5090, while warm medians change+0.55% and+0.12% with unchanged work.
These modest cold wins and noise-scale warm changes are not a statistical
distribution or a literal zero-regression claim. All32 retained independent
reference comparisons and the tighter paired-native observable gates pass.
Only those complete endpoints justify default promotion. Frozen warm SCF
has zero feature pair sweeps, so this does not target the old 1.576-second warm
gap and must not be reported as fixing it.

Before PR submission, master advances to `f1212d5aa` (#3021/#3022). The final
integrated source `85c68f6b...` is rebuilt and passes default-only complete
cold+one-warm checks on actual A100226125/RTX5090-16606, eight additional static
reference comparisons and tighter cross-image E/F gates. Do not mix these
integration observations with the original same-image paired timing. The A100
submission transport timeout is audited as an already-completed allocation;
no duplicate work is submitted.

PR #3024's initial type check rejects an optional reciprocal IR node even
though policy validation guarantees its presence whenever consumed. Construct
the node unconditionally instead of weakening the type check. A one-off full
`native_header()` comparison before/after is byte identical (SHA-256
`54a9b794e2262859984e7bf38afd4dd2fadab22d4cf773974a472f2b95edf1f6`): dead
IR nodes do not become extra runtime divisions. The checked native source
identity nevertheless changes to `dfb92b1b...`; preserve all measured source
and ELF identities, record the equality proof separately, and refresh generated
catalog metadata rather than relabeling old GPU runs. Pinned `ty==0.0.82` and
424 focused host cases pass (two explicit compiler-dependent skips). No repeated
dual-card timing is needed for this code-generation-neutral typing fix.

## Rejected alternatives and failed gates

Strict specialization alone improves the component roughly 6.8% on A100 but
only 0.75% on5090. It accompanies the actual algebraic work reduction; these
effects must not be conflated. Full-tile/fringe separation adds only about
0.04% beyond strict specialization on5090, so its extra kernel and template
complexity are not in production.

LSF226101 lacked optional `cuobjdump` and failed before any tests/timings. The
corrected probe records SASS unavailability rather than fabricating A100 SASS.
Slurm16598 compared intentionally different strict derivative orders bitwise.
The corrected test still requires the mixed-domain refusal path to equal a
separately executed incumbent strict closure bit for bit; new-order numerical
gates are unchanged. Slurm16599 passes1450 consumer/device cases. Memcheck and
synccheck Slurm16600 each pass eight focused cases with zero reported errors.

The local uninstalled checkout initially lacks a native library for two
energy-only consumer cases. Those cases pass against the measured built image;
the scoped local cohort reports313 passes with those two explicitly deselected.
No initial failed cohort is relabeled as a clean pass.

## Revisit when

Require independent numerical gates and complete same-image endpoints before
changing the bounded lowering, counter admission, mixed/geometry boundaries or
packet schedule. Larger gains need actual pair-algorithm/data-movement work,
not another spelling/width sweep. Preserve all scientific and resource fallbacks.

References: issue #2976; the rejected force experiment is recorded separately in
`.agents/notes/rejected/2026-11-06-vv10-force-output-demand.md`.
