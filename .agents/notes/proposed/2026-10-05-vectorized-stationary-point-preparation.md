# Proposal: separate point-parallel XC preparation from indexed AO pullback

Status: proposed; not implemented or performance-qualified
Date: 2026-10-05

## New mechanism evidence

Read the *actual loaded endpoint JIT artifacts*, not the separately packaged
full-primitive AOT probe. The clean Slurm 6115 48-atom receipt identifies the v6
runtime `e800d677.../runtime.so` (binary `00054bd3...`) and schedule-3 runtime
`621a2fef.../runtime.so`. Their cooperative kernel resources from static
`cuobjdump` inspection are respectively 255 registers / 72 stack bytes and
255 registers / 168 stack bytes, both with 1040 static shared bytes. These are
static resource declarations, not achieved occupancy, local-memory requests,
spill attribution, or timing. Full binary hashes and dumps are retained in
`.artifacts/stable-group/static-resources/`.

The emitted source executes `geometry_point_setup` only in thread 0 of a
128-thread point-worker CTA; that includes the shared nonlinear XC point model.
The rest of the CTA subsequently constructs and gathers the indexed AO gradient
panel. The audited 6090 2.675511-second aggregate includes all these stages, so
it cannot identify how much time belongs to setup, AO construction, atom gather
or owner addition. Register count alone cannot establish a useful optimization.

## Candidate to test

Compute the exact existing point model in a point-parallel preparation kernel,
then consume its transient FP64 point coefficients in a distinct indexed AO
kernel. Use only an already charged, dead scratch alias: phased Becke does not
use the cooperative Becke scratch during AO publication. Admission must prove
the point record and external-motion cells do not overlap and remain bounded.
No retained numerical cache, geometry reuse shortcut, new map inventory,
screening change, extra AO/jet evaluation, or arena growth is allowed.

Reuse `geometry_point_setup` and the existing graph; do not write another XC
algebra. In the phased lane-equals-point route, a scratch slice can provide the
already-zero owner-motion cells used by external seeds. A specialized prepared
AO entry must not retain a runtime fallback that still embeds the entire point
model in its register-heavy call graph. Nonphased, nonfitting and unsupported
routes must retain the old entry and resource policy.

The existing scan entry should also be insulated from experimental scheduling
code rather than assuming that default schedule 0 retains the old compiled
resources. The 72-to-168 stack change is a reason to measure/insulate it, not
proof of spilling or a measured default-path regression.

## Gates before implementation or promotion

- Preserve each AO/coordinate accumulation order and initial value, including
  external-motion seeds and signed zero; reject zero-based partial summation
  followed by a final merge that reassociates the existing sums.
- Compare identical state bits across old/prepared entries for large RKS/UKS
  LDA/GGA/meta-GGA, shuffled/noncontiguous maps, empty/tail domains, capture,
  geometry changes, and sticky producer/consumer errors.
- Prove scratch lifetime under the caller's stream/lease and test the fallback
  without additional allocation or downgrading a fitting cooperative panel.
- Measure preparation and AO/gather kernels separately on the actual public
  force invocation with CUDA event-completion tracing explicitly disabled.
- Record actual resources, traffic/work domains and cold/moved Fock histories;
  require clean complete 48/96 E+F improvement, not only lower registers or a
  synthetic pipeline win.

## Current experiment boundary

The 6115 48-atom schedule-3 experiment passes 72 same-geometry numerical pairs
per variant and changes warm median 8.951451 to 8.840963 seconds. Moved execution
uses 13 vs 12 Focks and takes 54.625284 vs 51.403380 seconds. That is a prohibited
observed regression for promotion; neither its Fock history nor its roughly
1.2% warm change completes P0-C. Do not ascribe the different Fock count to
integer grouping without evidence. The complete 96-atom pairing remains part
of the same finite allocation. It is now complete: 96-atom warm changes
24.775303 to 24.053450 seconds (-2.9136%); its moved endpoint improves with
13 vs 14 Focks. All 72 pairings per variant/size pass, but the 48-atom moved
regression still rejects promotion. No counts are normalized and P0-C remains
unfinished. Receipts: `.artifacts/n1-owner-endpoints-6115/paired-clean-summary.json`.

References: `../implemented/performance/2026-10-05-stable-owner-schedule-experiment.md`
and issue #1893 P0-C. This candidate is distinct from #1892 Direct-force work.
