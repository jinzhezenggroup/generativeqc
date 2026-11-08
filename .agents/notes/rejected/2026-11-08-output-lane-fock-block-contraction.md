# Decision: reject this output-lane combined-Fock block contraction probe

Status: rejected; no production accumulation or runtime change
Date: 2026-11-08

## Problem

Issue 2976 comment 6715999722 recommended a bounded actual-page experiment
before expanding combined-J/K block contraction. Publishing one reduced shell
block element can replace many component-wise global atomic calls, but the
extra local contraction, indexing and synchronization need to pay for themselves.
The previously rejected scalar-axis and cooperative-root experiments are closed.

## Experiment

A diagnostic interposer captures the first default HF96 DPPP materialized page
and density immediately before its original Fock consumer. Device-to-host copies
are ordered graph nodes, not host reads of unexecuted capture state. The
unmodified native process completes its cold energy/analytic-force/export and
three warm calls; these intrusive endpoint times are not benchmark samples.

Each card retains the first 8192 of 147800 admitted tasks, order 800, at the
observed first-page cutoff 8e-8. This is an early default-page diagnostic, not a
strict-final-Fock census. A100 uses 864 workers; 5090 uses 1360. Page order is
not deterministic across cards; these are separate card-local inputs, not a
cross-card timing comparison. On 5090 the runtime launch records the SM120 owner.
On A100 the named SM80 host-wrapper interposer fires first; its metadata owner
string is empty. The source-bound hook identifies that wrapper, but the missing
field is not represented as an independently recorded runtime kernel name.

One existing SM80 value-only Rys emitter materializes the admitted ERIs once.
The incumbent scatter and candidate then consume exactly those same values,
density and task set. Both persistent consumers load the same CTA-local values.
The candidate assigns output lanes to twelve directed J/K shell blocks, sums
their component-density products, and publishes nonzero block elements. It uses
the existing orbit ordering, uniqueness predicate and FP64 contribution helper;
it does not assume a symmetric density or omit repeated AO indices.

The probe is restricted combined HF without compensated output. It is not a new
UHF, response, force, packed-K or resident-KS implementation. No production
Fock algebra, schedule, manifest, native admission or default was modified.

## Evidence

For each density, one first validation launch is retained and reported, followed
by five complete unfiltered control/candidate/candidate/control groups. Every
launch produces a real full Fock matrix, which is copied and compared outside
the event interval. The second density is directed and has explicit zero weights.
All 42 samples per card remain available; including the first validation launch
in the control median does not change either reported control median.

| Card / density | Incumbent consumer / ms | Block consumer / ms | Incumbent / block |
| --- | ---: | ---: | ---: |
| A100 / actual | 0.168960 | 0.288256 | 0.586146 |
| A100 / directed | 0.236544 | 0.291840 | 0.810526 |
| 5090 / actual | 0.089088 | 0.284160 | 0.313514 |
| 5090 / directed | 0.128000 | 0.290816 | 0.440141 |

An independent NumPy unique-orbit contraction checks full matrices against the
fixed values at atol 2e-11/rtol 2e-10. All four engine/density comparisons pass
per card: maxima are 1.155e-14 on A100 and 2.043e-14 on 5090. The actual pages
contain 4768 and 6037 repeated-ket-shell tasks, respectively. This is independent
contraction validation, **not independent qualification of the native ERIs**.
There is no new Libcint, full E/F candidate endpoint, moved-geometry, lifetime,
sanitizer or final-default qualification for this rejected consumer.

PTXAS for the standalone SM80 consumer reports 55 to 78 registers; SM120 reports
46 to 54. Both variants use 1496 bytes shared and report zero static stack/spill
traffic. These descriptors do not measure occupancy, dynamic spills, atomic
requests, or a removable fraction of complete cold time. Publication-count
budgets are not hardware-atomic measurements. The isolated probe is not the
ordinary fused value-plus-scatter endpoint, and is not linearly extrapolated to
the HF96 cold deficit.

Local Slurm build 16799 and replay 16804 succeed; capture 16803 succeeds. LSF
CPU setup 228745 and exclusive A100 228746 both finish successfully. Earlier
capture attempts remain failed: 16792 mixed unsigned host types; 16794 wrong
authentication working directory; 16796 ambiguous overloaded function pointer;
16795/16798/16801 complete native solves but acquire no page. The latter are
not successful captures or performance populations. Runtime tracing identified
the 5090 SM120 owner and fixed the overly SM80-specific capture filter. No
replacement performance cohort was selected from these setup diagnostics.

## Identity and retention

The native image is unchanged control source
`50ca688e7f9dc3d51c05f9aa336fab14226372fcbe8f970536062753a87ff6e7`,
ELF `2977049443dd116779ad583f67da20c902fcede2fdb0ae721d2995d8aee147ff`.
Latest master 674672269 is integrated as 671f44f30; its change is an unrelated
nonfinite oracle test, not a relabelled measurement of a new native image.

Probe source SHA256:
`53f813c58bf12b85b3cbbaf89a53b5f0d2f0568737dcc738f865974404d3d2c3`.
Probe executable:
`b90de7ecddb68365e37caf53b46506df86de2acaf7e5bfdedead07a75f4f4144`.
Capture library:
`d9fb6712fe087b48e517ab52e3af90613033c4a775ac22003231f9f159b5f322`.
Authenticated A100 evidence archive:
`bd9ea76e16b6dd672b81f01a878e4ac8e3daf1a282f13bc3ae675319904b206a`;
original result archive:
`d3fd0e5a883e78e01500ea905d40cf938940ccb4395ef1593225db4de2fecfd2`.
5090 evidence archive:
`664285fdad5c354e33e927d071fd20dfae42121bca7bdfd69d635dba5332b8d3`.
Sources, arrays, all samples, setup failures, receipts and offline reducers remain
under `.artifacts/hf-block-contraction-v1-20261108/`.

## Decision and revisit conditions

Reject this twelve-block output-lane gather scheme before production expansion.
It is slower on both cards, including A100; do not retain a runtime opt-in, sweep
block dimensions, retime it for a better point or change the frozen full64 totals.
This does not prove all possible local contractions are unprofitable or that
global atomics never matter. Revisit only with a distinct causal design and a
measured relevant consumer, not the same static publication-count argument.

Next probe may reuse the existing shared-Rys-XY recurrence owner on the retained
A100 page. That is a separate existing mechanism, not a new ERI implementation
or H100-to-A100 default authority. Runtime/default admission, weighted Z, and
complete cold/each-warm protection remain separate qualification gates. The older
13-path adaptive experiment remains untouched and still lacks valid whole-work
counters; invalid is not zero.

The complete decision is synchronized to issue 2976 comment 6716350814.
