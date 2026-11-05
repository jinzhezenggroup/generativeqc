# Decision: ingredient-based occupied XC and an earlier strict tail

Status: implemented
Date: 2026-11-05

## Problem

The [functional survey](2026-11-05-occupied-xc-functional-survey.md) retains
MN15 RKS and M06-2X UKS endpoint regressions with additional Fock builds, despite
passing independent scientific gates. Late provisional-to-strict handoff resets
the energy baseline near convergence, where another strict energy observation
can outweigh the saved contractions. That survey does not qualify this change.
The former implementation also accepts only five-feature meta-GGA layouts.

## Decision

Start the strict energy epoch at residual RMS `1e-4` rather than `1e-6`.
This is an earlier initializer handoff, never a relaxed stopping tolerance.
Keep the eleven-proposal cap, two reserved strict evaluations, finite original
iteration budget, bounded predictor history and strict original-D publication.
The occupied-FP64 predictor uses the ordinary chronological DIIS ring, not a
blanket reset after six strict submissions. Its provisional entries are replaced
within the configured history depth (validated at most 64), or sooner when
dependent-error retirement shrinks the live window. No old entry certifies a
physical state. The distinct FP32 history experiment keeps its existing reset.

The first current-source candidate `8299cd88d1400931e67958e1ca6181912f49e196`
demonstrated why earlier handoff alone was insufficient. M06-2X UKS retained
24 vs 25 cold Focks and 18 vs 20 displaced Focks. Its physical residual history
agrees with the matrix control until the density proposed immediately after the
six-sweep history reset. That reset discarded six useful strict Focks together
with the remaining old provisional entries. Natural ring ageing removes the
unnecessary solver restart without a kernel, transfer or extra target rebuild.

The next candidate `ca8855d9dc5d64ffec46f011edf295ce04700811` removes those
extra Focks, but exposes a second independent handoff cost: calling
`prepare_phase_source` regenerates a same-precision exact ERI lease. This affects
hybrid cold endpoints even when their SCF/Fock counts agree. XC reassociation
does not change J/K arithmetic, geometry, basis, screening or source ownership;
therefore its handoff now preserves an active strict lease and leaves an absent
lease on the incumbent bounded direct path. A genuinely mixed active lease
still follows the original strict retirement/replacement path. No generic
cross-geometry cache identity policy is weakened.

The `4dd6b34b53b882688e18e4a0036acc20326a5704` cohort exposes a third,
separate cost on automatically incremental Direct endpoints: the shared
`strict_refinement` flag disables delta J/K after an XC-only handoff. A traced
48-atom M06-2X cold solve executes 48 rather than 32 structured basis actions,
despite both routes reporting 10 Focks. Reassociation changes XC features, not
the exact J/K density anchor, screening contract or precision. Preserve the
ordinary incremental schedule for this FP64-only handoff. Genuine precision
transitions, full-energy refinement and final closure still forbid deltas.
SCF/Fock totals alone would have hidden this semantic-work regression.

Dispatch by supported ingredient layout, not functional or molecule names:

- LDA: one occupied value projection and squared-orbital feature reduction.
- GGA: one value projection `p = C^T phi`, then reconstruct `C p` during the
  AO-row feature reduction using the existing compiler-owned bilinears and
  original AO gradients. Both rank contractions must be cheaper than `D phi`,
  requiring `2 * rank < active`, with no extra global panel.
- Tau-meta-GGA: retain the existing four projected jets and `rank < active`.

Each tile checks work and scratch admission independently. Missing or stale
canonical frames, nonlocal consumers, response, lower precision and unsupported
ingredient layouts keep matrix work. Hybrid exact exchange remains unchanged.
The lower-level exact mode remains meta-GGA-only with two panels and original-D
bitwise feature verification. Ordinary admitted provisional points do not
recompute the dense contraction; unsafe points retain the stored-D fallback.

## Rejected alternatives

- Blindly removing the five-feature check would overrun single-panel GGA/LDA
  scratch or reduce its projected jets incorrectly.
- Four occupied GGA jets can cost more than the original single `D phi` product;
  value projection plus reconstruction uses two rank contractions instead.
- Functional or atom-count exclusions hide rather than fix solver overhead.
- Comparing strict energies with a provisional baseline or relaxing numerical
  gates would compromise the final-state contract.
- Clearing the whole DIIS ring after six strict sweeps destroys useful target
  history. Truncating to six also unnecessarily discards strict directions for
  callers with a deeper ring; normal bounded chronological replacement already
  retires every old entry and requires no new device work.
- Re-preparing an unchanged exact ERI lease at an XC-only handoff repeats
  allocation and integral production. This cannot be justified by saving only
  density-feature contractions; preserve the same-geometry strict lease.

## Evidence

Independent dense-matrix host fixtures cover 1/4/5 features, restricted and
unrestricted occupations, empty spin channels and nonfinite-projection fallback.
Executable native-owner fixtures cover bounded handoff and work admission.
The scheduled GPU projection suite passes 272 cases, including mixed-capacity
mapped batches, standalone execution, scratch canaries and poisoned D proving
ordinary admitted points do not consult the matrix oracle.
The final scientific source is `3c41b36142d2fe52fc7b3cb7de89e42fef670b42`,
with source identity
`5db4c208f23bc6215e03241dd08214258c69995291b8c8aa99a85b3b47adc8a3`.
The [retained capsule](../../../../benchmarks/results/occupied-xc-families-20261105/README.md)
binds 624 new family E+F observations and twelve original-workload observations
to that image. The original 96-atom Direct Standard r2SCAN-3c cold median is
`30.449281 -> 29.303447 s` (3.763% less time), with 10 Focks on both routes.
M06-2X/48 cold improves 2.442% on node4 and independently 2.210% on node5;
both exceed the predeclared 2%/MAD floor. LDA and GGA examples are supported
but remain below the endpoint gain floor, not advertised as speedups.

Keep the negative evidence: the pre-incremental candidate populations, the
DIIS-restart counterexample and an isolated final PBE0/3 warm confirmation
flag (about 1.3 ms, 2.045%). The original/main PBE0 warm population is neutral;
neither this disagreement nor a functional name justifies changing gates or
adding a whitelist. Independent final native/GPU tests pass 320 cases,
including 16 complete RKS/UKS oracle/control tests. The emitted family subset
passes memcheck, racecheck and synccheck (36 cases each, zero findings).

### Master integration supplement

The later master integration (`051a157c6`, measured source `1a01e7109`) has
native identity `efe799a008482dad106f2d49e4926fdc2e47c26f915cb4a635308037772e8518`,
not the survey image above. The six relevant occupied-XC implementation/compiler
files remain byte-identical; upstream CPU/Hessian code changes the image identity.
Keep `integration.json.xz` separate rather than rewriting or pooling the earlier
measurements. Three fresh-process node4 pairs reconfirm original r2SCAN-3c/96
cold at `30.279007 -> 29.138219 s` (3.768%) and M06-2X/48 at
`9.820248 -> 9.611699 s` (2.124%), with 10/10 Focks. Other measured phases
remain neutral under the same floor; both independent M06-2X references and the
original-workload independent qualifier still pass. The integration GPU suite
passes 320 cases and the existing host/controller/capsule suite passes 293.
The supplementary replay preserves separate image/host identities and rejects
corruptions under normal and optimized Python. Earlier sanitizer results are
not relabelled as an integration-image sanitizer run.

The later automatic C++ formatting changes only whitespace in three files;
canonical clang-format output matches before and after. The rebuild at
`25e1039eb` (including the upstream CI-only merge) has native identity
`1ec87aa0b415e19b8af0a53450ab8b0e29358d7a6f25ce3dc26a0fd089ee4217`.
Its separate GPU rerun and expanded 486-case host suite are retained as
`formatted_followup`, not presented as new endpoint timings. Source-hash-bound
generated catalog snapshots must be refreshed even for such formatting changes;
the refresh preserves the old evidence's stale/zero-qualified status.

## Revisit when

Retain neutral and regressing complete E+F populations. Revisit scheduling on a
credible endpoint regression, not on isolated projection-kernel improvements.
Generalizing to Laplacian, nonlocal or response features requires their own
mathematical, resource and publication contracts, not just another ordinal.
