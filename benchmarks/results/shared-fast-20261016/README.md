# Shared Fast qualification, 2026-10-16

This change optimizes shared Direct J/K screening, nonlocal packet admission and
native FP64 force ownership. No new functional-name performance dispatch is
introduced. Genuine FP32 operator work, independent accuracy and complete
endpoint cost are separate requirements; Fast is not universally faster.

## Measurement contract

All GPU work uses finite Slurm allocations, preserving assigned device
visibility. Family comparisons use Direct restricted KS, spherical def2-SVP,
`GridSpec(48,16,32)`, moving-grid analytic forces and a 180-iteration ceiling.
Every timed call starts a fresh calculator and hcore density, includes creation,
SCF, forces, host export and teardown, and excludes only separately recorded
build attestation. The continuous span around that attestation is retained.
Process/disk artifacts are warm; no converged density is reused. First calls
remain in the raw evidence, separately from three-repeat medians.

Each family/size has an independent GPU4PySCF energy/force reference. Every
repeat, including first calls, must pass. Fast gates are `1e-5 Eh`,
`1e-4 Eh/bohr` maximum force and `3e-5 Eh/bohr` RMS force. Family-comparison FP64
gates are `1e-7 / 1e-6 / 3e-6` in the same units. Successful Fast samples require
one budget solve, actual positive FP32 J and recurrence work, positive FP32 K
when required, and positive FP32 nonlocal pair work for nonlocal models. Strict
fallbacks are never counted as accelerated samples.

The [machine-readable summary](summary.json) retains all family points, including slowdowns,
the same-card native-owner comparison, numerical maxima and semantic SCF work.
The earlier packet-v4 comparison is explicitly a different source snapshot,
not a measurement of the final native-force-v5 build.

## Complete endpoint results

Final native-force-v5 on node4 RTX 5090, three-repeat medians. Times are
**FP64 / Fast** seconds; speed is FP64 divided by Fast. Ratios near one are
descriptive measurements, not a statistical significance claim.

| Electronic model | 3 atoms: seconds | Speed | 12 atoms: seconds | Speed |
| --- | ---: | ---: | ---: | ---: |
| LDA | 1.908 / 2.075 | 0.919x | 4.838 / 4.106 | 1.178x |
| PBE | 0.533 / 0.749 | 0.711x | 3.497 / 2.750 | 1.272x |
| PBE0 | 0.566 / 0.554 | 1.020x | 1.952 / 1.945 | 1.003x |
| B3LYP | 0.612 / 0.602 | 1.016x | 2.122 / 2.022 | 1.049x |
| r2SCAN | 1.089 / 1.194 | 0.912x | 4.844 / 3.931 | 1.232x |
| M06-2X | 1.245 / 1.069 | 1.164x | 3.461 / 3.070 | 1.127x |
| MN15 | 8.824 / 8.696 | 1.015x | 10.081 / 9.773 | 1.032x |
| WB97M-V | 6.514 / 4.841 | 1.346x | 23.431 / 13.805 | 1.697x |

All 128 returned family samples pass their respective independent gates. Across
the 64 Fast calls, maximum errors are `2.56e-10 Eh`, `8.40e-6 Eh/bohr` maximum
force and `3.99e-6 Eh/bohr` RMS force. The FP64 maxima are `3.76e-12 / 7.13e-10 /
3.30e-10` in the same units. These are measured errors for these physical inputs,
not universal error bounds.

The same physical node4 GPU separately compares the packet-v4 incumbent with
native-force-v5. The change retains identical SCF iteration/mixed/strict work
populations in both arms; it removes compatibility force orchestration and
retains the qualified native FP64 force schedule.
The old/new snapshots run as separate sequential jobs, not randomized or
interleaved old/new arms. Within each snapshot the middle repeat reverses
FP64/Fast order. Observed operator-work populations also match in both arms.

| Endpoint | Previous Fast | New Fast | Elapsed reduction |
| --- | ---: | ---: | ---: |
| PBE, 3 atoms | 2.302 s | 0.749 s | 67.5% |
| PBE0, 3 atoms | 2.058 s | 0.554 s | 73.1% |
| r2SCAN, 3 atoms | 2.691 s | 1.194 s | 55.6% |
| PBE, 12 atoms | 4.385 s | 2.750 s | 37.3% |
| PBE0, 12 atoms | 3.574 s | 1.945 s | 45.6% |
| r2SCAN, 12 atoms | 5.583 s | 3.931 s | 29.6% |

Earlier same-card packet-only evidence improves 12-atom WB97M-V from 15.043 to
14.829 seconds (1.42%) and 96 atoms from 767.212 to 749.552 seconds (2.30%).
That packet-v4 build's separate 96-atom FP64 pilot is 1063.615 seconds, or
1.419x FP64/Fast using the two-repeat Fast median. The final v5 96-atom pilot
finishes in 810.377 seconds on a **different node5 GPU/driver**; it is not a
paired speed comparison or a regression inferred from the node4 number. It
executes 21 Focks, 17 mixed plus four strict, and passes the authenticated
retained independent reference with errors `6.68e-10 / 3.86e-6 / 1.17e-6`.

## Source identities

- Core scientific base: `4a36f6f3ef5d072eabba0f0fe02a4521d76ecdd0`.
- Packet-v4 source: `9e709e551c8261fad08c1c0011fbb3cec4e1ae259c9ca466c78c07d5a76b3b28`;
  core SHA-256: `b6a06237afe3f03b0eb416b0dc9f64c13bece29ad86297b4a2730610601975f7`.
- Native-force-v5 source: `0d351ff90f03b77b6f3cad07fef225e4a4ac13f72c32384172043ae10c0b2410`;
  core SHA-256: `caf921d07b7909f501e715419b3563a9a4660930d0be7260467c293d9cd6dde8`.
- Six existing generated force packages retain their own unchanged binary and
  manifest identities; a new core identity does not requalify their arithmetic.

Later master changes are not silently folded into these qualification receipts.
The retained reverse patch reconstructs packet-v4 from the v5 production source
and was checked against the complete canonical source identity.

## Scope and negative evidence

Native force ownership removes compatibility request/provenance orchestration
for already-qualified complete graphs. Relaxed SCF controls still select FP64
derivative arithmetic; they do not confer strict-density mixed-gradient
qualification. The native token, physical-state and memory checks remain active.
Independent tests forbid the Python compatibility consumer for admitted graphs,
including genuinely retained prepared warm/moved execution.

A displaced PBE/12 proposal fails the final physical-state proof on both the
incumbent and candidate and takes an explicit bounded FP64 fallback plus audit.
It remains a negative acceleration result, not a relabeled FP32 success. The
initial over-strong lifecycle assertion and both paired diagnostics are kept.

Small PBE and r2SCAN cases can still be slower than optimized FP64: mixed J uses
canonical whole-shell recurrence while strict J has density-contracted MD. Do
not disable MD in the comparator or enable a different FP32 density-product
contract to manufacture a win. LDA/MN15 and unsupported complete native force
graphs retain their existing bounded consumers.

The packet optimization reuses immutable point-scale proofs; radius admission
still runs per pair and FP64 weighted reductions remain unchanged. Its measured
incremental endpoint gains must not be combined across hardware with the earlier
density-screening gains. Unequal artifact warmups do not prove a cold-start gain.

## Selected validation

- Host precision/code-generation selections: 136 passed, 1062 GPU cases skipped;
  the final admission/inventory selection separately passes eight tests. These
  selections overlap and are not a whole-repository test count.
- Earlier native-owner/cold selection: 26 passed on this production core. The
  final file contains 28 cases and was not rerun in full after test expansion.
- Final scalar artifact-replay/moved and retained prepared warm/moved lifecycle
  selection: 12 passed on node4, Slurm 1045. Five families cover 3/12 atoms;
  prepared lifetime cases cover PBE0 at both sizes. Every returned endpoint is
  checked against the independent reference, including explicit moved recovery.
- Final `compute-sanitizer --tool memcheck` selection: two 3-atom PBE0
  scalar/prepared lifetime cases passed, zero errors, in the same finite Slurm
  allocation. Sanitized runtimes are not benchmark samples. A requested time
  extension was refused; the job still completed within its original 15 minutes.
- Seven existing Fast-label/fallback-count assertions also fail on the baseline,
  as retained in the density-screening decision. They are not waived, deleted or
  counted as passing; no whole-repository green claim is made.

## Evidence retention and reproduction

Full vectors, references, repeats, work inventories, source recovery inputs,
initial failures, XML, scheduler logs and diagnostic profiles stay outside the
tracked source tree in `../evidence-fast-goal-20261016/`, with predecessor
receipts in `../evidence-fast-packet-20261016/` and
`../evidence-fast-followup-20261016/`. The summary records the local archive and
manifest digests. Nothing is uploaded as a GitHub Release or external backup.

Verify and restore the local checksum-inventoried archive into a new directory:

```bash
PYTHONPATH=. python -m tools.unpack_evidence \
  ../evidence-fast-goal-20261016 --output /tmp/shared-fast-evidence-restored
```

The archived driver composes existing benchmark primitives, not a new permanent
benchmark framework. Reproduction must bind the matching source/core/force
packages and use the archived finite-`srun` wrapper. The ordinary maintained
entry point remains `python -m benchmarks.readme_dft --method <method> native
--policy fast|fp64`; independently verify matching physical protocol and cache
conditions before comparing with these historical receipts.

See the durable decisions for [density screening](../../../.agents/notes/implemented/performance/2026-10-16-density-screened-mixed-fock.md),
[packet proofs](../../../.agents/notes/implemented/performance/2026-10-16-packet-fp32-scale-proofs.md)
and [native force ownership](../../../.agents/notes/implemented/performance/2026-10-16-budget-fock-native-force.md).
