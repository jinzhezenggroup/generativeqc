# Decision: retain XC low words with a bounded warp reduction alongside corrected J

Status: implemented; component and complete cold/moved gates passed for r2SCAN-3c direct/96
Date: 2026-10-15

## Problem

The [Direct-J investigation](../../proposed/2026-10-15-r2scan-direct-j-roundoff-root-cause.md)
establishes the first differing operation in r2SCAN-3c direct/96. The
[J-only implementation](2026-10-15-generated-direct-j-compensation.md) improves
three moved endpoints from 19/21/13 to 13/13/13 Fock builds, but its three cold
calls take 26 builds versus 24/33/24 in the incumbent. Keep this complete cold
negative: compensating J alone does not close every phase's performance.

The [previous serial XC-only trial](../../rejected/2026-10-15-r2scan-cse-rounding-and-xc-totals.md)
left J noisy and did not identify the first differing operation. That rejection
does not establish that ordinary XC scalar reduction is sufficient once J is
corrected. A new isolated control therefore holds the corrected J fixed and
adds precisely the archived generic XC-total correction, with global CSE off.
No SCF gate, point expression, Vxc coefficient or force schedule changes.

## Component control

Slurm 8682, node1/main/gpu:5090:1, visible device 3, finite 01:10:00, runs a
new cold owner and one moved owner using the original independent reference
and unchanged original native moved seed. Both pass the independent 1e-8 Eh /
1e-7 Eh/Bohr gates. The combined serial control takes 24 cold Fock builds
(450.530 complete seconds) and 13 moved builds (294.120 seconds). Its cold
history no longer has any step with both recorded maximum density/residual
<=1e-10 and energy change >1e-12. J-only cold repeats have extra energy-blocked
iterations at 24 and 25. This is a new conditional control, not a retrospective
claim that the old XC-only experiment solved the J problem.

The serial scalar reducer has only one active lane per channel traversing every
point. Its cost is measurable even with the same Fock count, so do not promote
the serial implementation or infer an endpoint win from a sum microbenchmark.

## Decision

Retain three high and three low words in the existing runtime-owned XC arena.
Charge all six through the shared overflow-checked layout; an exact-size arena
must execute and one byte less must refuse before publication. Clear six words
at every strict-FP64 enqueue, accumulate across all point batches, fold once
before publishing the unchanged three-double scalar view. No full-grid retained
array, hidden allocation, CPU work or functional-specific path is introduced.
The existing FP32 schedules retain their original accumulation.

Replace the serial per-channel traversal with one full 32-thread warp. Each
lane compensates its strided points within the original compact tile layout.
A fixed shuffle tree merges each neighboring subtree's high **and** low word;
lane zero merges both resulting words into the persistent per-channel pair.
All lanes participate even for empty/partial tiles. The existing 32-thread
launches and stream ordering are unchanged. Nonfinite inputs remain subject to
the same error audit. Correction improves accuracy, not bitwise SCF determinism.

## Invariants

- Preserve point algebra, Vxc, spin/channel ordering, all screening and numerical
  acceptance gates. Keep J's correction owner separate from K/derivative views.
- Keep the three public scalar outputs and preserve low words until the final
  fold; never merge a subtree's high word while discarding its correction.
- Keep launch size exactly one full warp and avoid hidden full-grid work/storage.
- Qualify the coupled candidate with complete cold and moved SCF+force calls,
  fresh owners/processes, source/library receipts and Fock/XC work counts.
- Do not enable global SCF CSE or claim other consumers' performance by analogy.

## Component evidence

The private parallel image is
`/data/jzzeng/qc-r2scan-jxc-parallel-20261015` on n1. Build receipts reuse the
read-only XC-control CUDA/host objects and rebuild only the grid translation
unit containing the changed emitted reducer. The two Direct-J host replacements
are reused from the separately source-bound J+serial-XC image. Native integration
tests compile their required test hooks against the exact same generated grid
source, not the serial test object. The first image's reused stationary modules
are not admitted: their contract predates the warp emitter, as recorded below.

- The actual emitted warp reducer passes 23 independent `math.fsum`, dyadic
  cancellation, three-channel, empty/partial tile, cross-batch and full
  2,359,296-point-grid tests on a finite Slurm allocation. A four-case
  cancellation/nonfinite Compute Sanitizer subset has zero errors.
- The full native E/V/state suite passes, including dense/indexed/compact/ragged,
  optional provider fallback, FP32, response and graph behavior. Workspace,
  point-batch and indexed-density memcheck subsets each report zero errors.
- The generated/native/bounded J fixture passes against the same parallel
  image, preserving independent J oracles and two-plane reset checks.
- Compiler structure: 541 modules, zero dependency errors. Changed-line
  clang-format and Ruff pass before the frozen endpoint campaign starts.

## Rejected stale-module image and matched rebuild

Slurm 8713 stops after two of twelve requested points. Its baseline moved call
completes, but the candidate converges in 13 Fock builds and then fails the force
stage with `stationary CUDA AOT contract_identity identity mismatch`. Its
215.646 seconds do not include successful forces and are not an admitted timing
result. Preserve this failed population under the original image's `results/`;
do not combine it with completed calls from a later image.

The stationary contract hashes all compiler files in the common/integral/xc/dft/
method/tensor families, not just imported modules. Comparing the frozen serial
and warp source inventories identifies exactly one changed compiler input:
`dft/xc_totals_cuda.py`, from SHA256
`d81f322e12ab8899f5f936788818a3de5a0a1b2bd5013d99d1a81d0523b01c67`
to `318a7058a377116028570bef6c08f601feda22c226d8f7cccd8f6af40892cc63`.
The existing CMake manifest dependencies already include this source closure;
37 AOT-dependency/CMake ownership tests pass with the new helper present. This
failure comes from manually reusing the old modules/manifests in the private
selective image, not evidence that the normal build needs an extra dependency.

Create a separate frozen `matched-aot` prefix, with a byte-identical copy of the
qualified core and real regenerated modules. Run the official source generation,
wrapper compilation, device link, host link and manifest commands for both
`r2scan_rks` and `r2scan_rks_spd`. Reuse the 24 s/p and s/p/d primitive objects
only after regenerating and checking exact source-byte equality. The two wrapper
objects are rebuilt; no contract field, cache key or runtime check is overridden.
`matched-aot/evidence/build.json` binds the core receipt, commands, objects,
modules and manifests. The first candidate in Slurm 8738 passes complete SCF and
forces, confirming that this actual matched image satisfies the runtime checks.

## Complete molecular qualification

Slurm 8738 completes all twelve balanced baseline/candidate calls on node1,
main/gpu:5090:1, assigned visible device 1, eight CPU threads, finite 06:30:00.
The unchanged 1800-second per-point bound is not reached. The requested order is
`moved:B0,C0; cold:B0,C0,C1,B1; moved:C1,B1,B2,C2; cold:B2,C2`.
Every call uses a fresh process and owner. Cold calls have no checkpoint input
and export unique new checkpoints; moved calls all import the original native
seed, never a candidate-generated seed. No requested point is discarded.

The case is 96 atoms, 1088 spherical AOs, r2SCAN-3c/def2-mTZVPP, direct/no-DF,
48x16x32 atom-centered grid (2,359,296 points). The original energy 1e-12 Eh,
density 1e-10, screening 1e-12 and maximum 100 iterations remain unchanged.
The independent acceptance gates remain 1e-8 Eh and 1e-7 Eh/Bohr. All twelve
complete SCF+force calls pass; the largest candidate errors are
3.8654e-11 Eh and 6.0141e-11 Eh/Bohr. The largest errors across both arms are
4.2292e-11 Eh and 1.4518e-10 Eh/Bohr.

| Phase/arm | Complete seconds, repeats 0/1/2 | Fock builds = XC evaluations | Median seconds |
| --- | --- | --- | --- |
| Cold baseline | 558.633 / 545.019 / 446.713 | 33 / 32 / 25 | 545.019 |
| Cold coupled candidate | 446.207 / 445.712 / 445.380 | 24 / 24 / 24 | 445.712 |
| Moved baseline | 379.717 / 380.527 / 296.530 | 20 / 20 / 14 | 379.717 |
| Moved coupled candidate | 288.359 / 289.133 / 288.979 | 13 / 13 / 13 | 288.979 |

Complete seconds include preparation and the entire SCF+force execution. Median
candidate/baseline ratios are 0.817791664 cold (18.22% lower elapsed time) and
0.761038220 moved (23.90% lower). Each candidate history has no conservative
energy-only-blocked step; the baseline histories retain such steps in all six
calls. These are complete-endpoint improvements from avoiding extra operator
applications, not evidence that an individual compensated J build is faster.
Per-channel J/K application counters are unavailable (`complete: false`, both
counts null); only the actual complete Fock/XC counts above are claimed.

The unchanged moved seed SHA256 is
`84cb007bfbb43233565a782d6533ef87209a0fc04b2f408482d2f3d43944b7d1`;
the independent oracle SHA256 is
`a27d03119f2f7387db6d6bdc9abb306b4b137a3a8ac891431a26a25f5a456dfc`.
The source base is `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`. The qualified
core SHA256 is
`5c1902010a496cb2dfb656e5cc3b73bbed0c8508cce851b3abdf3fc6ae4b812f`;
the matched AOT build receipt SHA256 is
`1a4771bcf270c7cca06ffb9899c4a1a3c62569600fdf78025d54208058979c1d`.
The actual s/p and s/p/d modules have contract identities
`5054ecd618ef724857a1f11e2639cd5cdfa7221c168462866da667743fc35b88`
and `73662cad6bd0d038be14bc2e25f9bd56d4875379596ef1596f79a42c29bed99e`;
their binary SHA256 values are
`eee6366b428c13bbc038d5a816dd3ddbb0d8a05e5aed95f1b9996c239ec55066`
and `9aa5abd61db1dfcf5eb86502725ef340e86ec8b85e42bc58300dfa948d4b0432`.

## Final audit and retention

The focused host suites pass 1600 cases plus 37 AOT-dependency/CMake cases.
Slurm 8779 re-runs all 23 independent reduction cases using the final test file,
verified ccache 4.5.1, visible device 0 and a finite 00:05:00 allocation. The
original four-case sum sanitizer and three native DFT sanitizer subsets remain
zero-error. The J-only note retains its independent J sanitizer and the broad
canonical-routing failure reproduced in both incumbent and candidate; this is
not a claim that every repository test passes.

`matched-aot/evidence/audit.json` verifies all twelve calls, original seed,
protocol/reference/driver/library identities, nine final production source
hashes, all reused link inputs, the 24 regenerated byte-identical primitive
sources, actual new AOT modules/manifests, emitted grid helper, native test-hook
build lineage, host/compiler checks and real-GPU component logs. It preserves
the prior force-identity failure outside the timing populations. Raw binaries,
checkpoints and all journals remain in the private n1 roots above.

The checked local and remote compact archive is
`coupled-qualified-evidence.tar.xz`, 123 files, SHA256
`41967a180d0ca2242f0da97b8a0abcaeedae5f5a56bc9b98e554037ee4a57f27`.
It includes the frozen qualified source patch, independent driver/reference,
all complete and failed endpoint journals, build/audit/source receipts, emitted
grid source, regenerated wrappers/manifests and component logs, excluding large
unchanged binaries and checkpoints. The ignored local campaign is
`.artifacts/r2scan-j-compensation-20261015/`; the remote archive is under the
parallel image's `matched-aot/` directory.

## Consequences and boundaries

This is a generic FP64 accumulation correction, not a functional exception or
convergence relaxation. It adds one admitted Cartesian J plane and 24 XC arena
bytes. Existing bounded provider fallbacks remain explicit. Retain the J-only
cold 9.76% negative and the serial reducer's cost: numerical accuracy is not a
zero-cost kernel optimization, and these twelve calls do not prove a matched-
work speedup. The present evidence qualifies this direct/96 cold and original-
seed changed-geometry consumer, not DF, every functional/size, or bitwise SCF
determinism. Global SCF CSE stays off; this does not promote #2507's CSE policy
or duplicate #2520's separate force-only work.

Revisit if another provider dominates, optional capacity changes admission, or
an equally accurate bounded reduction reduces fixed-work cost. Any such change
needs fresh independent numerics and complete endpoint populations, not a
microbenchmark or an edited artifact identity.
