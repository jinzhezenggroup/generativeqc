# Decision: bounded native Gaussian ECP geometry sources

Status: implemented, with molecular/public qualification boundaries below
Date: 2026-11-11

## Problem

Checked radial quadrature does not reliably resolve tight residual projectors
on the original Cs-H Hessian fixture. Increasing a diagnostic grid cannot repair
a mismatch between the Hamiltonian used by SCF, its forces and response sources.
The preceding [Gaussian-projector investigation](../../proposed/2026-11-11-gaussian-ecp-projector-identity.md)
qualified complete private one-electron jets, but exporting their global
coordinate-squared AO tensor is not a scalable production implementation.

## Decision

`integral/ecp_gaussian.py` lowers the radial-power-two local and s/p/d projector
operators through the existing scalar `Graph` and `ScalarCEmitter`. Cartesian
raising differentiates the same primal seed. Generated shift blocks cover s--f
orbital values and first/second nuclear derivatives; their maximum shifted
orbital degree is five and pair degree is eight. The regular Bessel series has
analytic origin limits; scaled recurrence is selected above argument 50.

Generated functions use stages of at most 512 scalar assignments with one
endpoint-owned scratch buffer. The compiler rejects a lowered kernel exceeding
524,288 doubles of scratch. The current largest kernel uses 310,663 doubles.
This bound is a resource contract, not a generated-source fingerprint.

`src/integrals/ecp_gaussian.cpp` owns allocation, primitive/channel scheduling,
compensated contraction and physical-atom scatter. Common ECP center/exponent
groups share radial frontiers and shifted blocks. The checked CPU production
selector chooses this specialization for supported operators. Other legal
radial powers/projectors retain checked-grid execution; CUDA is unchanged.

The weighted second-order interface accepts arbitrary ordered real AO weights.
Triangular off-diagonal pairs use the sum of both ordered weights, with no
implicit occupation factor. It contracts before physical-atom scatter and
retains only local/nonlocal coordinate gradients and Hessians. It never owns a
global `(3N)^2 * nao^2` tensor. Workspace and conservative generated-block work
are admitted separately before primitive execution; default limits are 512 MiB
and 64 * 1024 * 1024 blocks. Actual semantic counters accompany the result.

The private KS snapshot ABI borrows immutable native integral geometry while
holding the context lock, validates the snapshot token again before publication,
and leaves caller output/work buffers untouched on failure. No copied grid or
density panels are needed for this geometry source. The Python response owner
caches the fixed-density contracted Hessian once and forms each component HVP
as `H_fixed @ direction + first : delta_density`. The frozen-Fock source uses
the same first derivatives. Stale tokens fail; different density weights cause
a bounded temporary recomputation rather than reusing an unrelated Hessian.

Shape-only HF/KS resource queries lack power/projector metadata, so CPU admission
reserves the larger of checked-grid and analytic storage. The latter includes
compensated output/publication, the compiler scratch cap, shifted-block caches,
primitive jets and conservative shell/group metadata. Native and portable sizing
share the compiler policy. CUDA retains its existing grid-tile inventory.

## Rejected alternatives

- Loosening the original Cs-H derivative gate or scaling its direction would
  conceal the Hamiltonian/source discrepancy rather than resolve it.
- A private hand-written derivative algebra duplicates the compiler's scientific
  owner and makes primal/force/Hessian consistency harder to establish.
- One enormous generated function overwhelmed compiler optimization resources;
  noinline bounded stages preserve the emitted evaluation order.
- Per-primitive generated scratch allocation or a global raw AO Hessian creates
  unnecessary allocation/work or coordinate-squared AO storage.
- Reference-engine derivatives or displaced SCFs remain independent oracles,
  never production providers.

## Evidence and artifact identity

These are actual completed processes from the pre-resource-inventory dirty
source cohort, not clean installed-wheel or later-source qualification:

- Native projector/checked-grid/Gaussian regression suite: three passed in
  50.97 seconds. Source-import Python response/fallback suite: 23 passed in
  2.82 seconds. Tests include independent PySCF value/first matrices, complete
  mixed-derivative directions, density response, frozen Fock, failure recovery,
  transactional caller buffers and stale-token rejection.
- Original unchanged nightly Cs-H RKS unit-Cs-x HVP: one passed in 408.10 seconds.
  Electronic errors at steps `[0.003, 0.001, 0.0003]` are
  `[1.121918442023584e-6, 1.2448700328904216e-7, 1.121267932668868e-8]`,
  below the original `1e-5` gate. Electronic plus independent D3 errors are
  `[1.1219208960128226e-6, 1.2448727595514817e-7, 1.1212703867335285e-8]`.
  Native HVP time is 400.0256073 seconds; native plus reference validation is
  406.5873610 seconds. Concurrent jobs prevent an isolated performance claim.
- The actual production library is also compared with the complete independent
  moment/radial reference for Ni/I/H and Ni/Cs/H, Cartesian and spherical.
  Complete primal/first AO panels and early-contracted 9x9 coordinate Hessians
  for nonsymmetric, transposed and positive-semidefinite weights pass the
  unchanged gates at all three steps. Maximum value/first/second errors are
  `2.132e-14`, `3.693e-14`, `5.615e-11`. This production test does not export a
  global raw AO second tensor; that private reference gate is separately retained.

The tested library SHA256 is
`90f39ce0385b5551767164bbf2d19bfdf6cf10566354a793d0ca0288b2fb29b7`.
The member/hash-verified local recovery
`build/qualification/ecp-gaussian-production-20261111/snapshot-acceptance-recovery.tar.gz`
contains 537 members / 75,057,243 bytes, SHA256
`67e27eb7b676ab978c385de8894635e4713df2e482ae99811c363e2980c3935b`.
Its manifest/receipt retain original sources, library, generated/JIT artifacts,
drivers, arrays and results. The failed old-installed-Python harness and initial
tuple/Atom contract harness remain failures, not relabeled successes. Later
resource changes require separately identified build/test evidence.

After adding the shared scratch cap and shape-only inventory, a new native
build completes in 233.70 seconds; only six compiler invocations are needed
(six misses, no cache errors), with generated kernel bodies unchanged. Its
library SHA256 is
`2436a788ed7b1c8f504872e4ffea7cee061389eeb6f1314abd7a93ee18bb42d7`.
The new resource/source/fallback suite passes 88 tests in 3.58 seconds,
including native/portable overflow and small-scratch/large-derivative shapes;
the three native regressions pass in 50.95 seconds. All nine existing
whole-molecule global-hybrid ECP force tests pass in 145.04 seconds, including
RKS/UKS, covariance, resource refusal and the native default-grid Cs endpoint.
The separately named `resource-complete-contracted-reference` driver repeats
the complete production/reference comparison in 8.80 seconds with unchanged
gates. These outputs are under the production qualification directory above,
not restamped over the earlier cohort. Compiler ownership checks report 604
modules / zero dependency errors, and all changed-file pre-commit hooks pass.
The original unit-Cs-x nightly HVP is also rerun against this new library:
one passed in 407.26 seconds, retaining the original direction and `1e-5` gate.

A first synthetic NaH2 doublet screening attempt does **not** pass matched-state
acceptance: the independent and native unseeded SCFs converge to different
stationary roots (energy difference `0.06657756592545849` Eh). The independent
energy at the native density agrees to `3.764e-14` Eh and its orbital residual
there is `2.357e-14`, so this is not evidence of differing Hamiltonians. Native
and independent spin-square values are `0.8400483455` and `1.1836111363`.
Retain the actual failed screen and diagnostic in
`build/qualification/uks-analytic-ecp-20261111/`; no density is cross-seeded and
no gate is relaxed. This attempt does not justify scalar-ECP UKS admission.
Removing the Na p shell on the same separated-H geometry also fails the
matched-state energy gate; its separate `s-only-screen` receipt remains failed.
A distinct Na + bonded-H2 geometry keeps both spin occupation spaces `(2, 1)`
and the original s/p basis, and passes the unseeded matched SCF/force screen in
0.71 seconds (density error `1.721e-11`, force error `5.466e-12`). The
`bonded-h2-screen` receipt qualifies only this synthetic SCF/force fixture, not
its second derivatives or the earlier failed roots, canonical def2 or public ECP.

## Consequences and remaining gates

This fixes a production one-electron/source mismatch and one original low-level
RKS HVP failure. It does not complete issues #2844--#2848. The low-level scalar-ECP
UKS candidate remains closed, as do installed public ECP Hessian capabilities.
Complete total matrices, spin/root/physical-response gates, mixed Ni/I/Cs and
25--40-atom resource/work qualification, installed execution and the requested
Python-free TS/frequency workflow remain separate requirements. In particular,
the simultaneously running installed all-electron Ni UKS endpoint has no bearing
on ECP admission. Do not promote capabilities from one successful integral test
or direction, and do not restamp the archived cohort as a newer build.

Revisit the retained fallback or projector/power bounds only with independent
operator, force, response and complete-endpoint gates. Revisit staging/storage
only with compiler-lifetime evidence and endpoint work/resource measurements.
