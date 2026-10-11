# Decision: share automatic point lowering and screen in physical spin coordinates

Status: implemented
Date: 2026-10-11

## Problem

The automatic CPU registry owned its physical-spin Libxc point lowering in a
build script. CUDA needs exactly the same imported Graph, symbolic E/vxc roots,
work-domain policy, Cartesian sigma chain rule, and native half-vtau convention;
copying that lowering would create a second scientific owner.

Independent point-output tests also exposed a pre-existing numerical defect not
visible in the already qualified zero-beta molecular force endpoints. The Maple
`screen_dens` predicate reconstructed a spin density from `rs` and polarization.
At the worker's density floor the FP64 inverse-coordinate round trip could put
the nominally empty spin just above its screening threshold. The inactive
MGGA_X_LTA kinetic coefficient then became approximately -1116 instead of zero.
An unoccupied spin channel can hide this error in energy/force contraction; that
does not make its potential or generalized-KS matrix scientifically correct.

## Decision

- `xc/automatic_point.py` owns automatic CPU/CUDA point source emission using the
  existing bulk Graph, `ScalarCEmitter`, and Libxc work-policy owner. The CPU
  generator only binds that source to its existing native registry descriptor.
- CPU retains domain exceptions. CUDA uses the existing `DevicePointValue`
  validity flag. Both backends retain the exact same semantic/output identity;
  their emitted-source identities differ by ABI and guard bytes.
- The bulk physical-coordinate adapter substitutes the exact inverse-coordinate
  expressions for alpha/beta densities with the corresponding input variables
  **before** symbolic differentiation. This uses the existing Graph replacement
  operation, applies to all registrations, and does not branch on functional names.
  For the adapter's three-dimensional `rs` and polarization definitions these
  expressions are algebraically the original spin-density coordinates. Avoiding
  their rounded reconstruction deliberately stabilizes the existing `<=` screen.
- Regenerate the catalog and its registered provenance through the existing
  importer. All 544 registrations, 221 imported registrations, and 197 automatic
  registry entries remain unchanged; affected Graph node counts change.
- CMake tracks the new point owner and existing work-policy source explicitly.
  Generated mathematical bytes have a complete eight-shard regression baseline.

## Identity and admission boundaries

The refactor by itself preserved all ten generated CPU products' executable and
registry bytes. Only expression fingerprints changed, because the existing
capability inventory conservatively binds the entire compiler source closure.
Do not exclude the new source from that closure to manufacture identity stability.

The subsequent physical-coordinate correction deliberately changes mathematical
source and Graph identities. Its regenerated shard baseline and independent
oracle checks are therefore required, not optional refactor cleanup.

This change adds no cache owner, runtime Libxc dependency, public CUDA selector,
ordinary CUDA XC/KS execution, force capability, Hessian/HVP capability, or AOT
package qualification. CUDA point probes are test-only consumers. Existing
automatic CUDA runtime/resource rejection stays closed pending actual native
integration and complete-endpoint qualification.

## Rejected alternatives

- Ignore inactive-spin potentials because their occupied-density contraction is
  zero: this misses the actual E/vxc contract and generalized-KS matrix semantics.
- Widen tolerances or clamp emitted vxc afterward: the error is a wrong piecewise
  branch, not small ordinary rounding, and derivatives must follow the Graph.
- Perturb work density floors with `nextafter`: this silently changes the pinned
  worker policy and lacks a generic scientific justification.
- Copy the CPU formulas into CUDA, or call a CPU/oracle provider from the device
  endpoint: both violate the shared scientific-owner/production contract.

## Evidence

Before correction, the independent CPU and CUDA matrices each had seven failed
UKS cases, with the same spin-screening defect. After correction each backend
passes 33 native point tests over 11 non-curated registrations: three LDA, five
GGA, and three tau meta-GGA, each in RKS/UKS physical layouts. Tests include 37-point
partial tiles, vacuum, both empty-spin orientations, minority-density floors,
zero gradients, FHC boundaries, required-input rejection, and independent
directional derivatives at interior points.

The oracle is independently installed PySCF 2.14.0 / Libxc 7.0.0. E/vxc gates are
absolute 2e-11 and relative 2e-9; directional gates are absolute 2e-10 and relative
2e-7. CPU maximum E/vxc absolute error is 1.4211e-12; CUDA maximum is 1.3074e-12.
Maximum independent directional error is 5.6708e-11 on both backends.

The final CUDA campaign runs on n5's RTX 5090 through Slurm job 1465, with
`--partition=main --gres=gpu:5090:1 --time=00:10:00`, preserving assigned device
visibility. NVCC 12.9 targets sm_120, FP64 with contraction disabled; verified
sccache 0.16.0 wraps each object compilation. Retained test receipts bind exact
source/object/binary hashes, compiler/options/cache launcher, physical device,
driver, Slurm job, output layout, and PTXAS resources. Observed kernels use
56--254 registers. The retained TPSS receipt has 120 stack bytes and 184 bytes
each of spill stores and loads; the other probes have zero stack/spills. These
are static point-probe resources, not profitability claims or production
schedule admission. The earlier zero-spill summary was a reporting error, not
what the retained TPSS measurement showed.

The broader compiler/independent-fixture campaign passes all 558 non-catalog
checks, including the 442 polarized/unpolarized imported E/vxc/fxc fixtures.
The catalog-reproduction failure caused by changed Graph node counts is resolved
by canonical regeneration; `import_libxc_bulk.py --check` and source-registry
verification protect freshness. Source-only ownership/docstring checks pass.

The complete CPU force regression initially reports 224 passes, 40 skips, and
one LTA finite-difference SCF failure. The displaced UKS He/H endpoint stopped
its DIIS loop at iteration 8, but the unshifted final closure returned density
RMS 1.0271e-10 against the requested 1e-10. Its physical residual was 3.0690e-11.
This is an intentionally fail-closed final-state audit, not a force discrepancy.
The qualification producer now requests a stricter native density tolerance of
1e-11, keeping its 200-iteration bound and all independent energy/force gates.
The three targeted LTA lifecycle/zero-beta/s-p force tests then pass. Increasing
only the iteration budget would not fix an already accepted DIIS proposal;
never bypass the bounded physical closure or relax its acceptance threshold.
The full final force campaign is retained separately from this failed diagnostic.

Final force/evidence regression has 42 passing tests; its one failed test read
the new compiler file during a simultaneous Git rebase that briefly removed it.
The isolated immutable-checkout rerun passes. All eleven functional receipts
then revalidate against the final compiler/capability bytes: 66 cold/replay/
changed-geometry CPU endpoint rows pass, with maximum independent force error
8.3072e-9 Hartree/Bohr and energy error 4.1880e-9 Hartree. Keep qualification
checkouts immutable while running source-bound campaigns; do not mistake a
checkout mutation for a scientific or solver failure.

Ignored local receipts are retained under `.cache/1118-point-cpu-final/`,
`.cache/1118-point-cuda-final/`, and `.cache/1118-point-force-qualified/`.
The final force receipts are in `.cache/1118-automatic-force-final/` plus the
stable LDA_C_VWN_4 rerun in `.cache/1118-vwn-diagnosis/`.
The pre-correction campaigns and pre-refactor complete CPU products remain in
separate ignored directories for reproducing the diagnosis. Raw test logs are
under `/tmp/qc-1118-point-*-tests.log`.

## Revisit when

Extend the existing native CUDA schedule, semantic descriptor, and AOT/cache
owners to consume this source. Qualify ordinary CUDA XC/KS and force endpoints,
prepared invalidation, complete supported-toolchain closure, and the combined
catalog before promoting or closing the remaining #1118 lanes. The TPSS point
probe's 254 registers also require explicit schedule/resource consideration;
passing a numerical point test is not sufficient evidence of a useful endpoint.

## Split-hybrid source-matching follow-up

PR #2255's CUDA and CuMetal builds exposed an interaction with the existing
M06-2X Stoll cancellation repair. The bulk importer now canonicalizes physical
spin densities before differentiation, but the repair reconstructed its source
terms with the old inverse-rs/zeta coordinates. Its exact reachability guard
therefore correctly rejected those terms even though the pinned PW source and
parameters had not changed. This reproduces as three failing split-hybrid
generation/work-policy tests and the same `perpendicular PW terms changed`
exception in the CUDA release-resource job.

The importer and source-term matcher now share one physical-spin-coordinate
canonicalizer. Restricted embedding preserves the importer's literal zero zeta
when both spins are the same Graph node; it must not introduce a new subtractive
round trip. The pinned-parameter, reachability, shared-PW and structural guards
remain intact, including a regression rejecting a deliberately absent source
term. The complete automatic CPU mathematical-byte baseline is not relaxed, and
catalog membership and Graph node counts remain unchanged.

The original measurements above refer to the pre-follow-up compiler closure.
The shared canonicalizer changes source-bound identities even though its
automatic executable mathematics is unchanged. Final-source point and force
receipts must therefore be regenerated, not relabeled or silently promoted from
those earlier measurements. The independent split-hybrid boundary oracle remains
the authenticated Libxc 7.0.0 source archive, including the existing 113-bit PW
cancellation check; successful code generation alone is not numerical evidence.

The follow-up at source commit `5b54d660bd9114fcf0177ce8b7eca3e948716daa`
passes 508 compiler/independent-oracle tests (including all 442 both-spin
E/vxc/fxc fixtures and the split-hybrid boundary oracle), 33 CPU point tests,
and 33 real CUDA point tests through Slurm job 1468. The complete CPU force/
evidence suite passes all 43 tests in one immutable-checkout run; eleven receipts
revalidate all 66 lifecycle rows, with maximum independent force error
8.307147636443801e-9 Hartree/Bohr and energy error 4.187928448473599e-9 Hartree.
These are measurements of that compiler closure, not future rebases or an
ordinary CUDA XC/KS/force endpoint.

An additional native release-resource compilation exposed cu++filt's typed
template literals: CUDA 12.9 prints `(long)4` and `(bool)0` where GNU prints `4l`
and `false`. The resource decoder now normalizes only those known integer casts
before passing names to the unchanged active-region classifier. Unknown casts/
values remain unsupported. In particular, mixed-precision kernels must not
substitute for a missing strict-FP64 kernel. Compiler logs keep the original
mangled names and resource measurements; no register/spill gate is relaxed.

The existing `census_libxc_aot.py` owner also completes a full both-spin E/vxc
CPU census: 442 registration variants form 433 exact-source groups, all 433
object probes compile, and alias grouping saves nine build tasks without merging
scientific provenance. Unique emitted CPU source is 9,461,460 bytes and aggregate
object size is 6,719,496 bytes. The corresponding CUDA source-only census has
442 variants, 433 groups, and 9,470,986 unique source bytes; its explicit 64-task
budget selects 928,647 source bytes. This latter census does not compile or run
those CUDA groups. Neither census measures installed AOT bundles, full supported
toolchain/header/library cache closure, cold compile cost, or public admission.
The existing cached object probes and their diagnostic recipe hashes are not
new persistent-cache keys. Raw reports are retained under `.cache/1118-ci/`.

## References

- Issues #1118, #1119, #1121, #1122, #1123, and #1124.
- CPU endpoint stage: PR #2242.
- `tests/python/test_libxc_automatic_point_source.py`.
- `tests/python/test_libxc_automatic_point_native.py`.
- `tests/python/test_libxc_physical_spin_screening.py`.
- `tests/python/test_libxc_automatic_forces.py`.
