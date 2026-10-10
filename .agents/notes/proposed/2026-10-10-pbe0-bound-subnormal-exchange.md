# Proposal: qualify a stable PBE0-only bound point before production dispatch

Status: proposed; experimental scalar CPU/GPU qualification passed, endpoint unqualified
Date: 2026-10-10

## Problem

The compile-time restricted point experiment has fewer instructions and
registers, but independent PBE0 tail cases failed in both the original and
restricted implementations. At spin density 1e-240, forming rho*cbrt(rho)
produces a quantized subnormal denominator. Dividing a representable gradient
by that rounded product causes approximately 1e-5 relative gradient error.
Baseline/candidate parity cannot establish scientific acceptance for this case.

The original supplemental 60-point PBE and PBE0 populations, their strict
failures and all prior evidence remain unchanged. General PBE additionally
has near-canceling component cases; this proposal does not resolve those.

## Experimental lowering

The existing exchange algebra is shared through a default-false
`StableSubnormalRatio` template option. Existing full-spin entries, the guarded
restricted entry and response consumers retain their original arithmetic.
Only the new private `evaluate_pbe0_restricted_bound` entry selects the option.
It has fixed PBE0 semilocal coefficients (0.75 exchange, 1.0 correlation) and
accepts one per-spin density and three Cartesian gradient values, constructing
equal features explicitly rather than assuming them from a label.

For the low-reduced-gradient branch with a subnormal rho^(4/3), compute

```text
u_k = (gradient_k / rho) / cbrt(rho)
```

instead of dividing by the quantized product. This is the same real-valued
ratio, avoids that intermediate precision loss and has bounded intermediates
in this branch. Normal-product arithmetic is unchanged. The reciprocal
high-gradient branch, energy underflow convention, input validation, spin
boundary extension and Cartesian derivative outputs are unchanged. There is
no density floor, input clipping, oracle substitution or tolerance relaxation.

This is an explicit new experimental numerical lowering, not a relabeling
of the old bit-identical prototype. The production header remains unchanged.

## CPU evidence

The actual retained original/candidate headers are compiled with verified
ccache, C++20/O2, `-ffp-contract=off` and the same output ABI. The existing
general entry is bit-identical on all 5,104 broad PBE0 inputs. Its existing
full-spin response is bit-identical on 32 targeted input/direction pairs;
that is a compatibility check, not complete response qualification. The bound
entry has matching validity and bit-identical outputs on the 5,104 broad
equal-spin constructed inputs (90 invalid statuses included).

All original 60 independent PBE0 inputs are retained. Another 27 unique cases
cover the normal/subnormal rho^(4/3) boundary, the reduced-gradient branch
boundary and minimum-subnormal gradient components. The unchanged original
rs/t2 formula and hybrid derivative helper produce the additional references
at 450 and 550 decimal digits with identical rounded outputs.

All 87 candidate points pass every one of the nine outputs under the unchanged
`5e-10*abs(reference)+1e-322` gate. The baseline still fails indices 8, 9, 65,
66, 67 and 68. Its failed results remain in the same retained record; no
incumbent pass is asserted. The broader original PBE qualification remains
failed and is not included in the claim for this PBE0-only entry.

The first CPU run stopped because importing the complete reference generator
would require unused PySCF dependencies in the local Python environment. Its
failed log/partial build remain under tail-host-v1. The succeeding v2 loads
only the original energy function and derivative helper verbatim by AST;
formula/helper hashes and independent fixture identity are retained.

## Real-device evidence

Slurm job 7029, node1/main, one RTX 5090, eight CPUs, finite 00:12:00 allocation.
Assigned CUDA_VISIBLE_DEVICES=0 is preserved. Verified ccache 4.5.1 wraps the
actual CUDA 12.9 compilation and link; existing shared cache stats show one
object cache miss and one uncacheable link, not cache hits. No cache was cleared.

The ordinary GPU run, compute-sanitizer memcheck and initcheck each pass:

- All 87 original-formula candidate points at the unchanged strict gate;
- All 5,104 broad CPU comparisons on valid outputs, plus all 90 invalid
  statuses, with no validity mismatch;
- 2,000 declared positive-density unpolarized Libxc 7 PBE0 interior points,
  checking energy and both copies of the rho/Cartesian-gradient coefficients.

Both sanitizer logs report zero errors. Original GPU failures remain indices
8, 9, 65, 66, 67 and 68, as on CPU. Source/data before-and-after identities,
binary hashes, outcomes and the completed marker are retained.

The point-only kernels in this same qualified module use 100 baseline and
84 candidate registers, with zero reported static local/shared bytes. These
are resource attributes, not executed memory traffic, production geometry
kernel resources or complete endpoint performance. No timing gain is claimed.

## Remaining production requirements

Lend a generation-checked proof from the grid producer before selecting a
separate restricted compiled entry; keep v1 GridTaskView ABI and the general
fallback intact. No experimental fast entry is yet selected by production.
Do not infer proof from functional labels or arbitrary external arrays.

Qualification here covers this bound PBE0 scalar entry, not production grid
ownership, complete forces, HVPs or unrelated PBE/UKS consumers. Full endpoint
energy/force gates and clean work-aware timing at 48 and 96 atoms remain
required before claiming a performance improvement or opening a stage PR.

Latest inspected master is 82c166cac. Subsequent #2187 adds resource/allocation
journals and #2180 adds GFN2 diagnostics; neither changes the point, grid or
stationary geometry sources relevant here. The baseline point header remains
byte-identical to the original. No old test matrix was repeated for these
merges, and the qualified production binary remains the distinct #2185 snapshot.

## Provenance and references

Experimental canonical point header SHA256:
`f9ddf9e8ee0c95680735bc812fe6b20e6d488025d8b7f4a4b44155bf4fd3a204`.
Namespace-only candidate probe header SHA256:
`c618d1e8cbbccc41cd322373966499a499f59a980c16d89d12f1039faa35fa83`.
GPU probe library SHA256:
`35244207b0c7e0c451d1acf957347b893f331f79a93f4d17e7c4da7d2bd62db8`.

- `.artifacts/xc-restricted-stage/evidence/tail-host-v2/qualification.json`
- `.artifacts/xc-restricted-stage/evidence/tail-device-v1/verified.json`
- `.artifacts/xc-restricted-stage/qualify-tail-host.py`
- `.artifacts/xc-restricted-stage/qualify-tail-device.py`
- `.agents/notes/proposed/2026-10-10-rks-point-compile-time-binding.md`
- `.agents/notes/proposed/2026-10-10-restricted-point-oracle-boundaries.md`
