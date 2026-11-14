# Decision: build-time CPU scalar Hessian leaves without endpoint promotion

Status: implemented (scalar package only; no complete native CPU owner)
Date: 2026-11-11

## Problem

The linked native Hessian package owns a CUDA WB97M-V endpoint. Its stream ABI
and CMake CUDA requirement do not supply a CPU B3LYP stationary/response owner.
The Python CPU composite owner already uses shared scalar AD, but calling that
owner from a native command would not satisfy a Python-free CPU endpoint.

## Decision

Add an explicitly selected, opt-in static CPU leaf target, independent of CUDA
and the public runtime library. The method-owned inventory combines existing
physical semilocal roots (`production=True`), compact coefficient pullbacks,
their directional response, mixed grid primitives and nuclear pair HVP roots.
The original Graph and `ScalarCEmitter` remain the sole scientific algebra and
lowering owners. A narrow scalar-client declaration permits that neutral
emitter dependency; it does not admit integral recurrence/schedule dependencies
into method policy or weaken the subsystem dependency matrix.

Each allocation-free FP64 function receives input-major and root-major panels
with independent contiguous lanes, explicit element counts and a lane count.
It refuses count/byte overflow and invalid nonempty pointers before reading.
Inactive Graph branches remain lazy. Nonfinite roots fail explicitly and
invalidate the complete output, including previously written roots/lanes.
Callers must not alias input/output storage. These are ABI checks, not a
substitute for a complete owner's chemical-domain admission.

The registry authenticates exact method/expression identities and reachable
input labels. Missing identities never trigger runtime compilation or another
method. CMake uses the existing generated-source dependency runner, including
imported Python depfiles, and strict host FP64 flags. Inventory edits reconfigure
the selected linked closure. The root source-identity inventory records the new
generator, inventory and CMake module as well as the existing compiler trees.

## Rejected alternatives

- Do not enable the native CLI capability after linking scalar arithmetic: no
  occupied/virtual response, complete integral/XC/grid assembly, SCF lifetime or
  native CPU stationary owner has been delivered by this package.
- Do not use Python callbacks, finite differences or an external optimizer as
  an alleged Python-free complete endpoint.
- Do not duplicate XC or nuclear derivative formulas in the C++ consumer.
- Do not impose the CUDA leaf package's toolchain/stream ABI on a CPU target.
- Do not run a toolchain at registry misses or quietly choose a reference engine.

## Evidence

`build/qualification/host-hessian-aot-20261111/unit-rerun.log` records an actual
zero exit: **27 passed in 22.71 seconds**. The tests compile all selected leaves
and their registry with a verified compiler cache, execute both spin layouts,
check overflow/count/null/zero-lane behavior and exact lookup, exercise an
input-free scalar and inactive invalid branch, and compare:

- physical B3LYP energy, every feature first/second derivative and inactive tau
  channels against an independent Libxc oracle at ordinary interior points;
- two minority-spin B88 controls against derivatives of the original primal at
  80-digit precision, without using Libxc as an assumed minority-node oracle;
- coefficient/response pullbacks against the elementary sigma chain rule;
- all four mixed grid leaves against independent 60-digit primal derivatives,
  including upstream mixed tangents; and nuclear HVPs against the Coulomb tensor.

An isolated CMake CXX-only consumer compiles/links/runs the target and removes
polarized entries after an inventory edit. A subprocess import guard runs the
real generator while refusing the public runtime, PySCF, CuPy and Torch.
The root CPU-only CMake target also compiled through verified sccache 0.16.0;
verbose commands and cache snapshots are retained. No frozen scientific-job
library was rebuilt. Shared cache statistics include concurrent work and must
not be interpreted as exclusively this target's hit counts.

The initial 19-pass/one-fail probe had an overbroad invalid-pointer assertion:
an input-free constant correctly accepts a null input. Its log is retained and
the assertion now uses a genuinely invalid output count. The first structure
audit exposed the undeclared neutral emitter dependency; it was not waived by
moving method policy into a lower subsystem. An initial formatter invocation
used a nonexistent virtualenv executable; the actual local Ruff then passes.

## Consequences and remaining work

The broader source-identity/compiler/AOT/B88/saddle regression run actually
exits zero: **173 passed, one explicitly opt-in skipped in 46.05 seconds**
(`build/qualification/host-hessian-aot-20261111/broad.log`). That skip is the
expensive UKS molecular saddle oracle, not an accepted TS result. Full
changed-file pre-commit and strict Sphinx (`-W --keep-going`) also exit zero.
The root CPU target's subsequent build retains unchanged generated arithmetic;
the original failed structure audit remains separately retained.

The final focused run also actually exits zero, **27 passed in 22.99 seconds**
(`unit-final.log`). It strengthens the byte-overflow test with matching element
counts and nonnull sentinel pointers, so unrelated count/null refusals cannot
mask a missing byte bound. The mixed-grid controls use nonparallel tangents,
preventing a constant ratio or purely radial norm direction from hiding terms.

The package is intentionally not installed, linked into the public core/CLI,
or reported as molecular capability. It reduces a build-time prerequisite only.
A complete native CPU stationary owner still needs bounded integral and grid
assembly, native response actions/solve, state/identity/lifetime and resource
contracts, independent full matrices/HVPs, then installed API/CLI qualification.
Requested Ni/I/Cs, mixed and 25–40-atom endpoint and TS/frequency gates remain
separate; none follows from these leaf tests or their execution time.

## Native integration boundaries checked during implementation

The apparent native response entry points are not an existing CPU B3LYP owner:
`generativeqc_ks_xc_response_create_v1` and
`generativeqc_ks_response_solve_v1` in `src/api/c_api_ks_snapshot.cpp` construct
their response consumers under `GENERATIVEQC_HAS_CUDA`. The host point-response
helpers in `src/dft/xc_point_response.hpp` accept an LDA/PBE selector and
exchange/correlation scales; their existence does not supply physical
B88/VWN-RPA/LYP second derivatives. Native B3LYP point energy/first-derivative
dispatch is likewise not a complete second-order response implementation.

The generic host GMRES and `response::spin_orbital_problem` already supply
method-neutral vector/orbital machinery. A future CPU stationary owner should
bind those to the physical, leased full-range hybrid Fock/XC action rather than
adding another response algebra. In particular,
`generativeqc_integral_ad_rhf_v1` in `src/api/c_api_integral_ad.cpp` accepts
explicit dense ERIs and all-electron Cartesian closed-shell RHF inputs; it is
not a spherical B3LYP/ECP integral-HVP provider and must not be relabeled as one.

## References

The subsequent [bounded native AO XC response consumer](2026-11-12-bounded-host-xc-response.md)
extends the optional target without promoting a complete endpoint. Its note
also identifies the retained, byte-identical scalar-only static archive; the
original scalar qualification receipts remain unchanged.

- Issues #2844–#2848; PR #3062.
- `docs/developer/dft_hessian.md`.
- `python/generativeqc_compiler/method/host_hessian_aot.py`.
- `src/dft/host_hessian_aot.hpp`.
- `tests/python/test_host_hessian_aot.py`.
- `tests/python/test_host_hessian_aot_cmake.py`.
