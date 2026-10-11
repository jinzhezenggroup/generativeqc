# Decision: reuse the prepared native force owner for CPU Direct PBE/PBE0

Status: implemented
Date: 2026-10-12

## Problem

The CPU DF PBE/PBE0 owner already assembled a complete token-qualified force,
but its Direct integral-source branch returned `NOT_IMPLEMENTED`. Python still
scheduled the admitted Direct sources even though their stationary TensorIR AD
weights and packaged CPU primitive derivatives already existed. Enabling eager
coordinate-by-AO-rank-four ERI derivatives would exchange an ownership gap for
an unacceptable retained derivative tensor.

## Decision

Keep the existing `KsPreparedCalculation` final-state authentication, complete
XC/AO/moving-Becke/nuclear assembly, force sign, batch lifecycle and transactional
publication. Add only a compact CPU Direct integral-source consumer:

- Generate portable weights by specializing the existing
  `StationaryGradientPlan.integral_block(...).weights` inline lowering, with no
  independently transcribed D/W, Coulomb or exchange equations. The CUDA emitter's
  default dimension spelling remains unchanged.
- Generate native bindings from the existing CPU first-derivative dispatch table
  and shard inventory. Restore every compiler-canonicalized center and axis slot
  before scattering to physical atoms, including repeated atoms and spherical AO
  expansions. Do not infer method coefficients in the integral runtime.
- Stream ordered shell-local weights; each primitive derivative is evaluated
  once for the union of nonzero J/K channels. Keep independent source blocks and
  each channel's accumulation order. Packaged and reference primitive arithmetic
  need numerical agreement, not bit-identical primitive values. Default reference
  single-channel and fused reference accumulation remain bit-identical.
- Admit native Direct forces only with packaged AOT enabled, all-electron s/p/d
  restricted PBE/PBE0, exact full-range J/K and the existing public work ceilings.
  Preserve conservative independent J/K work charges for admission despite fused
  execution. Unsupported contexts retain their previous separately qualified
  Python path; native calls never compile or substitute DF/reference science.
- Budget the three matrix weight buffers, retained coordinate sources, shell
  weights, returned channel arrays and offsets. Sample actual vector capacities
  with the simultaneously retained prepared owner/final frame and XC buffers.
  These are explicit numeric ownership samples, not a complete allocator/RSS
  census: recurrence/eigensolver scratch, metadata and opaque libraries remain
  excluded. R2 resource-authority completion is not implied.

The mathematical compiler remains Python/offline. Runtime allocation, scheduling,
token verification and complete force publication remain native. The public
Python router requires no new method whitelist: its existing per-prepared-item
capability query admits this source-complete owner automatically.

## Rejected alternatives

1. **Retain eager molecular ERI derivative tensors.** The prepared CPU Exact
   value owner deliberately has derivative order zero. Materializing
   `3*Natom*nbf^4` derivatives is unnecessary for stationary contractions.
2. **Use the compatibility Jet recurrence as the production fast path.** An
   initial implementation passed molecular/partial-source numerical checks, but
   a water diagnostic spent about 2.5 seconds per integral-force source call.
   Existing packaged generated derivatives avoid rebuilding dynamic recurrence
   scratch. The Jet path stays an independent numerical oracle/legacy consumer,
   not a silent production retry.
3. **Copy named PBE0 force factors into C++.** This would create another scientific
   equation owner. Generated AD supplies the factors and AO density pairings.
4. **Advertise all PBE-family forces.** UKS, RSH, ECP, meta-GGA, nonlocal/corrected
   models, larger angular domains and missing AOT storage are not authenticated
   by a family name. Neither the global method manifest nor an integral source
   alone advertises a complete force.

## Evidence and reproduction

- `test_dft_integral_sources.cpp`: Direct and DF PBE/PBE0, reconverged central
  differences, cold/warm/moved batches, failed-neighbor recovery, nonconvergence,
  stale token and exact/below-limit source budgets with unchanged failed outputs.
- `test_mp2_contract.cpp`: reference/fused/packaged angular shell derivatives,
  Cartesian and spherical d shells, repeated physical centers, independent dense
  and displaced-value oracles, zero channels and transactional invalid weights.
- `test_stationary_integral_gradient.cpp`: one-atom channel capacity accounting
  and shape-only near-limit primitive/grid/Becke work admission.
- Python gates compare public native forces to the retained Python driver and
  independent PySCF moving-grid analytical forces for asymmetric water and
  Cartesian/spherical synthetic s/p/d systems. Effect tests forbid invoking the
  Python scientific force driver on admitted paths. Generated portable weights
  match the independent TensorIR executor, and generation rejects runtime/oracle
  imports. PBE/PBE0 CUDA weight emission remains byte-identical to parent
  `f498134312e7d64e4562f81cd26dbefaa812b632` in both spin modes.
- Paired complete endpoint reproduction lives in ignored
  `.artifacts/2227/direct_force_endpoint.py`: construction, preparation and three
  cold/warm/genuinely moved water/STO-3G E+F calls followed by destruction. Compare
  alternating native and forced legacy-Python routes using one shared library,
  equal grid/convergence/precision and equal SCF iteration/Fock-build counts.
  Clean timings exclude diagnostic traces. Diagnostic journals report actual
  ERI shell submissions/primitive evaluations, not inferred AO-rank powers.
- For that fixture each force submits 625 ordered shell quartets and 194481
  primitive derivatives for either PBE or PBE0. Sharing J/K avoids a second
  primitive pass for PBE0. Source/weight capacity is 2400/3144 bytes respectively;
  complete endpoint samples additionally include the retained native/SCF/XC
  buffers and are explicitly not complete allocation peaks. Preserve measured
  ratios, library SHA and sample receipts in the exact-head PR evidence rather
  than generalizing this fixture's speed to arbitrary molecules.
- Installed C++ SDK and CLI Direct PBE/PBE0 force smoke tests run under
  `env -i PATH=/no-runtime-tools`; exec traces must contain only `env` and the
  native executable. They do not rely on a Python import or runtime compiler.

## Consequences and revisit conditions

The ordered quartet schedule is bounded and source-complete, but is not an
eightfold shell-orbit optimization. Consider symmetry folding or primitive
batching only with independent derivative normalization/scattering gates,
actual work counters and larger complete endpoints. Keep the explicit work
ceilings until their replacement has native authority and full endpoint evidence.

This is one R1/R6 slice of #2227, not completion of its R0–R6 tracker. CUDA forces,
response/HVP controllers, full native resource authority and the other runtime
policy/resolution/correction owners remain separate obligations. No GPU claim is
made: the local scheduler currently exposes only maintenance/drained `node3`.
