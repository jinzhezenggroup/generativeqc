# Decision: operator-matched long-range RI-K for molecular VV10 compositions

Status: implemented
Date: 2026-10-15

## Problem

The existing fitted WB97M-V composition fitted only full-range J/K. Its
long-range exchange correction still traversed exact four-center integrals.
Selecting DF therefore retained a potentially expensive Direct molecular owner.
Differentiating a Direct LR Hamiltonian after changing only SCF to fitted LR
would also produce inconsistent forces.

## Decision

For qualified molecular range-exchange/nonlocal graphs, prepare two independent
fitted owners. The primary uses 1/r; the correction uses erf(omega*r)/r in both
its three-center tensor and auxiliary metric. The physical exchange is the
short-range coefficient times full-range fitted K, plus the difference of
long-/short-range coefficients times fitted LR K. The two metrics have separate
rank selections with the same caller-supplied relative threshold.

The compiler's existing Gaussian moment polynomials also evaluate LR values
and center derivatives. Substituting t=sqrt(theta)*u, with
theta=omega^2/(rho+omega^2), rescales the polynomial geometry and radial
prefactor. The ratio omega/hypot(omega,sqrt(rho)) avoids squaring a large omega.
Exponents and omega are fixed under nuclear differentiation. All actual basis
factors retain their original exponents, normalization and public-AO pullbacks.
The value-only evaluation skips response contractions, rather than maintaining
another recurrence. CUDA's normalized-factor traversal accepts a policy object
so omega is immutable owner state, not device-global mutable state.

Response contracts the same fitted model, including metric pseudoinverse and
auxiliary-center derivatives. CUDA LR response uses the bounded generic
generated FP64 derivative consumer, not full-range-only shell specializations.
The existing five-source composer recombines fitted primary and fitted LR rows.
CPU does not instantiate its exact LR quartet consumer for this model.

`GENERATIVEQC_DF_RANGE_EXCHANGE=direct` retains the prior mixed Hamiltonian as
an explicit compatibility choice. The prepared execution plan freezes that
choice; a token-checked native proof determines Python's derivative route.
Missing or unrecognized native LR proofs fail closed instead of assuming Direct.
Unavailable LR coverage and failed resource admission never select Direct.
An explicit CUDA DF byte envelope is partitioned between both radial owners;
CPU retains its previous budget semantics. These response/resource contracts
still do not imply a complete process-memory bound.

## Rejected alternatives

- Reusing full-range whitened factors for LR changes the operator incorrectly.
- Keeping a hidden Direct LR fallback makes DF work and force semantics unclear.
- Calling PySCF to generate production LR factors introduces an oracle dependency.
- Calling the VV10 grid-pair evaluation "DF" confuses two separate algorithms.
  This change does not approximate or accelerate that nonlocal kernel.

## Invariants

- Omega, basis geometry, metric cutoff, backend and approximation bind every
  value/response owner and its live final-state token.
- Source recomposition applies physical spin/exchange coefficients once.
- Both metric and three-center derivatives use the same radial operator.
- Production CUDA factors and responses execute natively on CUDA, with existing
  provider-stream event ordering. CPU/PySCF is not a production CUDA fallback.
- Existing all-electron, precision, basis and complete-force admission gates
  remain in force. LR fitting currently requires orbital/auxiliary s/p/d/f.

## Evidence

- Raw contracted values and every independent-center derivative match libcint
  for metric and three-center signatures through f, including spherical
  projection, at omega 0.001, 0.3, 1.2 and 1000 (8e-11 absolute/relative gates).
- CPU RKS/UKS complete energies match independently reconstructed PySCF LR
  fitting within 2e-8 Eh; directional force checks use independently
  reconverged, displaced energies with a 3e-7 Eh/bohr gate.
- Water/def2-SVP Cartesian and spherical endpoints exercise the molecular
  auxiliary response beyond hydrogen-only cases. The independent LR oracle
  diagonalizes libcint's metric in normalized auxiliary coordinates with the
  declared relative cutoff. PySCF's default Cholesky otherwise keeps tiny
  Cartesian LR modes that the native model intentionally discards; comparing
  those different models is not a reason to loosen the energy gate.
- A spherical water/def2-TZVP endpoint also covers orbital f functions. Cold,
  warm and geometry-rebound calculations retain their fitted LR proof even if
  the process environment changes to the legacy selector after preparation;
  rebound energies and forces agree with a fresh full-RI calculation.
- Small auxiliary bases deliberately separate fitted LR from exact LR. Native
  approximation proofs and force checks protect that Hamiltonian boundary.
- Warm replay revokes old native approximation proofs. Explicit mixed-mode
  regressions remain independently tested.
- GPU tests must use a finite Slurm allocation and exclude maintenance nodes.
  Compile-only or CPU checks do not qualify real-device numerical behavior.
- Seven CUDA complete energy-plus-force endpoints passed on node2 (Slurm
  `gpu:pro6000`) under jobs 3044 and 3045, CUDA 12.9.86 and a pinned `g++-11` JIT host
  driver. They cover RKS/UKS, two auxiliary bases, Cartesian/spherical water,
  orbital f functions, cold/warm execution and geometry rebinding. The same
  independent energy and force gates above apply without relaxation.

### Validation build boundary

The numerical CUDA qualification uses Release DF kernels. To avoid spending
the qualification on unrelated, very expensive Direct device optimization,
the local validation build uses the supported standalone Direct ownership
(`GENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=OFF`), with `--Ofast-compile=max` on
only the unchanged `direct_jk_kernels.cu`, `direct_bounded_fallback.cu` and
`direct_force_class_domains.cu` files. No DF kernel uses that option or device
debug flags. This is numerical endpoint evidence, not release performance or
whole-process resource qualification.

Using `-G` to speed the old relocatable Direct compilation was ineffective:
device assembly remained expensive, and sccache 0.16.0 hit a preprocessing
`-E`/`-MD` incompatibility. Prefer the existing fast-compile mechanism for
iteration rather than changing production source to address this build cost.
Keep toolkit headers/libraries coherent. Pin a complete C++ host driver for
runtime CUDA shared-library compilation: this node's default `gcc` lacked
`cc1plus`, and selecting its C driver alone did not link the C++ runtime.

Reproduction and logs are retained as ignored artifacts under
`build/validation/`; the scheduler runner preserves Slurm's device visibility
and uses a finite ten-minute allocation. Verified compiler-cache launchers were
used throughout, and existing caches were not cleared.

## Revisit when

Subsequent source-matched endpoint measurements and newly discovered
practical-auxiliary force/grid-admission qualification limits are recorded in
`../performance/2026-10-15-range-df-endpoint-observations.md`. Their exploratory
timings do not broaden the small-auxiliary numerical acceptance above.

Qualify auxiliary g or a faster LR-specific CUDA response schedule; measure
complete cold/warm energy-plus-force endpoints before promoting that schedule.
Treat low-rank VV10 grid algorithms as a separate numerical approximation.
