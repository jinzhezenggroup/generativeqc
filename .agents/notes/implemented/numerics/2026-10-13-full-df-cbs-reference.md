# Decision: preserve RI-JK through complete correlated E+F

Status: implemented
Date: 2026-10-13

## Problem

Correlation-only DF-CBS still spends most of its large-basis time in exact
RHF orbital/nuclear response. Replacing only the SCF initial density does not
remove those four-center traversals. A genuine RI-JK model needs its own
consistent Hessian actions and auxiliary-metric/center derivatives.

## Decision

Expose `density_fitting="full_cuda"` for MP2/DF-CCSD(T) and
`hf_reference="df"` for CBS, keeping existing selectors unchanged. Retain the
native SCF RI-JK owner through same-basis correlation companions and response.
The shared response maps are independent of how the linear potential is
implemented. Use RI-JK for every physical action, including the fresh residual.
Use symmetric polarization of the existing generated DF quadratic derivative
for the nuclear cross term. The same metric rank/cutoff defines value and
response; no derivative of an independently refitted or exact reference enters.

## Invariants

- CUDA scientific work remains native; PySCF is an independent test oracle.
- Signed response densities use linear dense J/K, not occupied-factor guesses.
- Full-DF resource/numerical refusal never switches the Hamiltonian.
- Shared source lifetimes are charged through CC, triples, Lambda and response.
- Existing exact-source recycled images are not compatible with RI-JK.
- Fitting bases are explicit: this first interface uses one auxiliary record
  for both JK and correlation in each orbital-basis group. MP2-RI does not
  automatically qualify HF fitting error. Separate JK/RI records are not yet
  supported, and the correlation factor build does not yet borrow SCF factors.

## Evidence

`tests/python/test_full_df_cbs_native.py`: six real-device tests pass on node4
through finite Slurm allocations (20.60 s). They cover H2, water DZ/TZ,
Cartesian truncated-metric water, DF-CCSD(T) with HF/MP2 companions, and complete
CBS composition. All energies agree with independent PySCF within `5e-9 Eh`
(`1e-8` for the composite), and force projections agree within `3e-6 Eh/Bohr`
at reconverged central differences of `1e-4` and `3e-5 Bohr`. Tests check
translation, CUDA execution, repeated E+F and energy-only parity.

Two failed qualification attempts exposed important boundaries, now fixed:

- TZ JKFIT contains auxiliary g. The HF derivative adapter must explicitly
  admit g and select six-term metadata for **both** orbital and auxiliary
  views, including the metadata bound and device expansion stride. It uses
  the existing generated derivative algebra. Four-class s/p/d/f shell
  metadata cannot silently drop g: automatic selection keeps the generic
  generated sink; explicit incompatible shell selection refuses.
- Cartesian water DZ has 131 auxiliaries but retained rank 130; its smallest
  normalized-metric eigenvalue ratio is `9.063188291382534e-11`. PySCF's
  default Cholesky keeps a different space. The independent oracle uses
  libcint integrals and NumPy spectral filtering of unit-self-overlap
  normalized Cartesian auxiliaries at the native `1e-10` relative cutoff.
  Displaced-energy checks qualify the changing retained/discarded projectors;
  loosening the tolerance to hide the former `1.99e-7 Eh` mismatch is not valid.

Actual QM7 ethane full CBS also passes independent checks for all five
constituents and the composite: maximum energy error `3.965e-12 Eh`; maximum
directional errors `1.681e-9` and `5.455e-9 Eh/Bohr` at the two steps.
This is a directional check, not an independent full-vector or CBS-limit
accuracy certificate. The legacy exact response/DF-MP2 GPU regression passes
160 tests in 83.57 s. H2 and TZ water pass Compute Sanitizer memcheck and
synccheck with zero errors. Host/API, compiler algebra, hooks and Sphinx gates
pass; raw qualification lives under `.artifacts/cbs-response/`.

Qualified source is `b400ac1fb9ba232d29e0069ea2c759ac5b3bc926` plus frozen patch
`023d0739ada6df92ceb8acd2da7af280a62afceed8a63a7fb91f12351d2fac16`;
Release CUDA 12.9.1 sm120 DSO SHA-256 is
`30fc778fe44fd89100f5159e174b34faf5fb61a5d598de8ae73b753d47e6a01c`.
Later documentation-only edits do not relabel this binary as current master.

## Resource and performance boundaries

Timing began only after scientific qualification. The node1 RTX 5090 campaign
uses cc-pV[D,T]Z with **JKFIT for both JK and correlation**, not the earlier
exact-reference/RI-fit Hamiltonian. The two campaigns must not be described
as identical-Hamiltonian A/B speedups. Same-basis reference and correlation
companions are reused; independent orbital-basis groups remain sequential.

The current owner conservatively retains a host+device allowance of twice
`min(reference_budget / 4, 4 GiB)` plus identities and source metadata through
CC/Lambda/response. This is reserved capacity, not measured allocation. The
24-GiB campaign completes C1–C6 (4.853, 12.119, 36.347, 99.169, 256.444,
754.981 s), but C7 refuses Lambda's total numerical budget after 606.662 s.
That duration is **not** a completed E+F time or a hardware out-of-memory
measurement. A distinct 32-GiB campaign retains the same frozen binary/model
and records its explicit allowance and all prior failures.

## Revisit when

Independent endpoint profiles justify a bilinear DF nuclear consumer or
borrowing SCF AO factors into the correlation source. Do not optimize those
paths by dropping metric response, changing the residual model, or relabeling
the earlier correlation-only measurements as full DF.
