# Source-derived global-hybrid composition and J/K admission

The CUDA KS scientific composition is shared by the descriptor admission,
native method preparation, and final-state validator in
`src/dft/global_hybrid_composition.hpp`. The matching Direct/DF Fock identity
checks live in `src/dft/global_hybrid_fock.hpp`. A functional's Libxc point program
does not define a separate Coulomb/exchange implementation:

- **Scientific identity**: The pinned Libxc hybrid registration supplies
  B3LYP's exact-exchange fraction (1/5) and its four semilocal components.
  `tools/generate_native_semilocal_families.py` materializes the entire
  semilocal component graph and exact-exchange fraction from the SHA-checked
  generated Libxc C-owner metadata, without duplicate numeric entries in the
  native manifest; missing or incompatible source records fail closed. M06-2X and MN15 derive their split semilocal registrations and
  fractions (27/50, 11/25) through
  `tools/generate_xc_split_hybrid_registry.py`. The native PBE envelope also
  represents PBE0's MethodIR-selected 1/4 exact exchange; this is an explicit
  qualified composition, not a claim that every Libxc hybrid is executable.
- **Execution identity**: Prepared Fock J and K must both be full-range
  Direct/Exact or both be density-fitted. A mixed provider, changed coefficient,
  changed spin normalization, changed semilocal scale, range/nonlocal source, or
  stale final-state token cannot pass global-hybrid validation.
- **Backend capability**: Libxc provides XC *physics*, not supported CUDA
  kernels, RI response derivatives, resources, or the public E+F contract.
  Curated CUDA KS capabilities stay in
  `manifests/native_semilocal_families.json`; registered split-hybrid DF
  support remains explicitly gated by
  `manifests/cuda_split_hybrids.json`. M06-2X has its proposed CUDA DF
  admission; MN15 remains Direct-only.

The same final-state proof is consumed by prepared CUDA, Python snapshots,
and complete forces; neither a descriptor method label nor an XC family name
substitutes Direct derivatives for fitted Hamiltonian derivatives.

This source contract applies uniformly to the hybrid families it describes.
GPU4PySCF comparisons at 3/12 atoms and later 48/96 atoms are still required
before claiming new M06-2X DF force numerical qualification. No performance
number or larger-domain support is inferred from source-only review.
