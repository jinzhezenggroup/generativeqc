# Proposed: validate meta-GGA occupied projections before CUDA KS adoption

Status: proposed
Date: 2026-10-15

The existing CUDA XC path builds four work panels for tau-dependent
functionals: D*phi, D*d_x(phi), D*d_y(phi), and D*d_z(phi). The first panel
is also needed for rho and its Cartesian gradient. The last three can
mathematically be replaced **for density features only** by
B^T*d_k(phi), where D = B B^T includes exact occupations. The Vxc
matrix contraction still needs all four AO jets; merely setting
work_jets=3 is scientifically incorrect.

Existing code already supports occupied-factor CPU XC with generation
and density-witness checks (src/dft/xc.cpp) and a separately prepared
CUDA grid feature producer with factor panels (src/dft/cuda_grid.cu).
The ordinary CUDA KS XC producer does NOT consume that factor.

This PR adds a fixed-density, feature-only CUDA D-versus-B benchmark using
those existing producers. It uses synthetic PSD factors, representative
def2-SVP AO shapes at 3/48/96 atoms, sampled active-AO maps, independent
NumPy D*phi + 3 B^T*grad(phi) numerical comparisons, paired timing
samples, and actual map counts. There is **no molecular E+F speedup
claim, no new CUDA KS default, and no new hybrid CUDA kernel yet**.

For p points, a tile-active AOs, and r occupied orbitals:
- Four density panels: 4 p a^2 products.
- Hybrid density/tau: p (a^2 + 3 a r) products.
- Four occupied panels: 4 p a r products.

The hybrid count obeys 4 * hybrid = density + 3 * occupied.
Thus it is **never the cheapest route by product count**: when r < a,
four occupied panels are cheaper than hybrid; when r > a, four dense
panels are cheaper; at equality all three match. This motivates a
measured crossover between the existing *four D* and *four B* producers
rather than adding a dominated hybrid CUDA kernel.

With local-AO sparsity, a may be much smaller than full-basis nao.
Packing, occupancy, memory, potential construction, SCF work and
factor lifetime can change actual wall-time ordering even when the
arithmetic counts are ordered. No method-name specialization.

A subsequent native implementation requires a source generation/density
witness (not user-supplied or DIIS-mixed D), bounded device leases,
a fail-closed general-density fallback for stale/unsupported/response
states, retained four-jet Vxc, and source-matched full cold/warm/moved
M06-2X E+forces against independent 1e-8 Eh / 1e-7 Eh/Bohr references.
Only complete E+F tests may authorize a default performance claim.

Agent: ChatGPT
Model: GPT-6
