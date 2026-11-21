# Retain the accepted fitted legs across CUDA Hessian directions

## Decision

The raw DF nuclear map now prepares the accepted symmetric fitted leg
`B = A M⁻¹` and, for each spin with exchange, its half-dressed panel `X_p = D B_p`.
These belong to the authenticated geometry and density lease, not to any
particular HVP. Keep them in the same explicitly bounded device arena as the
borrowed-source response maps. CPU consumers use the same scientific graph
with the fit inlined. No response cap, metric cutoff, or audit tolerance is
increased, and admission does not fall back to Direct or host numerics.

Exchange uses `A_p X_p` for Fock, `X_p D` for the raw cotangent, and
`tr(X_p X_q)` for the full-rank auxiliary Gram. The latter works for signed
densities without an occupied factor. Trace cyclicity makes the two moving
Gram legs transposes, so compute one contraction. At fixed density the
full-rank exchange Fock contribution `A D dB` moves the symmetric inverse
onto `A`, giving `B D (dA - B dM)`. AO reflection then lets it share a single
exchange contraction with `dA D B`. Coulomb must retain both product-rule
legs: AO reflection does not interchange the two external density pairs.
Spectral response keeps the actual `d(M⁺)` and its separate metric pullback.

## Work admission and checks

The emitted action work estimate for hybrid HF/cc-pVTZ/full JKFIT
(`44 AO × 95 auxiliary`, RKS) goes from 206,133,404 to 98,162,928 term visits.
For H₂O⁺/cc-pVDZ plus O(f)/full JKFIT (`31 AO × 113 auxiliary`, UKS), it goes
from 159,046,910 to 86,183,027. This fits the existing 100-million-term
full-rank per-action cap. The spectral branch can still fall outside the
bound at these shapes; no radial or spectral method change is implied.
These are generated schedule counts, not timing measurements.

The host-only native probe checks admission of both shapes and rejection of
an oversized full-rank hybrid schedule. Compiler tests compare the retained
schedule against expanded independent fitted ERIs and recontract all tangent
weights. The CPU WB97M-V fitting regression passes with the current
full-rank metric-direction contract, including reconverged RKS/UKS energies.

On RTX 5090 Slurm job `1751`, both HF/cc-pVTZ RKS models used the production
100-million-term cap and passed independent PySCF/Libxc reconverged-gradient
references. Maximum Hessian errors were `6.78e-10` (M06-2X) and `5.10e-10`
(MN15) Ha/Bohr², with translation, raw symmetry, and dense/HVP replay checks.
The dense native endpoint took `140.63–140.91 s` for six directions, with six
raw-DF value and contraction passes each and 144 induced Fock actions. HVP
took `23.76–23.81 s` with one pass of each kind and 24–26 induced Fock actions.
No numerical host fallback was used. These are single-run timings and do not
establish an endpoint speedup or qualify the public angular gate on their own.

Jobs `1751` and `1753` also passed the respective M06-2X and MN15 UKS
H₂O⁺/cc-pVDZ plus O(f) full-JKFIT references. The largest raw Hessian element
errors were `4.98e-9` and `3.73e-9` Ha/Bohr². The dense endpoints took
`133.51–133.80 s`, with nine DF passes of each kind and 292–298 induced
Fock actions. The HVP took `15.45–15.57 s`, with one pass of each kind and
34 induced Fock actions. Translation, symmetry, and HVP consistency also
passed without host numerical fallback. The separate raw-integral suite
passed all 49 cases with PySCF/libcint loaded on the scheduled node (`1752`).

## Public spherical f scope

Job `1755` replayed the MN15 HF/cc-pVTZ RKS and M06-2X H₂O⁺/O(f) UKS
full-JKFIT references through both the native CLI and Python Hessian/HVP.
The first Direct replay in that job exposed a separate bug: Direct f still
selected the incumbent SCF model despite having the linked AOT Hessian leaves.
Select the variational SCF identity for spherical f with either provider,
independently of requested properties. Keep the existing Cartesian SCF policy
and the separate Direct force inventory, which still stops at d.

On the corrected build (`1756`), spherical H₂ and H₂⁺ with synthetic s/f
bases passed both public frontend Hessian/HVP paths against independently
reconverged PySCF/Libxc gradient references for both models. The maximum
element error was `3.63e-10 Ha/Bohr²`; no numerical host fallback was used.
These exact Direct fixtures evaluate 444,156 primitive components per dense
Hessian and 74,026 per HVP. The independent gradients are the oracle; they
do not imply that the standalone generated Direct f force endpoint is qualified.
Job `1756` also passed all 15 selected frontend regressions, including both
spin conventions and both providers for the new f-shell cases, fitted PBE,
WB97M-V fitted long-range exchange, and r²SCAN+rVV10.

Slurm `5488` replayed the four independent Direct fixtures on RTX PRO 6000
and passed six selected spherical-f frontend regressions. Both GPUs are SM120;
this is a second device check, not cross-architecture qualification.

For the separate pure Coulomb specialization, see
[auxiliary charge response](2026-11-21-cuda-df-coulomb-charge-response.md).
