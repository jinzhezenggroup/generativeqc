# Decision: Admit the bounded I28 saddle protocol in Python

Status: implemented
Date: 2026-11-20

The [bounded CLI protocol](2026-11-20-bound-iodine-def2-saddle-cli-steps.md)
addresses occupied-root loss at larger steps, not a Python-specific numerical
defect. The isolated Python candidate allowed only the frozen I28 fixture; it
passed its separate 108-point independent curvature/mode and occupied gates.
Both candidate Python and exact-source installed CLI had identical terminal
geometry, energy, gradient and raw Hessian. The one unscaled imaginary mode
was `−256.185940558 cm⁻¹`, with four accepted attempts and final independent
occupied overlap `0.9999999999999735`.

Use the same restricted protocol in the public Python entry point: pinned
resolved scalar I28 def2 operator on each ECP center, spherical canonical
B3LYP-D3(BJ) RKS, accepted-frame warm starts, no preliminary guess or cold
Fock damping, initial/maximum radius at most `0.015/0.03 Bohr`, and minimum
occupied overlap at least `0.99`. Cs46 retains its distinct damping protocol.
No lab-only geometry, fixture hash, or broad "any ECP with 28 core electrons"
allowlist belongs in production; the guard compares the actual resolved
operators. I28 and Cs46 mixtures remain outside the saddle domain.

Original terminal seals are `installed-full-gate.sha256` under
`installed-i28-python-candidate/i28/` (SHA-256 `f206075b4ea41b6f076d97f6350fe27aa35e1406f3c3c217522713e9b090395b`)
and `source-i28-cli-admission/i28/` (SHA-256 `e49ad6c57809e0b035b5a9316bf53719f5a56fd1eee5046a6695c5d8`),
at `/data/jzzeng/qualify-2844-2848-20261120/`. They certify those exact
installed endpoints, not an unrelated rebuild or arbitrary starting XYZ.
