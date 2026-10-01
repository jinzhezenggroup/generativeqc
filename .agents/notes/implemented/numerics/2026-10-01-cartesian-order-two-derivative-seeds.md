# Decision: preserve derivative seeds in HF Cartesian source shortcuts

Status: implemented; independent native derivative gates passed
Date: 2026-10-01

## Problem

The compiler-owned Cartesian source dispatcher has a full-range closed-form
shortcut for order-two shell classes 2, 3 and 6. Its output is an
`Order2IntegralVector` of `double` values. Returning `static_cast<Scalar>` is
valid for a value consumer but silently makes every derivative zero when
`Scalar` is `Dual3`. HF's qualified explicit low-order force workers bypass
this value shortcut; a new general RSH derivative consumer must not assume
the shortcut also propagates geometry jets.

Pure s+f force fixtures do not reach these order-two classes. Generic s+p/d
fixtures that do not create a Cartesian through-f owner also miss this route.

## Decision

Restrict the existing shortcut to arithmetic value scalars and
`MixedPrecisionFloat`. Derivative scalars retain the existing compiler-owned
seeded Cartesian recurrence, with the same angular class, radial operator,
normalization, geometry and contraction order. Do not implement another
derivative algebra in native runtime code or cast a value vector into a dual.

The native regression creates an s+p or s+d system with a spectator f shell
whose density is exactly zero. Its fused Cartesian J/SR/LR force is compared
against displaced energies of an independent compact CPU s+p/d system.
The gate covers both AO representations and spins, two geometries and unequal
SR/LR exchange coefficients. The existing through-f displaced-energy and
matrix gates remain unchanged. Acceptance is 3e-8 Eh/Bohr for the independent
central differences and 3e-12 for public value matrices.

## Evidence and boundaries

Slurm 11964 passes the order-two seed gate, through-f full/SR/LR derivatives,
screened public matrices, both projection implementations, row boundary/work
checks and prepared one-electron CPU-gradient/zero-H2D checks. Logs and exact
binary/source identity are retained locally in
`.artifacts/omol25-hf-cartesian/`. This evidence qualifies the numerical fix,
not a complete OMol25 energy/force speedup or the 100-atom domain.

See the [Cartesian source candidate](../../proposed/2026-10-01-through-f-hf-cartesian-source-reuse.md)
for its independent resource, endpoint and performance qualification.
