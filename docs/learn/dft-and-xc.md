# DFT and exchange-correlation functionals

Kohn-Sham density-functional theory (DFT) describes the electronic problem through the electron density plus an exchange-correlation (XC) approximation. The density is built from the occupied Kohn-Sham orbitals,

$$
n(\mathbf r)
=
\sum_i f_i |\phi_i(\mathbf r)|^2,
$$

with occupation numbers $f_i$.

A common form of the Kohn-Sham energy is

$$
E[n]
=
T_s[n]
+
\int v_\mathrm{ext}(\mathbf r)n(\mathbf r)\,d\mathbf r
+
E_\mathrm{H}[n]
+
E_\mathrm{xc}[n]
+
E_\mathrm{nn},
$$

where

$$
E_\mathrm{H}[n]
=
\frac{1}{2}
\iint
\frac{n(\mathbf r)n(\mathbf r')}
{|\mathbf r-\mathbf r'|}
\,d\mathbf r\,d\mathbf r'.
$$

Variation of the energy gives the ordinary Kohn-Sham equations,

$$
\left[
-\frac{1}{2}\nabla^2
+
v_\mathrm{ext}(\mathbf r)
+
v_\mathrm{H}(\mathbf r)
+
v_\mathrm{xc}(\mathbf r)
\right]
\phi_i(\mathbf r)
=
\varepsilon_i\phi_i(\mathbf r),
$$

with the XC potential

$$
v_\mathrm{xc}(\mathbf r)
=
\frac{\delta E_\mathrm{xc}}{\delta n(\mathbf r)}.
$$

This displayed multiplicative-potential form is the ordinary KS form. When orbital-dependent ingredients such as exact exchange or kinetic-energy-density dependence are treated directly, generalized-KS implementations can instead contain nonlocal or differential XC operators.

The functional is part of the method identity. Broad semilocal families differ in which density features enter the XC energy. Schematically,

$$
E_\mathrm{xc}^\mathrm{LDA}
=
\int F(n)\,d\mathbf r,
$$

$$
E_\mathrm{xc}^\mathrm{GGA}
=
\int F(n,\nabla n)\,d\mathbf r,
$$

and a meta-GGA may additionally depend on quantities such as the kinetic-energy density

$$
\tau(\mathbf r)
=
\frac{1}{2}\sum_i f_i |\nabla\phi_i(\mathbf r)|^2.
$$

A simple global hybrid illustrates exact-exchange mixing,

$$
E_\mathrm{xc}^\mathrm{hybrid}
=
a E_x^\mathrm{HF}
+
(1-a)E_x^\mathrm{DFA}
+
E_c^\mathrm{DFA},
$$

although individual hybrids such as B3LYP have their own detailed mixing formulas.

Range-separated hybrids split the Coulomb operator, for example,

$$
\frac{1}{r_{12}}
=
\frac{\operatorname{erfc}(\omega r_{12})}{r_{12}}
+
\frac{\operatorname{erf}(\omega r_{12})}{r_{12}},
$$

and assign method-specific short- and long-range exchange coefficients.

LDA, PBE, r2SCAN, PBE0, B3LYP, and range-separated hybrids therefore make different approximations; sharing the Kohn-Sham SCF structure does not make them the same scientific method. Composite methods may also add dispersion or other corrections,

$$
E_\mathrm{method}
=
E_\mathrm{electronic}
+
E_\mathrm{dispersion}
+
E_\mathrm{other\ corrections},
$$

with the required terms belonging to the named method identity.

Use the generated [public method table](../public_methods.md) for current GenerativeQC method identities instead of copied tutorial lists.

Next: [energies and derivatives](energy-force-hessian.md).
