"""Explicit molecular VV10 density domain; raw fixed-grid kernels stay strict.

GPU4PySCF 1.8.1 uses rho >= 1e-10 in both SCF and force pair domains.
Keep the molecular energy, potential and response on that same density domain;
CPU PySCF's older 1e-8 cutoff is not the GPU default. This is a quadrature
policy, not a density floor or a change to the raw VV10 kernel. Derivatives
remain local to a fixed active set. Points whose quadrature weight has absolute
value <= 1e-14 are also inactive in every consumer, as in GPU4PySCF's SCF and
force domains: they contribute below FP64 resolution of the energy yet carry
about a sixth of the quadratic pair work on large molecular grids.
"""

from fractions import Fraction

MOLECULAR_VV10_DENSITY_POLICY = "vv10-molecular-rho-ge-1e-10-absw-gt-1e-14-v3"
MOLECULAR_VV10_DENSITY_THRESHOLD = Fraction(1, 10000000000)
MOLECULAR_VV10_WEIGHT_THRESHOLD = Fraction(1, 100000000000000)
