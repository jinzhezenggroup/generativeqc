"""Explicit molecular VV10 density domain; raw fixed-grid kernels stay strict.

GPU4PySCF 1.8.1 uses rho >= 1e-10 in both SCF and force pair domains.
Keep the molecular energy, potential and response on that same density domain;
CPU PySCF's older 1e-8 cutoff is not the GPU default. This is a quadrature
policy, not a density floor or a change to the raw VV10 kernel. Derivatives
remain local to a fixed active set. Native retains every nonzero signed weight;
GPU4PySCF additionally drops weights with absolute value <= 1e-14.
"""

from fractions import Fraction

MOLECULAR_VV10_DENSITY_POLICY = "vv10-molecular-rho-ge-1e-10-v2"
MOLECULAR_VV10_DENSITY_THRESHOLD = Fraction(1, 10000000000)
