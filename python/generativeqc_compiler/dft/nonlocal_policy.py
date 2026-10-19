"""Explicit molecular VV10 density domain; raw fixed-grid kernels stay strict.

GPU4PySCF 1.8.1 uses rho >= 1e-10 for VV10 pair-domain admission in
SCF and gradients. The same threshold does not equate grid or pair semantics.
Source: https://github.com/pyscf/gpu4pyscf/blob/v1.8.1/gpu4pyscf/dft/numint.py
"""

from fractions import Fraction

MOLECULAR_VV10_DENSITY_POLICY = "vv10-molecular-rho-ge-1e-10-v2"
MOLECULAR_VV10_DENSITY_THRESHOLD = Fraction(1, 10000000000)
