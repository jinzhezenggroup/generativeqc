"""Compatibility alias for the installed RKS Hessian owner."""

import sys

from generativeqc import rks_hessian as _implementation

sys.modules[__name__] = _implementation
