"""Compatibility alias for the compiler-owned CC lambda equations module."""

import sys

from generativeqc_compiler.cc import lambda_equations as _canonical

sys.modules[__name__] = _canonical
