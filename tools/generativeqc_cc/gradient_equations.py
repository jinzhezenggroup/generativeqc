"""Compatibility alias for the compiler-owned CC gradient equations module."""

import sys

from generativeqc_compiler.cc import gradient_equations as _canonical

sys.modules[__name__] = _canonical
