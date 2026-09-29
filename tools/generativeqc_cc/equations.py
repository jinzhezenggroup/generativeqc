"""Compatibility alias for the compiler-owned RCCSD TensorIR equations."""

import sys

from generativeqc_compiler.cc import equations as _canonical

sys.modules[__name__] = _canonical
