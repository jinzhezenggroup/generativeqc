"""Compatibility alias for the compiler-owned canonical MP2 equations."""

import sys

from generativeqc_compiler.mp2 import equations as _canonical

sys.modules[__name__] = _canonical
