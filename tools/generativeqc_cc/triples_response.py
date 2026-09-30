"""Compatibility alias for the compiler-owned CC triples response module."""

import sys

from generativeqc_compiler.cc import triples_response as _canonical

sys.modules[__name__] = _canonical
