"""Explicit matched-domain variant of the pinned, independent PySCF VV10 oracle.

The default remains upstream CPU PySCF's 1e-8. The opt-in 1e-10 variant changes
only its density threshold, not its equations, grids, or SCF convergence gates.
No GenerativeQC numerical implementation or computed answer is used here.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def vv10_density_domain(cutoff: float) -> Iterator[None]:
    from pyscf import __version__
    from pyscf.dft import libxc, numint

    if cutoff not in (1e-8, 1e-10):
        raise ValueError("VV10 oracle cutoff must be explicitly 1e-8 or 1e-10")
    assert __version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"
    original = numint._vv10nlc
    if cutoff == 1e-10:
        source = inspect.getsource(original)
        assert source.count("thresh=1e-8") == 1, "upstream VV10 oracle changed"
        namespace = dict(numint.__dict__)
        # Execute only the pinned installed oracle function, never supplied code.
        exec(  # noqa: S102
            compile(
                source.replace("thresh=1e-8", "thresh=1e-10"),
                "<PySCF VV10 explicit rho >= 1e-10>",
                "exec",
            ),
            namespace,
        )
        numint._vv10nlc = namespace["_vv10nlc"]
    try:
        yield
    finally:
        numint._vv10nlc = original
