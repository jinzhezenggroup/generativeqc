"""Enable diagnostic events without changing the complete molecular protocol."""

import typing

from generativeqc import _stationary_cuda as runtime

from benchmarks.readme_pbe0 import main

original = runtime.complete_rks_cuda_gradient_diagnostic


def profiled(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Request per-tile event fences only for this separate intrusive run."""
    kwargs["profile_device"] = True
    return original(*args, **kwargs)


runtime.complete_rks_cuda_gradient_diagnostic = profiled
main()
