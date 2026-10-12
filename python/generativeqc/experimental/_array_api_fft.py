"""CPU Array API fft extension; no implicit TensorIR lowering."""

from __future__ import annotations

import typing


def _dispatch(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Resolve the public dispatcher lazily to avoid a namespace import cycle."""
    from .array_api import _dispatch as dispatch

    return dispatch(name, *args, **kwargs)


def fft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.fft, returning protocol-aware arrays."""
    return _dispatch("fft.fft", x, n=n, axis=axis, norm=norm)


def ifft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.ifft, returning protocol-aware arrays."""
    return _dispatch("fft.ifft", x, n=n, axis=axis, norm=norm)


def fftn(
    x: typing.Any,
    /,
    *,
    s: typing.Any = None,
    axes: typing.Any = None,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.fftn, returning protocol-aware arrays."""
    return _dispatch("fft.fftn", x, s=s, axes=axes, norm=norm)


def ifftn(
    x: typing.Any,
    /,
    *,
    s: typing.Any = None,
    axes: typing.Any = None,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.ifftn, returning protocol-aware arrays."""
    return _dispatch("fft.ifftn", x, s=s, axes=axes, norm=norm)


def rfft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.rfft, returning protocol-aware arrays."""
    return _dispatch("fft.rfft", x, n=n, axis=axis, norm=norm)


def irfft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.irfft, returning protocol-aware arrays."""
    return _dispatch("fft.irfft", x, n=n, axis=axis, norm=norm)


def rfftn(
    x: typing.Any,
    /,
    *,
    s: typing.Any = None,
    axes: typing.Any = None,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.rfftn, returning protocol-aware arrays."""
    return _dispatch("fft.rfftn", x, s=s, axes=axes, norm=norm)


def irfftn(
    x: typing.Any,
    /,
    *,
    s: typing.Any = None,
    axes: typing.Any = None,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.irfftn, returning protocol-aware arrays."""
    return _dispatch("fft.irfftn", x, s=s, axes=axes, norm=norm)


def hfft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.hfft, returning protocol-aware arrays."""
    return _dispatch("fft.hfft", x, n=n, axis=axis, norm=norm)


def ihfft(
    x: typing.Any,
    /,
    *,
    n: typing.Any = None,
    axis: typing.Any = -1,
    norm: typing.Any = "backward",
) -> typing.Any:
    """Execute standard CPU fft.ihfft, returning protocol-aware arrays."""
    return _dispatch("fft.ihfft", x, n=n, axis=axis, norm=norm)


def fftfreq(
    n: typing.Any,
    /,
    *,
    d: typing.Any = 1.0,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard CPU fft.fftfreq, returning protocol-aware arrays."""
    return _dispatch("fft.fftfreq", n, d=d, dtype=dtype, device=device)


def rfftfreq(
    n: typing.Any,
    /,
    *,
    d: typing.Any = 1.0,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard CPU fft.rfftfreq, returning protocol-aware arrays."""
    return _dispatch("fft.rfftfreq", n, d=d, dtype=dtype, device=device)


def fftshift(x: typing.Any, /, *, axes: typing.Any = None) -> typing.Any:
    """Execute standard CPU fft.fftshift, returning protocol-aware arrays."""
    return _dispatch("fft.fftshift", x, axes=axes)


def ifftshift(x: typing.Any, /, *, axes: typing.Any = None) -> typing.Any:
    """Execute standard CPU fft.ifftshift, returning protocol-aware arrays."""
    return _dispatch("fft.ifftshift", x, axes=axes)
