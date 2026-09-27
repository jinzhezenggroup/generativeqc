"""Public discovery/resolution for evidence-backed Libxc semilocal functionals."""

from __future__ import annotations

from vibeqc_compiler.method import BulkKsResolution, resolve_public_bulk_ks
from vibeqc_compiler.xc.public_inventory import (
    available_public_functionals,
    public_evidence,
)


def available_libxc_functionals() -> tuple[str, ...]:
    """Return installed CPU-energy Libxc registrations with retained public evidence."""

    return available_public_functionals()


def resolve_libxc_functional(
    name: str, *, spin: str = "unpolarized"
) -> BulkKsResolution:
    """Resolve one retained public Libxc registration into canonical MethodIR/KS.

    This is a public composition/discovery API.  It does not invent native
    Calculator admission: the returned resolution remains evidence-bound and
    can be consumed only by execution surfaces that support generic bulk point
    programs.
    """

    if spin not in ("unpolarized", "polarized"):
        raise ValueError("Libxc spin must be unpolarized or polarized")
    evidence = public_evidence(name)
    return resolve_public_bulk_ks(
        name,
        spin=spin,
        backend="cpu",
        evidence=evidence,
        identifier=f"LIBXC:{name.upper()}",
    )


__all__ = ["available_libxc_functionals", "resolve_libxc_functional"]
