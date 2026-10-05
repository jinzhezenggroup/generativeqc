"""Data-only local AO/grid domains shared by collocation and its consumers."""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.common.resources import checked_bytes


@dataclass(frozen=True)
class AoGridBlockLayout:
    """Local columns with one existing AO map and explicit jet provenance.

    Evaluated jet order does not qualify a sparse map's discovery order.
    Unknown map capability stays unknown. This descriptor owns no coordinates,
    device memory, scientific equation, or method policy. The method-layer
    bridge makes gather/projection/scatter explicit in generic TensorIR.
    """

    nao: int
    nactive: int
    npoint: int
    derivative_order: int
    basis_identity: str
    indexed: bool
    map_derivative_order: int | None = None
    point_start: int = 0
    basis_generation: int | None = None
    geometry_generation: int | None = None

    def __post_init__(self) -> None:
        for name in ("nao", "nactive", "npoint", "point_start"):
            checked_bytes(getattr(self, name), name)
        if not self.nao or self.nactive > self.nao:
            raise ValueError("active AO domain must lie within the global basis")
        if type(self.indexed) is not bool or (
            not self.indexed and self.nactive != self.nao
        ):
            raise ValueError("dense AO layout must use the full identity domain")
        if (
            type(self.derivative_order) is not int
            or not 0 <= self.derivative_order <= 3
        ):
            raise ValueError("AO derivative order must lie in [0,3]")
        if self.map_derivative_order is not None and (
            type(self.map_derivative_order) is not int
            or not 0 <= self.map_derivative_order <= 3
        ):
            raise ValueError(
                "map derivative capability must lie in [0,3] or be unknown"
            )
        if not isinstance(self.basis_identity, str) or not self.basis_identity:
            raise ValueError("AO layout requires an explicit basis identity")
        for name in ("basis_generation", "geometry_generation"):
            if getattr(self, name) is not None:
                checked_bytes(getattr(self, name), name)
        checked_bytes(self.point_start + self.npoint, "point interval end")
        checked_bytes(self.ao_jet_values * 8, "local AO jet bytes")
        checked_bytes(2 * self.nactive * self.nactive * 8, "local density bytes")

    @property
    def jet_count(self) -> int:
        return (1, 4, 10, 20)[self.derivative_order]

    @property
    def ao_jet_values(self) -> int:
        return self.jet_count * self.npoint * self.nactive

    def require_derivative_order(self, order: int) -> None:
        """Reject unproven map reuse rather than infer capability from shape."""
        if type(order) is not int or not 0 <= order <= 3:
            raise ValueError("required AO derivative order must lie in [0,3]")
        if order > self.derivative_order:
            raise ValueError("AO layout does not contain the requested derivative jets")
        if self.indexed and (
            self.map_derivative_order is None or order > self.map_derivative_order
        ):
            raise ValueError("indexed AO map lacks the required derivative capability")

    def logical_work(self, jets: int) -> dict[str, int]:
        """Logical two-spin schedule work, not capacity or measured DRAM traffic."""
        if type(jets) is not int or jets not in (1, 4) or jets > self.jet_count:
            raise ValueError("density projection requires one or four available jets")
        return {
            "ao_map_bytes": self.nactive * 8 if self.indexed else 0,
            "ao_jet_values": self.ao_jet_values,
            "density_gather_values": 2 * self.nactive**2 if self.indexed else 0,
            "projection_fma_pairs": 2 * jets * self.npoint * self.nactive**2,
            "local_potential_values": 2 * self.nactive**2,
        }
