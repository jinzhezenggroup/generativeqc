"""Compose a DFT producer's indexed domain into provider-neutral TensorIR."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor.indexed_layout import IndexedTensorLayout
from generativeqc_compiler.tensor.ir import Node, einsum, input_tensor, slice_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

if TYPE_CHECKING:
    from generativeqc_compiler.dft.indexed_layout import AoGridBlockLayout


@dataclass(frozen=True)
class AoGridBlockProgram:
    """Graph composition over the same layout; never a second sparse map owner.

    DFT owns collocation and its data-only capability contract. TensorIR owns
    independent runtime index axes. This bridge owns the density projection
    and its exact scatter adjoint without importing any public runtime.
    """

    layout: AoGridBlockLayout

    @cached_property
    def _indices(self) -> tuple[Index, ...]:
        block = self.layout
        name = "ao_" + block.basis_identity.encode().hex()
        global_ao = IndexSpace(name, "ao", block.nao)
        local_ao = (
            IndexSpace(name + "_local", "ao", block.nactive)
            if block.indexed
            else global_ao
        )
        return (
            Index("spin", IndexSpace("grid_spin", "spin", 2)),
            Index("mu", global_ao),
            Index("nu", global_ao),
            Index("mu_local", local_ao),
            Index("nu_local", local_ao),
            Index(
                "point",
                IndexSpace("grid_point", "batch", block.point_start + block.npoint),
                block.point_start,
                block.point_start + block.npoint,
            ),
            Index("jet", IndexSpace("ao_jet", "component", block.jet_count)),
        )

    @cached_property
    def ao_map(self) -> Node:
        return input_tensor(
            "ao_ids",
            TensorSpec(
                (self._indices[3],), dtype="int64", role="input", differentiable=False
            ),
        )

    @cached_property
    def density_layout(self) -> IndexedTensorLayout | None:
        if not self.layout.indexed:
            return None
        spin, mu, nu, local_mu, local_nu, _, _ = self._indices
        return IndexedTensorLayout(
            TensorSpec((spin, mu, nu), role="input", differentiable=True),
            ((1, self.ao_map, local_mu), (2, self.ao_map, local_nu)),
        )

    def density_program(self) -> Program:
        """Gather directly to D[I,I], never a local-by-global intermediate."""
        spin, mu, nu, _, _, _, _ = self._indices
        density = input_tensor(
            "density", TensorSpec((spin, mu, nu), role="input", differentiable=True)
        )
        local = self.density_layout.select(density) if self.layout.indexed else density
        return Program({"local_density": local})

    def projection_program(self, jets: int) -> Program:
        """Express the existing local density projection independently of provider."""
        block = self.layout
        if type(jets) is not int or jets not in (1, 4) or jets > block.jet_count:
            raise ValueError("density projection requires one or four available jets")
        spin, _, _, local_mu, local_nu, point, jet = self._indices
        density = self.density_program().outputs["local_density"]
        ao = input_tensor(
            "ao_jets",
            TensorSpec((jet, point, local_mu), role="input", differentiable=True),
        )
        ao = slice_tensor(ao, ((0, jets), (0, block.npoint), (0, block.nactive)))
        projected = einsum("jpm,smn->sjpn", ao, density)
        assert projected.spec.shape == (
            spin.extent,
            jets,
            point.extent,
            local_nu.extent,
        )
        return Program({"projected": projected})

    def scatter_program(self) -> Program:
        """Use exactly the density map for the local Vxc accumulation domain."""
        spin, _, _, local_mu, local_nu, _, _ = self._indices
        local = input_tensor(
            "local_potential",
            TensorSpec((spin, local_mu, local_nu), role="input", differentiable=True),
        )
        potential = (
            self.density_layout.scatter_add(local) if self.layout.indexed else local
        )
        return Program({"potential": potential})
