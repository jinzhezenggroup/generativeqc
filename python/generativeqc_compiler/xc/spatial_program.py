"""Libxc point programs for the shared resident spatial E/V contraction.

The existing point lowerers own mathematical expressions and Libxc work domains.
This adapter binds their Cartesian coefficients to the grid's bounded point
panel. It grants no molecular-method or exact-exchange capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.cuda import compile_cuda
from generativeqc_compiler.dft.xc_kernel import GridXcConsumer, GridXcKernel

from ._generated_split_hybrids import SPLIT_HYBRID_COMPONENTS, SPLIT_HYBRIDS
from .automatic_point import emit_automatic_point
from .contracts import DerivativeRequest, DiscreteEnergyContract
from .spec import COMPONENTS, FunctionalSpec, UnsupportedXC, functional


def _semilocal_spec(value: str | FunctionalSpec, spin: str | None) -> FunctionalSpec:
    """Resolve explicit XC inputs; named hybrids contribute only their XC part.

    The spatial E/V API never assembles exact exchange. A full method descriptor
    with exchange-operator metadata is rejected rather than silently truncated.
    """
    if isinstance(value, FunctionalSpec):
        if spin is not None and spin != value.spin:
            raise ValueError("spatial XC spin disagrees with FunctionalSpec")
        spec = value
    elif isinstance(value, str):
        spin = "polarized" if spin is None else spin
        if value in SPLIT_HYBRIDS:
            spec = FunctionalSpec(
                value + "_SEMILOCAL",
                tuple(
                    (name, Fraction(c))
                    for name, c in SPLIT_HYBRIDS[value]["components"]
                ),
                spin=spin,
            )
        elif value in COMPONENTS:
            spec = FunctionalSpec(value, ((value, Fraction(1)),), spin=spin)
        else:
            spec = functional(value, spin=spin)
    else:
        raise TypeError("spatial XC requires a registered name or FunctionalSpec")
    if spec.exact_exchange or spec.long_range_exchange or spec.range_omega:
        raise UnsupportedXC(
            "spatial Libxc programs require semilocal inputs without exchange operators "
            "or range-parameter overrides"
        )
    return spec


@dataclass(frozen=True)
class SpatialPointProgram:
    """Source-only point contract, independent of compiler/device availability."""

    spec: FunctionalSpec
    kernel: GridXcKernel
    point_contracts: tuple[dict[str, Any], ...]

    @property
    def metadata(self) -> dict[str, Any]:
        """Bind work semantics, scalar outputs, coefficients and emitted bytes."""
        return {
            "schema": "generativeqc.spatial-libxc-point.v1",
            "functional": self.spec.to_payload(),
            "points": self.point_contracts,
            "consumer": self.kernel.consumer.identity,
            "selector": self.kernel.consumer.evaluator - 2,
            "source": canonical_hash(self.kernel.source),
            "output": "semilocal energy and spin AO potentials; no exact exchange",
        }


def emit_libxc_spatial(
    value: str | FunctionalSpec, *, spin: str | None = None, shared_group: Any = None
) -> SpatialPointProgram:
    """Adapt existing Libxc work-domain lowerers without a functional whitelist.

    Ordinary components are admitted by their projected rho/sigma/tau ABI and
    existing work-policy lowerer. Split hybrids retain their exact registered
    component pair. Each component keeps its own thresholds and original-density
    energy weighting; derivatives are never taken through worker clipping.
    """
    spec = _semilocal_spec(value, spin)
    # Validate the registered composition before generating any component body.
    payload = spec.to_payload()
    active = tuple((name, c) for name, c in spec.components if c)
    ingredients = ("rho",)
    if "sigma" in spec.ingredients:
        ingredients += ("gradient",)
    if "tau" in spec.ingredients:
        ingredients += ("tau",)
    mask = (
        1 | (2 if "gradient" in ingredients else 0) | (8 if "tau" in ingredients else 0)
    )
    bodies, calls, contracts = [], [], []
    selector = 0
    if shared_group is not None:
        if (
            len(active) != 1
            or active[0][1] != 1
            or active[0][0] not in shared_group.names
        ):
            raise ValueError(
                "shared spatial worker requires a registered unit component"
            )
        selector = shared_group.names.index(active[0][0])
        rows = [p.point.parameters for p in shared_group.providers]
        if not rows[0] or any(
            p.point.parameters != p.restricted.parameters
            for p in shared_group.providers
        ):
            raise ValueError("shared spatial worker requires a paired parameter ABI")
        bodies.append(
            f"__device__ const double spatial_parameters[{len(rows)}][{len(rows[0])}] = {{\n"
            + ",\n".join(
                "{" + ",".join(float(value).hex() for _, value in row) + "}"
                for row in rows
            )
            + "\n};"
        )
    split = any(name in SPLIT_HYBRID_COMPONENTS for name, _ in active)
    if split:
        entry = next(
            (
                (name, record)
                for name, record in SPLIT_HYBRIDS.items()
                if dict(active) == {key: Fraction(c) for key, c in record["components"]}
            ),
            None,
        )
        if entry is None:
            raise UnsupportedXC(
                "split-hybrid spatial XC requires its registered component pair"
            )
        # Reuse the qualified FP64 native point differential, including its
        # bounded integer-power lowering and equal-spin derivative preservation.
        from .geometry_cuda import _emit_composed_point

        name, record = entry
        source = _emit_composed_point(record["functional_code"], spec)
        bodies.append("namespace spatial_libxc_0 {\n" + source + "\n}\n")
        calls.append(
            (
                (
                    "restricted ? spatial_libxc_0::stationary_evaluate_point<true>(rho, gradient, kinetic)"
                    " : spatial_libxc_0::stationary_evaluate_point(rho, gradient, kinetic)"
                ),
                Fraction(1),
            )
        )
        contracts.append({"split_hybrid": name, "source": canonical_hash(source)})
    else:
        for index, (name, coefficient) in enumerate(active):
            # LDA RKS has its own total-density worker: folding P2 inputs would
            # change floors. Other RKS workers retain independent P2 partials.
            if shared_group is None:
                point = emit_automatic_point(name, backend="cuda")
                restricted = emit_automatic_point(
                    name,
                    backend="cuda",
                    spin="unpolarized" if point.ingredient_mask == 1 else "polarized",
                    equal_spin=point.ingredient_mask != 1,
                )
            else:
                # The existing package planner proves both exact worker bodies
                # and parameter ABIs. Never infer sharing from a family label.
                provider = shared_group.providers[0]
                point, restricted = provider.point, provider.restricted
            parameters = ", spatial_parameters[selector]" if shared_group else ""
            namespace = f"spatial_libxc_{index}"
            bodies.append(
                f"namespace {namespace} {{\nusing DevicePointValue = GridXcPoint;\n"
                + point.source
                + "\n"
                + restricted.source
                + "\n}\n"
            )
            calls.append(
                (
                    (
                        f"restricted ? {namespace}::{restricted.function_name}(rho, gradient, kinetic{parameters})"
                        f" : {namespace}::{point.function_name}(rho, gradient, kinetic{parameters})"
                    ),
                    coefficient,
                )
            )
            contracts.append(
                {
                    "coefficient": str(coefficient),
                    "polarized": point.mathematical_contract,
                    "restricted": restricted.mathematical_contract,
                    "polarized_source": point.source_identity,
                    "restricted_source": restricted.source_identity,
                }
            )
    lines = [
        f"constexpr unsigned grid_custom_xc_mask = {mask};",
        *bodies,
        "__device__ inline GridXcPoint grid_custom_xc_point(int selector, bool restricted, const double* features, I npoint, I point) {",
        "  GridXcPoint out;",
        f"  if (selector < 0 || selector >= {len(shared_group.names) if shared_group else 1}) {{ out.valid = false; return out; }}",
        "  const double rho[2]{features[point], features[5*npoint + point]};",
        "  double gradient[2][3]{}, kinetic[2]{};",
        # Split workers assume physical inputs; reject before their fmax floors
        # could otherwise turn invalid values into finite scientific outputs.
        "  if (!isfinite(rho[0]) || !isfinite(rho[1]) || !isfinite(rho[0] + rho[1]) ||",
        "      rho[0] < 0.0 || rho[1] < 0.0) { out.valid = false; return out; }",
        "  if (restricted && rho[0] != rho[1]) { out.valid = false; return out; }",
    ]
    if "gradient" in ingredients:
        lines += [
            "  for (int spin = 0; spin < 2; ++spin)",
            "    for (int axis = 0; axis < 3; ++axis)",
            "      gradient[spin][axis] = features[(5*spin + axis + 1)*npoint + point];",
            "  for (int spin = 0; spin < 2; ++spin) {",
            "    double norm = 0.0;",
            "    for (double value : gradient[spin]) norm += value * value;",
            "    if (!isfinite(norm)) { out.valid = false; return out; }",
            "  }",
            "  for (int axis = 0; axis < 3; ++axis)",
            "    if (restricted && gradient[0][axis] != gradient[1][axis]) { out.valid = false; return out; }",
        ]
    if "tau" in ingredients:
        lines += [
            "  kinetic[0] = features[4*npoint + point]; kinetic[1] = features[9*npoint + point];",
            "  if (!isfinite(kinetic[0]) || !isfinite(kinetic[1]) ||",
            "      kinetic[0] < 0.0 || kinetic[1] < 0.0) { out.valid = false; return out; }",
            "  if (restricted && kinetic[0] != kinetic[1]) { out.valid = false; return out; }",
        ]
    for index, (call, coefficient) in enumerate(calls):
        factor = float(coefficient).hex()
        lines += [
            f"  const auto part{index} = {call};",
            f"  out.valid = out.valid && part{index}.valid;",
            f"  out.energy += {factor} * part{index}.energy;",
            "  for (int spin = 0; spin < 2; ++spin) {",
            f"    out.rho[spin] += {factor} * part{index}.rho[spin];",
            f"    out.kinetic[spin] += {factor} * part{index}.kinetic[spin];",
            "    for (int axis = 0; axis < 3; ++axis)",
            f"      out.gradient[spin][axis] += {factor} * part{index}.gradient[spin][axis];",
            "  }",
        ]
    lines += [
        "  out.valid = out.valid && isfinite(out.energy);",
        "  for (int spin = 0; spin < 2; ++spin) {",
        "    out.valid = out.valid && isfinite(out.rho[spin]) && isfinite(out.kinetic[spin]);",
        "    for (double value : out.gradient[spin]) out.valid = out.valid && isfinite(value);",
        "  }",
        "  return out;",
        "}",
        "",
    ]
    source = "\n".join(lines)
    if shared_group is not None:
        contracts = [
            {
                "name": provider.point.name,
                "polarized": provider.point.mathematical_contract,
                "restricted": provider.restricted.mathematical_contract,
                "parameters": provider.point.parameters,
            }
            for provider in shared_group.providers
        ]
    identity = canonical_hash(
        {
            "schema": "generativeqc.spatial-libxc-work.v1",
            "functional": payload
            if shared_group is None
            else {"package": shared_group.names},
            "points": contracts,
            "source": source,
        }
    )
    source = f'constexpr const char* grid_custom_xc_identity = "{identity}";\n' + source
    kernel = GridXcKernel(source, GridXcConsumer(2 + selector, ingredients, identity))
    return SpatialPointProgram(spec, kernel, tuple(contracts))


@lru_cache(maxsize=8)
def _spatial_groups(names: tuple[str, ...]) -> dict[str, Any]:
    """Reuse exact native parameter-worker discovery for this catalog snapshot."""
    from .parameter_cuda_package import parameter_worker_groups

    return {
        name: group for group in parameter_worker_groups(names) for name in group.names
    }


def shared_libxc_spatial(
    value: str | FunctionalSpec, *, spin: str | None = None
) -> SpatialPointProgram:
    """Bind a registration to its compatible immutable parameter package.

    Distinct functionals retain distinct metadata and selectors. Package source
    and its read-only coefficient table are common; no per-owner mutable CUDA
    symbol is used. Unmatched/composed graphs keep their specialized executable.
    """
    from .automatic_semilocal import automatic_registry_entries

    spec = _semilocal_spec(value, spin)
    active = tuple((name, c) for name, c in spec.components if c)
    group = None
    if (
        len(active) == 1
        and active[0][1] == 1
        and active[0][0] not in SPLIT_HYBRID_COMPONENTS
    ):
        names = tuple(entry.name for entry in automatic_registry_entries())
        group = _spatial_groups(names).get(active[0][0])
    return emit_libxc_spatial(spec, shared_group=group)


@dataclass(frozen=True, init=False)
class SpatialContractionProgram:
    """Compiled CUDA E/V program for any supported Libxc point worker.

    Pass ``artifact`` to PreparedSpatialGrid and this program to
    PreparedXCContractions. The resident path shares point caching, AO assembly,
    resource bounds and density/orbital leases with interior programs. This
    device-only contract rejects host schedules or mismatched grid artifacts.
    """

    point: SpatialPointProgram
    artifact: Any
    contract: DiscreteEnergyContract

    def __init__(
        self,
        value: str | FunctionalSpec,
        *,
        compiler: Any,
        cache: Any,
        spin: str | None = None,
        share_workers: bool = True,
        ao_radial_reuse: bool = True,
    ) -> None:
        if type(share_workers) is not bool or type(ao_radial_reuse) is not bool:
            raise ValueError("spatial XC compilation choices must be boolean")
        point = (shared_libxc_spatial if share_workers else emit_libxc_spatial)(
            value, spin=spin
        )
        artifact = compile_cuda(
            compiler, cache, xc_kernel=point.kernel, ao_radial_reuse=ao_radial_reuse
        )
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(
            self,
            "contract",
            DiscreteEnergyContract(point.spec, DerivativeRequest("potential")),
        )

    @property
    def spec(self) -> FunctionalSpec:
        """Return the exact semilocal expression; method exchange stays external."""
        return self.point.spec

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose the point/domain identity to prepared resource and reuse checks."""
        return self.point.metadata

    @property
    def consumer(self) -> GridXcConsumer:
        """Return the exact generated evaluator bound into this artifact."""
        return self.point.kernel.consumer
