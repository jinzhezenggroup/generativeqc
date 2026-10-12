"""Offline, data-driven Libxc Maple imports into the existing scalar Graph.

The bulk catalog is a compiler-facing mathematical inventory, not public SCF
admission. Its domain is the ordinary positive-density, positive-gradient
interior. No Libxc library, Maple executable, reference oracle or GPU is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from . import libxc_maple
from .libxc_maple import MapleImportError, MapleModule, import_maple_file

BULK_SEMANTICS = "libxc-bulk-interior/v1"
SOURCE_ASSET = "upstream/libxc-fulltree/7.0.0"
CATALOG_PATH = Path(__file__).with_name("libxc_bulk_catalog.json")
_INCLUDE = re.compile(r'^\s*\$include\s+"([^"]+)"\s*$', re.MULTILINE)
_SUPPORT = (
    "src/util.h",
    "src/util.c",
    "src/functionals.c",
    "src/xc.h",
    "maple/util.mpl",
)


def canonicalize_spin_density_coordinates(
    graph: Graph,
    roots: tuple[Expr, ...],
    rho_a: Expr,
    rho_b: Expr,
) -> tuple[Expr, ...]:
    """Express boundary predicates in exact physical input coordinates.

    Apply this before differentiation and when reconstructing source subgraphs
    for matching. Both consumers must see the same inclusive spin-screen nodes;
    a floating-point inverse-coordinate round trip is not a physical density.
    """
    density = rho_a + rho_b
    # Restricted imports use an exact zero, not a floating-point cancellation.
    zeta = (
        graph.constant(0)
        if rho_a.identifier == rho_b.identifier
        else (rho_a - rho_b) / density
    )
    rs_factor = graph.approximate_constant((3.0 / (4.0 * math.pi)) ** (1.0 / 3.0))
    rs = rs_factor * density.pow(-1.0 / 3.0)
    reconstructed_density = (rs_factor / rs).pow(3.0)
    roots = graph.replace_subexpressions(
        roots,
        {
            (1 + zeta) * reconstructed_density / 2: rho_a,
            (1 - zeta) * reconstructed_density / 2: rho_b,
        },
    )
    return _canonicalize_kinetic_ratio_predicates(graph, roots, rho_a, rho_b)


def _canonicalize_kinetic_ratio_predicates(
    graph: Graph,
    roots: tuple[Expr, ...],
    rho_a: Expr,
    rho_b: Expr,
) -> tuple[Expr, ...]:
    """Compare tau_W/tau to one without cancelling fractional density powers.

    The imported ``xt**2 / (8*t_total(z, ts_a, ts_b))`` is exactly
    ``sigma_total / (8*rho_total*tau_total)`` on positive work coordinates.
    Its normalized round trip can exceed one by an ulp at a physical rank-one
    density, selecting a constant branch with the wrong inward PSD derivative.
    Rewrite only the predicate before differentiation. Retain the original
    branch values and derivatives, including their raw Libxc work-boundary
    coefficients at empty/floored spin coordinates.
    """
    names = {node.payload for node in graph.nodes if node.operation == "variable"}
    if {"sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b"} <= names:
        sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b = (
            graph.variable(name)
            for name in ("sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b")
        )
    elif {"sigma", "tau"} <= names:
        sigma_aa = sigma_ab = sigma_bb = graph.variable("sigma") / 4
        tau_a = tau_b = graph.variable("tau") / 2
    else:
        return roots
    density = rho_a + rho_b
    zeta = (
        graph.constant(0)
        if rho_a.identifier == rho_b.identifier
        else (rho_a - rho_b) / density
    )
    sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    total_t = tau_a * rho_a.pow(-5 / 3) * ((1 + zeta) / 2).pow(5 / 3)
    total_t += tau_b * rho_b.pow(-5 / 3) * ((1 - zeta) / 2).pow(5 / 3)
    normalized_ratio = graph.multiply_many((sigma, density.pow(-4 / 3).pow(2))) / (
        8 * total_t
    )
    replacements = {}
    reachable = set(graph.topological_order(roots))
    for index in sorted(reachable):
        node = graph.nodes[index]
        if (
            node.operation == "select_le"
            and node.arguments[0] == normalized_ratio.identifier
            and graph.is_constant(Expr(graph, node.arguments[1]), 1)
        ):
            # Multiply density and tau first: 8*density can overflow even
            # when their product is representable. Outside the normal-product
            # range retain the existing comparison rather than introducing
            # an underflow-induced boundary change. Production work floors
            # keep this product many orders above the binary64 normal minimum.
            product = density * (tau_a + tau_b)
            physical = graph.select_le(
                sigma,
                8 * product,
                Expr(graph, node.arguments[2]),
                Expr(graph, node.arguments[3]),
            )
            guarded = graph.select_le(
                product,
                float.fromhex("0x1.0p-1022"),
                Expr(graph, index),
                physical,
            )
            replacements[Expr(graph, index)] = guarded
            # The original comparison remains inside its underflow fallback.
            # Retain an already-normalized wrapper verbatim on repeated calls,
            # while still normalizing other uses of the same original node.
            if guarded.identifier in reachable:
                replacements[guarded] = guarded
    return graph.replace_subexpressions(roots, replacements)


def read_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Read the generated catalog and reject incompatible importer semantics."""
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if (
        catalog.get("schema") != "generativeqc.libxc-bulk"
        or catalog.get("schema_version") != 1
        or catalog.get("bulk_semantics") != BULK_SEMANTICS
        or catalog.get("importer_semantics") != libxc_maple.IMPORTER_SEMANTICS
    ):
        raise MapleImportError("bulk catalog needs regeneration for this importer")
    records = catalog.get("registrations")
    if not isinstance(records, list) or not records:
        raise MapleImportError("empty or malformed bulk functional inventory")
    names = [item["name"] for item in records]
    if len(set(names)) != len(names):
        raise MapleImportError("duplicate bulk functional identifiers")
    return catalog


def available_functionals() -> tuple[str, ...]:
    """Return only registrations that actually lowered to a Graph."""
    return tuple(
        item["name"]
        for item in read_catalog()["registrations"]
        if item["graph_status"] == "imported"
    )


def _source_bytes(root: Path, name: str, files: dict[str, str]) -> bytes:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise MapleImportError("bulk source path escapes the pinned source root")
    if name not in files:
        raise MapleImportError(f"source is absent from the bulk manifest: {name}")
    candidate = root / name
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        raw = candidate.read_bytes()
    except (OSError, ValueError) as error:
        raise MapleImportError(
            f"bulk source is absent or outside its root: {name}"
        ) from error
    if hashlib.sha256(raw).hexdigest() != files[name]:
        raise MapleImportError(f"bulk source SHA-256 mismatch: {name}")
    return raw


def import_record(
    record: dict[str, Any], files: dict[str, str], source_root: Path
) -> MapleModule:
    """Import a verified source closure without changing shared frontend semantics.

    Upstream Maple uses a shared basename include search path. A short-lived,
    collision-checked flat projection preserves every source byte and lets the
    existing importer retain include ordering/redefinition/fail-closed behavior.
    Only the selected closure is copied; its immutable module outlives the files.
    """
    by_name: dict[str, str] = {}
    for path in files:
        if path.endswith(".mpl"):
            name = PurePosixPath(path).name
            if name in by_name:
                raise MapleImportError(f"ambiguous Maple include basename: {name}")
            by_name[name] = path
    selected: dict[str, bytes] = {}

    def visit(path: str) -> None:
        if path in selected:
            return
        raw = _source_bytes(source_root, path, files)
        selected[path] = raw
        try:
            text = libxc_maple._strip_comments(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise MapleImportError(f"bulk Maple source is not UTF-8: {path}") from error
        for name in _INCLUDE.findall(text):
            if PurePosixPath(name).name != name or name not in by_name:
                raise MapleImportError(f"unknown or non-basename Maple include: {name}")
            visit(by_name[name])

    visit(record["entry"])
    support = (*_SUPPORT, record["owner"])
    for path in support:
        selected[path] = _source_bytes(source_root, path, files)
    basenames = [PurePosixPath(path).name for path in selected]
    if len(basenames) != len(set(basenames)):
        raise MapleImportError("ambiguous source basename in import closure")
    with tempfile.TemporaryDirectory(prefix="generativeqc-libxc-") as directory:
        flat = Path(directory)
        for path, raw in sorted(selected.items()):
            (flat / PurePosixPath(path).name).write_bytes(raw)
        return import_maple_file(
            flat,
            PurePosixPath(record["entry"]).name,
            bindings=record["bindings"],
            support_files=tuple(PurePosixPath(path).name for path in support),
            allow_duplicate_includes=True,
        )


@dataclass(frozen=True)
class BulkProgram:
    """One imported energy density with physical feature-coordinate derivatives."""

    name: str
    family: str
    spin: str
    features: tuple[str, ...]
    graph: Graph
    variables: tuple[Expr, ...]
    energy: Expr
    identity: str

    def roots(self, derivative_order: int = 1) -> tuple[Expr, ...]:
        """Return E, vxc, then upper-triangular fxc in feature order."""
        if type(derivative_order) is not int or derivative_order not in (0, 1, 2):
            raise ValueError("bulk derivative order must be 0, 1 or 2")
        result = [self.energy]
        if derivative_order:
            first = tuple(
                self.graph.differentiate(self.energy, var) for var in self.variables
            )
            result.extend(first)
            if derivative_order == 2:
                result.extend(
                    self.graph.differentiate(first[i], self.variables[j])
                    for i in range(len(self.variables))
                    for j in range(i, len(self.variables))
                )
        return tuple(result)

    def emit_source(self, derivative_order: int = 1, *, cuda: bool = False) -> str:
        """Emit a scalar point function through the existing C/CUDA emitters."""
        variables = {name: f"features[{i}]" for i, name in enumerate(self.features)}
        if cuda:
            from generativeqc_compiler.integral.cuda import CudaEmitter

            emitter = CudaEmitter(self.graph, variables)
        else:
            emitter = ScalarCEmitter(self.graph, variables)
        roots = self.roots(derivative_order)
        emitter.emit(roots)
        prefix = 'extern "C" __device__ ' if cuda else ""
        lines = [
            "/* Generated from pinned MPL-2.0 Libxc sources by GenerativeQC. */",
            "#include <math.h>",
            f"{prefix}void bulk_xc_point(const double *features, double *outputs) {{",
            *["  " + line for line in emitter.lines],
            *[
                f"  outputs[{i}] = {emitter.reference(root)};"
                for i, root in enumerate(roots)
            ],
            "}",
            "",
        ]
        return "\n".join(lines)


def build_record(
    record: dict[str, Any],
    files: dict[str, str],
    source_root: Path,
    *,
    spin: str = "polarized",
) -> BulkProgram:
    """Lower a bound registration into physical density/gradient/laplacian/tau IR."""
    if spin not in ("polarized", "unpolarized"):
        raise ValueError("bulk spin must be polarized or unpolarized")
    polarized = spin == "polarized"
    family = record["family"]
    if family not in ("lda", "gga", "mgga"):
        raise MapleImportError("bulk Graph construction requires an energy family")
    names = ["rho_a", "rho_b"] if polarized else ["rho"]
    if family != "lda":
        names += ["sigma_aa", "sigma_ab", "sigma_bb"] if polarized else ["sigma"]
    if family == "mgga":
        names += (
            ["lapl_a", "lapl_b", "tau_a", "tau_b"] if polarized else ["lapl", "tau"]
        )
    graph = Graph()
    variables = tuple(graph.variable(name) for name in names)
    rho_a, rho_b = variables[:2] if polarized else (variables[0] / 2,) * 2
    density = rho_a + rho_b
    zeta = (rho_a - rho_b) / density if polarized else graph.constant(0)
    rs = graph.approximate_constant((3 / (4 * math.pi)) ** (1 / 3)) * density.pow(
        -1 / 3
    )
    arguments = [rs, zeta]
    if family != "lda":
        sigma_aa, sigma_ab, sigma_bb = (
            variables[2:5] if polarized else (variables[1] / 4,) * 3
        )
        xt = (sigma_aa + 2 * sigma_ab + sigma_bb).pow(0.5) * density.pow(-4 / 3)
        xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4 / 3)
        xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4 / 3)
        arguments.extend((xt, xs_a, xs_b))
    if family == "mgga":
        lapl_a, lapl_b, tau_a, tau_b = (
            variables[5:]
            if polarized
            else (
                variables[2] / 2,
                variables[2] / 2,
                variables[3] / 2,
                variables[3] / 2,
            )
        )
        arguments.extend(
            (
                lapl_a * rho_a.pow(-5 / 3),
                lapl_b * rho_b.pow(-5 / 3),
                tau_a * rho_a.pow(-5 / 3),
                tau_b * rho_b.pow(-5 / 3),
            )
        )
    module = import_record(record, files, source_root)
    energy = density * module.call(graph, "f", *arguments)
    # Inverse rs/zeta coordinates are exactly the input spin densities. Their
    # FP64 round trip can land on either side of screen_dens at its threshold,
    # activating a nominally empty-spin exchange/kinetic branch. Substitute the
    # physical coordinates before differentiation, not a post-hoc vxc clamp.
    (energy,) = canonicalize_spin_density_coordinates(graph, (energy,), rho_a, rho_b)
    identity = canonical_hash(
        {
            "bulk_semantics": BULK_SEMANTICS,
            "source_provider": SOURCE_ASSET,
            "record": {
                k: v
                for k, v in record.items()
                if k not in ("graph_status", "graph_nodes")
            },
            "spin": spin,
            "features": names,
            "source_manifest_sha256": canonical_hash(files),
            "transitive_sha256": module.transitive_sha256,
            "adapter_sha256": file_hash(Path(__file__)),
            "importer_sha256": file_hash(Path(libxc_maple.__file__)),
        }
    )
    return BulkProgram(
        record["name"], family, spin, tuple(names), graph, variables, energy, identity
    )


def build_bulk_program(
    name: str,
    *,
    spin: str = "polarized",
    source_root: Path | None = None,
    catalog_path: Path = CATALOG_PATH,
) -> BulkProgram:
    """Build one named imported registration, never a handwritten fallback."""
    catalog = read_catalog(catalog_path)
    records = {item["name"]: item for item in catalog["registrations"]}
    if not isinstance(name, str) or name.upper() not in records:
        raise MapleImportError(f"unknown bulk Libxc registration: {name!r}")
    record = records[name.upper()]
    if record["graph_status"] != "imported":
        raise MapleImportError(
            f"bulk Libxc registration is blocked: {record['reason']}"
        )
    root = asset_path(SOURCE_ASSET) if source_root is None else Path(source_root)
    return build_record(record, catalog["source_files"], root, spin=spin)
