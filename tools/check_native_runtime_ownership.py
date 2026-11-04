#!/usr/bin/env python3
"""Validate source anchors in the Python/native runtime call-path inventory.

The historical baseline and subsequent delivery snapshot have different source
revisions. Neither is a numerical qualification or a native parity checker.
Capability admission and exact-head execution evidence remain owned by the
existing method manifest and production-path ledger.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "manifests/maintenance/python_native_runtime_ownership.json"
DELIVERY = "manifests/maintenance/python_native_runtime_delivery.json"
CLASSIFICATIONS = {
    "wrapper",
    "compiler-ad",
    "compatibility",
    "reference-oracle",
    "scientific-runtime",
}
AXES = {
    "backends": {"cpu", "cuda"},
    "representations": {"cartesian", "spherical", "native-xtb"},
    "spins": {"restricted", "unrestricted"},
}
DELIVERY_STATES = {
    "native-dispatch",
    "partial-cutover",
    "intentional-python",
    "compatibility",
}


def _anchor(root: Path, value: str) -> Path:
    """Require repository-local files, not mutable external evidence links."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
        raise ValueError(f"missing or nonlocal source anchor: {value}")
    return root / path


def _python_symbol(path: Path, symbol: str) -> None:
    """Bind dotted symbols to real declarations, rather than text in comments."""
    body = ast.parse(path.read_text(encoding="utf-8")).body
    for component in symbol.split("."):
        node = next(
            (
                node
                for node in body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == component
            ),
            None,
        )
        if node is None:
            raise ValueError(f"missing Python symbol: {path}:{symbol}")
        body = node.body


def validate(inventory: dict, root: Path = ROOT) -> None:
    """Check matrix coverage fields without inferring any qualified endpoint."""
    if inventory.get("schema") != "generativeqc.python-native-runtime-ownership.v1":
        raise ValueError("unsupported runtime ownership inventory schema")
    if not re.fullmatch(r"[0-9a-f]{40}", inventory.get("source_sha", "")):
        raise ValueError("baseline needs an exact source SHA")
    if inventory.get("coverage") != "baseline-call-paths-not-qualification":
        raise ValueError("source inventory must not claim numerical qualification")
    if inventory.get("required_slices") != [f"R{index}" for index in range(7)]:
        raise ValueError("runtime cutover must retain the complete R0-R6 scope")
    catalog = json.loads(
        _anchor(root, inventory["catalog"]).read_text(encoding="utf-8")
    )
    families = {method["family"] for method in catalog["methods"]}
    methods = {method["name"] for method in catalog["methods"]}
    identities: set[str] = set()
    classified: set[str] = set()
    covered_families: set[str] = set()
    if not inventory.get("rows"):
        raise ValueError("empty runtime inventory")
    for row in inventory["rows"]:
        identity = row["id"]
        if not identity or identity in identities:
            raise ValueError("runtime inventory IDs must be unique and nonempty")
        identities.add(identity)
        if row.get("qualification") != "source-inventory-only":
            raise ValueError(
                f"{identity}: wrapper/source coverage is not native parity"
            )
        for axis, choices in AXES.items():
            values = row[axis]
            if (
                not values
                or len(values) != len(set(values))
                or not set(values) <= choices
            ):
                raise ValueError(f"{identity}: invalid {axis} axis")
        if not row["method_families"] or not set(row["method_families"]) <= families:
            raise ValueError(f"{identity}: unknown catalog method family")
        covered_families.update(row["method_families"])
        for axis in (
            "properties",
            "approximations",
            "method_selectors",
            "tests",
            "trackers",
        ):
            if not row.get(axis) or not all(
                isinstance(value, str) and value for value in row[axis]
            ):
                raise ValueError(f"{identity}: missing {axis} axis/evidence")
        for selector in row["method_selectors"]:
            if selector.startswith("catalog-family:"):
                if selector.removeprefix("catalog-family:") not in families:
                    raise ValueError(f"{identity}: unknown method-family selector")
            elif selector not in methods and not selector.startswith("method-ir:"):
                raise ValueError(f"{identity}: unknown method selector")
        for key in ("native_scientific_sources", "native_callers", "tests"):
            for value in row[key]:
                _anchor(root, value)
        for site in row["python_sites"]:
            if site["classification"] not in CLASSIFICATIONS:
                raise ValueError(f"{identity}: unclassified Python call site")
            classified.add(site["classification"])
            path = _anchor(root, site["path"])
            if path.suffix != ".py" or not site["symbol"]:
                raise ValueError(f"{identity}: Python sites need precise symbols")
            _python_symbol(path, site["symbol"])
            if (
                site["classification"] == "scientific-runtime"
                and not row["remaining_python_runtime"]
            ):
                raise ValueError(
                    f"{identity}: Python runtime incorrectly marked retired"
                )
        for site in row.get("retired_python_sites", []):
            historical_path = Path(site["path"])
            if (
                historical_path.is_absolute()
                or ".." in historical_path.parts
                or historical_path.suffix != ".py"
            ):
                raise ValueError(f"{identity}: retired Python path must be local")
            if not re.fullmatch(
                r"[0-9a-f]{40}", site.get("last_present_source_sha", "")
            ):
                raise ValueError(f"{identity}: retirement needs an exact source SHA")
            if not site.get("symbols") or not all(
                isinstance(symbol, str) and symbol for symbol in site["symbols"]
            ):
                raise ValueError(f"{identity}: retirement needs historical symbols")
            if not isinstance(site.get("reason"), str) or not site["reason"].strip():
                raise ValueError(f"{identity}: retirement needs a reason")
            _anchor(root, site["replacement"])
    if classified != CLASSIFICATIONS:
        raise ValueError(
            "wrapper, compiler, compatibility, oracle and runtime must stay separate"
        )
    if covered_families != families:
        raise ValueError("baseline omits a registered method family")


def validate_delivery(delivery: dict, inventory: dict, root: Path = ROOT) -> None:
    """Keep all baseline domains visible without treating source review as science.

    Source revisions identify audits, not production-body fingerprints. Anchors
    are file-level navigation/evidence pointers; this check deliberately does
    not require particular function names, AST shapes, or a match to HEAD.
    Listed behavior tests are not execution receipts or capability admission.
    """
    if delivery.get("schema") != "generativeqc.python-native-runtime-delivery.v1":
        raise ValueError("unsupported runtime delivery schema")
    if delivery.get("baseline") != MANIFEST:
        raise ValueError("delivery must reference the historical baseline")
    if not re.fullmatch(r"[0-9a-f]{40}", delivery.get("source_sha", "")):
        raise ValueError("delivery needs an exact audit source SHA")
    audited_on = delivery.get("audited_on", "")
    try:
        if date.fromisoformat(audited_on).isoformat() != audited_on:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("delivery needs an ISO audit date") from error
    if delivery.get("coverage") != "source-delivery-not-qualification":
        raise ValueError("delivery must not claim numerical qualification")
    if delivery.get("required_slices") != inventory["required_slices"]:
        raise ValueError("delivery must retain the complete R0-R6 scope")
    baseline_ids = {row["id"] for row in inventory["rows"]}
    rows = delivery.get("rows", [])
    identities = [row["id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("delivery row IDs must be unique")
    if set(identities) != baseline_ids:
        raise ValueError("delivery must account for every baseline row exactly once")
    for row in rows:
        identity = row["id"]
        if row.get("state") not in DELIVERY_STATES:
            raise ValueError(f"{identity}: unsupported delivery state")
        slices = row.get("slices", [])
        if not slices or not set(slices) <= set(delivery["required_slices"]):
            raise ValueError(f"{identity}: missing or unknown required slice")
        roles = row.get("python_roles", [])
        if not roles or not set(roles) <= CLASSIFICATIONS:
            raise ValueError(f"{identity}: unclassified Python role")
        if row["state"] == "intentional-python" and (
            not set(roles) <= {"compiler-ad", "reference-oracle"}
            or row.get("retired_python_runtime")
        ):
            raise ValueError(f"{identity}: retain independent compiler/oracle work")
        for field in (
            "delivered_scope",
            "retired_python_runtime",
            "remaining_python_runtime",
            "remaining_acceptance",
            "source_anchors",
            "tests",
        ):
            values = row.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"{identity}: invalid {field}")
        for field in ("delivered_scope", "source_anchors", "tests"):
            if not row[field]:
                raise ValueError(f"{identity}: missing {field}")
        if row["state"] == "partial-cutover" and not row["remaining_acceptance"]:
            raise ValueError(f"{identity}: partial cutover needs remaining acceptance")
        if "scientific-runtime" in roles:
            if row["state"] != "partial-cutover" or not row["remaining_python_runtime"]:
                raise ValueError(f"{identity}: Python science is not retired")
        elif row["remaining_python_runtime"]:
            raise ValueError(
                f"{identity}: remaining Python science needs classification"
            )
        for field in ("source_anchors", "tests"):
            for anchor in row[field]:
                _anchor(root, anchor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        inventory = json.loads((args.root / MANIFEST).read_text(encoding="utf-8"))
        validate(inventory, args.root)
        delivery = json.loads((args.root / DELIVERY).read_text(encoding="utf-8"))
        validate_delivery(delivery, inventory, args.root)
    except (ValueError, KeyError, OSError) as error:
        print(f"native runtime ownership inventory failed: {error}")
        return 1
    print(
        f"Runtime ownership source anchors checked: {len(inventory['rows'])} rows; no parity claim"
    )
    print(
        f"Delivery snapshot checked: {len(delivery['rows'])} rows at "
        f"{delivery['source_sha']}; source audit, not current-head qualification"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
