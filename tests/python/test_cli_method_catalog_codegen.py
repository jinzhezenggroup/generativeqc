"""Validate that CLI generated rows reproduce the canonical MethodIR primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from generate_cli_method_catalog import render, rows


class NativeMethodirProjectionTests(unittest.TestCase):
    def test_automatic_bulk_selectors_preserve_the_public_resolver(self) -> None:
        from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks

        table = {row.name: row for row in rows()}
        for selector, spin in (
            ("libxc:gga_x_apbe", "unpolarized"),
            ("libxc-uks:gga_x_apbe", "polarized"),
        ):
            row = table[selector]
            resolved = resolve_bulk_ks("GGA_X_APBE", spin=spin)
            self.assertEqual(row.identity, resolved.method.identity)
            self.assertEqual(row.components, (("GGA_X_APBE", 1.0),))
            self.assertTrue(row.cpu)
            self.assertFalse(row.cuda)
            self.assertEqual(row.exchange, ())

    def test_semantic_projection_and_backend_gates(self) -> None:
        table = {row.name: row for row in rows()}
        self.assertGreater(len(table), 10)
        pbe0 = table["pbe0-rks"]
        self.assertEqual(pbe0.source, "PBE0")
        self.assertEqual(
            dict(pbe0.components),
            {"GGA_X_PBE": 0.75, "GGA_C_PBE": 1.0},
        )
        self.assertEqual(pbe0.exchange, ((1, 0.25, 0.0),))
        self.assertEqual(len(pbe0.identity), 64)
        self.assertTrue(pbe0.cpu)
        self.assertTrue(pbe0.cuda)
        self.assertEqual(table["pbe0-uks"].spin, 2)
        self.assertNotEqual(pbe0.identity, table["pbe0-uks"].identity)

        b3lyp = table["b3lyp-rks"]
        self.assertEqual(len(b3lyp.components), 4)
        self.assertEqual(b3lyp.exchange, ((1, 0.2, 0.0),))
        self.assertTrue(b3lyp.cpu)
        self.assertTrue(b3lyp.cuda)

        for name in ("m06-2x-rks", "mn15-uks"):
            with self.subTest(name=name):
                self.assertFalse(table[name].cpu)
                self.assertTrue(table[name].cuda)

        for name in (
            "b3lyp3-rks",
            "b3lyp3-uks",
            "mb3lyp-rc04-rks",
            "mb3lyp-rc04-uks",
        ):
            with self.subTest(nonserializable=name):
                row = table[name]
                self.assertFalse(row.cpu)
                self.assertFalse(row.cuda)
                self.assertEqual(row.identity, "")
                self.assertIn("automatic bulk components cannot mix", row.reason)

        nonlocal_row = table["wb97m-v-rks"]
        self.assertTrue(nonlocal_row.cpu)
        self.assertTrue(nonlocal_row.cuda)
        self.assertEqual(nonlocal_row.range_omega, 0.3)
        self.assertEqual(nonlocal_row.exchange, ((2, 0.15, 0.3), (3, 1.0, 0.3)))
        self.assertEqual(nonlocal_row.nonlocal_correlation, (1, 6.0, 0.01, 1.0))
        for name in ("pbe-d3(bj)-rks", "pbe0-d3(0)-uks", "b3lyp-d4(bj-eeq-atm)-rks"):
            self.assertTrue(table[name].cpu)
            self.assertNotEqual(table[name].dispersion[0], 0)
        self.assertFalse(table["r2scan-3c-rks"].cuda)
        self.assertTrue(table["r2scan-3c-rks"].cpu)
        self.assertTrue(table["r2scan-3c-rks"].bound_basis)
        self.assertIn("gCP", table["r2scan-3c-rks"].reason)

    def test_render_is_deterministic_and_complete(self) -> None:
        catalog = rows()
        self.assertEqual(
            tuple(sorted(item.name for item in catalog)),
            tuple(item.name for item in catalog),
        )
        generated = render(catalog)
        self.assertEqual(generated, render(catalog))
        self.assertIn('{"pbe0-rks", "PBE0"', generated)
        self.assertIn('{"b3lyp-uks", "B3LYP"', generated)
        self.assertIn("no qualified native semilocal lowerer", generated)
        self.assertIn("constexpr const Method* find_method", generated)


if __name__ == "__main__":
    unittest.main()
