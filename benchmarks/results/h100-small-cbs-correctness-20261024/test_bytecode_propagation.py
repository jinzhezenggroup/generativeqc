"""One targeted flag only; never enumerate the environment or touch Rpub."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

import transport


class BytecodePropagation(unittest.TestCase):
    def setUp(self) -> None:
        original = os.environ.get("PYTHONDONTWRITEBYTECODE")
        flag = sys.dont_write_bytecode

        def restore() -> None:
            if original is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = original
            sys.dont_write_bytecode = flag

        self.addCleanup(restore)

    def test_readonly_flag_reaches_child_and_grandchild_without_B(self) -> None:
        transport.inherit_no_bytecode_writes()
        grandchild = (
            'import sys,json;print(json.dumps({"dont_write":sys.dont_write_bytecode}))'
        )
        child = (
            'import json,subprocess,sys; r=subprocess.run([sys.executable,"-c",'
            + repr(grandchild)
            + "],capture_output=True,text=True,check=True);"
            'print(json.dumps({"child":sys.dont_write_bytecode,"grandchild":json.loads(r.stdout)["dont_write"]}))'
        )
        result = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(json.loads(result.stdout), {"child": True, "grandchild": True})

    def test_only_required_control_is_set(self) -> None:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "0"
        sys.dont_write_bytecode = False
        transport.inherit_no_bytecode_writes()
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), "1")
        self.assertTrue(sys.dont_write_bytecode)


if __name__ == "__main__":
    unittest.main()
