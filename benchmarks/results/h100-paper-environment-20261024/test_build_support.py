"""CPU engineering checks; no compiler or scientific endpoint is invoked."""

from __future__ import annotations

import os
import subprocess
import unittest

from build_support import local_workers_only, stop_owned_process_group


class BuildSupportTests(unittest.TestCase):
    def test_actual_disabled_json_and_null_accepted(self) -> None:
        for raw in [
            '{"Disabled":"disabled"}',
            '{"Disabled":null}',
            '{"Disabled":"no remote scheduler configured"}',
        ]:
            self.assertTrue(local_workers_only(raw))

    def test_enabled_ambiguous_missing_malformed_refused(self) -> None:
        for raw in [
            '{"Enabled":{}}',
            '{"Disabled":null,"Enabled":{}}',
            "{}",
            "[]",
            "null",
            "true",
            "broken",
            '{"Disabled":{}}',
        ]:
            self.assertFalse(local_workers_only(raw))

    @unittest.skipUnless(os.name == "posix", "POSIX process-group ownership check")
    def test_owned_cleanup_preserves_unrelated_process(self) -> None:
        owned = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        unrelated = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        try:
            stop_owned_process_group(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            stop_owned_process_group(owned)
            stop_owned_process_group(unrelated)


if __name__ == "__main__":
    unittest.main()
