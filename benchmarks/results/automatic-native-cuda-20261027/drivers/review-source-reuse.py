"""Audit an explicit source-closure transition before reusing independent FD data."""

import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

sys.path[:0] = ["python", "."]
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.method.stationary_cuda import STATIONARY_AOT_ASSETS

root = Path.cwd()
e = root / "evidence"
profiles = {}
for p in (e / "qualified-forces").glob("*-h4.json"):
    for row in json.loads(p.read_text())["rows"]:
        receipt = row["artifact"]
        profile = receipt["force_profile"]
        old = receipt["force_source_identity"]
        before = root / "evidence/reference-generated-force" / f"native_{profile}.cu"
        after = root / "build/native-cuda/generated/stationary_force" / before.name
        a, b = before.read_text(), after.read_text()
        new = re.findall(r'"([a-f0-9]{64})"', b)[-1]
        assert a.count(old) == 1 and a.replace(old, new) == b, profile
        normalized = a.replace(old, "<SOURCE_IDENTITY>")
        profiles[profile] = {
            "previous_identity": old,
            "current_identity": new,
            "previous_source_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "current_source_sha256": hashlib.sha256(b.encode()).hexdigest(),
            "normalized_source_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "only_changed_literal": "native binding source_identity",
        }
changed = []
current = source_hashes(
    "common", "integral", "xc", "dft", "method", "tensor", assets=STATIONARY_AOT_ASSETS
)
with tarfile.open(e / "qualified-matrix-source.tar.gz", "r:gz") as archive:
    for name, digest in current.items():
        original = hashlib.sha256(archive.extractfile(name).read()).hexdigest()
        if original != digest:
            changed.append(
                {"path": name, "previous_sha256": original, "current_sha256": digest}
            )
assert [r["path"] for r in changed] == [
    "python/generativeqc_compiler/method/rhf_integral_ad.py"
], changed
payload = {
    "schema": "generativeqc.native-automatic-reference-source-review.v1",
    "scope": "independent FD reuse only; no cache hit or unchanged artifact claim",
    "reason": "stationary contract hashes all method compiler inputs; upstream RHF integral AD change invalidated force artifact identity while emitted source changed only its identity literal",
    "compiler_closure_inputs_checked": len(current),
    "changed_inputs": changed,
    "profiles": profiles,
}
(e / "reference-reuse-source-review.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(len(profiles), "profiles checked;", len(changed), "compiler input changed")
