"""One-off cross-revision comparison; do not pin generated production hashes in CI."""

import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "python"), str(root)]
# Import from the selected checkout only after installing its source paths.
from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider  # noqa: E402
from generativeqc_compiler.xc.automatic_semilocal import automatic_registry_entries  # noqa: E402

rows = {}
start = time.perf_counter()
for entry in automatic_registry_entries():
    p = emit_automatic_cuda_provider(entry.name)
    rows[entry.name] = {
        key: {
            "source_sha256": hashlib.sha256(point.source.encode()).hexdigest(),
            "source_bytes": len(point.source.encode()),
            "expression_identity": point.expression_identity,
            "contract": {
                k: v
                for k, v in point.mathematical_contract.items()
                if k != "expression_identity"
            },
        }
        for key, point in [("polarized", p.point), ("restricted", p.restricted)]
    }
Path(sys.argv[2]).write_text(
    json.dumps({"seconds": time.perf_counter() - start, "rows": rows}, indent=2) + "\n"
)
