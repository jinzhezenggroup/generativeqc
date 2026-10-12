"""CC owners must import with their documented runtime dependencies."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cc_response_import_does_not_require_typing_extensions() -> None:
    code = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "python"), str(root)]
sys.modules["typing_extensions"] = None
from tools.generativeqc_cc import (
    PreparedCUDALambda,
    CudaTriplesResponseTiles,
    PreparedUCCSD,
    PreparedUCCSDLambda,
    PreparedUCCSDT,
)
assert PreparedCUDALambda.__module__ == "tools.generativeqc_cc.lambda_cuda"
assert CudaTriplesResponseTiles.__module__ == "tools.generativeqc_cc.triples_response_cuda"
assert PreparedUCCSD.__module__ == "tools.generativeqc_cc.uccsd_solver"
assert PreparedUCCSDLambda.__module__ == "tools.generativeqc_cc.uccsd_lambda"
assert PreparedUCCSDT.__module__ == "tools.generativeqc_cc.uccsd_t"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
