"""Print the checked qualification without a GPU or writing output files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.generativeqc_validation.xc_diagonal_evidence import summarize

if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, allow_nan=False))
