"""Authenticate the actual endpoint source and loaded core before GPU work."""

import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

from tools.generate_build_identity import _inventory, _source_identity

root = Path.cwd()
identity = _source_identity(root, _inventory(root, root / "cmake/GenerativeQCSourceIdentity.json"))
expected = json.loads(Path(sys.argv[2]).read_text())
assert identity == Path(sys.argv[1]).read_text().strip() == expected["source_identity"]
library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == identity
digest = hashlib.file_digest(Path(os.environ["GENERATIVEQC_LIBRARY"]).open("rb"), "sha256").hexdigest()
assert digest == expected["library_sha256"]
expected["scheduler"] = {name: os.environ.get(name) for name in (
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "LSB_JOBID", "LSB_QUEUE", "LSB_HOSTS", "CUDA_VISIBLE_DEVICES"
)}
Path(sys.argv[3]).write_text(json.dumps(expected, indent=2) + "\n")
